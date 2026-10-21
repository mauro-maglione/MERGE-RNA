def forward_model(reads, N_reads, logweights, logweights_0, mutation_probs, mask = None):
    """Compute loss and derivatives approximately using ensemble of suboptimal structures.
    Masked positions (mask False) are unobserved: they contribute a factor 1 to every read."""

    # I'm assuming that reads comes in a form that has a the mutated sites -
    # - in reads and the number of reads with these mutations in N_reads
    if mask is None:
        mask = np.ones(mutation_probs.shape[1], bool)
    observed_sites = np.where(mask)[0]

    normalization = np.sum(np.exp(logweights))
    log_normalization = np.log(normalization)

    # compute the forward model for the mutation probabilities
    loss = 0.
    grad = np.zeros(len(logweights),float)

    for r in range(len(reads)):
        read_likelihood = 0.
        der_read_likelihood = np.zeros(len(logweights),float)
        for s in range(mutation_probs.shape[0]):
            mut_prob_per_struct = 1.
            for mut_site in reads[r]:
                if mask[mut_site]:
                    mut_prob_per_struct *= mutation_probs[s,mut_site]

            for non_mut_site in observed_sites - set(reads[r]):
                mut_prob_per_struct *= (1-mutation_probs[s,non_mut_site])

            read_likelihood += np.exp(logweights[s] - log_normalization) * mut_prob_per_struct   

            der_read_likelihood[s] += mut_prob_per_struct * np.exp(logweights_0[s]-log_normalization)
            der_read_likelihood += - mut_prob_per_struct * np.exp(logweights_0 + logweights[s] - 2*log_normalization) 

        # i'm considering a negative log likelihood

        loss += - N_reads[r]*np.log(read_likelihood)
        grad += - N_reads[r]*der_read_likelihood / read_likelihood
            
    return loss, grad