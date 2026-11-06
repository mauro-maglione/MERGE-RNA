# `com_maxent.py`: matrix formulation and correctness check

This file writes the algorithms in [`com_maxent.py`](com_maxent.py) as matrix operations and checks each one against the math.
Status as of 2026-11-06. The loss, the gradient and the chunked version were checked by finite differences on random toy data (S=7 structures, N=12 sites, R=40 reads, 3 masked sites).

---

## 0. Notation

| Symbol | Shape | Code | Meaning |
|---|---|---|---|
| $N$ | – | `len(seq)` | sequence length |
| $\mathcal O\subseteq\{1..N\}$, $N_o=\lvert\mathcal O\rvert$ | – | `position_mask` | observed (fitted) sites |
| $S$ | – | `states.shape[0]` | number of structures in the truncated ensemble |
| $R$ | – | `len(reads)` | number of **unique** reads |
| $\Sigma\in\{0,1\}^{S\times N}$ | $S\times N$ | `pairing_states` | $\Sigma_{si}=1$ if site $i$ is paired in structure $s$ |
| $a,b$ | $N$ | `a_i`, `b_i` | affine mutation model $M_i = a_i + b_i\,\sigma_i$ |
| $X\in\{0,1\}^{R\times N_o}$ | sparse | `X` | $X_{ri}=1$ if read $r$ is mutated at observed site $i$ |
| $n\in\mathbb N^R$ | $R$ | `N_reads` | multiplicity of each unique read |
| $\ell^0$ | $S$ | `logweights` (global) / `logweights_0` (arg) | $\log P_0(s)$, Boltzmann log-weights normalized over the truncated set |
| $\theta=\log W$ | $S$ | `logweights` (arg of `forward_model`), `res.x` | log of the reweighting factors (optimizer variable) |

Beware: the global `logweights` is $\ell^0$, but inside `forward_model` the argument named `logweights` is $\theta=\log W$.

---

## 1. Prior ensemble: `structures_sampling`, `structures_sampling_random`

**Enumeration.** `structures_sampling` lists every structure within $\Delta$ kcal/mol of the MFE under the energy

$$E(s) = E_{\text{Turner}}(s) + \sum_i (\text{penalty} + \lambda^{sc}_i)\,\Sigma_{si}.$$

It returns

$$\ell^0_s = -\beta E(s) - \log\sum_{s'\in\text{set}} e^{-\beta E(s')}, \qquad \textstyle\sum_s e^{\ell^0_s}=1,$$

which is the Boltzmann distribution **restricted to the truncated set and renormalized** over it.

- ✅ The convention matches merge_rna's: in bussi, `lambdas1d[i]` adds $\lambda_i+\lambda_j$ to pair $(i,j)$, and merge_rna's `sc_add_bp(i,j,λ_i)` loop does the same. For $\lambda<0$, merge_rna uses `sc_add_up(-λ)` instead. That differs only by a constant energy shift, so the distribution is the same.
- ⚠️ The temperature is hard-coded to its default of 37 °C, not taken from `exp.temp_C`. This is fine for the current data, which is at 37 °C, but it is not tied to the experiment.
- ⚠️ The prior includes the phase-1 $\lambda^{sc}$ (`np.loadtxt(phase1_params)[8:]`). That is a modelling choice: KL is then measured relative to the λ-biased ensemble, not plain Vienna.

**Sampling.** `structures_sampling_random` draws $K$ Boltzmann samples and deduplicates them. bussi's `sample_to_numpy` then returns $\ell^0_s=\log(c_s/K)$, where $c_s$ is the number of times structure $s$ was drawn. The prior is the **empirical** distribution, not the Boltzmann weights of the unique structures. Both choices are valid, but they are different priors.

- ❌ The reported coverage is computed **before** deduplication:
  $\text{cov}=\sum_{k=1}^{K} e^{-\beta(E_k - F)}$.
  Each structure is counted once per draw, so this overestimates the partition-function coverage and can exceed 1. Compute it over the unique structures instead:
  ```python
  uniq = list(set(structures))
  energies = mol.evaluate(uniq)
  log_coverage = logsumexp(-(energies - mol.total_free_energy()) / (bussi._KB * (temperature_C + 273.15)))
  ```

`possible_structures_number` $=2.1614\,N^{-3/2}\,1.96798^N$ is an asymptotic count for a homopolymer with no pairing rules. For a real sequence it is a large overestimate, so `coverage_of_structures` is only an order-of-magnitude indicator.

---

## 2. Mutation-probability matrix: `forward_model_fixed_params`

$$\mathbf P = \mathbf 1_S a^\top + \Sigma\,\mathrm{diag}(b) \quad(\text{then masked columns}\to 0), \qquad P_{si}=a_i+b_i\Sigma_{si}.$$

✅ Correct. The masked columns are zeroed but never used, because §3 keeps only the columns in $\mathcal O$.

---

## 3. Read likelihood under one structure: `precompute_read_factors`, `precompute_log_read_likelihood`

Assume sites mutate independently given the structure, and restrict to observed sites:

$$M_{rs} = P(\text{read } r\mid s) = \prod_{i\in\mathcal O} P_{si}^{X_{ri}}(1-P_{si})^{1-X_{ri}}.$$

Take logs and split the sum into mutated and unmutated sites:

$$\log M_{rs} = \sum_{i\in\mathcal O} X_{ri}\underbrace{\big[\log P_{si}-\log(1-P_{si})\big]}_{D_{si}} + \underbrace{\sum_{i\in\mathcal O}\log(1-P_{si})}_{c_s}.$$

In matrix form:

$$\boxed{\log\mathbf M = X D^\top + \mathbf 1_R c^\top}\qquad X: R\times N_o \text{ (sparse)},\; D: S\times N_o,\; c: S.$$

✅ Checked against a brute-force product over sites (max error $4\times10^{-15}$). Mutations at masked sites are dropped. Those sites therefore contribute a factor 1, which is the correct marginalization over unobserved sites.

- Note: $0<P_{si}<1$ is required. This holds as long as $a_i>0$ and $a_i+b_i<1$ on observed sites.
- Note: after masking, two different raw reads can map to the same row of $X$. They stay separate rows, which is harmless.

---

## 4. Reweighted ensemble and loss: `forward_model`, `_loss_and_ratio_sum`

**Parametrization.**

$$p_s = \frac{P_0(s)\,W_s}{Z},\qquad Z=\sum_s P_0(s)W_s,\qquad \log p = \ell^0+\theta-\log Z.$$

**Mixture likelihood of each read.**

$$L_r = \sum_s p_s M_{rs},\qquad \log L = \operatorname{logsumexp}_s(\log\mathbf M + \mathbf 1_R \log p^\top)$$

**Loss.** This is the binomial/multinomial NLL plus α·KL to a reference ensemble $q$ (by default $q = P_0$):

$$\mathcal L(\theta) = -\,n^\top\log L \;+\;\alpha\,\mathrm{KL}(p\Vert q),$$

$$\mathrm{KL}(p\Vert q)=\sum_s p_s(\log p_s-\log q_s) = p^\top(\theta+\ell^0-\log q)-\log Z.$$

✅ Loss value: matches a direct evaluation exactly (error 0) for α = 0 and α = 0.5. The code's `kl` line is exactly the last form above.

**Gradient with respect to $W$.** Define the ratio vector

$$\rho = \big(\mathbf M\oslash L\mathbf 1_S^\top\big)^\top n,\qquad \rho_j=\sum_r n_r \frac{M_{rj}}{L_r}\qquad(\texttt{ratio\_sum}).$$

Using $\partial p_s/\partial W_j = \frac{P_0(j)}{Z}(\delta_{sj}-p_s)$:

$$\frac{\partial \mathcal L}{\partial W_j} = -\frac{P_0(j)}{Z}\big(\rho_j - \textstyle\sum_r n_r\big) + \alpha\frac{P_0(j)}{Z}\Big(\log\frac{p_j}{q_j}-\mathrm{KL}\Big).$$

✅ This is exactly what `grad` in `forward_model` computes. The bug from the earlier review (`logweights_0` in the second term) is fixed, and so is the missing $1/Z$ normalization.

### ❌ Bug A: the gradient is with respect to $W$, but the optimizer variable is $\theta=\log W$

`minimize(obj, np.zeros(S))` optimizes $\theta$, and `res.x + logweights` is treated as $\log(P_0W)$. So the Jacobian passed to L-BFGS-B must be

$$\frac{\partial\mathcal L}{\partial\theta_j}=W_j\frac{\partial\mathcal L}{\partial W_j} = p_j\Big[\big(1-\tfrac{\rho_j}{\sum_r n_r}\big)\textstyle\sum_r n_r + \alpha\big(\log\tfrac{p_j}{q_j}-\mathrm{KL}\big)\Big].$$

Finite-difference check on toy data:

| | α = 0 | α = 0.5 |
|---|---|---|
| ‖`grad` − ∂L/∂θ‖∞ (as written) | **26.6** | **26.8** |
| ‖`grad`·W − ∂L/∂θ‖∞ | 1.6e-7 | 2.1e-7 |

At the starting point θ = 0, $W=1$ and the gradient happens to be right. Once the optimizer moves away from it, L-BFGS-B receives an inconsistent gradient. Expect line-search failures (`ABNORMAL_TERMINATION_IN_LNSRCH`) or premature stops. The code comment "multiply by W = exp(logweights − logweights_0)" is left over from an older parametrization; in the current one, $W=e^{\theta}$.

Fix (it is cleaner to write it directly in $p$):
```python
log_p = logweights + logweights_0 - log_Z
p = np.exp(log_p)
grad = -p * (ratio_sum - np.sum(N_reads))
if alpha > 0:
    grad += alpha * p * (log_p - logweights_ref - kl)
```
Sanity check: $\sum_j \partial\mathcal L/\partial\theta_j = 0$. This holds because $\sum_j p_j\rho_j=\sum_r n_r$ and $\sum_j p_j(\log p_j/q_j - \mathrm{KL})=0$. It reflects the gauge freedom θ → θ + const.

---

## 5. Chunked version: `forward_model_chunked`

Intended to be identical to §4, with $\log\mathbf M$ rebuilt $B$ rows at a time:

$$\log L_{[r_0:r_1]} = \operatorname{logsumexp}_s\big(X_{[r_0:r_1]}D^\top + \mathbf 1c^\top + \mathbf 1\log p^\top\big),\qquad \rho=\sum_{\text{chunks}}\rho_{\text{chunk}}.$$

The chunking is correct, because both $n^\top\log L$ and $\rho$ are sums over reads.

### ❌ Bug B: the normalization omits the prior

```python
log_Z = logsumexp(logweights)          # should be logsumexp(logweights + logweights_0)
log_p = logweights - log_Z             # should be logweights + logweights_0 - log_Z
```
So in the likelihood term the chunked version uses $p\propto W$ instead of $p\propto P_0W$. It is also inconsistent with itself: the KL and gradient lines below it do add `logweights_0`, but they use this wrong `log_Z`. On toy data, chunked vs. `forward_model` differs by ΔL = 4.7 and Δgrad = 28. The script takes this path automatically when $S\ge 2\times10^5$ (e.g. Δ = 10 kcal/mol). It also needs the Bug A fix.

---

## 6. Post-processing (fitted ensemble)

Define $\ell^{\rm fit}=\theta^*+\ell^0$ and $F=\operatorname{logsumexp}(\ell^{\rm fit})$, so that $p=e^{\ell^{\rm fit}-F}$.

| Quantity | Matrix form | Code | Status |
|---|---|---|---|
| Pairing probabilities | $\langle\sigma\rangle = \Sigma^\top p$ | `pairing_probs` | ✅ |
| Prior pairing probabilities | $\Sigma^\top e^{\ell^0}$ | `pairing_probs_0` | ✅ (truncated ensemble) |
| Second moment | $C=\Sigma^\top\mathrm{diag}(p)\,\Sigma$ | `correlation_matrix` (Python loop over $S$) | ✅. Vectorize as `pairing_states.T @ (p[:,None]*pairing_states)` |
| Covariance | $C-\langle\sigma\rangle\langle\sigma\rangle^\top$ | `unconn_correl_matrix` | ✅ value, ⚠️ name: this is the **connected** correlation (covariance) |
| KL to the prior | $p^\top(\log p-\ell^0)$ | `d_kl` | ❌ **Bug C** |

**Bug C.** The script computes
`d_kl = exp(ℓfit + ℓ0 − F) @ (ℓfit − F)` $=\sum_s P_0(s)\,p_s\log p_s$.
That is neither a probability-weighted average nor a log-ratio. On toy data it gives −0.278 where the true value is 0.385; a KL divergence can never be negative. Fix:
```python
log_p = logweights_fitted - free_en
d_kl = np.exp(log_p) @ (log_p - logweights)
```

---

## 7. "True-structures" test block

`mol_0` biases structures with state biases to build `good_states`. It then calls `vienna_logweights_of_good = mol_00.evaluate(good_states_raw)`.

### ❌ Bug D: `evaluate` returns free energies (kcal/mol), not log-weights

`bussi.Molecule.evaluate` returns $E(s)$. Passing it as $\ell^0$ uses $P_0(s)\propto e^{+E(s)}$: the wrong sign, the wrong scale (missing $\beta$), and not normalized. Fix:
```python
E = mol_00.evaluate(good_states)          # or good_states_raw
lw = -E / (bussi._KB * (37 + 273.15))
vienna_logweights_of_good = lw - logsumexp(lw)
```
`forward_model` normalizes via $Z$, so normalization is optional for the loss. The sign and the β factor are not.

---

## 8. Minor points

- **Test inputs (lines 309–317).** `test_logweights` is built as $\ell^0$ + noise and normalized, which is a $\log(P_0W)$. It is then passed as θ, so $P_0$ is applied twice. This only affects the smoke test.
- **Optimizer.** The objective is invariant under θ → θ + const, so the Hessian has a zero mode. L-BFGS-B tolerates this. If you want, you can pin it, for example by adding $\tfrac{\epsilon}{2}(\operatorname{logsumexp}(\theta+\ell^0))^2$.
- **Memory.** `logM` is computed twice (lines 306 and 322), and `X, D, base` twice (lines 307 and 325). This is harmless but redundant.
- **Truncation.** The previous review found that the true synthetic ensemble is poorly covered at Δ = 4 kcal/mol: only ~25% of the true weight lies inside the prior's window. Reweighting cannot put mass on structures that are not in the set.

---

## Summary

| # | Location | Status |
|---|---|---|
| – | $\log\mathbf M = XD^\top+\mathbf 1c^\top$, masking | ✅ correct |
| – | `forward_model` loss value, KL, $\partial/\partial W$ | ✅ correct |
| A | `forward_model` gradient handed to L-BFGS-B on θ = log W | ❌ missing factor $W$ |
| B | `forward_model_chunked` normalization | ❌ omits $P_0$ in `log_Z`, `log_p` |
| C | `d_kl` post-processing | ❌ wrong formula (can be negative) |
| D | `vienna_logweights_of_good` | ❌ energies used as log-weights |
| – | random-sampling coverage | ⚠️ counts duplicate draws |
