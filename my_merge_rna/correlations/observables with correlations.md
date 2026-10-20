We assume that the frozen dynamic regime of [[accurate_reaction_modelling_formatted]] holds (see [[info on RNA dynamics]]).

Convention: $s_i=1$ means nucleotide $i$ is paired, $s_i=0$ unpaired (same as mergeRNA).

## 1. Reaction profile given the structure

The probability that a molecule shows a certain reaction profile $d$ ($d_{i}=1$ means a reaction with DMS happened at the $i$-th nucleotide) is
$$
	P(d,t \mid s) = \prod_{i:d_{i}=0} e^{-k_{i}\,t\,(1-s_{i})}\prod_{j:d_{j}=1} \left(1-e^{-k_{j}\,t\,(1-s_{j})}\right)
$$
If $d_j=1$ and $s_j=1$ for some $j$, then $P(d,t\mid s)=0$: a paired site cannot react. A paired site with $d_j=0$ contributes a factor $1$.

Conditioned on the initial (frozen) structure $s$, the $d_i$ are independent, non-identical Bernoulli trials with
$$
	P(d_i=1\mid s_i) = (1-u_i)(1-s_i),\qquad u_i \equiv e^{-k_i t}.
$$
The number of reacted sites $\sum_i d_i$ is then [Poisson-binomial](https://en.wikipedia.org/wiki/Poisson_binomial_distribution) distributed.

## 2. Averaging over structures

Averaging over the structures couples the Bernoulli trials, since the pairing states of different nucleotides are not independent:
$$
	P(d,t ) =\sum_{s} P^{\text{eq}}(s)\, P(d,t \mid s) = \sum_{s} P^{\text{eq}}(s)  \prod_{i:d_{i}=0} e^{-k_{i}\,t\,(1-s_{i})}\prod_{j:d_{j}=1} \left(1-e^{-k_{j}\,t\,(1-s_{j})}\right)
$$

*Partition-function form.* Using $e^{-k_it(1-s_i)} = s_i + (1-s_i)\,u_i$, every factor is a per-nucleotide weight that depends only on whether $i$ is paired:
$$
	w_i(d_i,s_i)=
	\begin{cases}
	d_i=0: & 1 \text{ if paired},\quad u_i \text{ if unpaired} \\
	d_i=1: & 0 \text{ if paired},\quad 1-u_i \text{ if unpaired}
	\end{cases}
$$
Hence
$$
	P(d,t) = \frac{Z(d)}{Z},\qquad Z(d)=\sum_s e^{-\beta F(s)}\prod_i w_i(d_i,s_i).
$$
Here $Z(d)$ is a partition function with a soft unpaired pseudo-energy $-k_BT\ln u_i = k_BT\,k_it$ on the $d_i=0$ sites and a hard "unpaired" constraint (times the constant $1-u_i$) on the $d_i=1$ sites. For the $d_j=0$ sites,
$$
	\frac{\partial \ln P(d,t)}{\partial k_j} = -t\,\langle 1-s_j\rangle_{d},
$$
where $\langle\cdot\rangle_d$ is the average in the fictitious ensemble defined by $Z(d)$.

## 3. The factorized (mergeRNA-like) assumption

Read as a model for whole reads, mergeRNA treats the sites as independent, each averaged with its own marginal:
$$
	P^{\text{merge}}(d,t ) =  
	\prod_{i : d_{i}=0} \; \left[\sum_{s_i} P^{\text{eq}}(s_i)\,  e^{-k_{i}\,t\,(1-s_{i})}\right]  
	\prod_{j:d_{j}=1} \left[\sum_{s_j}  P^{\text{eq}}(s_j) \left(1-e^{-k_{j}\,t\,(1-s_{j})}\right) \right]
$$
With $\langle s_i\rangle = P^{\rm eq}(s_i=1)$ this is
$$
	P^{\text{merge}}(d,t) =  
	\prod_{i} \left[ e^{-k_{i}\,t} +d_{i} \left(1-2e^{-k_{i}\,t} \right)+ 
	(1-2d_{i})\left( 1- e^{-k_{i}\,t} \right) \langle s_{i} \rangle
	\right]
$$
Check:
- $d_i=0$ gives $e^{-k_it}+(1-e^{-k_it})\langle s_i\rangle$.
- $d_i=1$ gives $(1-e^{-k_it})(1-\langle s_i\rangle)$.
- The two sum to 1.

## 4. Sequencing errors

Both expressions then go through the further averaging that accounts for errors in the sequencing procedure. Call $m$ the mutational profile and define $\epsilon_i(d_{i})$ through $P(m_i=1\mid d_i)=1-e^{-\epsilon_i(d_i)}$. Here $\epsilon_i(0)$ controls false positives and $\epsilon_i(1)$ controls false negatives (false-negative probability $e^{-\epsilon_i(1)}$), analogous to the quantities in mergeRNA. The probability of a mutational profile after an experimental time $T$ is
$$
	P(m)  = \sum_{d} P(d,T)  \prod_{i:m_{i}=0} e^{-\epsilon_{i}(d_{i})}\prod_{j:m_{j}=1} \left( 1-e^{-\epsilon_{j}(d_{j})} \right)
$$

Notation for what follows:
$$
	u_i \equiv e^{-k_iT},\qquad E^0_i\equiv e^{-\epsilon_i(0)},\qquad E^1_i\equiv e^{-\epsilon_i(1)}.
$$

## 5. Mutation probability given the structure

Insert $P(d,T)=\sum_s P^{\rm eq}(s)P(d,T\mid s)$ and exchange the sums. Given $s$, the $d_i$ are independent (§1), and given $d$, the $m_i$ are independent. So the sum over $d$ factorizes site by site:
$$
	P(m\mid s) = \prod_i r_i(m_i\mid s_i),\qquad
	r_i(m_i\mid s_i) = \sum_{d_i\in\{0,1\}} P(d_i\mid s_i)\,P(m_i\mid d_i).
$$
For $m_i=1$:
$$
\begin{aligned}
	r_i(1\mid s_i) &= (1-E^0_i)\,\big[1-(1-u_i)(1-s_i)\big] + (1-E^1_i)\,(1-u_i)(1-s_i)\\
	&= (1-E^0_i) + (E^0_i-E^1_i)(1-u_i)(1-s_i).
\end{aligned}
$$
Since $s_i\in\{0,1\}$, this is exactly linear in $s_i$:
$$
	\boxed{\;r_i(1\mid s_i) = a_i + b_i\,s_i,\qquad
	a_i = (1-E^0_i) + (E^0_i-E^1_i)(1-u_i),\qquad
	b_i = -(E^0_i-E^1_i)(1-u_i)\;}
$$
and $r_i(0\mid s_i)=1-a_i-b_i s_i$.

Limits:
- Paired, $s_i=1$: $a_i+b_i = 1-E^0_i$, the pure false-positive probability.
- Unpaired, $s_i=0$: $a_i = 1-E^0_i+(E^0_i-E^1_i)(1-u_i)$. This tends to $1-E^1_i$ as $T\to\infty$.
- $b_i<0$ whenever $E^0_i>E^1_i$, i.e. when a reacted site is more likely to read as mutated than an unreacted one, so pairing lowers the mutation probability.

## 6. Averaging over structures: full profile and marginals

Write the single-site factor as
$$
	r_i(m_i\mid s_i)=\alpha_i(m_i)+\beta_i(m_i)\,s_i,\qquad
	\alpha_i(1)=a_i,\ \beta_i(1)=b_i,\quad \alpha_i(0)=1-a_i,\ \beta_i(0)=-b_i .
$$
Then
$$
	P(m) = \Big\langle \prod_i \big[\alpha_i(m_i)+\beta_i(m_i)\,s_i\big]\Big\rangle_{P^{\rm eq}}
	= \sum_{S\subseteq\{1,\dots,L\}}\ \prod_{i\notin S}\alpha_i(m_i)\prod_{i\in S}\beta_i(m_i)\ \Big\langle \prod_{i\in S}s_i\Big\rangle .
$$
So $P(m)$ depends on pairing correlators of all orders.

As in §2, it is also a ratio of partition functions, $P(m)=Z(m)/Z$, with per-nucleotide weights
- paired: $\alpha_i(m_i)+\beta_i(m_i)$
- unpaired: $\alpha_i(m_i)$

These are soft constraints (hard only if a weight vanishes, e.g. $E^0_i=1$ with $m_i=1$). One constrained partition function per read therefore gives the exact read likelihood.

**Marginal mutation probability:**
$$
	P(m_i=1) = a_i + b_i\,\langle s_i\rangle .
$$

**Pair correlations:**
$$
	P(m_i{=}1,m_j{=}1)-P(m_i{=}1)P(m_j{=}1) = b_i b_j\big(\langle s_is_j\rangle-\langle s_i\rangle\langle s_j\rangle\big).
$$
With $b_i,b_j<0$ the mutation covariance has the sign of the pairing covariance $\mathrm{Cov}(s_i,s_j)$.

## 7. Comparison with mergeRNA

### 7.1 The mergeRNA model

mergeRNA (Sacco et al., arXiv:2512.20581, SI eqs. 16–27) chains three steps.

1. **Physical binding**, Langmuir/Boltzmann:
$$
	P(k_i=1\mid s_i=0)=\sigma(\beta\mu),\qquad P(k_i=1\mid s_i=1)=\sigma(\beta\mu'),
$$
	with $\sigma(x)=\frac{e^x}{1+e^x}$, $\mu'=\mu-\Delta\mu_{\rm pairing}$ and $\mu=\mu_r+k_BT\ln([{\rm DMS}]/[{\rm DMS}]_r)$.
2. **Chemical modification:**
$$
	P(c_i=1\mid s_i,n_i)=p_{\rm bind}(n_i,s_i)\,P(k_i=1\mid s_i).
$$
	For DMS, $p_{\rm bind}(A,1)=p_{\rm bind}(C,1)=0$.
3. **Mutational profiling:**
$$
	P(\text{mut}_i\mid c_i)=1-e^{-(m(c_i)+\epsilon^{\rm M}_i)},
$$
	where $m_0, m_1$ are global false-positive/false-negative parameters and $\epsilon^{\rm M}_i$ is a per-site background fitted on the no-reagent control. (Superscript M to avoid a clash with my $\epsilon_i$.)

Define $q^u_i\equiv P(c_i=1\mid s_i=0)$ and $q^p_i\equiv P(c_i=1\mid s_i=1)$. The mutation probability given the pairing state is then
$$
	M_i(s_i) = \sum_{c_i}P(c_i\mid s_i)\,P(\text{mut}_i\mid c_i)
	= 1-e^{-(m_0+\epsilon^{\rm M}_i)} + \big(e^{-(m_0+\epsilon^{\rm M}_i)}-e^{-(m_1+\epsilon^{\rm M}_i)}\big)\big[q^u_i+(q^p_i-q^u_i)\,s_i\big],
$$
so $M_i(s_i)=a^{\rm M}_i+b^{\rm M}_i s_i$ with
$$
	a^{\rm M}_i = 1-e^{-(m_0+\epsilon^{\rm M}_i)} + \big(e^{-(m_0+\epsilon^{\rm M}_i)}-e^{-(m_1+\epsilon^{\rm M}_i)}\big)\,q^u_i,
	\qquad
	b^{\rm M}_i = -\big(e^{-(m_0+\epsilon^{\rm M}_i)}-e^{-(m_1+\epsilon^{\rm M}_i)}\big)\,(q^u_i-q^p_i).
$$
Ensemble averaging (their eq. 20, then eq. 27) gives
$$
	M_i = a^{\rm M}_i + b^{\rm M}_i\,\langle s_i\rangle .
$$

### 7.2 Dictionary

| this note | mergeRNA |
|---|---|
| $s_i$ ($1$ = paired) | $s_i$ ($1$ = paired) |
| $d_i$ (reacted) | $c_i$ (chemically modified) |
| $P(d_i=1\mid s_i=0)=1-e^{-k_iT}$ | $q^u_i = p_{\rm bind}(n_i,0)\,\sigma(\beta\mu)$ |
| $P(d_i=1\mid s_i=1)=0$ | $q^p_i = p_{\rm bind}(n_i,1)\,\sigma(\beta\mu')$ |
| $\epsilon_i(0)$ | $m_0+\epsilon^{\rm M}_i$ |
| $\epsilon_i(1)$ | $m_1+\epsilon^{\rm M}_i$ |
| $r_i(1\mid s_i)=a_i+b_is_i$ | $M_i(s_i)=a^{\rm M}_i+b^{\rm M}_is_i$ |

With $q^u_i=1-u_i$, $q^p_i=0$ and the $\epsilon$ identification above, $a_i=a^{\rm M}_i$ and $b_i=b^{\rm M}_i$ exactly. The error model has the same functional form in both: $P(\text{mut}\mid\cdot)=1-e^{-(\cdot)}$, with false negatives $\propto e^{-\epsilon_i(1)}$ resp. $e^{-(m_1+\epsilon^{\rm M}_i)}$.

### 7.3 Differences at the single-site level

1. **Reaction physics.**
   - This note: kinetic, a first-order irreversible reaction during the probing time on a frozen structure, giving $1-e^{-k_iT}$, with $k_i\propto[\rm DMS]$ (pseudo-first-order).
   - mergeRNA: equilibrium binding occupancy times a conversion probability, $p_{\rm bind}\,\sigma(\beta\mu)$.
   - Both saturate, the first at $1$ and the second at $p_{\rm bind}$, and both are linear in $[\rm DMS]$ at low dose: $k_iT$ versus $p_{\rm bind}\,e^{\beta\mu_r}[{\rm DMS}]/[{\rm DMS}]_r$. The probing time enters explicitly only here.
2. **Paired reactivity.**
   - This note: zero for every paired nucleotide.
   - mergeRNA: zero only for A, C. For G, U, $q^p_i\neq0$.
   - This note can be generalized by giving paired sites their own rate, $k_i\,t\,(1-s_i)\to t\,[k^u_i(1-s_i)+k^p_is_i]$. This gives $q^u_i=1-e^{-k^u_iT}$, $q^p_i=1-e^{-k^p_iT}$, and all results above hold with $(1-u_i)\to(q^u_i-q^p_i)$ in $b_i$.
3. **Error parametrization.**
   - This note: $\epsilon_i(0),\epsilon_i(1)$ are free per site.
   - mergeRNA: additive, $m_d+\epsilon^{\rm M}_i$, so the ratio $E^1_i/E^0_i=e^{-(m_1-m_0)}$ is the same at every site.
4. **Ensemble.**
   - mergeRNA's $P(s)$ includes the probe-induced perturbation $\Delta F_{[\rm DMS]}(s)$.
   - In the frozen regime, $P^{\rm eq}$ here is the ensemble at the start of probing, so the same $\Delta F_{[\rm DMS]}$ can be included in it if binding equilibrates before reaction.

### 7.4 Differences at the read level

The per-site marginals coincide:
$$
	P(m_i=1)=a_i+b_i\langle s_i\rangle = M_i .
$$
mergeRNA's step $\langle P(c_i\mid s)\rangle_s=\langle P(c_i\mid s_i)\rangle_{s_i}$ (their eq. 20) is exact for single-site quantities, so the two models agree on every per-site mutation rate and are indistinguishable from per-position mutation counts alone.

They differ only in the joint distribution of a read. With $F_m(s)=\prod_i[\alpha_i(m_i)+\beta_i(m_i)s_i]$:
$$
	P(m)=\langle F_m(s)\rangle \qquad\text{vs.}\qquad P^{\rm merge}(m)=F_m(\langle s\rangle)=\prod_i\big[\alpha_i(m_i)+\beta_i(m_i)\langle s_i\rangle\big].
$$
In the expansion of §6, the factorized model replaces every correlator $\langle\prod_{i\in S}s_i\rangle$ by $\prod_{i\in S}\langle s_i\rangle$. This is exact only if the pairing states are independent under $P^{\rm eq}$. In particular,
$$
	\mathrm{Cov}^{\rm merge}(m_i,m_j)=0
	\qquad\text{vs.}\qquad
	\mathrm{Cov}(m_i,m_j)=b_ib_j\,\mathrm{Cov}(s_i,s_j).
$$
The co-mutation signal is therefore exactly the pairing covariance, scaled by the per-site sensitivities $b_i$. These are the same coefficients that mergeRNA already fits from the marginals.
