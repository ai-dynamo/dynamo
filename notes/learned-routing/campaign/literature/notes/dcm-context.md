# DCM with context effects and variable choice sets: notes for the learned-routing campaign

Scout angle: dcm-context. Written 2026-10-02 for the campaign in
`<campaign-root>/CONTRACT.md` and `<repo>/notes/learned-routing/PLAN.md`.
PDFs: `/tmp/learned-routing-lit/pdfs/dcm-context/`. Citation keys below match file stems.

Labels used in this file:
- **[paper]**: a claim stated in the cited paper and section.
- **[derivation]**: follows mechanically from the model's algebra. The anchoring equation is cited, but the inference is mine.
- **[hypothesis]**: not verified. It needs a campaign measurement.
- **[code]**: read from the campaign worktree (`<worktree>`, read-only).

## 0. Reading list and depth

| Key | Paper | Depth |
|---|---|---|
| tomlinson2021-lcl-dlcl | Tomlinson & Benson, *Learning Interpretable Feature Context Effects in Discrete Choice*, KDD 2021, arXiv:2009.03417 | full |
| seshadri2019-cdm | Seshadri, Peysakhovich & Ugander, *Discovering Context Effects from Raw Choice Data*, ICML 2019, arXiv:1902.03266 | full (main text) |
| rosenfeld2020-set-dependent-aggregation | Rosenfeld, Oshiba & Singer, *Predicting Choice with Set-Dependent Aggregation*, ICML 2020, arXiv:1906.06365 | key sections (§1–4, App. D.4–D.5) |
| pfannschmidt2019-fate-feta | Pfannschmidt, Gupta, Haddenhorst & Hüllermeier, *Learning Context-Dependent Choice Functions*, IJAR 2022, arXiv:1901.10860 | key sections (§4–6) |
| aouad2023-rumnet | Aouad & Désir, *Representing Random Utility Choice Models with Neural Networks* (RUMnet), Mgmt Sci, arXiv:2207.12877 | key sections (§1, 2.3–2.4, 4.2, 6.3–6.4) |
| maragheh2018-halo-mnl | Yousefi Maragheh, Chronopoulou & Davis, *A Customer Choice Model with HALO Effect*, arXiv:1805.01603 | key sections (§1–3) |
| ko2024-choice-self-attention | Ko & Li, *Modeling Choice via Self-Attention*, arXiv:2311.07607 | key sections (§4–5) |
| zhang2026-deephalo | Zhang, Wang, Gao & Li, *DeepHalo: A Neural Choice Model with Controllable Context Effects*, arXiv:2601.04616 | key sections (§3, App.) |
| train2009-gev-ch4(.clean) | Train, *Discrete Choice Methods with Simulation* 2nd ed., Ch. 4 "GEV" (the `.clean.pdf` strips a MacBinary header from the original download) | key sections (§4.1–4.2.4) |
| seshadri2020-iia-testing-limits | Seshadri & Ugander, *Fundamental Limits of Testing the IIA in Discrete Choice*, arXiv:2001.07042 | abstract + §1 |
| webb2021-divisive-normalization | Webb, Glimcher & Louie, *The Normalization of Consumer Valuations*, Mgmt Sci 2021 (author PDF) | key sections (§2–3.2, 4.2) |
| jain2020-generalization-new-actions | Jain, Szot & Lim, *Generalization to New Actions in RL*, ICML 2020, arXiv:2011.01928 | key sections (§3–4.3, 6.2.2) |
| jain2022-agile-action-relations | Jain, Kosaka, Kim & Lim, *Know Your Action Set* (AGILE), ICLR 2022. Only the ICLR slides could be fetched; OpenReview blocked curl. | abstract + slides |
| mitzenmacher2000-old-information | Mitzenmacher, *How Useful Is Old Information?*, IEEE TPDS 2000 (SRC TN 1998-002) | key sections (abstract, §3–4, §6) |

Out of scope here, already in other scouts' folders: Decima, input-driven baselines, ARS, the CMA-ES tutorial
(`/tmp/learned-routing-lit/pdfs/learned-systems-policies/`).

## 1. How the campaign's model sits in the DCM taxonomy

The contract's utility is `u_i = θ·x_i + Σ_k (p_k·x_i)(q_k·x̄_S)`, with `x̄_S` the candidate mean. This equals
`x_iᵀ(θ + A x̄_S)` with `A = Σ_k p_k q_kᵀ` (rank r).

- **This is exactly the linear context logit (LCL)** of tomlinson2021 §3.1, eq. (4): `Pr(i,C) ∝ exp([θ + A x̄_C]ᵀ x_i)`.
  T&B obtain it from two assumptions:
  - the set effect decomposes additively over items;
  - each item's effect is diluted by `1/|C|` (§3, "we assume that the effect of each item is diluted in large choice
    sets").

  T&B also note that a low-rank `A` makes the parameter count linear in d (§3.1) [paper].
- **LCL is a "set-dependent weights" model.** The choice set shifts the coefficient vector:
  `θ(S) = θ + A x̄_S` (tomlinson2021 §3; rosenfeld2020 §2 eq. (1), SDW `g(x,s)=⟨w(s),F(x)⟩`) [paper].
- **CDM is the item-level, sum-pooled relative.** `u(x|C) = Σ_{z∈C\x} c_zᵀ t_x` (seshadri2019 §2.2 eq. (2)). Context
  vectors are summed over the set, not averaged [paper].
- **Halo MNL is item-level pairwise.** Pairwise halo terms are defined between product identities (maragheh2018 §3).
  Low-rank Halo MNL is `H = diag(α) + UVᵀ` (ko2024 §4 eq. (2)) [paper].
- **FETA and FATE** (pfannschmidt2019 §4):
  - FETA is a first-plus-pairwise expansion averaged with weight `1/(|Q|−1)`.
  - FATE scores each item against a mean-pooled set embedding.

  Both are designed to "generalize beyond the task sizes encountered in the training data" (§5) [paper].
- **DeepHalo's first interaction layer is the campaign's M2 with a linear φ.** The layer is
  `z_j¹ = z_j⁰ + (1/H) Σ_h Z̄_h¹ · φ_h(z_j⁰)`, with `Z̄¹ = (1/|S|) Σ_k W¹ z_k⁰` (zhang2026 §3.2) [paper]. With linear φ
  and H = r heads, this equals the contract's low-rank term [derivation]. Each further residual layer adds one
  interaction order in a controlled way (zhang2026 §3.2 eqs. (4)–(5)) [paper].
- **Nested logit and GEV.** Errors are correlated within nests; IIA holds within each nest (train2009 §4.2.1). The
  choice factors into a nest choice, driven by the inclusive value, times a within-nest logit (§4.2.3) [paper].

## 2. Per-paper notes

### 2.1 Tomlinson & Benson 2021 (LCL / DLCL), the closest match to M2

**Model**
- LCL has d²+d parameters. `A_pq` is the effect of the set-mean of feature q on the coefficient of feature p
  (§3.1) [paper].
- Diagonal `A_pp < 0` matches the similarity effect; `A_pp > 0` matches asymmetric dominance (§3.1) [paper].
- DLCL is a mixture of d logits. Component k carries only the context effect exerted by feature k:
  `B_k + A_k (x̄_C)_k`, with mixture weights π (§3.2 eq. (5)). This gives each effect its own intercept [paper].

**Identification (§4)**
- **Lemma 4.3.** The log-probability ratio is `β_{i,C} = (θ + A x̄_C)ᵀ(x_i − x̄_C)`. Choice depends only on
  within-set deviations `x_i − x̄_C`, weighted by the set-dependent coefficient [paper].
- **Theorem 4.1.** The model is identifiable iff the vectors `[x̄_C; 1] ⊗ (x_i − x̄_C)` span `R^{d²+d}` [paper].
- **Proposition 4.5.** It is not identifiable unless the data has d+1 choice sets with affinely independent mean
  feature vectors [paper].
- **Proposition 4.6.** A sufficient condition is d+1 sets with affinely independent means, each containing d+1 items
  with affinely independent features [paper].
- **§6.1.** LCL was not identifiable in 4 of 7 general datasets. L2 weight decay identifies it in all of them [paper].

**Estimation**
- **§5.1.** The LCL negative log-likelihood is convex in (θ, A), with a Lipschitz gradient. Gradient descent finds the
  global optimum [paper].
- **§5.2.** The DLCL NLL is not convex. EM reduces it to convex M-steps, and EM beat SGD on 18/22 datasets [paper].
- **§6, estimation details.**
  - Adam, batch 128, 500 epochs or 1 h.
  - Weight decay and learning rate grid-searched on validation likelihood.
  - Features standardized to zero mean and unit variance.

  [paper]
- **Footnote 3.** "the true LCL optimum would have better likelihood than MNL … but the complexity introduced by
  additional parameters means that LCL does not beat MNL within the 500 training epochs." On a finite optimizer
  budget, the larger nested model can lose to the smaller one [paper].

**Diagnostics and findings (§6.3)**
- **Binned-MNL diagnostic (Fig. 2).** They bin observations by set-mean feature, fit one MNL per bin, and plot the
  coefficient against the bin's set mean. They find:
  - clean linear dependence on a *log*-transformed set mean (mathoverflow: r² 0.61–0.86);
  - visibly non-linear dependence (email-enron);
  - none (synthetic-mnl).

  [paper]
- **Table 3.** Likelihood-ratio tests of LCL vs MNL are significant on all 13 real network datasets and on 4 of 7
  general datasets. They are not significant on synthetic-mnl (p = 1.0) [paper].
- **Table 4.** Held-out prediction gains are often modest: mean relative rank 6% lower on facebook-wall, 24% lower on
  bitcoin-otc. Some datasets show no significant gain [paper].
- **Tables 5–7.** Single-entry constrained LCL re-estimates each `A_pq` alone, with the rest of `A` set to 0. In
  sushi, the large entries of the full-`A` fit mostly shrink toward 0 when estimated alone, "a sure sign of null
  effects". Only oiliness-on-oiliness survives (p = 1.5e-6) [paper].
- **Choice-set confounding.** In expedia and car-alt, choice sets were built to match preferences. This produced
  large "context effects" that are really assignment bias (§6.3) [paper].
- **Fig. 3.** Under an L1 path on `A`, the context-free synthetic data drops `A` to 0 at once with no likelihood loss.
  Real effects persist until a likelihood jump [paper].
- **Fig. 4.** A t-SNE of learned `A` matrices clusters by domain: context effects are domain-specific [paper].

### 2.2 Seshadri, Peysakhovich & Ugander 2019 (CDM)

- **§2.1.** Universal (mother) logit gives every item a free utility in every set. Batsell–Polking (Lemma 1)
  expands it into 1st-order, pairwise, triple, … terms. MNL is order 1; CDM keeps order 2 [paper].
- **§2.2.** CDM has `n(n−1)−1` free parameters. Low-rank CDM uses target and context vectors `t_x, c_z ∈ R^r`, and the
  context is a *sum* over `C \ x` (eq. (2)) [paper].
- **Theorem 1.** CDM is identifiable if the data covers all sets of two sizes k and k′, at least one of which is not 2
  or n [paper].
- **Theorem 2.** No rank-r CDM is identifiable from choice sets of a single size [paper].
- **§3, §4.** Use L2 regularization to select the minimum-norm solution when the model is not identified [paper].
- **Theorem 3 and §3.1.** The full-rank CDM log-likelihood is log-concave, and the MLE error is O(d/m) [paper].
- **§3.2 and §4, Fig. 1.** The nested CDM-vs-MNL likelihood-ratio test of IIA is well calibrated (rejects slightly
  under 5% under the null) and powerful. A test against the universal logit is "highly anti-conservative" in finite
  samples [paper].
- **§4.**
  - CDM optimization "is initialized with values corresponding to a Luce MLE" (MNL warm start).
  - Low-rank CDMs beat the full-rank unfactorized CDM out of sample (SFwork/SFshop; nature-photo triplets).

  [paper]

### 2.3 Rosenfeld, Oshiba & Singer 2020 (set-dependent aggregation)

- **§1.** For any item-score model, argmax predictions "are clearly independent of s". Making the score function more
  complex cannot fix this. Set dependence must be built into the architecture [paper].
- **§2, eqs. (1)–(3).** SDW changes the scoring direction with the set; SDE moves items in embedding space. The
  inductive bias (§2.1 eq. (4)) is `ϕ(x,s) = µ(F(x) − r(F(s)))`:
  - r is a set reference point;
  - µ is an asymmetric s-shaped nonlinearity (a kinked tanh, App. D.4);
  - w and r are small mean-pooled set networks.

  [paper]
- **§4.2, Fig. 3 left.** Accuracy rises with aggregation dimension ℓ, but about 90% of the gain arrives by ℓ = 4
  [paper].
- **§4.2, violation capacity κ (eq. (8)).** κ is how often the prediction changes when one item is removed. It tracks
  accuracy, and SDA spends its violation budget mostly on examples MNL gets wrong [paper].
- **Table 1.** Deep Sets, the unconstrained set-function baseline, "over-utilizes its unconstrained capacity", shows
  high variance, and loses to the structured SDA [paper].

### 2.4 Pfannschmidt et al. 2019/2022 (FATE / FETA)

- **§4.1.** FETA averages pairwise sub-utilities with `1/(|Q|−1)` within each interaction order and sums across
  orders. The stated aim is to keep scores "in roughly the same scale, which is advantageous when the model is
  applied to choice tasks Q of varying size". It also contrasts this with CDM, which sums and imposes sum-to-zero
  constraints [paper].
- **§4.2.** FATE mean-pools a learned embedding into a set representative µ_Q and scores each item against it
  [paper].
- **§6.4.3, Fig. 8.** Models trained on sets of size 10 and tested on sizes 3–21 "generalize quite well". FETA-Net
  improves with size on Medoid, where the task becomes less context-dependent. On Hypervolume, baselines peak at size
  3 while FETA and FATE decay more slowly [paper].
- **§6.4.1.** SDA had the worst accuracy on LETOR and Expedia. The authors suspect it was trained on a fixed set size
  and evaluated on varying sizes, so its set-dependent aggregation "does not generalize well to the larger choice
  tasks" [paper].

### 2.5 Aouad & Désir (RUMnet)

- **§2.3.**
  - Argmax-based choice layers give gradients that are zero almost everywhere ("a small perturbation of the utility
    parameters cannot reverse the highest-utility alternative unless there are ties").
  - Adding Gumbel noise turns the argmax into a softmax and makes training possible.

  [paper]
- **§1, §4.2.** RUMnet is a mixture over K sampled utility networks. Its generalization bound does not grow with K and
  scales sub-quadratically in assortment size κ [paper].
- **§6.3, §6.4, Fig. 5.**
  - Model-free classifiers (random forests) overfit on Expedia, which has large and varying assortments.
  - Their error grows as input dimension grows with the number of products.
  - The authors attribute this to the difficulty of encoding variable-size assortments and argue that RUM structure
    helps generalization.

  [paper]

### 2.6 Halo MNL (Maragheh et al. 2018) and low-rank Halo / self-attention (Ko & Li 2024)

- **maragheh2018 §3–4.** Pairwise halo terms between product identities, with identifiability conditions [paper].
- **ko2024 §4, Prop. 1.** The full Halo MNL needs Ω(m²) samples for m products [paper].
- **ko2024 §4, eq. (2).** Low-rank Halo keeps the MNL diagonal separate from the low-rank interactions:
  `H = diag(α) + UVᵀ` [paper].
- **ko2024 §5, eq. (3).**
  - The estimator adds `λ(‖α‖² + ‖U‖²_F + ‖V‖²_F)`.
  - λ > 0 "solves an identifiability issue (otherwise the parameters are only unique up to a multiplicative factor)".
  - It also strengthens the guarantees, giving O(rm) sample complexity.

  [paper]

### 2.7 DeepHalo (Zhang et al. 2026)

- **§3.**
  - Utility is decomposed by interaction order.
  - The first interaction layer modulates a mean-pooled context summary with a per-item nonlinear transform.
  - Each residual layer adds one order (eqs. (4)–(5)).
  - With quadratic activations, L layers reach order 2^(L−1) (§4.2).

  [paper]
- **§4 and experiments.**
  - The authors pitch interpretability as "controllable interaction order".
  - On synthetic data at a fixed parameter budget, depth (interaction order) matters more than width.

  [paper]

### 2.8 Train Ch. 4 (GEV, nested logit)

- **§4.2.1.** IIA holds within each nest but not across nests [paper].
- **§4.2.2.** λ_k measures within-nest independence. λ_k ∈ (0, 1] is consistent with utility maximization for all
  data; λ_k → 0 approaches elimination-by-aspects [paper].
- **§4.2.3.** The choice factors into nest-choice and within-nest logits. The nest-choice logit uses
  `W_nk + λ_k I_nk`, where I is the log-sum (inclusive value) of the nest [paper].
- **§4.2.4.** The nested-logit log-likelihood "is not globally concave". Sequential estimates are recommended as
  starting values for full maximum likelihood [paper].
- **[derivation]** GEV models are RUMs (§4.1). As the overall utility scale grows (temperature → 0), every GEV choice
  rule tends to the argmax of the systematic utility. Nest correlations λ_k therefore change nothing about a
  deterministic (temperature 0) policy. Only nest-level attributes `W_nk` would.

### 2.9 Seshadri & Ugander 2020 (limits of IIA testing)

- **Abstract and §1.**
  - "any general test for IIA with low worst-case error would require a number of samples exponential in the number
    of alternatives".
  - Sample complexity grows at least with the square root of the sum of subset cardinalities in the collection.
  - Restricting to violations on a specific collection of sets (e.g. pairs) gives much milder bounds.

  [paper]

### 2.10 Webb, Glimcher & Louie 2021 (divisive normalization)

- **§2, eq. (2).** Divisive normalization: `z_i = v_i / (σ + ω(Σ_n v_n^β)^{1/β})` [paper].
- **§3, eq. (6).** Normalization is equivalent to a set-dependent error variance: the noise scale depends on the
  number and value of alternatives [paper].
- **§3.2.**
  - "Local" range normalization `v_i / (max v − min v)` predicts the *opposite* substitution pattern to divisive
    normalization.
  - Heteroskedasticity of this kind makes estimates from non-normalized discrete choice models inconsistent.

  [paper]
- **§4.2, Fig. 12.**
  - In a set-size experiment, P(best) drops by about 20% at 12 alternatives.
  - Divisive normalization fitted on the trinary experiment predicts the set-size data out of sample better than
    Probit or range normalization fitted in sample.

  [paper]
- **[derivation]**
  - Dividing *all* utilities by one positive set-level scale never changes the argmax. It only changes stochastic
    choice.
  - Dividing *individual features* by their own set statistics, before a linear combination, does change the argmax.
    It amounts to a diagonal, reciprocal context effect `θ_k(S) = θ_k / s_k(S)`.

### 2.11 Choice models as policies: Jain et al. 2020 and 2022, Mitzenmacher 2000

- **jain2020 §3.2, §4.**
  - The policy is a shared utility over (state, action-representation) followed by a softmax over the available
    actions. This is a conditional logit over a variable action set.
  - Generalizing to unseen action sets needed three regularizers (§4.3):
    - random subsampling of the action set in each training episode;
    - max-entropy regularization;
    - validation-based model selection on held-out action sets (Alg. 1 line 17).
  - Ablations show the generalization gap grows without subsampling or entropy (§6.2.2, Fig. 6).

  [paper]
- **jain2022 (AGILE), abstract and slides.** With varying action sets, the best action depends on which other actions
  are available. A graph-attention action-set summary beats non-relational (IIA-style) utility policies on
  recommender and tool tasks [paper; slide-level evidence only].
- **mitzenmacher2000.**
  - Abstract and §3: with stale load information (periodic updates), "go to the apparently least loaded server" herds
    and "can significantly hurt performance", even with 8 servers (Fig. 3). Least-of-two-random-choices is robust
    over a wide range of staleness.
  - §4: randomizing the information age (exponential rather than fixed delay) breaks the herding.
  - §6: "the importance of using some randomness in order to prevent customers from adopting the same behavior."

  [paper]

## 3. Invariances that matter for the `learned-choice` parameterization (derivations)

Algebra from tomlinson2021 Lemma 4.3 and the softmax form, applied to the contract's utility:

| Change | Effect under softmax(u/τ) | Effect under argmax (τ = 0) |
|---|---|---|
| Add any set-level constant to every u_i (e.g. features `x_i − min_S x`, `x_i − mean_S x` entering linearly) | none | none |
| Multiply every u_i by one set-level scale (utility-level divisive or range normalization) | changes stochasticity only | none |
| Joint scaling (θ, P) → (cθ, cP) together with τ → cτ | none | none (no τ) |
| Low-rank gauge P → RP, Q → R^{-T}Q (r = 1: p → cp, q → q/c) | none | none |
| Per-feature normalization by set statistic (`x_ik / s_k(S)`), ranks, hinge(`x_ik − r_k(S)`) | changes decisions | changes decisions |
| Feature constant within S (request covariates, e.g. ISL) placed in `x_i` | none (cancels) | none |
| The same covariate placed in the context `x̄_S` (q-side) | changes coefficients per request | changes decisions |
| Nest correlations λ_k (nested logit, M3) | changes stochasticity | none (train2009 §4.2.3 limit) |

Each "none" row is an exact flat direction for CMA-ES if the parameter is left free.

Also:
- Mean-pooling a *sparse* or one-hot feature makes the context scale with 1/N. Example: `session_affinity` is 1 on at
  most one worker, so `mean_S = 1/N`. Under the PLAN's N-scaled loads, *dense* per-worker load features have
  N-stable means by design [derivation].
- Z-scores degenerate at N = 2: with distinct values, `z ∈ {−1, +1}` (population sd) for any gap, and the support
  widens with N [derivation].

## 4. Code fact relevant to baselines

- **[code]** `lib/router-plugins/builtin/src/default/picker.rs` (lines 15–60):
  - With temperature > 0, the default picker's softmax normalizes cost differences by the set's cost range:
    `scale = −1/((max_cost − min_cost)·temperature)`.
  - This is local range normalization in the sense of webb2021 §3.2.
  - The contract's `learned-choice` samples `softmax(u/temperature)` with no range normalization.
- **[derivation]** The two "temperature" knobs are not the same parameter. Parity with θ = −e₀ holds only at
  temperature 0. A range-normalized softmax becomes *more* random among the leading workers when an extra bad worker
  joins the set, and its randomization changes with N.

## 5. Possible sim-to-real gap (hypothesis)

- **[code]** A grep of `lib/mocker/src/replay/offline/extensions/kv_router/` in the worktree found no KV-event or
  load-update delay knob.
- **[hypothesis]** Offline replay gives the router a fresher state than a live multi-replica deployment.
  mitzenmacher2000 (§3, §6) shows that deterministic argmax over stale load herds. A τ = 0 learned policy tuned on
  fresh information could herd live.

## 6. Lessons (ranked by expected reward) with campaign actions

Each lesson gives its evidence and a concrete action, keyed to PLAN/CONTRACT stages.

**L1. Remove exact flat directions from the CMA-ES search space (high).**
- Evidence:
  - utility-scale and temperature non-identifiability: tomlinson2021 §2 eq. (1) and §4; ko2024 §5 ("unique up to a
    multiplicative factor");
  - low-rank gauge: seshadri2019 §2.2; ko2024 §5;
  - §3 above.
- Action for `lr-train` space.yaml, for M1 and M2:
  - Fix θ₀ = −1, the default-logit anchor, or normalize θ to the unit sphere when writing the policy.
  - Learn τ only when θ's scale is fixed.
  - At τ = 0, do not search τ at all.
  - For r = 1, constrain ‖q‖ = 1, or parameterize A directly as d×d with an L2 or L1 penalty.
  - Never search p and q freely in raw form.

**L2. Use a decomposed context: pick q from a few named, N-stable set statistics, and leave per-row p free (high).**
- Evidence:
  - DLCL builds each context effect from one feature's set mean (tomlinson2021 §3.2);
  - single-entry tests show full-`A` entries are often null (tomlinson2021 Tables 5–7);
  - most of the gain arrives at a small aggregation dimension (rosenfeld2020 Fig. 3);
  - low rank beats full rank out of sample (seshadri2019 §4).
- Action:
  - Define the context input as `z_S = [mean_S kv_load_frac, mean_S active_prefill_tokens_k, mean_S active_requests_s, max_S overlap_frac, isl_k]`.
  - Set q_k to unit vectors over z_S (2–3 sources), so the term is linear in the free p_k.
  - Add sources one at a time and keep a source only if validation improves.

**L3. Keep set-constant request covariates out of θ; put them in the context; drop the hand-made interaction they subsume (high).**
- Evidence: tomlinson2021 Lemma 4.3 (set-constant terms cancel) and Prop. 4.5 (identification needs variation in
  x̄); §3 above.
- Action:
  - Feed ISL (and any session-turn covariate) only through z_S.
  - If ISL enters z_S, freeze θ₇ (`isl_x_prefill_load`) at 0. Otherwise `A[active_prefill, isl]` and θ₇ are the same
    function: an exact ridge.
  - Freeze in v2 before calibration.

**L4. Do not mean-pool sparse features into the context (high, generalization to N).**
- Evidence:
  - LCL's 1/|C| dilution assumption (tomlinson2021 §3);
  - CDM sums instead (seshadri2019 eq. (2));
  - FETA normalizes by 1/(|Q|−1) specifically to stay scale-stable across sizes (pfannschmidt2019 §4.1);
  - SDA trained at one size failed at larger sizes (pfannschmidt2019 §6.4.1);
  - §3 above.
- Action:
  - Exclude `session_affinity` from x̄_S. Its mean is 1/N, which would let A learn an explicit 1/N dependence from
    N ∈ {4, 8} and then extrapolate it to N = 2 or 32.
  - Use `max_S` for `overlap_frac`.
  - Mean-pool only dense load features, whose per-worker level the PLAN matches across N.

**L5. Set-relative features must be nonlinear to do anything (high).**
- Evidence:
  - additive shifts cancel (tomlinson2021 Lemma 4.3);
  - SDA's reference-point features matter only through the kinked nonlinearity µ (rosenfeld2020 §2.1 eq. (4),
    App. D.4);
  - divisive normalization predicts across set sizes (webb2021 §4.2);
  - z-scores degenerate at N = 2 (§3 above).
- Action, for v2 and the PLAN's M2 row:
  - Drop `x_i − min_S`; it is a no-op, and next to `x_i` it is an exact collinearity.
  - Use instead:
    - `x_ik / (ε + mean_S x_k)` for load features;
    - the leave-one-out rank fraction `#{j≠i: x_j < x_i}/(N−1)`;
    - `relu(x_ik − mean_S x_k)`.
  - Avoid z-scores, because the N = 2 test cells make them degenerate.

**L6. Before spending budget on M2, run a cheap context-drift check (medium-high).**
- Evidence: the binned-MNL diagnostic (tomlinson2021 §6.3, Fig. 2), where coefficient dependence on set means was
  linear in the log mean, nonlinear, or absent depending on the dataset.
- Action, in the pilot or early full stage:
  - Train M1 separately on L1-only and L3-only train cells (and N = 4 vs 8) at a small budget.
  - Cross-evaluate: θ*_L1 on L3 cells against θ*_L3, and so on.
  - If the cross-play loss stays inside the noise floor, deprioritize M2.
  - If it does not, include the drifting load statistic in z_S, log-transformed when the drift looks multiplicative.

**L7. Warm-start M2 from M1 with small initial σ on the context coordinates (medium-high).**
- Evidence:
  - CDM initialized from the MNL MLE (seshadri2019 §4);
  - LCL failed to beat MNL within a fixed optimizer budget (tomlinson2021 footnote 3).
- Action:
  - Start the M2 CMA-ES mean at (θ*_M1, context = 0).
  - Use per-coordinate `CMA_stds`, smaller on the context coordinates.
  - Report M2 at equal *additional* budget versus continuing M1.

**L8. Optional convex warm start by distilling a privileged teacher (medium; hypothesis).**
- Evidence:
  - the LCL NLL is convex in (θ, A) (tomlinson2021 §5.1);
  - the nested LR test is well calibrated (seshadri2019 Fig. 1);
  - choice-set confounding warning (tomlinson2021 §6.3, expedia and car-alt).
- Action:
  - Log candidate tables (router-observable v1 features for every worker) plus the teacher's pick from replays of the
    simulator-signal ablation or the best tuned baseline.
  - Fit M1 and M2 by convex MLE as the CMA-ES initial mean, with the inverse Fisher information as the initial
    covariance.
  - Use the LR test M2-vs-M1 to decide whether M2 is worth the budget.
  - Use this only as initialization, because the states come from the teacher's own trajectory.

**L9. Treat temperature as a deployment choice, not a smoothing device, and check its N-transfer (medium).**
- Evidence:
  - the argmax gradient problem is solved by Gumbel smoothing only for gradient training (aouad2023 §2.3);
  - CMA-ES already smooths in parameter space;
  - range-normalized default softmax (§4 above);
  - softmax mass on non-best workers grows with N at fixed τ [derivation];
  - herding under stale information (mitzenmacher2000 §3).
- Action:
  - Train and deploy `learned-choice` at τ = 0 by default.
  - Tune τ > 0 only as a separate variant, and report how it transfers to N = 32.
  - When tuning the default baseline's temperature, record that it is range-normalized and not comparable to
    learned-choice τ.

**L10. Skip nested logit (M3) unless the policy is stochastic; use nest-level features instead (medium).**
- Evidence: train2009 §4.2.2–4.2.3 (decomposition and λ); §2.8 derivation.
- Action: at τ = 0, implement topology awareness as features such as node-aggregate load added to `x_i`. Do not build
  GEV machinery.

**L11. Select the final policy on validation, not as the best train sample; probe N-extrapolation before test (medium).**
- Evidence:
  - action-set subsampling plus validation-based model selection (jain2020 §4.3, Alg. 1, Fig. 6);
  - FATE/FETA size generalization (pfannschmidt2019 §6.4.3);
  - SDA's failure at larger sizes (pfannschmidt2019 §6.4.1).
- Action:
  - Select among the last k CMA-ES means and the top-k samples by fresh-repeat validation goodput (N ∈ {4, 6, 8}).
  - Add one validation-only N = 12 probe cell. It is not in the test set, which keeps test cells at {2, 16, 32}
    clean.
  - If the probe degrades, record a deviation and consider adding training cells at more N values.

**L12. Interpret coefficients by ablation, not by raw entries (medium, reporting).**
- Evidence:
  - single-entry constrained fits (tomlinson2021 Tables 5–7);
  - L1 path (tomlinson2021 Fig. 3);
  - domain-specific A (tomlinson2021 Fig. 4).
- Action:
  - In REPORT.md, report Δ validation goodput when each context source or θ_k is zeroed and briefly re-tuned.
  - Report per-family stability of A.
  - Do not read magnitudes off a jointly tuned A.

**L13. Pre-register the M2 variants; avoid free search over A on validation (medium-low).**
- Evidence:
  - general IIA tests need samples exponential in set size, while structured alternatives are cheap
    (seshadri2020 abstract);
  - the universal-logit test is anti-conservative (seshadri2019 Fig. 1).
- Action: fix the list of context sources and ranks before calibration, and count every M2 variant tried on
  validation in the multiplicity accounting.

**L14. No worker-identity terms (low; confirms the design).**
- Evidence:
  - item-level context needs Ω(m²) data and cannot transfer to new items (ko2024 Prop. 1; maragheh2018);
  - feature-based models transfer (tomlinson2021 §1);
  - model-free models over variable assortments overfit (aouad2023 §6.3–6.4).
- Action: keep the shared per-worker utility. Never add per-worker biases or N-length input vectors.

**L15. If M2 helps and residual structure remains, the next rung is one more DeepHalo-style layer (low).**
- Evidence: zhang2026 §3.2 eqs. (4)–(5).
- Action: an optional M2b rung, a second mean-pooled interaction layer with linear φ, instead of an MLP or nested
  logit.

## 7. What does not transfer

- Every DCM paper here estimates human choice probabilities by likelihood. The campaign optimizes a *policy* for
  goodput. Context effects in the policy are justified only if they raise held-out goodput, not likelihood.
  Identification and flat-direction results still apply to CMA-ES because they are properties of the decision map.
- The likelihood-ratio tests (tomlinson2021 Table 3; seshadri2019 §3.2) assume i.i.d. choices. Routing decisions
  within a replay are serially dependent, so they apply only to the optional teacher-distillation dataset, with
  caution.
- RUMnet's latent-class mixtures model population heterogeneity. A single router has no such heterogeneity, so
  RUMnet's main contribution does not carry over. Only its smoothing and structure arguments do.

## 8. Files

- PDFs (15):
  - `/tmp/learned-routing-lit/pdfs/dcm-context/*.pdf`;
  - `train2009-gev-ch4.pdf` is MacBinary-wrapped, so use `train2009-gev-ch4.clean.pdf`;
  - `jain2022-agile-action-relations.pdf` is the ICLR slide deck, not the paper.
- Text extracts used for reading are in the session scratchpad (not needed downstream).
