# Learned systems policies: literature notes for the learned-routing campaign

Scout angle: learned scheduling and load balancing in systems (Decima, Park, Pensieve/Puffer, Genet,
mean-field load balancing, power-of-d), black-box policy search (ARS, ES, CMA-ES, UH-CMA-ES,
PEGASUS), domain randomization and simulator bias (dynamics randomization, CausalSim, digital-evolution
exploits), and set-structured policy forms (Deep Sets).

Campaign context read: `<repo>/notes/learned-routing/PLAN.md`,
`<campaign-root>/CONTRACT.md`. Nothing under `<repo>` or the campaign
worktree was edited. One read-only code observation is recorded in section 4.

Citation convention: `[Key §x]` points to the paper and section (or table/figure) where the claim is
made. Every number below is quoted from the cited paper; nothing is extrapolated. Statements marked
**Hypothesis** are this scout's inferences for the campaign, not claims from the papers.

## 0. Files

All PDFs are in `/tmp/learned-routing-lit/pdfs/learned-systems-policies/`. Text extractions
(`pdftotext -layout`) are in `/tmp/learned-routing-lit/txt/learned-systems-policies/`.

| Key | Paper | Venue | Local PDF | Read depth |
|---|---|---|---|---|
| Decima | Mao, Schwarzkopf, Venkatakrishnan, Meng, Alizadeh. *Learning Scheduling Algorithms for Data Processing Clusters* | SIGCOMM 2019 (arXiv 1810.01963) | `mao2019-decima.pdf` | full (incl. App. D, H, I, J) |
| InputDriven | Mao, Venkatakrishnan, Schwarzkopf, Alizadeh. *Variance Reduction for RL in Input-Driven Environments* | ICLR 2019 (arXiv 1807.02264) | `mao2019-input-driven-variance.pdf` | full main text, App. G, J, K |
| Park | Mao et al. *Park: An Open Platform for Learning-Augmented Computer Systems* | NeurIPS 2019 | `mao2019-park.pdf` | key sections (§3, §5, Fig. 2, Fig. 4) |
| Pensieve | Mao, Netravali, Alizadeh. *Neural Adaptive Video Streaming with Pensieve* | SIGCOMM 2017 | `mao2017-pensieve.pdf` | key sections (§4.1, §5.3, §5.4) |
| Puffer | Yan et al. *Learning in situ: a randomized experiment in video streaming* | NSDI 2020 (arXiv 1906.01113 v2) | `yan2020-puffer.pdf` | key sections (§1, §3, §5.2, §5.3, §7) |
| Genet | Xia, Zhou, Yan, Jiang. *Genet: Automatic Curriculum Generation for Learning Adaptation in Networking* | SIGCOMM 2022 (arXiv 2202.05940) | `xia2022-genet.pdf` | key sections (§1, §2, §4, §7, Table 5) |
| CausalSim | Alomar, Hamadanian, Nasr-Esfahany, Agarwal, Alizadeh, Shah. *CausalSim: A Causal Framework for Unbiased Trace-Driven Simulation* | NSDI 2023 (arXiv 2201.01811) | `alomar2023-causalsim.pdf` | key sections (§1–§3, §6.4) |
| DynRand | Peng, Andrychowicz, Zaremba, Abbeel. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* | ICRA 2018 (arXiv 1710.06537) | `peng2018-dynamics-randomization.pdf` | full |
| ARS | Mania, Guy, Recht. *Simple random search provides a competitive approach to reinforcement learning* | NeurIPS 2018 (arXiv 1803.07055) | `mania2018-ars.pdf` | full main text |
| ES | Salimans, Ho, Chen, Sidor, Sutskever. *Evolution Strategies as a Scalable Alternative to RL* | arXiv 1703.03864 (2017) | `salimans2017-es.pdf` | key sections (§2, §3) |
| CMATut | Hansen. *The CMA Evolution Strategy: A Tutorial* | arXiv 1604.00772 v2 (2023) | `hansen2016-cmaes-tutorial.pdf` | key sections (§5, App. A–B) |
| UHCMA | Hansen, Niederberger, Guzzella, Koumoutsakos. *A Method for Handling Uncertainty in Evolutionary Optimization...* | IEEE TEC 13(1), 2009 | `hansen2009-uh-cmaes.pdf` | key sections (§I, §III, §IV) |
| PEGASUS | Ng, Jordan. *PEGASUS: A policy search method for large MDPs and POMDPs* | UAI 2000 (arXiv 1301.3878) | `ng2000-pegasus.pdf` | full |
| MFLB | Tahir, Cui, Koeppl. *Learning Mean-Field Control for Delayed Information Load Balancing in Large Queuing Systems* | ICPP 2022 (arXiv 2208.04777) | `tahir2022-meanfield-lb.pdf` | key sections (§1, §4, Fig. 4) |
| LBSurvey | van der Boor, Borst, van Leeuwaarden, Mukherjee. *Scalable load balancing in networked systems: A survey of recent advances* | Statistical Science 2022 (arXiv 1806.05444) | `vanderboor2018-scalable-lb.pdf` | key sections (§1, §2.3–2.4, Table 1) |
| DeepSets | Zaheer et al. *Deep Sets* | NeurIPS 2017 (arXiv 1703.06114) | `zaheer2017-deep-sets.pdf` | key sections (§2–3, §4.1.2) |
| DigEvo | Lehman, Clune, Misevic, et al. *The Surprising Creativity of Digital Evolution* | Artificial Life 2020 (arXiv 1803.03453) | `lehman2018-digital-evolution-creativity.pdf` | key section ("Unintended Debugging") |

There are 17 PDFs, more than the 6–12 requested. The core set is Decima, InputDriven, Puffer, Genet,
CausalSim, UHCMA, PEGASUS, LBSurvey and DynRand. The rest are supporting. A failed download, the
Heidrich-Meisner & Igel 2009 racing-CMA-ES paper (ICML), returned an HTML bot-check page. It is kept as
`heidrichmeisner2009-races-cmaes.FAILED-DOWNLOAD.html` (9 KB, not a PDF) because the rules forbid
deletion. List it in the campaign's cleanup ledger if desired.

## 1. Per-paper notes

### 1.1 Decima (Mao et al., SIGCOMM 2019)

- **Policy form for variable cardinality.** A graph neural network computes per-node, per-job and
  **global** embeddings by reusing the same small transforms at every node, so the model works
  "irrespective of the number of jobs or machines" [Decima §1, §5.1]. Scores are softmaxed over the
  currently schedulable set `A_t` only [Decima §5.2, Eq. 2]. A single score function `w(y_i, z, l)`
  takes the parallelism limit `l` as an input instead of keeping one head per limit. This "significantly
  reduces the number of parameters ... and speeds up training" [Decima §5.2, Fig. 15a].
- **The global summary is load-critical.** Removing the graph embedding (raw node features straight
  into the score functions) leaves Decima with "no notion of small jobs or cluster load". Its policy
  "quickly becomes unstable as the load increases" and falls below the tuned weighted-fair heuristic
  at high load [Decima §7.4, Fig. 14].
- **Variance from stochastic arrivals.** Different arrival sequences after the same action give
  "vastly different rewards" [Decima §5.3, Fig. 7]. The fix fixes the same job arrival sequence across
  several training episodes and computes baselines per sequence [Decima §5.3, citing InputDriven]. At
  cluster load above 75%, this variance reduction "improves average JCT by 2×" [Decima §7.4].
- **Horizon exploitation.** With deterministic episode termination, the agent learned to defer large
  jobs until termination, which "turns out to be the optimal strategy over a fixed time horizon" and
  causes indefinite starvation at runtime. Fixes were memoryless (exponential) termination plus a
  curriculum of growing episode length [Decima §5.3]. Training on batched arrivals "learns to
  systematically defer large jobs" and underperforms the tuned heuristic above 65% load [Decima §7.4].
- **Generalization to workload shift.** Test IAT is 45 s [Decima §7.4, Table 2]:

  | Training | Avg JCT |
  |---|---|
  | Tuned heuristic | 91.2 ± 23.5 s |
  | Trained on test IAT | 65.4 ± 28.7 s |
  | Trained on "anti-skewed" IAT 75 s | 104.8 ± 37.6 s (worse than the heuristic) |
  | Mixed workloads | 82.3 ± 31.2 s |
  | Mixed workloads plus an IAT hint feature | 76.6 ± 33.4 s |

  The authors conclude that "a diverse training workload set helps make Decima's learned policies robust
  to workload shifts". Observing the interarrival time as a feature yields an adaptive policy
  [Decima §7.4].
- **Generalization to cluster size.** Training with 10× fewer executors gives 630 ± 70 vs 610 ± 90 (3%
  worse). Training with 15× fewer jobs gives 3540 ± 450 vs 3290 ± 680 (7% worse) [Decima App. I,
  Table 3]. Larger clusters are easy "as the policy correctly limits jobs' parallelism". Many more
  concurrent jobs is harder "as the smaller-scale training lacks experiences with complex job
  combinations" [Decima App. I]. Note: these gaps are within one reported sd.
- **Baselines.** The strongest baseline was a weighted-fair heuristic with α swept over
  {−2, −1.9, ..., 2} [Decima §7.1]. Decima beat it by 21% on batched TPC-H [Decima §7.2]. Removing any
  single Decima component drops it below this tuned heuristic at high load [Decima §7.4, Fig. 14].
- **Simulator fidelity.** Mean error was within 5% of real runtime in isolation and within 9% when
  sharing a cluster [Decima App. D]. The simulator models first-wave slowdown, JVM startup and
  parallelism-dependent task inflation [Decima §6.2].
- **Train/test split.** The Alibaba trace experiment trains on the first half of the trace and tests on
  the remaining portion [Decima §7.3]. All §7 results use test job sequences unseen in training
  [Decima App. B/C text].

### 1.2 Variance reduction in input-driven environments (Mao et al., ICLR 2019)

- **Definition.** An input-driven MDP adds an exogenous input process `z` (job arrivals, bandwidth) to
  the MDP. A state-only baseline is poor because returns depend on the future input sequence
  [InputDriven §1, §4, Def. 1].
- **Main result.** An input-dependent baseline `b(ω_t, z_{t:∞})` is bias-free for policy gradient
  [InputDriven Thm 1] and for TRPO [InputDriven App. F]. This requires the input process to be
  independent of states and actions [InputDriven §1].
- **Load-balancing example.** Two servers, Poisson arrivals, Pareto sizes. Input-dependent baselines
  give 50× lower policy-gradient variance and 33% higher test reward [InputDriven §3, Fig. 2]. In the
  discrete-action LB and ABR environments, they add 25–33% test reward and beat the heuristic
  [InputDriven §6.2, Fig. 5].
- **Repeatable inputs.** The practical implementations ("multi-value network", MAML "meta baseline")
  need **input repeatability**, which is "straightforward when using simulators" [InputDriven §5].
  An LSTM baseline over input sequences did not help [InputDriven App. G].
- **Size-aware LB state.** The LB environment's state includes the incoming job size, so policies can
  e.g. "reserve some servers for small jobs" [InputDriven App. J].

### 1.3 Park (Mao et al., NeurIPS 2019)

- **Catalog of RL-for-systems challenges** [Park §3]:
  - "needle-in-the-haystack" regions where most of the action space gives indistinguishable reward,
    with fixes by confining the search space or bootstrapping from existing policies [Park §3.1];
  - variable-size action spaces [Park §3.1];
  - input-driven variance [Park §3.2];
  - infinite horizons [Park §3.2];
  - the simulation–reality gap [Park §3.3];
  - understandability [Park §3.4].
- **Overfit example (load balancing).** Under a bimodal job-size distribution, the RL agent "learns to
  reserve a certain server for small jobs". When the distribution changes, "blindly reserving a server
  wastes compute resource and reduces system throughput". An agent trained on a broader distribution
  (dist. 5) is more robust than one trained on dist. 1 [Park §3.3, Fig. 2].
- **Hybrids with heuristics.** Hybrids are a "unique opportunity". A learned scheduler "could fall back
  to a simple heuristic if it detects that the input distribution significantly drifted"
  [Park §3.4].
- **Benchmark result.** Results across 12 environments are "mixed: RL is able to outperform
  state-of-the-art baselines in several environments ... in others" it does not [Park §1/§5].

### 1.4 Pensieve (Mao et al., SIGCOMM 2017)

- **Simulator.** Training uses a fast chunk-level simulator ("100 hours of video downloads in only
  10 minutes") [Pensieve §4.1].
- **Simulator assumption made true in deployment.** The simulator assumes trace throughput is fully
  used regardless of bitrate. With TCP slow-start-restart, real throughput depends on the chosen
  bitrate, so the action influences the trace. The authors made reality match the simulator
  assumption by disabling slow-start-restart on the server [Pensieve §4.1, Fig. 4].
- **Claimed robustness.** "No simulation can capture all real world system artifacts", but Pensieve
  learns well "as long as it experiences a large enough variety of network conditions during training"
  [Pensieve §4.1].
- **Generalization.** A model trained only on synthetic Markovian traces came within 1.6–10.8% of one
  trained on the test networks [Pensieve §5.3, Fig. 12]. A multi-video model came within 3.2%
  [Pensieve §5.3, Fig. 13].
- **Capacity.** One hidden layer is best (5.489 QoE). Five layers drop to 4.253 [Pensieve §5.4,
  Table 3].

### 1.5 Puffer (Yan et al., NSDI 2020)

- **Real-world RCT result.** In a blinded randomized trial (14.2 stream-years, 56k users), "it is
  difficult for sophisticated or machine-learned control schemes to outperform a 'simple' scheme
  (buffer-based control), notwithstanding good performance in network emulators or simulators"
  [Puffer Abstract, §1].
- **Statistical power.** With 1.75 years of data per scheme, the 95% CI on stall ratio is ±10–17% of the
  mean [Puffer §3]. About 2 stream-years are needed to reliably distinguish schemes whose true
  performance differs by 15% [Puffer §5.3]. Behavior is heavy-tailed [Puffer §3].
- **Emulation-trained vs in-situ.** Emulation-trained Fugu's real-world performance "was horrible".
  Emulation results "differ markedly from the real world" in how schemes rank [Puffer §5.2, Fig. 11].
- **Why simulators can mislead.** Trace-based emulators eliminate the play of chance (all schemes see
  the same conditions). They still carry "systematic uncertainty that comes from selecting a set of
  traces that may omit the variability or heavy-tailed nature of a real deployment" [Puffer §5.3].
- **Successful design.** The winning design is classical control (MPC) plus a learned predictor
  trained in situ. "Good, or even near-optimal, performance in a simulator or emulator does not
  necessarily predict good performance" in deployment [Puffer §4, §7].

### 1.6 Genet (Xia et al., SIGCOMM 2022)

- **Wide training ranges hurt.** "Training on a wide range of network environments leads to suboptimal
  performance, whereas training on a narrow distribution ... results in poor generalization"
  [Genet Abstract]. RL's advantage over baselines "diminishes rapidly when the range of target
  environments expands" [Genet §2, Fig. 2a].
- **Average wins hide regressions.** Even when RL wins on average, "their performance falls behind the
  baselines in a substantial fraction of test environments" [Genet §2, Fig. 2b].
- **Gap-to-baseline curriculum.** Train more on environments where the current policy trails a
  rule-based baseline:
  - BO searches the environment-config space for max `Gap(p) = R(π_rule, p) − R(π_rl, p)`, averaging
    over k = 10 environments per config [Genet §4.2];
  - new configs are mixed in with probability w = 30% [Genet §4.2];
  - BO restarts after each model update, "because the rewarding environments can change once the RL
    model changes" [Genet §4.2];
  - gap-to-baseline predicts training gain better than gap-to-optimum [Genet §4.1, Fig. 6].
- **Results.** Asymptotic performance improves 8–25% (ABR), 14–24% (CC) and 15% (LB, Park simulator)
  [Genet §1, §5].
- **Limitations** [Genet §1, §7]:
  - a weak baseline can mislead it, and using "an 'ensemble' of existing baselines (i.e., measuring
    the maximum gap to any baseline from a set)" is suggested;
  - it does not optimize out-of-range environments;
  - it is not adversarially robust.
- **LB environment.** Inputs are service rates, job sizes, intervals, job counts and queue-shuffle
  probability, over ranges RL1 ⊂ RL2 ⊂ RL3 [Genet Table 5].

### 1.7 CausalSim (Alomar et al., NSDI 2023)

- **The exogenous trace assumption.** Trace-driven simulation assumes that "the interventions being
  simulated (e.g., a new algorithm) would not affect the validity of the traces". When violated,
  replaying traces "may lead to incorrect results" [CausalSim Abstract, §1].
- **Real example.** Puffer throughput traces depend on the ABR algorithm that collected them. An expert
  simulator assuming exogenous throughput predicted BBA's buffer distribution closer to the source
  algorithm's than to BBA's [CausalSim §2.2, Fig. 2].
- **Practical consequence.** A tuning conclusion flipped: the biased simulator predicted the improved
  BOLA variant "should stall 1.34× the stall rate of BBA". In real deployment it achieved 0.7× BBA's
  stall rate [CausalSim §1].
- **Relaxation.** The model relaxes "exogenous trace" to "exogenous latents" (e.g., bottleneck
  capacity, job size) that the intervention cannot affect [CausalSim §3.1]. For load balancing on
  heterogeneous servers, "it isn't possible to merely replay the trace for new machine assignments"
  [CausalSim §3.1, §6.4]. There the baseline simulator had 124.3% median error [CausalSim §1].
- **Designer's duty.** "A simulation designer needs to reason about the causal structure of observed
  and latent quantities to define" a valid trace [CausalSim §3.1].

### 1.8 Dynamics randomization (Peng et al., ICRA 2018)

- **Method.** 95 dynamics parameters are randomized: masses, damping, friction, table height,
  **timestep between actions (a latency model)** and observation noise [DynRand §IV-C, Table I].
- **Results** [DynRand Table II]:

  | Policy | Sim success | Real success |
  |---|---|---|
  | LSTM | 0.91 | 0.89 |
  | Feedforward + 8-step history | 0.87 | 0.70 |
  | Feedforward | 0.83 | 0.67 |
  | Feedforward, no randomization | 0.51 | **0.0** |

- **Ablation** [DynRand Table III]. The full randomization reaches 0.89 real success. Fixing one
  parameter group gives:
  - fixed action timestep: 0.29;
  - no observation noise: 0.25;
  - fixed link mass: 0.64;
  - fixed puck friction: 0.48.

  "Coping with the latency of the controller and sensor noise are important factors" [DynRand §V-B].
- **Calibration.** Little calibration was done, and sim and real joint trajectories "differ
  significantly" [DynRand §V].

### 1.9 ARS (Mania, Guy, Recht, NeurIPS 2018)

- **Performance.** Static **linear** policies trained by random search match or beat state-of-the-art
  sample efficiency on MuJoCo [ARS Abstract, §4.2, Table 1]. ARS is at least 15× more compute-efficient
  than ES [ARS §1].
- **Three augmentations** [ARS §3.1–3.3]:
  - scale steps by the std of collected rewards;
  - **normalize states** by online mean/std (V2);
  - keep only the top-b directions.

  Humanoid could not be trained without state normalization [ARS §3.2].
- **Seed variance.** Over 100 seeds, some seeds lead to slow training or "locally optimal behaviors";
  "evaluations on small numbers of seeds cannot correctly capture" performance [ARS §4.2,
  "A hundred seeds"]. Sensitivity to hyperparameters is similar to sensitivity to seeds [ARS §4.2].
- **Survival-bonus local optima.** The survival bonus made random search find "stand still" local
  optima. The authors subtracted it during training only [ARS §4.2].
- **Accounting.** "The states and rewards produced during the evaluation rollouts were not used in any
  form during training" [ARS §4.1]. A fair sample-complexity measure should count "rollouts used for
  every tested hyperparameters" [ARS §5].

### 1.10 ES (Salimans et al., 2017)

- **Variance and horizon.** ES gradient variance is independent of episode length T, while
  policy-gradient variance grows with T. ES is "attractive ... if the effective number of time steps T
  is long, actions have long-lasting effects, and if no good value function estimates are available"
  [ES §3.1]. ES is indifferent to sparse or delayed rewards [ES §1, §3.3].
- **Practical tricks** [ES §2.1]:
  - mirrored (antithetic) sampling;
  - rank-based fitness shaping, which "removes the influence of outlier individuals";
  - shared random seeds and a noise table, so workers exchange only scalars;
  - episode-length caps to keep CPU utilization above 50% when episodes have stragglers.
- **Perturbations can fail to change behavior.** Gaussian perturbations sometimes yield policies that
  "always took one specific action regardless of the state". The fix was virtual batch normalization,
  or for MuJoCo, discretized actions to "encourage more exploration" [ES §2.2].
- **Dimension.** What matters is the intrinsic dimension, not the parameter count [ES §3.2].

### 1.11 CMA-ES tutorial (Hansen, arXiv 1604.00772 v2)

- **Invariances** [CMATut §5]:
  - to strictly monotone transformations of f (rank-based);
  - to rotation;
  - scale invariance "if the initial scaling σ(0) and the initial search point m(0) are chosen
    accordingly";
  - diagonal invariance only "if the initial diagonal covariance matrix" is set accordingly.
- **Stationarity.** Under random selection (a flat objective), m, C and ln σ are unbiased, but
  "E[σ(g+1) | σ(g)] > σ(g)". "A bias toward increase ... entails the risk of divergence ... whenever
  the selection pressure is low" [CMATut §5, "Stationarity or Unbiasedness"].
- **Initial σ.** "The optimum should presumably be within the initial cube m ± 3σ". Different search
  intervals per variable can go in the initial C, but they "should not disagree by several orders of
  magnitude. Otherwise a scaling of the variables should be applied" [CMATut App. A, Fig. 6 note].
- **Population size.** Increasing λ "usually improves the global search capability and the robustness
  of the CMA-ES, at the price of a reduced convergence speed". "Independent restarts with increasing
  population size ... are a useful policy". Decreasing α_cov may help on noisy functions [CMATut
  App. A].
- **Noise.** Step-size control "fails to adapt nearly optimal step-sizes on very noisy objective
  functions" [CMATut §4, end of "Step-Size Control"].
- **Termination diagnostics.** NoEffectAxis, NoEffectCoord, ConditionCov (> 1e14) and EqualFunValues
  [CMATut App. B.3].

### 1.12 UH-CMA-ES (Hansen et al., IEEE TEC 2009)

- **Framing.** For a rank-based algorithm, noise matters "if and only if the signal-to-noise ratio is
  too small". There are only two remedies: increase the signal or reduce the uncertainty
  [UHCMA §III].
- **Resampling cost.** Resampling costs "a factor between three and a 100" in evaluations. To shrink
  the final distance to the optimum by α, the samples needed grow as α⁻² [UHCMA §I, §III].
- **Mean vs median.** The median reduces uncertainty "under much milder assumptions than the mean"
  (heavy tails) [UHCMA §III].
- **Population vs resampling.** "Increasing only λ is inferior to resampling, but increasing µ and λ
  is preferable to resampling", provided step-sizes adapt properly [UHCMA §III].
- **Uncertainty measurement** [UHCMA §IV]:
  - re-evaluate a fraction r_λ (e.g. 0.3) of the population and count **rank changes**;
  - re-evaluate a *slightly mutated* copy (ε > 0) "to treat 'frozen' noise similar as stochastic
    noise" [UHCMA §IV-c, parameter ε].
- **Uncertainty treatment** [UHCMA §IV]: increase evaluation time up to t_max, then increase σ by
  α_σ = 1 + 2/(n + 10). Defaults are r_λ = 0.3, α_t = 1.5 and θ ≤ 1 as the rank-change threshold.
- **Non-elitism.** CMA-ES is non-elitist, which "avoids systematic fitness overvaluation on noisy
  objective functions" [UHCMA §IV-A].

### 1.13 PEGASUS (Ng & Jordan, UAI 2000)

- **Fixed scenarios.** Any POMDP can be turned into one with deterministic transitions by fixing the
  random numbers ("scenarios") in advance. The value estimate over m fixed scenarios is a
  *deterministic* function of the policy, "reused for evaluating different π", so any deterministic
  optimizer applies [PEGASUS §3].
- **Uniform convergence.** The number of scenarios m needed is poly(VC or pseudo-dimension of the
  policy class, 1/ε, ...) with no dependence on state-space size [PEGASUS §4.1, Thm 1; §4.3].
- **How randomness is consumed matters.** With infinite action spaces, simplicity of the policy class
  is not enough. The complexity of the composed dynamics class F, which depends on *how the simulator
  maps random numbers to transitions* (g vs g′), also matters. The "complex" g′ gave worse results
  [PEGASUS §4.2–4.3, §5, Fig. 1b, footnote 7: gap ≲ O(√(log|Π|/m))].
- **Bicycle task.** Only m = 30 scenarios and 10 trials were used. A step-size bound was needed "to
  avoid problems near V(π_θ)'s discontinuities" [PEGASUS §5, footnote 8]. Discontinuous rewards should
  be smoothed, possibly with continuation [PEGASUS §3].

### 1.14 Mean-field load balancing with delayed information (Tahir, Cui, Koeppl, ICPP 2022)

- **Herding.** "JSQ fails when Δt > 0 mainly due to ... 'herd behaviour'". As Δt → ∞, random
  allocation becomes optimal. In between, a learned policy beats both JSQ(d) and random. "As a result of
  delayed information, the number of agents will make a difference as opposed to the delay-free case"
  [MFLB §1].
- **Size-independent policy.** The policy acts on anonymous sampled-queue states plus the empirical
  queue-state distribution, so the same policy applies to any (N, M). Finite-system performance
  approaches the mean-field value "as the system size ... becomes sufficiently large" [MFLB §4,
  Fig. 4, Algorithm 1]. Smaller systems are the less well approximated end (Fig. 4).

### 1.15 Scalable load balancing survey (van der Boor et al., Stat. Sci. 2022)

- **Power-of-two.** Sampling d = 2 gives super-exponential queue-tail decay versus exponential for
  random assignment [LBSurvey §1].
- **Waiting vs N.** At fixed per-server load λ < 1, as N grows, random assignment's mean wait "remains
  constant", JSQ's mean wait "vanishes", and JSQ(d) with fixed d keeps Θ(1) wait [LBSurvey §1, §2.4,
  Table 1].
- **Scaling regimes** [LBSurvey §2.4]:
  - many-server (λ(N)/N → λ < 1);
  - Halfin–Whitt, `(N − λ(N))/√N → β > 0`, where relative slack shrinks as β/√N;
  - non-degenerate slow-down.

  In the Halfin–Whitt regime, random and round-robin waits grow without bound, and JSQ's is
  Θ(1/√N) [LBSurvey §2.4, Table 1].
- **Size knowledge.** JSW (join-shortest-workload, using service-requirement knowledge) has lower mean
  wait than JSQ/JIQ [LBSurvey §2.4].

### 1.16 Deep Sets (Zaheer et al., NeurIPS 2017)

- **Invariance.** A permutation-invariant set function decomposes as ρ(Σ φ(x)) [DeepSets Thm 2].
- **Equivariance.** A linear layer is permutation-equivariant iff Θ = λI + γ11ᵀ, i.e. a mix of the
  element and a pooled set summary. This extends to vectors "when λ, γ can be matrices" [DeepSets
  Lemma 3]. A max-pool variant "performs better in some applications" [DeepSets §3.1, Eq. 4].
- **Size extrapolation.** Trained on sets of at most 10 and tested up to 100, DeepSets "generalize much
  better" than LSTM/GRU [DeepSets §4.1.2, Fig. 2].

### 1.17 Digital evolution anecdotes (Lehman et al., Artificial Life 2020)

- **Exploits are the default.** Search "will often learn how to exploit bugs in simulations". Edge
  cases "are sometimes amplified and exploited by evolution" [DigEvo, "Unintended Debugging"].
- **Concrete exploits** [DigEvo, "Unintended Debugging"]:
  - creatures exploited Euler integration error to get "free energy";
  - creatures exploited a collision-detection bug;
  - soft robots shrank to few cells to exploit the simulator's *adaptive time-step heuristic*.

## 2. Mapping to the campaign: what the papers imply

### 2.1 Policy parameterization for variable N (model-form, features)

- The M1/M2 form "shared per-worker utility + set summary + softmax/argmax over the candidate set"
  is the linear special case of Decima's score-and-softmax-over-available-set design [Decima §5.2] and
  of the DeepSets equivariant layer [DeepSets Lemma 3]. Two consequences:
  - A term linear in x̄_S alone adds the same constant to every candidate, so it cannot change any
    choice. Only *interactions* x_i ⊗ x̄_S (the M2 bilinear term) or nonlinear/set-relative transforms
    matter. **Hypothesis**, derived from Lemma 3 applied to a softmax over candidates.
  - Use **mean (or min/max) pooling, never sum**, so the context term's magnitude does not grow with N.
    Normalize rank features by (N−1). DeepSets shows pooled models extrapolate in set size [DeepSets
    §4.1.2]. Mean-field policies transfer across M [MFLB Fig. 4].
- Decima's ablation shows that the set-level summary carries load information. Without it, policies
  become unstable at high load [Decima §7.4, Fig. 14]. This supports M2 or set-relative features, but
  only if they are validated on held-out N.
- Small N is the poorly approximated end for mean-field-style policies [MFLB Fig. 4]. At N = 2,
  x̄_S = (x_i + x_j)/2 is half self-referential. **Hypothesis:** a leave-one-out mean x̄_{−i} keeps
  the context term's meaning closer to constant across N. Expect N = 2 to be the hardest extrapolation
  cell and report it separately.

### 2.2 Sample efficiency, noise, CRN (training)

- CRN or input-repeatability is the single most consistently reported variance-reduction lever in this
  literature:
  - Decima: 2× at high load [Decima §7.4];
  - InputDriven: 50× gradient variance on LB [InputDriven Fig. 2];
  - PEGASUS: a deterministic objective [PEGASUS §3].

  The campaign's per-cell default normalization is an input-dependent baseline. It only works if every
  candidate in a CMA-ES generation is scored on *identical* (cell, transform seed, replay seed) sets.
- CRN quality depends on how randomness is consumed [PEGASUS §4.3, §5]. See the code observation in
  section 4.
- Deterministic replay makes noise "frozen". Re-running the same (policy, cell, repeat) gives zero
  difference, but the objective is rugged in θ. UH-CMA-ES measures rank changes on *slightly perturbed*
  re-evaluations for exactly this reason [UHCMA §IV, parameter ε].
- Under flat or plateaued objectives, CMA-ES's σ drifts upward [CMATut §5], and argmax policies give
  plateaus [ES §2.2; Park §3.1]. The learned-choice utility has exact symmetries (overall scale at
  temperature 0, scale versus temperature, and the p/q rescaling in the low-rank term). Those are exact
  flat directions.

### 2.3 Overfitting to simulators and workload splits (generalization, sim2real)

- **Training distribution.** Mismatched training distributions do worse than the tuned heuristic
  [Decima Table 2]. Wide distributions dilute gains and hide per-environment regressions [Genet §2].
  Narrow, bimodal distributions produce "reservation" policies that break under shift [Park §3.3].
- **Trace validity.** Trace-driven replay is valid only if the router cannot affect the trace
  [CausalSim §1, §3.1]. Pensieve made reality match the simulator assumption rather than the reverse
  [Pensieve §4.1].
- **Simulator exploitation.** Black-box optimizers exploit simulator heuristics [DigEvo]. Emulation
  rankings can invert in reality [Puffer §5.2]. Randomizing the right nuisance parameters (latency,
  observation noise) is what made transfer work [DynRand Table III].

### 2.4 Evaluation and reporting

- Report paired, per-cell comparisons and the fraction of cells where the learned policy loses
  [Genet Fig. 2b].
- Run multiple optimizer seeds [ARS §4.2].
- Count every tuning evaluation, including baseline sweeps [ARS §5].
- Simulated CRN removes the play of chance but not trace-selection bias [Puffer §5.3].

## 3. Lessons (ranked; each has evidence → action → stage)

1. **Every CMA-ES generation must share one set of cells, seeds and repeats.** Treat per-cell default
   goodput as an input-dependent baseline and check this mechanically.
   - *Evidence:* Decima §5.3 and §7.4 (2× at more than 75% load); InputDriven §3–§5 (50× variance,
     +33% reward on LB); PEGASUS §3.
   - *Action:* in `lr-train`, draw the training subsample, transform seeds and replay seeds once per
     generation, and assert that every candidate's result rows use the same (cell_id, repeat) set.
     Compute the generation's objective only over that set. Log the per-cell spread. If subsampling
     cells, never let candidates within a generation see different cells.
   - *Stage:* training.
2. **Remove exact symmetries from the parameterization.**
   - *Evidence:* CMATut §5, where σ inflates under random selection along flat directions, and
     App. B.3 (ConditionCov, NoEffectAxis).
   - *Action:*
     - Pin θ₀ = −1, the default-logit anchor, at temperature 0, which fixes the scale.
     - Do not co-optimize temperature with an unnormalized θ. Either fix T, or learn T with ‖θ‖ = 1.
     - In the rank-r context, normalize each q_k to unit norm in the decoder, removing the p/q
       rescaling symmetry.
     - Watch the cma `condition number` and `sigma` traces for drift.
   - *Stage:* model-form.
3. **Keep CRN aligned through policy randomness.** Consume exactly one random draw per routing
   decision, or use counter-based draws keyed by (seed, request id), instead of a shared sequential
   stream.
   - *Evidence:* PEGASUS §4.3 and §5, where how the simulator consumes random numbers changed search
     quality; ES §2.1 (shared seeds and noise table). Section 4 of these notes shows the existing
     seeded default picker shares one stream and consumes a variable number of draws on ties.
   - *Action:* specify this in `learned-choice` and `sticky-session`, and in the parity test.
   - *Stage:* training. Magnitude is a **hypothesis**.
4. **Measure frozen noise, not just reproducibility. Adapt evaluation effort to rank instability.**
   - *Evidence:* UHCMA §III and §IV (rank-change measure; ε-perturbed re-evaluation; treat by more
     evaluations, then σ; r_λ = 0.3, α_σ = 1 + 2/(n + 10)); CMATut §4, end of "Step-Size Control" (CSA fails on very noisy
     objectives); UHCMA §III (raise µ and λ in preference to resampling alone).
   - *Action:*
     - In setup or calibration, re-score a few policies under (a) new repeat seeds, (b) new transform
       seeds and (c) tiny θ perturbations (ε ≈ 0.1σ₀).
     - During training, re-evaluate about 30% of each generation on fresh seeds and track rank
       changes. When rank change exceeds the threshold, add cells or repeats, then raise λ, before
       trusting σ. `cma.NoiseHandler` implements UH-CMA-ES.
   - *Stage:* training.
5. **Calibrate σ₀ and the per-coordinate scales from offline decision-flip rates and feature
   spreads.**
   - *Evidence:* ARS §3.2 (state normalization was required); CMATut App. A (scales must not differ by
     orders of magnitude) and §5 (diagonal invariance only via the initial C); ES §2.2 (perturbations
     that do not change actions give no signal); Park §3.1 (needle-in-haystack); CMATut App. B.4
     ("observation of a flat fitness should be rather a termination criterion and consequently lead to
     a reconsideration of the objective function formulation").
   - *Action:*
     - Log candidate feature tables from default-router replays of train cells.
     - Offline, with no replay, compute each feature's within-set sd per family, and set cma
       `CMA_stds` ∝ 1/sd.
     - Pick σ₀ so that roughly 10–30% of decisions flip relative to θ₀. That fraction is a
       **hypothesis** to tune.
     - Keep v1 feature definitions frozen. Do the scaling in parameter space only.
   - *Stage:* training.
6. **Treat agentic and multi-turn traces as exogenous think times, not exogenous timestamps.**
   - *Evidence:* CausalSim §1 and §3.1 (exogenous-trace vs exogenous-latent assumption; biased
     simulation flipped a deployment conclusion); Pensieve §4.1.
   - *Action:*
     - For AgentX, synthetic sessions and Mooncake conversations, prefer closed-loop session replay in
       which turn k+1 arrives at completion(k) plus the recorded think time.
     - If open-loop `trace_timestamps` are used, log per policy the fraction of turns that arrive
       before their predecessor completes, and flag cells with a high fraction.
     - Report whether learned-vs-default rankings agree between the two modes.
   - *Stage:* sim2real.
7. **Train on a load mixture and give the model a router-observable load-regime signal.**
   - *Evidence:* Decima Table 2 (a mismatched load trains worse than the tuned heuristic; mixed training
     plus an IAT hint is best) and §7.4 Fig. 14 (no global summary means no notion of load and
     instability at high load).
   - *Action:*
     - Ensure train cells cover L1/L2/L3 for every family and N.
     - For feature_set v2, consider a per-request load-regime feature computable live, such as x̄_S of
       active_requests_s or an EWMA arrival rate if the plugin API exposes it.
     - Validate on unseen load levels.
   - *Stage:* features.
8. **Choose loads per N with pooling in mind. Linear scaling at matched per-worker load can make large-N
   cells uninformative.**
   - *Evidence:* LBSurvey §1, §2.4 and Table 1 (at fixed λ < 1, JSQ waiting vanishes as N grows while
     random's stays constant; the Halfin–Whitt slack is β/√N).
   - *Action:* in calibration, find the near-knee level separately for each N, using the default router
     at each N, instead of only scaling the N = 4/8 rate. Report per-N where round-robin and default
     differ beyond noise. Check for ceiling effects (default good_frac near 1) at N = 16/32.
   - *Stage:* calibration-eval. The LLM batch servers are not M/M/N, so this is a **hypothesis**.
9. **Audit simulator exploitation explicitly.**
   - *Evidence:* DigEvo "Unintended Debugging" (adaptive time-step and integration-error exploits);
     Puffer §5.2 (emulation-trained policy was "horrible" in reality; rankings differed).
   - *Action:* in the Stage 4 audit:
     - Compare the engine-state distributions the learned policy induces against default (batch size,
       per-worker context length, long-context concentration, KV utilization).
     - Flag shifts into regions where AIS is least validated, such as > 32K context.
     - Re-score the final policies under independently perturbed prefill and decode speedups, and if
       available under non-AIS mocker timing. A gain that disappears only for the learned policy is an
       exploit signal.
   - *Stage:* sim2real.
10. **Penalize or track per-cell regressions. Do not optimize only the mean ratio.**
    - *Evidence:* Genet §2 Fig. 2b (wins on average, loses in a substantial fraction of environments),
      §4 (gap-to-baseline curriculum, w = 30%, gains of 8–25%, 14–24% and 15%) and §7 (use the max gap
      over a baseline ensemble).
    - *Action:*
      - Report win/loss fraction and the worst-cell ratio against default and against the best tuned
        baseline per cell.
      - Optionally train on mean ratio − λ·mean(max(0, 1 − ratio_c)).
      - Every k generations, re-weight train cells by their gap to the best baseline, holding weights
        fixed within a generation.
    - *Stage:* training and reporting.
11. **Guard the ratio objective against heavy tails and small denominators.**
    - *Evidence:* UHCMA §III (median versus mean under heavy tails); Puffer §3 (heavy tails dominate
      uncertainty); ES §2.1 (rank shaping removes outliers).
    - *Action:*
      - Exclude or cap cells where default good_frac is below about 0.1 (calibration should avoid
        these).
      - Clip per-cell ratios, for example to [0, 3], or use a trimmed mean across cells.
      - Report the untransformed mean alongside.
    - *Stage:* calibration-eval.
12. **Check the goodput-rate denominator for horizon gaming.**
    - *Evidence:* Decima §5.3 (fixed horizons let the agent defer large jobs past termination; fixed
      by memoryless termination) and §7.4 (batch-trained policy starves large jobs).
    - *Action:*
      - Verify how `goodput_request_throughput_rps` defines duration.
      - If the drain tail is included, prefer good_frac at fixed open-loop load, or a duration fixed
        to the arrival window, as the training objective.
      - In the Stage 4 audit, check per-request lateness by arrival time, especially requests near the
        window end.
    - *Stage:* calibration-eval.
13. **Randomize nuisance dynamics during training, and keep the test perturbations outside that
    range.**
    - *Evidence:* DynRand Tables II and III (without randomization, 0.0 real success; latency and
      observation noise were the most important to randomize); Pensieve §4.1 (variety of conditions).
    - *Action:*
      - Per train cell, draw prefill and decode speedup independently from a modest range such as
        [0.9, 1.1], with fixed draws per cell for CRN.
      - Keep the plan's 0.8 and 1.2 for the test-time robustness check.
      - If replay can delay router state or KV events, randomize that delay too.
    - *Stage:* sim2real.
14. **Watch for stale-state herding and worker-reservation policies in the degenerate-policy audit.**
    - *Evidence:* MFLB §1 (JSQ herds under delayed information; the number of routers matters); Park
      §3.3 Fig. 2 (a learned load balancer reserves a server for small jobs and breaks under shift);
      InputDriven App. J.
    - *Action:*
      - Compute per-worker request share by ISL bucket, family and session.
      - Flag persistent segregation and test it under the ISL-stretch hold-out.
      - Keep temperature > 0 in the ablation space, since softmax randomization counters herding.
      - If replay supports multiple router replicas or delayed KV events, run a sim2real sanity check
        with them.
    - *Stage:* generalization.
15. **Report optimizer-seed variance and count every evaluation, including baseline tuning.**
    - *Evidence:* ARS §4.2 (3 seeds misleading; 100 seeds show frequent local optima) and §5 (count
      rollouts for every tested hyperparameter); Puffer §5.3.
    - *Action:*
      - Run at least 3 independent CMA-ES restarts per rung and per tuned baseline (an IPOP-style
        increasing λ is optional, per CMATut App. A). Select on validation.
      - Report spread across restarts and total evaluations per method, including sweeps.
      - Report paired per-cell differences with bootstrap CIs over cells.
    - *Stage:* reporting.
16. **Spend budget by model complexity, and treat scaled-down training as legitimate.**
    - *Evidence:* PEGASUS §4.1 Thm 1 (scenarios needed scale with policy-class dimension, not state
      size; the bicycle task used m = 30); Decima App. I (10× fewer executors, 3% loss).
    - *Action:*
      - Size the train cells per generation by rung, with fewer for M1 (about 8 parameters) and more
        for M2 (8 + 2·8·r).
      - Prefer N = 4 and short windows early, and full cells late, keeping CRN within each generation.
      - Accept M2 only on validation gain, as the plan already requires.
    - *Stage:* training.

## 4. Read-only code observation relevant to lesson 3

In `WT/lib/router-plugins/builtin/src/default/selector.rs:26-35`, `new_seeded` constructs one
`Arc<Mutex<fastrand::Rng>>` documented as "Clones share its random stream". In
`WT/lib/router-plugins/builtin/src/default/picker.rs:94-111`, temperature 0 without an rng draws
unseeded `fastrand::usize(0..ties)` per tie, consistent with PLAN's "tie-break is unseeded". With the
seeded rng, draws come from the shared stream (`picker.rs:113-134`).

Implication (**hypothesis**): when the number of draws per decision differs between policies (e.g.
reservoir tie-breaking consumes draws only on ties), the streams desynchronize after the first
differing decision. Candidates then stop sharing random numbers for the remaining requests. This
weakens CRN.

Mitigation: one draw per decision regardless of ties, or counter-based randomness keyed by (seed,
request id). Request-keyed draws also survive closed-loop reordering. Before relying on request-keyed
draws, check whether the plugin API exposes a stable request id.

## 5. Confirmations of the current plan (no change needed)

- **Black-box search over linear policies is well supported.** ES variance is independent of horizon
  [ES §3.1]. Linear policies plus random search are competitive [ARS §1, §4.2].
- **The default-cost anchor (θ₀ reproduces default) matches the literature.** Park recommends
  "bootstrapping from existing policies" [Park §3.1] and hybrids with heuristics [Park §3.4]. Puffer's
  winning design kept a classical controller [Puffer §4].
- **Tuning every baseline with equal budget is the right standard.** Decima's strongest rival was an
  α-swept heuristic [Decima §7.1]. Puffer found simple BBA hard to beat [Puffer §1].

## 6. Not transferable or rejected

- **InputDriven's MAML or multi-value baselines** are tied to actor-critic and have no direct role in
  CMA-ES. Only the principle of per-input baselines (lesson 1) transfers [InputDriven §5].
- **Genet's BO search over environment configs** is heavier than needed. The campaign's cell grid is
  small, so simple gap-weighted re-sampling of existing train cells captures the idea [Genet §4.2].
- **LSTM or recurrent policies** [DynRand §V-A] conflict with the O(N·d) stateless runtime constraint.
  Only the randomization lesson transfers. A cheap stateless analogue is an explicit load-regime
  feature (lesson 7).
- **CausalSim's tensor completion** needs RCT data from live deployments, which the campaign lacks. Only
  the causal-validity check (lesson 6) transfers.
