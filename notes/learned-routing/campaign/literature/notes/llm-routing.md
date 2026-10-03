# Literature notes: LLM request routing (learned and heuristic) for the learned-routing campaign

Scout angle: llm-routing. Written 2026-10-02 for the campaign in
`<repo>/notes/learned-routing/PLAN.md` and `<campaign-root>/CONTRACT.md`
(Qwen3-32B, vLLM 0.24.0, H100 SXM, TP2, aggregated; conditional logit over per-worker features,
CMA-ES on normalized episodic goodput, train N in {4, 8}, test N in {2, 6, 16, 32}).

Citation format: (FirstAuthor Year, section, PDF page). Page numbers are PDF page indices of the
downloaded file, located with `pdftotext` per page. Claims marked **[campaign inference]** are my
own reasoning from the cited evidence, not a claim made by the paper.

## 0. Method

- Searches (WebSearch + arXiv API): paper titles from the brief (Intelligent Router, Preble, DLPM,
  LMetric, DualMap, Ramjet, llm-d precise prefix, Autellix, ThunderAgent); keyword searches for
  "reinforcement learning request routing LLM inference KV cache", "learned load balancer LLM
  generalize instance count", "session affinity agentic multi-turn routing", "simulator trained
  router"; citation trails from Lodestar, LMetric, SMetric, GORGO, and Calibrate-then-Route
  related-work sections.
- Ramjet is not a paper: the campaign port cites Helix's GitHub load balancer
  (`WT/lib/router-plugins/builtin/src/ramjet.rs` header, `https://github.com/helixml/ramjet`).
- 12 PDFs downloaded to `/tmp/learned-routing-lit/pdfs/llm-routing/` (text in
  `/tmp/learned-routing-lit/txt/llm-routing/`). Four more sources read as web pages (Autellix routing
  section, llm-d KV-cache blog, "A Year in LLM Serving" load-balancing section, llm-d scheduler docs).
- Read depth: full for Calibrate-then-Route, Lodestar, GORGO, CacheRoute; method + evaluation +
  limitations for Jain, LMetric, SMetric, DualMap, Wu, Preble, ThunderAgent, AgentServeSim.

| File | Paper | Venue |
|---|---|---|
| `tumkur2026-calibrate-then-route.pdf` | Tumkur et al., "Calibrate, Then Route: A Measured Study of Learned Request Routing for Disaggregated LLM Serving", arXiv 2609.16206 | preprint, Sep 2026 |
| `lim2026-lodestar.pdf` | Lim et al., "Lodestar: An Online-Learning LLM Inference Router", arXiv 2606.00946 | preprint, May 2026 |
| `jain2024-intelligent-router.pdf` | Jain et al., "Intelligent Router for LLM Workloads", arXiv 2408.13510 | MSR; preprint |
| `zhang2026-lmetric.pdf` | Zhang et al., "Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling" (LMetric), arXiv 2603.15202 | OSDI'26 (per acknowledgments) |
| `wang2026-smetric.pdf` | Wang et al., "SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling", arXiv 2607.08565 | preprint, Jul 2026 |
| `toniolo2026-gorgo.pdf` | Ricci Toniolo et al., "GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving", arXiv 2602.11688v3 | preprint |
| `wu2026-randomized-eviction-learned-routing.pdf` | Wu, Silwal, Zhang, "Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective", arXiv 2601.18999 | ICLR 2026 |
| `cheng2026-cacheroute.pdf` | Cheng, "CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving", arXiv 2608.19677 | preprint, Aug 2026 |
| `yuan2026-dualmap.pdf` | Yuan et al., "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving", arXiv 2602.06502 | ICLR 2026 |
| `srivatsa2024-preble.pdf` | Srivatsa et al., "Preble: Efficient Distributed Prompt Scheduling for LLM Serving", arXiv 2407.00023 | ICLR 2025 |
| `kang2026-thunderagent.pdf` | Kang et al., "ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System", arXiv 2602.13692 | ICML 2026 |
| `rajib2026-agentservesim.pdf` | Rajib et al., "AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent Programs", arXiv 2606.09613 | preprint |

---

## 1. Learned / RL routers

### 1.1 Lodestar (Lim et al. 2026) — online-learned reward predictor router

**Setup.** Per-request greedy routing: a shared MLP f_θ([x_i ‖ r]) predicts reward (−TTFT) for each
instance i, route to argmax (§4.1, p5). Built on AIBrix; 8×A30 homogeneous and 8×A30 + 8×V100
heterogeneous clusters, Llama3-8B on vLLM v1 with chunked prefill and prefix caching (§5.1, p9–10).

**Model form.**
- MLP, 3 hidden layers × 128, ReLU, dropout 0.1 (§4.1, p5).
- "The same parameters θ are shared across all instances, and instance identity is never an input."
  The paper states two consequences: instance-count independence (works as instances are added or
  removed) and no "herding effect" where the model would associate an instance ID with good
  performance and oscillate (§4.1, p5).
- Linear regression on the same features predicted TTFT much worse than the MLP (Fig. 5, p6). Note
  this is predictor accuracy, not routing quality.

**Features** (§4.1, p6): input token length; expected prefix KV hit ratio per instance (from a
gateway radix tree of its own routing history); running requests; queued requests; in-flight prefill
tokens and in-flight decode tokens kept separate ("prefill is compute-bound and decode is
memory-bandwidth-bound ... Collapsing them would hide which bottleneck the instance is approaching");
GPU memory (KV) utilization; GPU model (categorical). Deliberately excluded: GPU util, SM activity,
memory-bandwidth util, because sampling artifacts add "more noise than signal" (p6).

**Training.** Online only, no offline pretraining in the headline results (§5.1, p10). Retrain every
θ = 1000 new samples (§4, p5). Training data: FIFO buffer of 5000 recent samples plus a 5000-sample
replay buffer admitted by a gradient-coreset criterion for diversity (§4.3.2, p8). Learns "within ≈5
minutes" (abstract, p1).

**Key failure analysis — circular dependency** (§3.3, p4; Fig. 3a): a TTFT predictor trained offline
on logs from Least-Request and Prefix-cache-and-load-aware routing is accurate on held-out offline data,
but "when the same model is deployed to route requests, its predictions collapse ... systematically
over-optimistic and loses all ranking signal between instances—fatal for a routing rule that picks
arg min." The variant "Lodestar offline train only" performed very poorly when trained on a different
cluster (§5.3, p10–11). A "mid-frozen" model trained during a 5% prefix-sharing phase failed after the
workload shifted to 50% sharing (§5.3, p11).

**Greedy myopia fix** (§4.1, p6): when cluster GPU memory utilization > 80%, first filter candidates to
k = 2 instances by consistent hashing on the shared-prefix group, then take the max predicted reward.
Ablation: removing the filter raises average TTFT and produces tail excursions near 12 s versus under
~7 s (Fig. 14, p11).

**OOD guard** (§4.3.1, p8): fall back to the heuristic on cold start, on features outside "the ranges
observed in the training buffer", or on RPC timeout.

**Evaluation.** Mooncake conversation, toolagent, synthetic traces plus synthetic prefix-sharing
workloads at 10/30/50/70%/mixed; RPS from medium load to saturation; metrics mean and P99 TTFT
(§5.1, p9–10). Headline: 1.41× lower mean TTFT and 1.47× lower P99 versus AIBrix
prefix-cache-and-load-aware (abstract, p1). Motivation figure: the best prefix-hit threshold τ differs
by workload (ToolAgent τ = 20%, Conversation τ = 60%), and even the best τ is ~300–400 ms behind
Lodestar (Fig. 2, §3.2, p3).

**Relevance.** Strongly supports the campaign's shared per-worker utility (N-agnostic), its
on-policy training (CMA-ES in replay sidesteps the circular dependency), and suggests a
consistent-hash candidate feature and an OOD guard.

### 1.2 Calibrate, Then Route (Tumkur et al. 2026) — learned router developed in a simulator, validated on hardware

**Setup.** Disaggregated vLLM 0.12 + NIXL on 8×A40, Qwen2.5-3B, KV budget 60k tokens per engine,
primary topology 2 prefill + 4 decode (§IV-A, p2–3). Router scores each candidate by marginal
completion time: w_b·backlog·σ(SLO) + w_s·ô_r·τ_dec·(1 + ρ·press⁺)·σ(SLO) + w_q·queue (§III-A, p2).
Weights hand-set; only constants (τ_pre, τ_dec) fitted (§VI Scope, p6).

**Protocol** (§IV-B, p3–4): token-count correctness gates, cross-GPU greedy-output equivalence, prefix
cache reset before every run "after we observed identical configurations measuring 20+ points apart
from session ordering"; each workload driven where round-robin is "mid-collapse" because "routing
policies only separate under contention". SLOs: TTFT 175 ms / TPOT 39.7 ms (tight), 3× loose.

**Results.**
- Mixed bursty: learned 0.864 goodput vs 0.835–0.847 for RR, JSQ, length heuristic (Table II, p3).
- Calibration: simulator-derived constants cost 4.5 goodput points and ~40% of the tail advantage;
  "with mispriced constants the router balances request counts almost perfectly while letting token
  load skew worse than JSQ" (§V-B, p4). Simulator τ_dec was 7.4× smaller than measured (§III-B, p2).
- Pool width: ties JSQ at width 3, leads at 4, reconverges at 6 (§V-D, p4).
- Starved regime (1P + 2D): learned collapses to 0.292 while RR holds 0.680; "greedy cost minimization
  concentrates load on whichever instance momentarily prices cheapest"; reproduced four times,
  "intrinsic"; mitigation suggested: power-of-two sampling or admission guard (§V-E, p4; §VI, p6).
- Feature ablation: predicted output length is load-bearing (−3.2 pts steady, −4.7 saturated); cache
  pressure inert because the KV budget never saturated; SLO term helps at moderate load, backfires
  under deep saturation (§V-G, p4). Predictor error injected up to 150% leaves goodput within noise
  (§V-H, p4–5).
- Homogeneous chat: learned 0.760 vs heuristic 0.855 — "a queue count is close to a sufficient
  statistic and the additional signals add only noise" (§V-C, p4).

**Sim-to-real findings** (§VI, p5): simulator constants 7–13× off; after re-fitting physics and adding
continuous batching, the simulator reproduces goodput with mean absolute residual 0.068 "but it does
not reproduce the measured winner on any workload"; the simulator's earlier apparent skill "was an
artefact of serialisation". "The simulator sets each instance's backlog signal from the cluster's own
clock, so the router receives its true remaining work; the deployed proxy must estimate that backlog
... a simulator can hand a policy information no deployment will have" (p5).

**Relevance.** The closest analogue to this campaign (policy developed in a discrete-event simulator,
goodput objective). Its warnings on information leakage through simulator state, ranking transfer,
small-N parity and scarcity herding all apply.

### 1.3 Intelligent Router (Jain et al. 2024, MSR) — heuristic-guided RL

**Setup.** 4× Llama-2-7B on V100 with vLLM FCFS; 2000 requests at 20 RPS drawn from five task types;
RL action every 0.02 s (§6, p8–9).

**MDP** (§5.3, p7–8): state = queue length, next request's prompt tokens and predicted decode bucket,
per-instance histograms of prompt and decode buckets (m × n_p, m × n_d matrices), capacity, earliest
completion. Action ∈ {instance 1..m, no action} — the router may hold a request. Reward: queue
penalty, completion reward, plus a "workload impact estimator" mixing penalty applied as
heuristic guidance whose weight decays over episodes (λ_k = e^{−0.5k}) (§5.3–6, p8–9).

**Mixing-penalty features** (§5.2, p7–8): impact on prefill ∝ prompt tokens × decode tokens running;
impact on existing decodes ∝ number of requests. Profiled gradients 3.2e−4 (prefill) and 3.3e−5
(decode) per token (§4, p4). Interference evidence: a p=1000/d=1000 request's latency rises from 17 s
to 31 s when p=500/d=500 requests arrive every 50 iterations (§4, p4–5; Fig. 1a).

**Output-length predictor** (§5.1, p7): DistilBERT bucket classifier; 5.5% accuracy on unequal buckets
without a task-type hint, 79.15% with it.

**Results.** End-to-end latency vs RR: baseline RL −4.35%, workload-augmented −7.79%,
workload-guided −11.43% (§6.1, p9). JSQ, max-capacity, min-min only 0.46%, 2.60%, 1.50% better than RR
(p9). Guided RL held requests at the router 2.05 s on average and reduced preemption; baseline RL held
0.59 s and suffered preemptions (p9). 8 instances: −11.62% vs RR (p10). Retrained per hardware/model.

**Limitations [campaign inference].** State is a fixed m × buckets matrix, so the policy is tied to m
and must be retrained per N; baselines are weak (RR only after the first table); gains are on E2E,
not SLO goodput.

### 1.4 GORGO (Ricci Toniolo et al. 2026) — evolution-strategy tuning of an additive routing cost

**Policy** (§3, p2–3): TTFT cost = W_rtt·RTT + W_queue·(in-flight tokens tracked at the proxy) +
prefill(uncached tokens); W_prefill fixed to 1 because only weight ratios matter.

**Training** (§3, p3): (1+1)-ES with log-space multiplicative perturbations, bounds [0.05, 2.0] and
[0.05, 0.5], Rechenberg 1/5 rule, objective p95 TTFT of a rolling 128-request window. Converged after
672 samples (18 ES steps) (Fig. 3, p5).

**Held-out evaluation** (§4, p5): tune on one 30-min window (Apr 5), freeze weights, evaluate on Apr 6
and Apr 7 with `time_scale` raised to 2.0 and 3.0 "to control replica saturation". Held-out p95 TTFT
improves 6.9–15.5% and p95 E2E 14.3–30.9% over session affinity / prefix-cache / least-load
(abstract, p1). Recommendation: "sweep across timescales to find a clean under-saturated traffic
profile" first (p5).

**Failure — reward hacking** (§5 and App. D, p7–8): optimizing TTFT alone, ES set w_queue → 0 and sent
"100% of requests to the closest replica"; best TTFT at every percentile, worst E2E and ITL tails
(E2E p95 12.6 s). "Adding a lower bound to the queue term mitigates but does not eliminate this
trade-off" (Limitations, p8). Also: "a request log generated under one policy does not contain the
counterfactual cache and queue evolution that would have arisen under" another (§2, p2).

**Relevance.** Methodologically identical to the campaign's M0 (ES on a cost-form router, frozen,
evaluated on held-out time windows), with a documented reward-hacking failure.

### 1.5 Wu, Silwal, Zhang (ICLR 2026) — KV eviction + learning-based greedy routing (LBGR)

**Model** (§3, p4–5): makespan formulation; service cost α_cached·hits + α_miss·misses + output cost.
Proves LRU leaf eviction (SGLang) is O(n)-competitive; randomized leaf eviction (RLT) is O(log n) and
optimal among randomized algorithms (§3–4, p5–6).

**LBGR routing** (§4.2, p6–7): estimated latency = service-time estimate (global radix tree hits) +
queue-load estimate (sum of assigned service-time costs, decayed by ρ every Δt, released on completion)
+ an online per-worker linear residual regression θ_iᵀφ_ij. Criticizes "raw counts of pending queries"
as "overly simplistic" (§1, p2).

**Evaluation** (§5, p7–9): Llama-3.1-8B on L40 (1–10 workers, 4 default) and 70B / Mixtral on H200;
GSP, ShareGPT, UltraChat, Loogle; outputs 4–128 tokens (4 default); random and adversarial
round-robin arrival order; 4–20 RPS. Claims up to 11.96× median latency and 14.06× median TTFT vs
SGLang cache-aware. Worker sweep 2–10: throughput converges for ≥6 workers at fixed 12 RPS (p9).

**Caveat [campaign inference].** Tiny default outputs (4 tokens) make this a prefill-only regime; the
magnitudes do not transfer. The transferable points are (i) routing and eviction policy are coupled,
so the learned router will exploit whatever eviction the mocker implements, and (ii) decayed
assigned-work load estimates.

### 1.6 AgentServeSim (Rajib et al. 2026) — validated agent-program simulator + LLM-driven policy search

- The unit of simulation is the agent program; "a Program Orchestrator causally releases successor
  turns from simulated predecessor completions"; request-stream simulators with externally supplied
  arrivals "cannot jointly represent the cross-turn state and policy-dependent successor releases"
  (abstract, p1).
- Validation: 20 paired sim–real cells (B200, RTX PRO 6000; Llama-3.1-8B/70B; SWE-bench and BFCL
  agents; five arrival rates): mean JCT error within 5.5% / 5.2% (abstract, p1; §4, p6–7).
- Policy search: OpenEvolve, 50 iterations per axis, each evaluation 19 min on one CPU core; best
  evolved policies improve mean JCT only 0.5% (KV retention) and 2.8% (scheduling) over hand-written
  seeds; "deployment-specific by design" (§5, p8).

### 1.7 Not downloaded, abstract-level

- Jha et al. 2024, "Learned Best-Effort LLM Serving" (arXiv 2401.07886): DQN that adjusts service
  quality (model variant) by task distribution and load; claims robustness to arrival and task shift.
  It routes across quality levels, not replicas of one model, so it is tangential.
- RouteBalance (Da & Kalyvianaki 2026, arXiv 2606.17949): fuses model routing and load balancing; its
  four-arm isolation finds "the learned predictors contribute calibration and SLO headroom rather than
  the headline frontier" (abstract).

---

## 2. KV-cache-aware heuristic routers (the campaign's baselines and their evidence)

### 2.1 LMetric (Zhang et al. 2026)

- Score = P-token × BS, argmin; P-token = new prefill tokens if routed there, including queued prefill
  on that instance; BS = running + queued batch size (§5, Fig. 17, p9). Hyperparameters cancel in the
  comparison (§1, p2).
- Linear combinations need per-workload tuning: optimal KV weight 0.7 on ChatBot, 0.55 on API (§4.4,
  p6); "a statically tuned weight cannot always achieve competitive performance ... the optimal weight
  may vary over time. Intuitively, if the GPUs are idle, we need to prioritize KV$ hits ... if
  prioritizing KV$ results in an imbalance, we need to reduce the weight" (p6).
- Filter-based (AIBrix-style) needs workload-specific thresholds (p6). Simulation-based (llm-d /
  Vidur-style) with a non-tuned simulator is much worse; even a tuned one has ~10% of requests with
  >20% TTFT error (§4.6, p7–8).
- Indicator choice: P-token beats (1 − hit ratio): 14.4% lower P50 and 42.8% lower P95 TTFT, because
  P-token also sees queued prefill (p9). BS beats total tokens (the Dynamo choice) as the decode-load
  indicator because decode time is more stable across batch sizes (Fig. 19, p9).
- Failure condition: KV hotspots where class popularity exceeds cache coverage (x/x̄ > |M|/|M̄|); rare
  in their traces; a two-phase detector filters suspected hotspot instances (§5.2, p10–11).
- Evaluation: 16× H20, vLLM v1, Qwen2-7B and Qwen3-30B (MoE); traces ChatBot, Agent (Qwen), Coder
  (BAILIAN), ToolAgent (Kimi); default load = half of the testbed's maximum rate "because when the
  arrival rate approaches serving capacity, BAILIAN commonly reroutes requests" (§3, p4). Baselines
  vLLM, Dynamo (tuned per workload), llm-d, BAILIAN (tuned), Preble, PolyServe, all reimplemented in
  one Rust router (§6, p11–12). Production canary: −39% mean TTFT and −51% mean TPOT (p13). On
  ToolAgent, llm-d's simulator had 10% lower mean TTFT but 30% higher TPOT (p12).

### 2.2 SMetric (Wang et al. 2026) — session-centric routing for agentic serving

- Agentic workload findings from two BAILIAN coding-agent traces (May 2026): KV reuse > 80%
  (Finding 1); > 65% of reuse is intra-session (Finding 2); system prompt gives 18–20% of reuse to first
  turns (Finding 3); ~90% of reuses recur within ~100 s (Finding 4); top 25% of sessions carry > 80% of
  tokens (Finding 5); but load can still be balanced at scale because requests per minute are 43–48×
  the number of instances (Finding 6) (§3, p4–5).
- "a single 128K-token request consumes about 32 GB of KV$ on the popular Qwen3-32B model" (§2, p3).
- Without a global (CPU/remote) KV tier, LMetric's reuse ratio is only 45% vs 74% for the KV-favoring
  linear BAILIAN scheduler, and LMetric serves 36% less TPS within SLO (§4.1, p6). Load-balance-only
  is within 7% of the best when the global tier is fully provisioned (p7).
- Sticky-by-session (llm-d session) still has mean imbalance 2.17× because sessions grow and their
  size is unknown at first placement (§4.2, p7).
- Policy (§4.2, Fig. 13, p8): first turn → least-loaded (LMetric-like score, still KV-aware via
  hits); follow-up → instance with the highest local hit, unless it cannot meet the TTFT SLO × SLACK, in
  which case migrate to the least-loaded instance. Session detection is stateless: turn index from the
  number of historical messages; eviction detection by comparing the best local hit to the expected hit.
- Load model: prefill time ≈ c_lin·n + c_attn·n(L − n/2), L = context length, n = new tokens, "from
  causal (lower-triangular) attention" (p9). SMetric criticizes Dynamo's token-count load as inaccurate (p9). Ablation: token-count → this cost model +4.2% / +1.9% TPS;
  load-aware filter → SLO-aware migration +7.2% / +24.7% (p12).
- Evaluation: 32× Qwen3-Coder-30B-A3B (PD colocated and disaggregated) and 8× Qwen3-235B; global tier
  Mooncake + LMCache; TPS within SLO with TTFT SLO = b + a·L (b = 1 s, a = 62.5 ms per 1K prompt tokens)
  and TPOT ≤ 20 ms (p10). Replay forces exact recorded output lengths and rewrites follow-ups so KV
  reuse stays consistent (p9–10). Gains vs best baseline grow with load: within 1% at 6× load, +4% at
  8×, +13% at 9×, +26% past saturation (p10). Peak +9% (30B) and +15% (235B) (p10). Robust to SLO
  settings (Fig. 17) and to its two hyperparameters (p12–13).

### 2.3 DualMap (Yuan et al., ICLR 2026)

- Two independent prefix hashes give two candidates; choose by cache affinity until expected TTFT
  exceeds the SLO, then the less-loaded candidate (§3.2, p6–7). Threshold expressed as max pending
  prefill tokens clearable within the TTFT SLO (p7).
- Why only two: "A global strategy that 'collects information from all instances and selects the best
  one' is effectively equivalent to using d = n choices ... negligible improvement ... in terms of
  maximum load, while severely degrading KV cache reuse" (§2, p5).
- Min-TTFT (per-request optimal, Mooncake-style) "may oscillate between cache-aware and load-aware
  decisions, leading to frequent cache misses under load fluctuation" (p6).
- Adaptive hash-prefix length by prefix hotness (ρ > 2/n means hot) (p6); hotspot migration within the
  candidate pair (p7); rendezvous/consistent hashing for elasticity (p7).
- Load-balance metric: coefficient of variation of pending prefill tokens across instances (§2, p3).
- Evaluation: 8 instances, Qwen2.5-7B/14B on Ascend 910B, Mooncake Conversation and Tool&Agent; effective
  request capacity (fraction with TTFT < 5 s) and goodput; up to 2.25× capacity (§4, p7–8).

### 2.4 Preble (Srivatsa et al., ICLR 2025)

- E2: exploit (route to the GPU holding the prefix) when matched prefix > remaining tokens; otherwise
  explore with cost L_i + M_i + P_i (§3.2, p5):
  - L_i = load over a recent window H (default 3 min; results insensitive to H), from regressions of
    prefill time on missed tokens and average output length in H. Uses windowed history rather than
    instantaneous load because "the placement of a prefix has a longer-term effect" (p5).
  - M_i = eviction cost: Σ over nodes that would be evicted of prefill time × the node's hit rate (p5).
  - P_i = prefill time of the missed tokens.
- Post-assignment rebalancing and prefix autoscaling (p5–6). Prefill/decode balancing: send
  prefill-heavy (explored) requests to decode-heavy GPUs, which "have unused computation capacity"
  (p6).
- Evaluation: vs round-robin SGLang/vLLM data parallel; 1.5×–14.5× average latency (abstract, p3).
  LMetric found Preble "falls back to the linear-combination branch most of the time" (LMetric p13).

### 2.5 CacheRoute (Cheng 2026) — planned affinity

- Periodic routing table: admit highest-rate prefix keys to a warm set, place them by
  longest-processing-time-first on expected load; others use power-of-two choices (§3, p2–3).
- 60 H100 (30 TP2 destinations), Llama-3.3-70B fp8: 176 ± 11 QPS at p99 ≤ 3.5 s, 2.3× Preble; served KV
  hit 93.2% vs 64.1% cache-blind (abstract, p1). With top-K128 that fits the warm allocation, the three
  balanced cache-aware policies tie; top-K256 separates them (Table 2, p4).
- Ablation (8B): affinity alone raises KV hit 56% → 88% but imbalance to 3.46× and leaves capacity flat;
  LPT placement brings imbalance to 1.24× and moves the knee (p1, p5).
- Counterexamples: on two 32B workloads affinity moves KV hit only 1.1% → 11.8% and capacity falls to
  0.50–0.67× flat-LB (p5). Recommendation: gate deployment with a shadow replay at one load below and
  one near the knee; "a cache-hit increase is not enough" (p5).
- Analytic residency prediction missed served hit rate by 14.3 pp median, 44.7 pp p90 (p6).
- Statistics: SLO-capacity knee per run, five paired seeds, 95% Student-t CIs, censoring noted (p4).

### 2.6 llm-d precise prefix-cache scorer (blog, web)

- `https://llm-d.ai/blog/kvcache-wins-you-can-see`: Qwen-32B, 16 H100 as 8 pods × 2 GPUs, 150 customer
  groups with 6,000-token system prompts and 1,200-token queries, 3–60 QPS Poisson, cache demand 73% of
  cluster capacity. Precise (KV-event-indexed) vs approximate (routing-history) prefix scoring: TTFT
  p90 0.542 s vs 31.083 s; throughput 8,730 vs 6,944 tok/s; load-aware only 4,429 tok/s.
- Default llm-d scheduler profile weights: prefix-cache scorer 3, queue scorer 2 (llm-d / Red Hat docs
  via search).

### 2.7 Other heuristics read at abstract or web level

- Autellix (Luo et al. 2025, arXiv 2502.13965), load balancer §4.3 (web): calls ≤ 2048 tokens go to the
  least-used engine; longer calls stick to the program's engine (new programs → least used). Rationale:
  within-program cache hit > 90%, across programs it decays (§3.2); short requests hit ≥ 75% anywhere due
  to shared system prompts. Up to 1.4× throughput vs RR and least-used (§6.4). Threshold is not
  sensitivity-tested.
- "A Year in LLM Serving" (Nixon et al. 2026, arXiv 2608.13573), §7.2 (web): simulated round_robin,
  load_first, sticky, cache_first at N ∈ {5, 10, 15, 20}; metrics token hit ratio, session replication
  ratio (distinct instances per session) and max/mean token/s imbalance. Cache-first imbalance stays
  within 5–7% because many single-turn sessions act as fillers, "suggest[ing] that imbalance may grow
  for workloads with fewer one-off requests"; replication rises with N under cache-blind routing.
- DLPM/D2LPM (Cao et al. 2025, arXiv 2501.14312): locality-aware fair scheduling; fairness is out of
  scope here.
- Llumnix (Sun et al., OSDI 2024): live migration across instances; Dynamo places once, so not
  directly applicable.
- Continuum (Li et al. 2025, arXiv 2511.02230): TTL pinning of KV across tool calls plus program-level
  FCFS; engine-side.
- SkyWalker (EuroSys'26) and GORGO address cross-region routing; network RTT is not a factor in this
  campaign.
- Efficient LLM Scheduling by Learning to Rank (Fu et al., NeurIPS 2024): relative output-length ranks
  are predictable even when exact lengths are not; it is used for in-engine SJF, not routing.

---

## 3. Agentic / program-level serving

### 3.1 ThunderAgent (Kang et al., ICML 2026)

- Problem evidence (§3, p5): request-level KV-aware routing (vLLM router) "sends all requests from the
  same agentic workflow" to one node; memory load "between two DP nodes diverges by more than 20% for
  over 37 minutes, reaching a peak imbalance of 51%" in 90-min rollouts. SGLang's router-side radix tree
  "approximates worker KV state. Under high agentic concurrency, worker-side evictions from KV thrashing
  leave this tree stale, yielding both poor cache reuse and cross-node imbalance" (p5). KV thrashing
  raises E2E latency up to 7.14× (p2).
- Scheduler (§4, p6–7): cost = decode + prefill + recompute + unused + caching (space-time product);
  periodic thrashing detection; pause programs in the acting (tool) phase, shortest context first
  (recompute cost quadratic in context); global program-aware waiting queue that restores paused
  programs on whichever replica has memory. Exponential time-decay of idle programs' memory weight.
- 1.5–3.6× serving throughput (abstract).

### 3.2 Takeaways across agentic papers

- Session affinity captures most reuse (SMetric Finding 2, p4; Autellix §3.2 web), but unconditional
  stickiness imbalances memory and load because sessions grow (SMetric p7; ThunderAgent p5).
- Successful designs gate affinity on request properties: turn index / first turn (SMetric), call
  length (Autellix), SLO feasibility (SMetric, DualMap).
- Simulation of agentic workloads must release successor turns causally (AgentServeSim p1).

---

## 4. Cross-paper synthesis

### 4.1 Features that were informative

| Feature | Evidence | In campaign v1? |
|---|---|---|
| New prefill tokens after cache hits, including queued prefill (P-token) | LMetric p9 (beats 1 − hit ratio); DualMap p7 (pending prefill tokens); Lodestar p6 | yes, split as features 2 and 3; product not representable linearly |
| Batch size (running + queued) as the decode-load indicator | LMetric p9 (beats total tokens); Lodestar p6 | yes (`active_requests_s`) |
| Separate in-flight prefill and decode tokens | Lodestar p6 | yes |
| KV memory utilization | Lodestar p6; Preble eviction cost p5 | yes (`kv_load_frac`) |
| Product / multiplicative interaction of cache and load | LMetric §5 | no |
| Turn index / first-turn indicator | SMetric p8 | no (only `session_affinity`) |
| SLO-feasibility hinge (backlog vs TTFT budget) | DualMap p7; SMetric p8 | no |
| Consistent-hash candidate membership | Lodestar p6; DualMap p5; CacheRoute | no |
| Attention-aware prefill cost n(L − n/2) | SMetric p9, ablation p12 | no |
| Prefill × decode-load interaction (interference or spare compute) | Jain p4, p7–8; Preble p6 | partly (`isl_x_prefill_load` is prefill × prefill) |
| Predicted output length | Calibrate-then-Route p4 (load-bearing, error-tolerant); Jain p7 | excluded (leakage); non-leaky proxy possible |
| Windowed or decayed assigned work instead of instantaneous load | Preble p5 (window H); Wu p7 (decay ρ) | no |
| Per-prefix popularity / rate | CacheRoute p2–3; DualMap hotness p6; LMetric hotspot p10 | no |
| Excluded as noisy | GPU util, SM activity, memory-bandwidth util (Lodestar p6) | n/a |

### 4.2 How they trained

- Direct policy search on the end metric: GORGO (1+1)-ES on p95 TTFT, 672 samples for 2 weights (p5);
  AgentServeSim LLM-evolved code, 50 iterations, 19 CPU-min per evaluation (p8).
- Online supervised reward predictor plus greedy argmax: Lodestar (MLP, retrain every 1000 samples);
  Wu LBGR (online linear residual).
- RL with a heuristic-guided reward: Jain (decaying guidance).
- Hand-set structure with fitted constants: Calibrate-then-Route; SMetric's c_lin/c_attn profiling.
- Tuning-free scores: LMetric.

### 4.3 How they evaluated

- Metrics: SLO goodput / TPS within SLO (SMetric, Calibrate-then-Route, DualMap); SLO capacity knee
  (CacheRoute); mean and P99 TTFT (Lodestar, LMetric); diagnostics: KV hit ratio, load-imbalance
  max/mean or CV, session replication ratio.
- SLO definitions: fixed TTFT (DualMap 5 s; CacheRoute p99 3.5 s; Calibrate 175 ms) or
  length-scaled TTFT b + a·L (SMetric).
- Loads: half of max capacity (LMetric); RR mid-collapse (Calibrate); ladders up to and past saturation
  (SMetric, CacheRoute, Lodestar); time_scale re-scaling for held-out windows (GORGO).
- Held-out: later days (GORGO), a second independently collected distribution (CacheRoute), a second
  model size (SMetric, LMetric), worker-count sweeps (Wu 2–10; Calibrate widths 3–6). None of the
  learned routers report training on some N and testing on unseen N, except Lodestar's architectural
  N-independence.
- Statistics: paired seeds with Student-t CIs (CacheRoute); three traces with per-trace spread
  (Calibrate); 20 episodes (Jain).

### 4.4 What failed

- Offline-trained predictors deployed as policies (Lodestar §3.3).
- Simulator-derived cost constants; simulator rankings (Calibrate §V-B, §VI).
- Optimizing TTFT alone (GORGO App. D).
- Greedy argmin under extreme scarcity (Calibrate §V-E) and greedy best-of-N dispersing prefixes
  (DualMap p5–6; Lodestar CH filter).
- Affinity when the recoverable KV is small (CacheRoute 32B counterexamples).
- Unconditional session stickiness (SMetric p7; ThunderAgent p5).
- Multiplicative LMetric at very high reuse without a global KV tier (SMetric p6).
- Linear combinations with one static weight across workloads (LMetric p6; Lodestar Fig. 2).

---

## 5. Lessons for the campaign (ranked by expected reward)

### L0. The campaign's `lmetric` port omits the queued-prefill term of the paper's P-token (baselines)

- **Evidence.**
  - LMetric defines its KV indicator as "the number of queued new prefill tokens when routing a
    request to an instance, considering KV$ hits" (§1, p2). It credits this over (1 − hit ratio)
    "as it additionally considers the queued prefill tokens in each instance" (§5.1, p9).
  - SMetric restates LMetric as f = (q_i + L − c_i) × l_i, where q_i is "the prefill work already
    queued at instance i" (§4.1, p6).
  - The port scores only `uncached_prompt_tokens(context, cache).max(1) × (active_requests + 1)`
    (`WT/lib/router-plugins/builtin/src/lmetric.rs:152-155`; `signals.rs:20-28` subtracts cached
    blocks from the prompt only). It never adds the worker's active or queued prefill tokens.
- **Why it matters.** A weakened LMetric baseline inflates any "beats every heuristic" claim, and
  LMetric is one of the strongest published baselines (§2.1).
- **Action.**
  - Before Stage 4 baseline tuning, fix or add a paper-faithful variant: P-token = worker active
    prefill tokens + this request's uncached tokens, floored at 1.
  - Record it in `facts/UPSTREAM_FOLLOWUPS.md`.
  - Report both the as-ported and the paper-faithful variants.

Each lesson gives evidence, an action, and a campaign stage.

### L0b. Derive baselines' physical constants from this deployment and SLO, even at "published defaults" (baselines)

- **Evidence.**
  - Running the same router with simulator-derived constants cost 4.5 goodput points: it "collapses
    the scorer into queue counting" (Calibrate-then-Route §V-B, p4). The paper treats calibration as
    "part of the method" (§III-B, p2).
  - DualMap's switching threshold is "the maximum number of pending prefill tokens that a GPU can
    process within the SLO" (§3.2, p7).
- **Campaign state.**
  - The `dualmap` port defaults `pending_prefill_token_budget: 65_536` (`WT/.../dualmap.rs:57`).
  - The `llm-d-optimized-baseline` port defaults `peak_prefill_tokens_per_second: 15_928.0`, which its
    header calls "one calibration point for one model, GPU, and parallelism"
    (`WT/.../llm_d/optimized_baseline.rs:17,79`).
- **Action.**
  - For the "defaults" arm, set these constants from the AIS prefill-throughput profile of Qwen3-32B
    TP2 H100 and the family's frozen TTFT SLO, which is what a real operator would do. Keep the CMA-ES
    tuning arm separate.
  - Record the derivation in `facts/`.

### L1. Put LMetric inside the M1 hypothesis class by adding log features (features / model-form)

- **Evidence.**
  - LMetric (argmin of P-token × BS) beat per-workload-tuned linear schedulers, including tuned Dynamo,
    on 16 GPUs (§6, p11–12) and in production (p13).
  - A linear utility over raw features cannot represent a product. In log space it can:
    argmin P·B = argmin(ln P + ln B) **[campaign inference]**.
- **Action.**
  - Add `log_ptok = ln(1 + (active_prefill_tokens + new_prefill_tokens)/1024)` and
    `log_bs = ln(1 + active_requests)`, in v1 if it is not yet frozen, otherwise as v2.
  - Add a parity test: θ = −(e_log_ptok + e_log_bs) reproduces `lmetric`'s argmin with the hotspot
    detector off.
  - Run CMA-ES as a multi-start from both θ_default and θ_LMetric.

### L2. Use a length-scaled TTFT SLO and report goodput by ISL bucket (calibration-eval)

- **Evidence.**
  - SMetric uses TTFT ≤ 1 s + 62.5 ms per 1K prompt tokens and TPOT ≤ 20 ms, and found results robust
    across SLO settings (p10; Fig. 17).
  - GORGO shows that a latency-only objective is gamed by sacrificing other requests (App. D, p7–8).
  - The campaign fixes one T per family. AgentX runs to 128K tokens, so long prompts may be unable to
    meet T even on an idle worker. A goodput objective then ignores them, or learns to sacrifice them
    to protect short requests **[campaign inference]**.
- **Action.**
  - In Stage 2, set T(L) = b + a·L per family, with a calibrated from AIS uncontended prefill time per
    token on this deployment (for example a slack multiple of it).
  - Record the fraction of requests whose SLO is unattainable on an idle worker.
  - Report goodput per ISL bucket in lr-report.

### L3. Guard the normalized-goodput objective against near-zero denominators and saturated cells (calibration-eval / training)

- **Evidence.**
  - Policy differences explode near and past saturation: SMetric's gap is within 1% at 6× load and 26%
    past saturation (p10). Calibrate-then-Route's starved regime gives 0.292 vs 0.680 (p4).
  - Light-load cells tie: Calibrate width 6 reconverges (p4); Wu's throughput converges once capacity
    suffices (p9).
  - The mean of goodput / default-goodput is therefore dominated by heavy cells where default goodput
    is small **[campaign inference]**.
- **Action.**
  - Floor the denominator, for example at max(default goodput, 0.2 × offered good-request rate), or
    use differences in good_frac or log-ratios with a floor.
  - Require the default router's good_frac ≥ ~0.3 at L3 during calibration.
  - Drop or down-weight L1 cells where round-robin and default do not differ beyond noise.

### L4. Add a consistent-hash "home" feature so the logit can learn DualMap/CHWBL-style concentration (features)

- **Evidence.**
  - DualMap: best-of-all-N "is effectively equivalent to using d = n ... severely degrading KV cache
    reuse" (p5). Per-request Min-TTFT oscillates (p6).
  - Lodestar: a greedy reward argmax needed a k = 2 consistent-hash filter under > 80% cluster KV use,
    which cut tail TTFT (p6; Fig. 14, p11).
  - CacheRoute: reactive spills create cold copies; planned affinity reached 2.3× capacity (p1).
- **Action.**
  - Add `hash_home` ∈ {0, 1}: 1 if the worker is among the top-2 rendezvous winners for the request's
    prefix-root or session key.
  - Reuse `signals::rendezvous` (`WT/lib/router-plugins/builtin/src/signals.rs:53`, already used by the
    dualmap and chwbl ports).
  - The feature is N-agnostic and router-observable. In the context term, let it interact with mean KV
    load.

### L5. Avoid mean-pooled context over one-hot features; it breaks N-extrapolation (model-form / generalization)

- **Evidence.** Lodestar makes its scorer instance-count independent by sharing parameters and never
  using instance identity (§4.1, p5).
- **Analysis [campaign inference].**
  - In M2, x̄_S of `session_affinity` (one worker has 1) equals 1/N: 0.25 and 0.125 at the training
    N, but 0.031 at N = 32. The same holds for `hash_home` (2/N).
  - The low-rank term then extrapolates in N for reasons unrelated to the state.
- **Action.**
  - Exclude indicator features from x̄_S, or pool them by max or sum instead of mean.
  - Prefer request-level scalars: `is_first_turn`, max_S overlap_frac, ISL.
  - Use set-relative load features (x_i − min_S, x_i / mean_S).
  - Add a unit test that a state with the same per-worker distribution at N = 4 and N = 32 gives the
    same argmin class.

### L6. Audit every feature for information the live router would not have (sim2real)

- **Evidence.**
  - Calibrate-then-Route: the "simulator sets each instance's backlog signal from the cluster's own
    clock ... a simulator can hand a policy information no deployment will have" (p5). Mispriced
    constants cost 4.5 points (p4). A re-fitted simulator matched goodput (MAE 0.068) but not the
    winner on any workload (p5).
  - ThunderAgent: router-side radix trees go stale under eviction (p5).
  - llm-d: precise (event-indexed) vs approximate prefix state is the difference between TTFT p90
    0.54 s and 31 s on 8× TP2 H100 Qwen-32B (blog).
- **Action.**
  - In the Stage 1 static audit, add a lens: for each v1 feature, show that replay computes it from the
    same router-side state, with the same update timing, as the live router. This covers KV-event lag,
    `accounting_cache_estimate` vs device events, and when active prefill tokens are decremented.
  - In Stage 5, add a cell with injected KV-event delay if replay supports it.
  - Report rank stability (learned vs best baseline per cell) under the 0.8/1.2 speedup perturbations,
    not only goodput deltas.

### L7. Evaluate AgentX in closed loop with causal successor release (calibration-eval / generalization)

- **Evidence.**
  - AgentServeSim: successor turns must be "causally release[d] ... from simulated predecessor
    completions"; request-stream simulators with external arrivals cannot represent policy-dependent
    successor releases (p1).
  - SMetric had to force output lengths and rewrite follow-ups to keep KV reuse consistent in replay
    (p9–10).
- **Action.**
  - Make `agentic_lanes` the primary AgentX load mode, and report trajectory latency.
  - For `trace_timestamps`, verify that replay does not release turn k+1 before turn k completes.
    Otherwise label it an open-loop stress test, not a primary objective cell.

### L8. Add a session-commitment memory feature for agentic and 128K workloads (features)

- **Evidence.**
  - Session-sticky KV routing produced > 20% cross-node memory divergence for 37 minutes, peaking at 51%
    (ThunderAgent p5).
  - Sticky-by-session had mean imbalance 2.17× (SMetric p7).
  - A 128K request on Qwen3-32B is about 32 GB of KV (SMetric p3). The campaign's
    `CR/config/engine.json` pins 301,808 KV tokens per worker, so one 128K session holds about 43% of
    a worker.
  - ~90% of reuse recurs within ~100 s (SMetric Finding 4, p5).
- **Action.** In v2, add:
  - `affined_ctx_frac_i` = Σ over sessions whose last request went to worker i within a TTL of about
    120 s, of their last prompt tokens, divided by worker KV capacity;
  - `is_first_turn` (no session or turn 0), as a request-level gate for overlap and load features.

### L9. Add an SMetric-style baseline and report against the per-cell best baseline (baselines / reporting)

- **Evidence.**
  - SMetric beat LMetric, llm-d, Preble and Dynamo by 9–15% peak TPS (p10).
  - The best heuristic flips by regime: LMetric drops to 45% reuse and −36% TPS vs a KV-favoring linear
    score at 82% reuse without a global tier (SMetric p6).
  - The best threshold or weight differs per workload (Lodestar Fig. 2, p3; LMetric p6).
- **Action.**
  - Add an `smetric` baseline from existing signals plus `session_context()`: first turn → LMetric
    score; follow-up → highest-overlap worker unless its predicted prefill backlog exceeds SLACK × TTFT
    SLO, then least-loaded.
  - In REPORT, compare against the per-cell best tuned baseline (oracle of baselines) as well as each
    baseline.

### L10. Keep training on-policy; never ship a predictor fitted to default-router logs (training)

- **Evidence.**
  - Offline-trained TTFT predictors collapse when they drive routing: over-optimistic and no ranking
    signal (Lodestar §3.3, p4). Offline-only Lodestar performed poorly (p10–11).
  - Logs lack counterfactual cache and queue evolution (GORGO p2).
- **Action.**
  - Use any regression warm start (θ fitted from default-router replays) only as a CMA-ES initial
    mean.
  - If a predictor rung is added, train it with iterated on-policy replay data (DAgger-style), never
    with one-shot logs.

### L11. Constrain load-coefficient signs and audit concentration (training / calibration-eval)

- **Evidence.**
  - ES zeroed the queue weight and sent 100% of traffic to one replica (GORGO App. D, p7–8). A lower
    bound on the queue weight mitigated this (p8).
  - Imbalance diagnostics from the literature: max/mean (SMetric, Nixon et al.), CV of pending prefill
    (DualMap p3), session replication ratio (Nixon et al. §7.2.2).
- **Action.**
  - In `space.yaml`, use sign-constrained transforms (load weights ≤ −ε).
  - In the Stage 4 audit, compute per-cell max/mean per-worker token throughput, worker share and
    session replication ratio. Flag any policy whose max worker share exceeds 2/N.

### L12. Expect parity at N = 2 and herding under scarcity; pre-register this (generalization)

- **Evidence.**
  - The learned router ties JSQ at width 3, leads at 4 and reconverges at 6 (Calibrate p4).
  - It collapses under scarcity from greedy concentration; the suggested mitigations are power-of-two
    sampling or an admission guard (p4, p6).
- **Action.**
  - Write in the frozen test plan, before results, that |Δ| within noise at N = 2 counts as parity,
    not as an extrapolation failure.
  - For L3 cells, test temperature > 0 or a two-candidate restriction (`hash_home`).

### L13. Calibrate cells by cache-pressure regime (working set / cluster KV) as well as by load (calibration-eval)

- **Evidence.**
  - Cache-aware policies tie while the hot set fits the warm allocation, and separate when it outgrows
    it (CacheRoute Table 2, p4).
  - llm-d's large win was at cache demand of 73% of cluster capacity (blog).
  - Hit rate and replication depend on working set vs capacity (Nixon et al. §7.2.2).
- **Action.**
  - In Stage 2, compute per cell the unique-prefix tokens touched within a ~100 s window, divided by
    N × 301,808. Ensure train, val and test span < 0.5, ≈ 1 and > 1.
  - Plot learned-vs-default gains against this ratio in REPORT.

### L14. Add an attention-aware prefill-cost feature for long context (features)

- **Evidence.** SMetric's load model n(L − n/2) beat token counts (+4.2% / +1.9% TPS, p12). Dynamo's
  token count is criticized as inaccurate because per-token cost depends on context (p9).
- **Action.** In v2, add `prefill_attn = n_i·(L − n_i/2)/8192²` per worker, where n_i is the uncached
  tokens on worker i and L is the prompt length.

### L15. Add a new-prefill × decode-load interaction and let the sign be learned (features)

- **Evidence.**
  - Incoming prefills inflate co-running decodes' latency, which hurts ITL goodput (Jain p4; mixing
    penalty p7–8).
  - Preble routes prefill-heavy requests to decode-heavy GPUs, which have spare compute (p6).
- **Action.** In v2, add `new_prefill_k × active_requests_s` (or × kv_load_frac). Report the learned
  sign as an interpretable result.

### L16. Clamp features to their training range and log OOD rates at test (generalization)

- **Evidence.** Lodestar falls back to a heuristic when features leave "the ranges observed in the
  training buffer" (p8).
- **Action.**
  - Record per-feature quantiles in lr-train.
  - Add an optional clamp ([q0.001, q0.999]) to `learned-choice` parameters.
  - Report the fraction of clamped decisions for N = 16/32 and 128K test cells.

### L17. Size the noise floor for 1–3% effects; gate model-ladder escalation on it (training / reporting)

- **Evidence.**
  - Evolutionary search over a validated simulator gained only 0.5–2.8% over hand-written seeds
    (AgentServeSim p8).
  - Classic heuristics gained 0.46–2.6% over RR (Jain p9).
  - Learned 0.864 vs best baseline 0.847 (Calibrate Table II, p3).
- **Action.**
  - Set a target minimum detectable effect of ~1–2% normalized goodput and size repeats and cells so
    that 2 × SE of the val mean Δ is below it.
  - Escalate M1 → M2 → M3 only when the previous rung beats tuned M0 by more than this MDE.

### L18. Tune the dispatch-or-hold knob jointly in agentic cells (training / baselines)

- **Evidence.**
  - Holding at the router (a "no action" choice) reduced preemption and latency (Jain p9).
  - ThunderAgent's gains come mainly from program-level pause/restore and a global waiting queue
    (p6–7).
- **Action.** Promote ablation (a) (joint `router_queue_threshold` tuning for the learned model and the
  best baseline) to a reported arm for the AgentX and 128K cells.

### L19. Held-out windows: re-match load, and train across prefix-sharing regimes (generalization)

- **Evidence.**
  - GORGO froze its weights and re-scaled `time_scale` to keep held-out windows comparable (p5).
  - Lodestar's model frozen in a 5%-sharing phase failed at 50% sharing (p11).
- **Action.**
  - Match per-worker load for held-out Mooncake windows (already planned).
  - Make sure train cells include both low- and high-reuse regimes.
  - Report each test cell's prefix-reuse ratio next to its result.

### L20. A non-leaky output-length proxy could recover the load-bearing feature (features, lower priority)

- **Evidence.**
  - Predicted OSL was the most important feature, and tolerated 150% error (Calibrate p4). That setting
    is disaggregated decode pools.
  - A text-only predictor is weak without task hints (Jain p7).
- **Action.** As an optional v2 ablation, add per-worker expected remaining decode tokens, estimated
  only from completed turns of the same session (EWMA). Leakage audit: no field derived from the
  current request's true OSL.

---

## 6. Sources

- https://arxiv.org/abs/2609.16206 (Calibrate, Then Route)
- https://arxiv.org/abs/2606.00946 (Lodestar)
- https://arxiv.org/abs/2408.13510 (Intelligent Router)
- https://arxiv.org/abs/2603.15202 (LMetric)
- https://arxiv.org/abs/2607.08565 (SMetric)
- https://arxiv.org/abs/2602.11688 (GORGO)
- https://arxiv.org/abs/2601.18999 (Randomization Boosts KV Caching, ICLR 2026)
- https://arxiv.org/abs/2608.19677 (CacheRoute)
- https://arxiv.org/abs/2602.06502 (DualMap)
- https://arxiv.org/abs/2407.00023 (Preble)
- https://arxiv.org/abs/2602.13692 (ThunderAgent)
- https://arxiv.org/abs/2606.09613 (AgentServeSim)
- https://arxiv.org/html/2502.13965 (Autellix; load balancer section read on the web)
- https://arxiv.org/html/2608.13573 (A Year in LLM Serving; §7 read on the web)
- https://llm-d.ai/blog/kvcache-wins-you-can-see (llm-d precise prefix scheduling benchmark)
- https://arxiv.org/abs/2401.07886, https://arxiv.org/abs/2606.17949, https://arxiv.org/abs/2501.14312,
  https://arxiv.org/abs/2511.02230, https://arxiv.org/abs/2408.15792 (abstract-level only)
- https://github.com/helixml/ramjet (Ramjet is a GitHub project, not a paper)
