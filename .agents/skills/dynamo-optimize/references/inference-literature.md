<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference Literature — Regression Conditions

Reference content for `dynamo-optimize/SKILL.md`. Distills six core LLM
inference papers down to the regression conditions a recipe selector
needs to refuse an inappropriate mode.

Source of truth: `dynamo-skills/corpus/papers/<short>/extracts.yaml`.
Every claim here carries a verbatim quote at the cited extract row.

## Mooncake — KV-cache-centric scheduling

[mooncake/extracts.yaml](../../../../../dynamo-skills/corpus/papers/mooncake/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| KV-aware routing fetches a remote cache only when transfer is cheaper than re-prefill. | `mooncake-kv-aware-routing-condition` (§6.2 Cache Load Balancing) | Don't enable `disagg-kv-router` if your KV transfer path is slow relative to local prefill — the router will keep computing locally and you pay the disagg overhead without the benefit. |
| Router prefers local recompute unless the remote prefix exceeds the local match by a threshold. | `mooncake-prefix-match-threshold` (§6.2) | The KV router has internal hysteresis; a workload with marginal prefix reuse may still route mostly locally. |
| The Kimi production workload has a ~50% theoretical reuse ceiling. | `mooncake-theoretical-reuse-ceiling` (§9 Related Work) | Even ideal infrastructure caps KV reuse around 50%; below ~25% measured reuse, KV-aware routing is unlikely to pay its costs. |
| Suboptimal P:D ratio (2P+2D vs 3P+1D) regresses TTFT when prefill becomes the bottleneck. | `mooncake-pd-ratio-regression` (§8.1.1 Public Datasets) | Recipe P:D ratios are tested for a specific workload shape; if your ISL/OSL differs, the tested envelope may not transfer. |
| Kimi traffic is long-input (avg 7,590 tokens) short-output (avg 182 tokens). | `mooncake-workload-shape` (§4.2) | Provides a workload-shape anchor for "disagg/KV-router are well-suited" — long inputs, short outputs, repeated prefixes. |

**Decision-point text for `SKILL.md`:** "Mooncake (Moonshot AI's production
serving platform for Kimi) demonstrates that KV-aware routing matters
most when measured prefix reuse approaches the per-workload theoretical
ceiling (~50% on their workload), and that suboptimal P:D ratios
*regress* TTFT — they don't merely fail to improve it. If your workload
has prefix reuse well below 25%, prefer non-KV modes."

## DistServe — Prefill/Decode disaggregation, goodput optimization

[distserve/extracts.yaml](../../../../../dynamo-skills/corpus/papers/distserve/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| Goodput is the max request rate per GPU that meets an SLO attainment goal (e.g. 90%). | `distserve-goodput-definition` (§1 Introduction) | Anchors AIPerf's `--goodput` grammar: declaring an SLO defines what "good" means for this run. |
| 7.4× more requests or 12.6× tighter SLO vs vLLM aggregated baseline. | `distserve-headline-improvement` (Abstract) | Upper bound for what disagg+routing-style optimization can deliver on a friendly workload. |
| Disaggregation effectiveness is compromised when goodput is NOT the optimization target — offline / non-latency-sensitive workloads. | `distserve-when-disaggregation-regresses` (§7 Discussion) | **Core refusal condition.** If the user's `opt_target` is raw throughput and they don't care about TTFT/ITL, don't pick a disagg mode; pick `agg-*`. |
| Fault propagation: a fault in one decoding instance can cripple multiple prefill instances mapped to it. | `distserve-fault-propagation-risk` (§4.3 Online scheduling) | Operational caveat to flag when chaining to `dynamo-troubleshoot`. |

**Decision-point text for `SKILL.md`:** "DistServe explicitly notes
disaggregation regresses in offline / non-latency-sensitive workloads
where the optimization target shifts to raw throughput. If your
`opt_target` from Phase 1 is throughput-not-latency, prefer `agg-*`."

## SplitWise — Phase splitting, workload-shape sensitivity

[splitwise/extracts.yaml](../../../../../dynamo-skills/corpus/papers/splitwise/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| Two machine pools: prompt machines produce KV cache; token machines continue generation. | `splitwise-cluster-design` (§IV-A Cluster-level scheduling) | This is exactly Dynamo's `disagg-*` topology with prefill vs decode workers. |
| Optimal P:T pool sizing depends on workload: coding (heavy prompt) uses 35P/5T; conversation uses 25P/15T. | `splitwise-workload-shape-dependency` (§VI-B Cluster provisioning) | **Core decision-point input.** If the user's ISL >> OSL, expect a prefill-heavy split; if ISL ≈ OSL, expect more balanced. Recipe P:D ratios encode an assumed workload shape. |
| 1.4× throughput at 20% lower cost, or 2.35× throughput at same power+cost budget. | `splitwise-cost-efficiency-headline` (Abstract) | Upper bound for the cost-efficiency improvement disagg can deliver. |
| On prompt or token machine failure, restart from scratch. | `splitwise-fault-recovery` (§IV-E) | Fault tolerance is at-request-restart granularity; flag for `dynamo-troubleshoot`. |

**Decision-point text for `SKILL.md`:** "SplitWise's empirical pool sizing
(35P/5T for coding, 25P/15T for conversation) confirms that the P:D
ratio in a disagg recipe is workload-shape-specific. If you're trying to
re-use a coding-trace recipe on conversation traffic, expect a P:D
mismatch that won't show up until you measure."

## vLLM (PagedAttention) — KV memory management

[vllm-pagedattention/extracts.yaml](../../../../../dynamo-skills/corpus/papers/vllm-pagedattention/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| Block size 16 is the practical default; 16-128 best on ShareGPT. | `vllm-block-size-recommendation`, `vllm-block-size-sweet-spot` (§7.2) | If a recipe pins a non-default block size, treat it as a tuning hint, not a free knob. |
| 2-4× throughput vs SOTA at same latency. | `vllm-abstract-claim` (Abstract) | Frames the vLLM-vs-FasterTransformer baseline; useful context when comparing engines. |
| PagedAttention kernel has 20-26% higher attention latency vs FasterTransformer. | `vllm-attention-kernel-overhead` (§7.1) | The block-indirection has a per-token cost that disappears in batched throughput but shows up in single-stream latency. |
| Non-memory-bound workloads can degrade — vLLM's indirection overhead can hurt. | `vllm-limitation-when-not-helpful` (§8 Discussion) | Refusal condition: if your workload has very short prompts and very small batch sizes, vLLM may not be the right backend. |

**Decision-point text for `SKILL.md`:** "vLLM's PagedAttention amortizes
the block-indirection overhead across batched requests. On non-memory-
bound workloads — short prompts, small batch — the overhead can dominate.
If your workload is exclusively short-prompt + low-batch, consider a
non-vLLM framework (sglang, trtllm)."

## SGLang (RadixAttention) — Cache-aware scheduling for prefix reuse

[sglang-radixattention/extracts.yaml](../../../../../dynamo-skills/corpus/papers/sglang-radixattention/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| RadixAttention retains the KV cache after generation completes, in a radix tree. | `sglang-radixattention-retention` (§5.1) | This is the mechanism behind sglang's prefix-cache win; recipes that pick sglang inherit this regardless of routing. |
| Higher cache hit rate REQUIRES cache-aware scheduling. | `sglang-cache-aware-scheduling` (§5.2) | If you pick sglang but disable cache-aware scheduling (e.g. by using `agg-round-robin` instead of `agg-kvbm` or `disagg-kv-router`), you leave RadixAttention's benefit on the table. |
| LMP (longest-matched-prefix) policy outperforms others; non-cache-aware policies suffer lower hit-rate and lower throughput. | `sglang-policy-impact` (§6.4.1 Ablation) | Confirms that the routing-policy choice (not just the cache mechanism) matters. |
| 4.4× throughput vs closest competitor on MMLU; 5.6× vs vLLM on ReAct agent workload. | `sglang-mmlu-speedup`, `sglang-react-agent` (§6.2) | Upper bounds for the prefix-cache + RadixAttention combination on prefix-heavy workloads. |
| GPU memory bounds the size of the retained KV cache. | `sglang-capacity-limit` (§5.1) | Capacity ceiling — if your workload has too many distinct prefixes to fit in HBM, RadixAttention will evict and the hit rate falls. KVBM helps here. |

**Decision-point text for `SKILL.md`:** "SGLang's RadixAttention turns
prefix reuse into raw throughput, but only when (a) the scheduler is
cache-aware and (b) the working-set fits in HBM. If your workload's
prefix-set is too large for HBM, pair sglang with `agg-kvbm` (KV
offloading) or with a KV-aware router that affines requests to workers."

## Orca — Iteration-level batching

[orca/extracts.yaml](../../../../../dynamo-skills/corpus/papers/orca/extracts.yaml)

| Claim used here | Quote anchor | Implication for `dynamo-optimize` |
|---|---|---|
| Iteration-level scheduling decides what to batch per model iteration, not per request. | `orca-iteration-level-scheduling-definition` (Abstract) | Modern engines (vLLM, SGLang, TRT-LLM) all do continuous batching; the recipe's framework choice inherits it. |
| 36.9× throughput vs FasterTransformer at the same latency on GPT-3 175B. | `orca-headline-result` (Abstract) | Upper bound; the static-batching baseline is gone in modern engines, but the principle (per-iteration scheduling) is foundational. |
| Static batching wastes compute on already-finished requests; early-finished requests can't return until the whole batch finishes. | `orca-static-batching-limitation`, `orca-static-batch-engine-overhead` (§3 C1) | If your engine doesn't do continuous batching, you regress to this failure mode. All Dynamo-supported engines do continuous batching. |
| At low load with very large models, no engine has an advantage — latency is engine-bound. | `orca-low-load-no-advantage` (§6.2) | Refusal condition: at low load, the optimization knob is hardware-level (KV memory, prefix reuse), not scheduling-level. |
| Increasing max batch size does not always help; tradeoff depends on workload. | `orca-batch-size-not-free` (§6.2) | If the user is asking "should I just turn up the batch size", the answer is workload-dependent — measure. |

**Decision-point text for `SKILL.md`:** "Orca's iteration-level scheduling
is foundational; all Dynamo-supported engines (vLLM, SGLang, TRT-LLM) do
continuous batching, so the static-batching regressions Orca documents
don't apply to the recipe set. But Orca's low-load observation does
apply: at low request rate, you can't get scheduling wins because
there's nothing to schedule. The optimization knob shifts to memory-tier
(KVBM) or routing (KV-aware vs round-robin)."

## Mode-to-paper crosswalk

| Recipe mode | Primary paper anchor | Primary regression condition |
|---|---|---|
| `agg`, `agg-round-robin` | Orca (continuous batching baseline) | Low load — nothing to optimize. |
| `agg-kvbm` | vLLM (memory mgmt) + SGLang (prefix retention) | Workload doesn't have repeated prefixes — KVBM offload is dead weight. |
| `agg-embedding-cache` | vLLM (memory mgmt) | Non-multimodal workload — embedding cache is dead weight. |
| `agg-eagle-*`, `disagg-eagle-*` | Custom — EAGLE speculative decoding (not in this corpus); kimi-k2.5 recipe is the in-tree anchor. | Workload doesn't have predictable token distributions — speculative decoding regresses. |
| `disagg`, `disagg-single-node`, `disagg-*gpu` | DistServe + SplitWise | Offline / non-latency-sensitive workload (DistServe §7); or P:D ratio mismatched to workload shape (SplitWise §VI-B). |
| `disagg-kv-router` | Mooncake (G12, G13) + DistServe | Prefix reuse below ~25% — KV transfer cost beats local recompute (Mooncake); or offline workload (DistServe). |
| `disagg-multi-node` | Mooncake + SplitWise + Dynamo NIXL | KV transport (NIXL/UCX) not configured — disagg degenerates to no-op cross-node. |

## See Also

- [k8s-recipe-workflow.md](k8s-recipe-workflow.md) — Where mode selection happens.
- [slo-shape.md](slo-shape.md) — How the SLO encodes "latency-vs-throughput preference".
- [known-issues.md](known-issues.md) — Operational failure patterns; this file covers algorithmic regressions.
