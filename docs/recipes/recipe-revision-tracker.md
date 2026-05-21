# Recipes Docs Revision Tracker

This is the working tracker for turning the recipes prototype into a real
replacement for the current `recipes/**/README.md` experience.

Status key:

- `todo`: not yet ported into Fern docs.
- `draft`: represented by the current prototype, but still needs source review.
- `blocked`: needs recipe owner or benchmark clarification before docs can be honest.
- `done`: ported, reviewed against source README and manifests, and visually checked.

## Porting Rules

Use these rules when moving README content into the new Fern recipe pages.

| README element | New home | Porting decision |
| --- | --- | --- |
| Model/provider/runtime/hardware/status | Catalog card and detail hero | Port, but derive from manifests/CSV where possible. |
| Available configurations table | Detail selector and variant list | Port, but collapse duplicate rows into comparable variants. |
| Topology | Detail summary cards and variant rows | Port; include GPU counts and P/D shape. |
| Dataset and workload description | Workload type, traffic facts, Benchmark Evidence | Port; avoid inventing workload labels not supported by `perf.yaml` or README. |
| Results, plots, expected metrics | Benchmark Evidence and Artifacts | Port if source gives numbers or image; mark benchmark status clearly. |
| Prerequisites | Before You Run | Port only recipe-specific requirements; link generic platform setup. |
| Deploy commands | Deploy tabs | Port only runnable commands for chosen variants. |
| Benchmark commands | Benchmark tabs | Port when `perf.yaml` exists; label deployment-only otherwise. |
| Test requests | Verify section | Port model-specific reasoning/tool/multimodal examples. |
| Key configuration notes | Configuration Notes | Port if they explain why the recipe works or what must be edited. |
| Cluster-specific networking | Platform Notes or Configuration Notes | Port, but separate GKE/AWS/IB details from the happy path. |
| Monitoring, artifact retrieval, cleanup | Operate section | Add this home; currently missing from prototype. |
| Container build instructions | Image section or separate build guide | Port only for recipes that require BYO image; otherwise summarize image provenance. |
| Directory layout and ownership | Maintainer appendix or evict | Usually evict from customer page unless layout affects usage. |
| Generic namespace/HF/storage boilerplate | Global setup guide | Evict from each recipe page; link once. |
| Troubleshooting boilerplate | Existing troubleshooting docs | Evict unless recipe-specific and actionable. |
| Contributing/quality standards | Maintainer docs | Evict from customer recipe pages. |

## Landing Page Tracker

| Status | Item | Notes |
| --- | --- | --- |
| draft | Model-first catalog | Current prototype has 15 model-family cards. Need confirm whether every family with deploy manifests appears. |
| draft | Filtering | CSS-only filters work for current cards. Need decide if filters should remain CSS-only or use generated metadata + client behavior later. |
| draft | Workload type taxonomy | Current buckets: agentic coding, long-context reuse, multimodal reuse, static ISL/OSL, long output, deployment only. Needs owner review against all `perf.yaml` shapes. |
| draft | Technique taxonomy | Current buckets are reasonable but incomplete for TRT-LLM/V4/Nemotron: add or normalize WideEP, frontend decoding, offload, MNNVL/ComputeDomain, reasoning/tool parsing? |
| draft | Status labels | Current labels are Verified, Preview, Experimental, Deploy only. Need formal criteria. |
| todo | Catalog counts | Replace hardcoded "15 model families" and "30 assessed variants" with generated metadata or documented manual source. |
| todo | Card ordering | Decide sort rule: verified first, recency, business priority, hardware priority, or model/provider alpha. |
| todo | Search | Prototype omitted search because Fern page scripts were fragile. Need decide generated static search facets vs native Fern search only. |
| todo | Empty state | CSS filters cannot show dynamic empty state today. Need implement only if we add JS/generated component. |
| todo | Cross-link source recipes | Each card/detail page should link to source README and manifests. |
| todo | Accessibility review | Confirm radio controls, labels, tab order, contrast, and mobile behavior after final taxonomy. |
| todo | Metadata source | Decide whether durable source is `recipe.yaml` per recipe, one central catalog file, or generated from manifests + perf YAML. |
| todo | Visual density | Current 4-column desktop grid is compact; verify with all final cards and longer names. |
| todo | Provider logos | Current logos are copied WebP assets. Confirm source/license and fallback style. |
| todo | Deployment-only handling | Decide whether deploy-only families appear in same grid or a separate "examples" / "functional recipes" band. |

## Required Page Template

Every real recipe detail page should make an explicit decision for each section.

| Section | Required? | Purpose |
| --- | --- | --- |
| Hero | Yes | What this model recipe is, recommended path, source link. |
| Configure | Yes | Real selectable variants only; read-only facts must not look selectable. |
| Selected recipe summary | Yes | Topology, runtime, hardware, benchmark status. |
| When to use | Yes | Customer decision guidance, not marketing copy. |
| Variants | Yes | All supported deploy/perf variants for that model family. |
| Before you run | Yes | Recipe-specific prerequisites only. |
| Deploy | Yes | Commands per variant. |
| Verify | Yes when request examples differ by model/backend | Smoke tests, tool calling, reasoning, multimodal examples. |
| Benchmark Evidence | Yes when `perf.yaml` exists | Dataset, traffic, concurrency, SLA, artifacts, results. |
| Operate | Yes when README includes lifecycle details | Monitor, logs, artifacts, cleanup. |
| Configuration Notes | Yes when README has engine/cluster caveats | Parser flags, RDMA, KVBM, EAGLE, NIXL, storage, image requirements. |
| Artifacts and References | Optional | Plots, papers, model cards, sibling recipes. |
| Evicted Content | Internal tracker only | Generic setup, redundant boilerplate, maintainer-only docs. |

## Homeless Content Tracker

These README elements do not yet have a satisfying customer-facing home in the
prototype. Each one needs either a new page section, a shared guide, or an
explicit eviction decision before the revised Recipes section is real.

| Status | Homeless element | Appears in | Proposed resolution |
| --- | --- | --- | --- |
| todo | Monitoring, logs, artifact retrieval, cleanup | Benchmark-heavy READMEs, especially Qwen3-32B and DeepSeek V3.2 | Add `Operate` section to detail pages; keep generic cleanup in setup/troubleshooting docs. |
| todo | Container/image build flows | DeepSeek V4 container, Nemotron Omni, Kimi TokenSpeed | Create shared image guide, then add short image provenance notes per recipe. |
| draft | Dataset generation and trace preparation | Qwen3-VL, Kimi K2.5, Qwen3.6 | Qwen3-VL now has a `Dataset` section; Kimi and Qwen3.6 still need first-class homes. |
| todo | Cluster-specific networking and fabric setup | DeepSeek V4 Pro, DeepSeek-R1, GLM-5 | Keep recipe-specific required settings in `Configuration Notes`; link generic platform setup. |
| todo | GAIE accessory manifests | Llama 3.3 70B | Move to optional integration note; do not count as a separate recipe variant. |
| todo | Script-driven benchmark workflows | Qwen3.6 | Give the detail page a `Run Scripts` section instead of forcing it into deploy-command tabs. |
| todo | Maintainer authoring rules and repo layout | Root `recipes/README.md` | Move to contributor/maintainer docs; evict from customer-facing recipe pages. |
| todo | Compatibility notes for older Dynamo releases | Nemotron Super, backend-specific READMEs | Verify against current branch; keep only if still relevant. |
| todo | Raw result images and benchmark artifacts | Kimi, DeepSeek V3.2, Qwen3-VL | Link from `Benchmark Evidence`; avoid embedding stale screenshots without date/source. |

## Recipe Family Tracker

| Status | Target Fern page | Source README(s) | Variants to cover | Must port | Likely evict | Open scrutiny |
| --- | --- | --- | --- | --- | --- | --- |
| todo | `deepseek-r1.mdx` | `recipes/deepseek-r1/README.md`, `recipes/deepseek-r1/sglang/README.md`, `recipes/deepseek-r1/vllm/disagg/README.md` | SGLang disagg 8 GPU, SGLang disagg 16 GPU, TRT-LLM GB200 WideEP, vLLM disagg | Backend/hardware matrix, huge-model prerequisites, backend-specific notes, TRT-LLM perf row | Repeated HF/storage setup | Whether all variants are production-ready and whether SGLang READMEs are too thin for detail page claims. |
| draft | `deepseek-v3-2-nvfp4.mdx` | `recipes/deepseek-v32-fp4/README.md` | TRT-LLM agg round-robin, TRT-LLM disagg KV router | Mooncake synthetic coding trace, dataset stats, WideEP topology, goodput SLA, benchmark jobs, artifact path, cleanup/operate, ComputeDomain notes | Generic setup boilerplate | Needs visual check and reconciliation of README tmux wording vs Job-based `perf.yaml`. |
| todo | `deepseek-v4-flash.mdx` | `recipes/deepseek-v4/deepseek-v4-flash/README.md` | vLLM B200, vLLM GB200, SGLang B200, SGLang GB200 | Deploy-only variants, reasoning/tool verification, startup caveats, model details | Benchmark evidence, because no `perf.yaml` exists | Needs explicit "deployment-only" treatment and no benchmark-backed claims. |
| todo | `deepseek-v4-pro.mdx` | `recipes/deepseek-v4/deepseek-v4-pro/README.md`, SGLang disagg B200/GB200 READMEs | vLLM B200 agg, vLLM GB200 agg, vLLM GB200 disagg, SGLang agg, SGLang disagg B200, SGLang disagg GB200 | Variant table, Day-0 status, GB200 ComputeDomain, RDMA/GKE/AWS/IB notes, perf for vLLM GB200 disagg, reasoning/tool verification | Duplicated setup | Need reconcile top-level SGLang agg plus nested SGLang disagg pages without overwhelming customers. |
| todo | `glm-5-nvfp4.mdx` | `recipes/glm-5-nvfp4/README.md` | SGLang disagg | Topology, published runtime image, EAGLE MTP, KV cache, NIXL/UCX, recovery/rollouts, 1K/8K benchmark result | Generic storage/HF setup | Need preserve performance number with caveat if not in artifact. |
| todo | `gpt-oss-120b.mdx` | `recipes/gpt-oss-120b/README.md`, `recipes/gpt-oss-120b/trtllm/disagg/README.md` | TRT-LLM agg, TRT-LLM disagg | Available configs, P/D topology, engine config differences, KV transfer, quantization, benchmark optional | Repeated top-level quickstart | GPU count ambiguity: README says 5 Blackwell for family, CSV inferred 6 from benchmark env. Verify. |
| todo | `kimi-k2-5.mdx` | `recipes/kimi-k2.5/README.md`, TokenSpeed README | TRT-LLM agg Eagle KV router, agg Eagle round-robin, agg round-robin, disagg Eagle KV router, TokenSpeed experimental agg | Agentic coding workload, result plot, expected metrics, Eagle/KV/offload comparison, trace copy requirement, TokenSpeed caveats | Generic setup | Decide whether TokenSpeed is a variant on same page or separate experimental subpage. |
| todo | `llama-3-3-70b.mdx` | `recipes/llama-3-70b/README.md` | vLLM agg, vLLM disagg single-node, vLLM disagg multi-node, GAIE accessory | Basic config table, FP8 model details, GAIE note, static 8K/1K perf rows | Simple duplicated quickstart | Needs richer "why choose each variant"; README is thin. |
| todo | `nemotron-3-nano-omni.mdx` | `recipes/nemotron-3-nano-omni/README.md` | vLLM agg | Custom container build, multimodal smoke tests, prefix-hash routing, tool/reasoning parsers, optional no-NATS | Benchmark section, because no `perf.yaml` | Since build is required, page needs an Image section or it will mislead. |
| todo | `nemotron-3-super-fp8.mdx` | `recipes/nemotron-3-super-fp8/README.md` | vLLM agg, TRT-LLM disagg, SGLang agg, SGLang disagg | Parser config, routing, backend notes, Dynamo 0.9.1 compatibility if still needed | Old compatibility if no longer supported | Need verify whether compatibility notes belong in current docs. |
| todo | `qwen3-235b-a22b-fp8.mdx` | `recipes/qwen3-235b-a22b-fp8/README.md` | TRT-LLM agg Hopper, agg Blackwell, disagg Hopper, disagg Blackwell | Hardware-specific MoE backend differences, static 4K/200 perf rows, hardware requirements | Boilerplate setup | Need explain Hopper vs Blackwell as customer choice, not implementation trivia. |
| draft | `qwen3-32b.mdx` | `recipes/qwen3-32b/README.md`, `recipes/qwen3-32b/vllm/agg-kvbm/README.md` | vLLM agg round-robin, vLLM disagg KV router, vLLM agg KVBM | Mooncake trace, P/D vs agg comparison, KVBM deploy-only caveats, traffic facts, benchmark commands, verify commands, operate/artifacts/cleanup, KVBM config notes | Generic setup boilerplate | Needs visual check and final decision on whether deploy commands should auto-follow the selector. |
| todo | `qwen3-32b-fp8.mdx` | `recipes/qwen3-32b-fp8/README.md` | TRT-LLM agg, TRT-LLM disagg, vLLM disagg | FP8 variants, port-forward variants, static perf rows | Boilerplate setup | Needs stronger guidance on why choose FP8 page vs BF16 Qwen3-32B page. |
| draft | `qwen3-vl-30b.mdx` | `recipes/qwen3-vl-30b/README.md` | vLLM agg embedding cache | Cache-on/off result, dataset generation, 80% image reuse, cache env vars, artifact path, helper script behavior, operate/cleanup | Generic setup | Visually checked; still needs final scrutiny of cache-off deploy instructions. |
| todo | `qwen3-6-35b.mdx` | `recipes/qwen3.6-35b/README.md` | vanilla vLLM serve, Dynamo frontend decoding, Dynamo FD plus embedding cache | 3-way comparison, run scripts, sliding-window dataset, hardware env files, shared-model-cache, aiperf pin | Maintainer layout rule unless needed | Page must explain why this uses scripts rather than direct manifest tabs. |
| todo | Maintainer/global page | `recipes/README.md` | Root recipes index, recipe structure, troubleshooting, contributing | Only if we create a maintainer recipe authoring page | Most of it from customer-facing page | Decide whether this belongs under contributor docs rather than Recipes. |
| todo | Container build guide | `recipes/deepseek-v4/container/README.md` plus build-required recipe sections | DeepSeek V4 SGLang overlay, TokenSpeed, Nemotron Omni | Build flow, image provenance, build args | None if build is required | Decide whether each recipe page owns build steps or links to a shared image guide. |

## Variant Coverage Tracker

Each row must end as one of: represented on a model page, intentionally evicted,
or moved to a separate operational/build guide.

| Status | Variant | Source path | Perf? | Current catalog? | Required action |
| --- | --- | --- | --- | --- | --- |
| todo | DeepSeek-R1 SGLang disagg 8 GPU | `recipes/deepseek-r1/sglang/disagg-8gpu/deploy.yaml` | No | Family only | Add as deployment variant; no benchmark claims. |
| todo | DeepSeek-R1 SGLang disagg 16 GPU | `recipes/deepseek-r1/sglang/disagg-16gpu/deploy.yaml` | No | Family only | Add as deployment variant; port WideEP/backend notes. |
| todo | DeepSeek-R1 TRT-LLM WideEP GB200 | `recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200` | Yes | Family only | Add benchmark evidence from `perf.yaml`. |
| todo | DeepSeek-R1 vLLM disagg | `recipes/deepseek-r1/vllm/disagg` | No | Family only | Add as deployment variant; preserve DEP setup notes. |
| draft | DeepSeek V3.2 agg round-robin | `recipes/deepseek-v32-fp4/trtllm/agg-round-robin` | Yes | Yes | Represented with deploy, verify, benchmark, operate, and cleanup commands; needs visual review. |
| draft | DeepSeek V3.2 disagg KV router | `recipes/deepseek-v32-fp4/trtllm/disagg-kv-router` | Yes | Yes | Represented with deploy, verify, benchmark, operate, cleanup, and WideEP configuration notes; needs visual review. |
| todo | DeepSeek-V4-Flash vLLM B200 | `recipes/deepseek-v4/deepseek-v4-flash/vllm/agg_b200/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Flash vLLM GB200 | `recipes/deepseek-v4/deepseek-v4-flash/vllm/agg_gb200/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Flash SGLang B200 | `recipes/deepseek-v4/deepseek-v4-flash/sglang/agg/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Flash SGLang GB200 | `recipes/deepseek-v4/deepseek-v4-flash/sglang/agg-gb200/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Pro vLLM agg B200 | `recipes/deepseek-v4/deepseek-v4-pro/vllm/agg/b200/deploy.yaml` | No | Family only | Add deploy-only variant and cold-start caveat. |
| todo | DeepSeek-V4-Pro vLLM agg GB200 | `recipes/deepseek-v4/deepseek-v4-pro/vllm/agg/gb200/deploy.yaml` | No | Family only | Add deploy-only variant and MNNVL caveat. |
| todo | DeepSeek-V4-Pro vLLM disagg GB200 | `recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200` | Yes | Family only | Add benchmark-backed variant. |
| todo | DeepSeek-V4-Pro SGLang agg B200 | `recipes/deepseek-v4/deepseek-v4-pro/sglang/agg/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Pro SGLang agg GB200 | `recipes/deepseek-v4/deepseek-v4-pro/sglang/agg-gb200/deploy.yaml` | No | Family only | Add deploy-only variant. |
| todo | DeepSeek-V4-Pro SGLang disagg B200 | `recipes/deepseek-v4/deepseek-v4-pro/sglang/disagg-b200` | No | Not explicit | Add or intentionally fold into V4-Pro page. |
| todo | DeepSeek-V4-Pro SGLang disagg GB200 | `recipes/deepseek-v4/deepseek-v4-pro/sglang/disagg-gb200` | No | Not explicit | Add cluster-specific config notes. |
| todo | GLM-5 NVFP4 SGLang disagg | `recipes/glm-5-nvfp4/sglang/disagg` | Yes | Yes | Create detail page. |
| todo | GPT-OSS-120B TRT-LLM agg | `recipes/gpt-oss-120b/trtllm/agg` | Yes | Family only | Add benchmark-backed variant. |
| todo | GPT-OSS-120B TRT-LLM disagg | `recipes/gpt-oss-120b/trtllm/disagg` | Yes | Family only | Add benchmark-backed variant and key config notes. |
| todo | Kimi-K2.5 TRT-LLM agg Eagle KV router | `recipes/kimi-k2.5/trtllm/agg-eagle-kv-router` | Yes | Family only | Add variant to Kimi detail page. |
| todo | Kimi-K2.5 TRT-LLM agg Eagle round-robin | `recipes/kimi-k2.5/trtllm/agg-eagle-round-robin` | Yes | Family only | Add variant to isolate routing effect. |
| todo | Kimi-K2.5 TRT-LLM agg round-robin | `recipes/kimi-k2.5/trtllm/agg-round-robin` | Yes | Family only | Add baseline variant. |
| todo | Kimi-K2.5 TRT-LLM disagg Eagle KV router | `recipes/kimi-k2.5/trtllm/disagg-eagle-kv-router` | Yes | Family only | Add recommended variant. |
| todo | Kimi-K2.5 TokenSpeed agg | `recipes/kimi-k2.5/tokenspeed/agg/nvidia` | No | Not explicit | Decide separate experimental subpage vs Kimi variant. |
| todo | Llama-3.3-70B vLLM agg | `recipes/llama-3-70b/vllm/agg` | Yes | Family only | Add benchmark-backed variant. |
| todo | Llama-3.3-70B vLLM disagg single-node | `recipes/llama-3-70b/vllm/disagg-single-node` | Yes | Family only | Add benchmark-backed variant. |
| todo | Llama-3.3-70B vLLM disagg multi-node | `recipes/llama-3-70b/vllm/disagg-multi-node` | Yes | Family only | Add benchmark-backed variant. |
| todo | Llama-3.3-70B GAIE manifests | `recipes/llama-3-70b/vllm/*/gaie/deploy.yaml` | No | No | Move to optional integration note, not a recipe card. |
| todo | Nemotron 3 Nano Omni vLLM agg | `recipes/nemotron-3-nano-omni/vllm/agg` | No | Yes | Add deploy-only detail page with build/image section. |
| todo | Nemotron-3-Super FP8 vLLM agg | `recipes/nemotron-3-super-fp8/vllm/agg` | No | Family only | Add deploy-only variant. |
| todo | Nemotron-3-Super FP8 TRT-LLM disagg | `recipes/nemotron-3-super-fp8/trtllm/disagg` | No | Family only | Add deploy-only variant. |
| todo | Nemotron-3-Super FP8 SGLang agg | `recipes/nemotron-3-super-fp8/sglang/agg` | No | Family only | Add deploy-only variant. |
| todo | Nemotron-3-Super FP8 SGLang disagg | `recipes/nemotron-3-super-fp8/sglang/disagg` | No | Family only | Add deploy-only variant. |
| todo | Qwen3-235B-A22B TRT-LLM agg Blackwell | `recipes/qwen3-235b-a22b-fp8/trtllm/agg/blackwell` | Yes | Family only | Add benchmark-backed variant. |
| todo | Qwen3-235B-A22B TRT-LLM agg Hopper | `recipes/qwen3-235b-a22b-fp8/trtllm/agg/hopper` | Yes | Family only | Add benchmark-backed variant. |
| todo | Qwen3-235B-A22B TRT-LLM disagg Blackwell | `recipes/qwen3-235b-a22b-fp8/trtllm/disagg/blackwell` | Yes | Family only | Add benchmark-backed variant. |
| todo | Qwen3-235B-A22B TRT-LLM disagg Hopper | `recipes/qwen3-235b-a22b-fp8/trtllm/disagg/hopper` | Yes | Family only | Add benchmark-backed variant. |
| draft | Qwen3-32B vLLM agg round-robin | `recipes/qwen3-32b/vllm/agg-round-robin` | Yes | Yes | Represented with deploy, verify, benchmark, operate, and cleanup commands; needs visual review. |
| draft | Qwen3-32B vLLM disagg KV router | `recipes/qwen3-32b/vllm/disagg-kv-router` | Yes | Yes | Represented with deploy, verify, benchmark, operate, cleanup, and configuration notes; needs visual review. |
| draft | Qwen3-32B vLLM agg KVBM | `recipes/qwen3-32b/vllm/agg-kvbm` | No | Yes | Represented as deployment-only variant with KVBM memory, connector, metrics, verify, and cleanup notes; needs visual review. |
| todo | Qwen3-32B FP8 TRT-LLM agg | `recipes/qwen3-32b-fp8/trtllm/agg` | Yes | Family only | Add benchmark-backed variant. |
| todo | Qwen3-32B FP8 TRT-LLM disagg | `recipes/qwen3-32b-fp8/trtllm/disagg` | Yes | Family only | Add benchmark-backed variant. |
| todo | Qwen3-32B FP8 vLLM disagg | `recipes/qwen3-32b-fp8/vllm/disagg` | Yes | Family only | Add benchmark-backed variant. |
| draft | Qwen3-VL-30B vLLM agg embedding cache | `recipes/qwen3-vl-30b/vllm/agg-embedding-cache` | Yes | Yes | Represented with cache on/off selector, dataset generation, deploy, benchmark, artifact, and cleanup sections; visually checked. |
| todo | Qwen3.6-35B vanilla vLLM serve | `recipes/qwen3.6-35b/deploy/vllm-serve.yaml` | Yes | Family only | Add baseline variant; explain script-driven workflow. |
| todo | Qwen3.6-35B Dynamo frontend decoding | `recipes/qwen3.6-35b/deploy/dynamo-fd.yaml` | Yes | Family only | Add comparison variant. |
| todo | Qwen3.6-35B Dynamo FD plus embedding cache | `recipes/qwen3.6-35b/deploy/dynamo-fd-ec.yaml` | Yes | Family only | Add recommended/interesting variant. |

## Immediate Implementation Order

1. Finish `qwen3-32b.mdx` because it is already the exemplar:
   - Visual check the updated page.
   - Decide whether Deploy tabs should auto-follow the technique selector or stay explicit.
2. Finish `deepseek-v3-2-nvfp4.mdx` as the second benchmark-heavy exemplar:
   - Visual check the updated page.
   - Confirm README benchmark-monitoring wording should follow the Job-based `perf.yaml`.
3. Finish `qwen3-vl-30b.mdx` to prove the multimodal/reuse pattern:
   - Confirm cache-off deploy guidance should stay as an inline patch or move entirely to the helper script.
4. Add `kimi-k2-5.mdx` to prove agentic coding workload navigation.
5. Fill deployment-only pages after benchmark-backed pages, with no implied perf claims.
6. Revisit the landing page after pages 1-4 exist:
   - Confirm taxonomy from real pages.
   - Replace hardcoded counts.
   - Decide deploy-only grouping.
   - Decide generated metadata source.
