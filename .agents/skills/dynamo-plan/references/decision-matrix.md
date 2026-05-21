# Path-Selection Decision Matrix

Picking the right planning path for a given model/hardware combination.

---

## Quick Reference

| Scenario | Path |
|---|---|
| Recipe exists for exact model+framework+config | Recipe |
| Cluster not available; planning is offline | AIConfigurator |
| Cluster available; iterating quickly | DGDR `rapid` |
| Cluster available; production rollout | DGDR `thorough` |
| Custom backend or unsupported model | Manual DGD authoring (out of scope; see [dynamo-deploy](../../dynamo-deploy/SKILL.md)) |

## Worked Examples

### Example 1: Llama-3-70B on 4× H200 SXM

- A recipe exists: `recipes/llama-3-70b/vllm/agg/`.
- Path: **Recipe**. Skip AIConfigurator and DGDR. Apply the recipe DGD directly via `dynamo-deploy` Phase 2.4.

### Example 2: Qwen3-235B-A22B-FP8 on 16× B200 SXM

- A recipe exists: `recipes/qwen3-235b-a22b-fp8/trtllm/agg/blackwell/`.
- Path: **Recipe**. Apply directly.

### Example 3: Internal model variant, no recipe, no cluster yet

- Cluster not provisioned.
- Path: **AIConfigurator**. Plan offline, capture `planning.json`, save for when cluster is ready.

### Example 4: Internal model, cluster up, iterating on parallelism

- Cluster available; want fast feedback.
- Path: **DGDR `rapid`**. ~30s per iteration; reads from AIC simulator.

### Example 5: Internal model, production rollout, large fleet

- Want best-possible config; cost amortises over the release.
- Path: **DGDR `thorough`**. Reserve 2-4h of GPU; the result drives the entire release line.

## Comparison: AIConfigurator vs DGDR `rapid`

Both use the AIC simulator under the hood. Differences:

| | AIConfigurator | DGDR `rapid` |
|---|---|---|
| Where it runs | Local Python (or a remote API call) | In-cluster Kubernetes Job |
| Requires cluster | No | Yes |
| Result format | `planning.json` | DGDR `.status.profilingResults.selectedConfig` |
| Auto-deploy | No (manual conversion to DGDR/DGD) | Yes if `autoApply: true` |
| Best for | Pre-cluster planning, iteration on a laptop | In-cluster iteration once the platform is installed |

The two are not redundant — they fit different workflow phases.

## Comparison: `rapid` vs `thorough`

Both run as in-cluster profiling Jobs.

| | `rapid` | `thorough` |
|---|---|---|
| Duration | ~30s | 2-4h |
| Data source | AIC simulator | Real-GPU sweeps |
| GPU consumption | Negligible | Reserves `numGpusPerNode × replicas` for the duration |
| Result quality | Good | Best |
| Use case | Iteration | Production rollout |

A common pattern: iterate with `rapid` to narrow the search space, then run one `thorough` pass at the chosen parallelism to lock in the exact config.

## When None of the Paths Apply

If you have a model that is not in `recipes/`, the AIC profiler doesn't support, and DGDR rejects (e.g. an architecture the profiler has no training data for), the planning skill cannot help. Fall back to manual DGD authoring (see [dynamo-deploy](../../dynamo-deploy/SKILL.md) Phase 2.3) — the human picks parallelism from first principles or from a similar model's recipe as a starting point.
