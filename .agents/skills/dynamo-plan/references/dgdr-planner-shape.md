# DGDR Fields Relevant to Planning

A narrower view of [`dgdr-shape.md` in `dynamo-deploy`](../../dynamo-deploy/references/dgdr-shape.md) focused on the fields that drive **planning decisions**. The deploy skill carries the full reference.

Source: `deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go` and `docs/kubernetes/dgdr.md` on the target release branch.

---

## 1. `searchStrategy`

Two values, enum per:

| Value | Cost | Use when |
|---|---|---|
| `rapid` | ~30s; AIC simulator | Iterating; cluster small; cost matters |
| `thorough` | 2-4h; real GPUs | Production rollout; one-time planning per release line |

Default is `rapid`. Always set explicitly when planning — implicit defaults are not auditable.

## 2. `hardware`

| Field | Required | Notes |
|---|---|---|
| `gpuSku` | Yes in namespace-restricted; auto in cluster-wide | Enum per |
| `numGpusPerNode` | Yes in namespace-restricted; auto otherwise | Drives feasibility of TP/PP topologies |
| `totalGpus` | No | Capped at 32 by the operator; set if you want a specific budget |
| `interconnect` | No | NVLink / PCIe / RoCE / IB; affects the parallelism the profiler can pick |
| `rdma` | No | Disagg KV transfer over RDMA when true |

Namespace-restricted operator installs require `gpuSku`, `numGpusPerNode`, and `vramMb` explicitly — auto-detect is disabled because the operator cannot enumerate cluster nodes.

## 3. `workload`

| Field | Default | Planning note |
|---|---|---|
| `isl` | 4000 | Set this to the **mean** of your production ISL distribution, not the max |
| `osl` | 1000 | Same — mean of the distribution |
| `concurrency` | — | Required (or `requestRate`) when `features.planner` is **disabled** |
| `requestRate` | — | Alternative to `concurrency` |

The defaults match a generic chat workload. Real production deploys should always override.

## 4. `sla`

| Field | Notes |
|---|---|
| `ttft` (ms) | Time to first token |
| `itl` (ms) | Inter-token latency |
| `e2eLatency` (ms) | End-to-end latency; **cannot** be combined with explicit `ttft`/`itl` |
| `optimizationType` | `latency` or `throughput` |

`ttft + itl` is the recommended composition. `e2eLatency` is only appropriate when ISL/OSL are highly variable.

## 5. `features.planner`

When set, the generated DGD includes a Planner pod that adjusts replicas to track the SLA envelope at runtime. The Planner consumes the worker metrics emitted on `/metrics` and scales between `min-workers` and `max-workers`.

For pre-deployment planning, `features.planner` is opt-in. The decision factors:

| Factor | Direction |
|---|---|
| Workload is bursty / variable | Enable Planner |
| Workload is steady-state | Optional |
| Cluster has Grove gang scheduling | Required for the Planner's latency-mode scale-up; otherwise Planner may stall at `num_workers=1` |

## 6. `autoApply`

| Value | Meaning |
|---|---|
| `true` (default) | Operator deploys the result automatically once profiling finishes |
| `false` | Operator stops at `Ready`; DGD spec is held on `.status.profilingResults.selectedConfig` for review |

Planning workflows use `autoApply: false` so a human reviews the recommended config before any workload runs.

## 7. Reading the Result (`autoApply: false`)

```bash
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.profilingResults.selectedConfig}' | python3 -m json.tool
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.profilingResults.pareto}'    # Pareto-optimal alternates
```

The `selectedConfig` is the operator's recommended DGD spec. The `pareto` field carries alternates that meet the SLA at different cost/throughput points.

## 8. PCIe Caveat (per)

`h100_pcie`, `a100_pcie`, `v100_pcie` are admitted by the CRD but the profiler does not yet ship training data for them. Planning on PCIe SKUs:

- The DGDR will run.
- The result will use defaults rather than profiled data.
- The planning result is best-effort; verify with a benchmark after deploy.
