# Multi-DGD Power Budget Operator Playbook

This guide covers running multiple DynamoGraphDeployments (DGDs) on the same
GPU cluster with the power-aware planner enabled.

---

## Scope of per-DGD power accounting

Each planner instance manages exactly the workers **it owns** via its
DGD name selector. It has no visibility into other DGDs' GPU allocations.
This means operators must maintain the invariant:

```
Σ total_gpu_power_limit (all DGDs on shared rack) ≤ facility_capacity_W − non_gpu_overhead − headroom
```

The planner cannot detect over-commitment across DGDs. That is the operator's
responsibility.

---

## Recommended deployment topologies (in order of preference)

### 1. One DGD per node pool — Fully isolated (recommended)

Each DGD targets a disjoint set of GPU nodes via Kubernetes node labels or
taints. Power budgets never overlap.

```yaml
# DGD-A targets node pool "pool-a"
nodeSelector:
  pool: pool-a
# total_gpu_power_limit = budget for pool-a only

# DGD-B targets node pool "pool-b"
nodeSelector:
  pool: pool-b
# total_gpu_power_limit = budget for pool-b only
```

This is the only topology where the planner's invariants hold without any
cross-DGD coordination.

**Sizing formula per DGD:**
```
total_gpu_power_limit = (rack_capacity_W × headroom_factor) − non_gpu_overhead_W
```

Where:
- `rack_capacity_W` — PDU-rated capacity for the node pool's rack(s)
- `headroom_factor ≤ 0.85` — recommended safety margin for simultaneous pod restarts
- `non_gpu_overhead_W` — CPUs, NVSwitches, NICs, storage, cooling fans (typically 500–1500 W per node)

**Example — 8× H200 SXM nodes, 2× 30 kW PDUs:**
```
rack_capacity_W    = 60_000
headroom_factor    = 0.85
non_gpu_overhead_W = 4_000   # 2 nodes × ~2 kW overhead each

total_gpu_power_limit = 60_000 × 0.85 − 4_000 = 47_000 W
```

### 2. Multiple DGDs, disjoint GPUs, shared nodes

Two or more DGDs run on the same physical nodes but use non-overlapping GPUs
(enforced by the NVIDIA device plugin or CDI). Both planners independently
annotate their own pods.

Requirements:
1. GPU selection must be disjoint. Use `nvidia.com/gpu.count` resource requests
   or explicit device UUIDs via CDI to guarantee separation.
2. Operator must manually apportion the rack budget:
   ```
   DGD-A.total_gpu_power_limit + DGD-B.total_gpu_power_limit
       ≤ facility_capacity_W − non_gpu_overhead_W
   ```
3. Set `headroom_factor ≤ 0.85` for each DGD independently.

The planner cannot enforce or detect the cross-DGD constraint — this must be
maintained in your deployment tooling (Helm values, GitOps pipeline, etc.).

### 3. Shared-GPU multi-tenancy (MIG / MPS / time-slicing) — NOT SUPPORTED v1

Two DGDs that share a single physical GPU will race to annotate the same pod's
GPU. The Power Agent uses last-writer-wins semantics; the resulting cap is
non-deterministic. **Do not use this topology.**

If you need shared-GPU multi-tenancy, consolidate into a single DGD or wait
for v2 tracking in [#TBD].

---

## Budget apportionment worksheet

```
# Fill in your cluster values:

RACK_CAPACITY_W   = ____     # PDU nameplate (total, all phases)
HEADROOM_FACTOR   = 0.85     # reduce to 0.80 for tighter racks
NON_GPU_OVERHEAD  = ____     # estimate: num_nodes × 1500 W typical

USABLE_W = RACK_CAPACITY_W × HEADROOM_FACTOR − NON_GPU_OVERHEAD

# Split USABLE_W across DGDs proportional to GPU count:
DGD_A_GPU_FRACTION = num_gpus_A / total_gpus
DGD_B_GPU_FRACTION = num_gpus_B / total_gpus

DGD_A.total_gpu_power_limit = USABLE_W × DGD_A_GPU_FRACTION
DGD_B.total_gpu_power_limit = USABLE_W × DGD_B_GPU_FRACTION
```

---

## Prometheus alerts for multi-DGD misconfiguration

Add these rules to your cluster alert manager.

```yaml
groups:
  - name: dynamo_power_multi_dgd
    rules:

      # 1. Conflicting annotations — two DGDs writing to the same GPU
      - alert: DynamoPowerAgentConflictingAnnotation
        expr: |
          increase(dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"}[5m]) > 0
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "Power Agent annotation conflict on {{ $labels.node }}"
          description: >
            Two pods (likely from different DGDs) are competing for the same GPU's
            power cap. The agent has applied power_agent_safe_default_watts.
            Check topology immediately — this is unsupported.

      # 2. Safe-default fallback (catch-all)
      - alert: DynamoPowerAgentSafeDefault
        expr: |
          increase(dynamo_power_agent_safe_default_applied_total[15m]) > 0
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "Power Agent safe-default applied on {{ $labels.node }}"
          description: >
            Either a multi-DGD conflict or an annotation parse failure.
            GPU is running at power_agent_safe_default_watts, not the planned cap.

      # 3. Cross-DGD over-commitment (DCGM required)
      - alert: DynamoClusterPowerOverCommit
        expr: |
          sum(DCGM_FI_DEV_POWER_USAGE) > <FACILITY_CAPACITY_W> * 0.90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Cluster GPU power approaching facility limit"
          description: >
            Sum of all GPU power exceeds 90% of facility capacity.
            Review Σ total_gpu_power_limit across all DGDs.

      # 4. Single-DGD budget exceeded
      - alert: DynamoPlannerBudgetExceeded
        expr: |
          dynamo_planner_power_projected_watts > dynamo_planner_power_budget_total_watts
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "DGD {{ $labels.dgd }} power budget exceeded"
          description: >
            _apply_power_budget clamping is active but projected power still
            exceeds total_gpu_power_limit. Either traffic outgrew sizing or
            the budget was set too tight.
```

---

## Escalation: `multi_pod_gpu_total{disposition="conflict"}` increments

1. Identify the conflicting pods:
   ```bash
   kubectl get pods --field-selector spec.nodeName=<node> \
     -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.nvidia\.com/dynamo-graph-deployment}{"\n"}{end}'
   ```

2. Determine if they share a physical GPU:
   ```bash
   kubectl get pod <pod-name> -o json \
     | jq '.status.containerStatuses[].allocatedResources'
   ```
   Cross-reference device plugin assignment or CDI device IDs.

3. **If pods belong to different DGDs and share a GPU** — this is the
   unsupported topology (case 3 above). Move the DGDs to disjoint GPU pools.

4. **If pods belong to the same DGD** — a planner bug or external mutator
   wrote conflicting annotations. Check:
   ```bash
   kubectl get pod <pod-name> -o jsonpath=\
     '{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}'
   ```
   And review planner logs for annotation PATCH errors.

5. **While diagnosing:** Affected GPUs run at `power_agent_safe_default_watts`
   (conservative, hardware-safe). No emergency GPU intervention needed, but
   SLA may degrade — consider manual scale-out.

---

## Cold-start power spike window

Between when a pod is *scheduled* and when the Power Agent *applies its cap*
there is a window of up to **15 seconds** where the GPU runs at TGP.
During this window:

| Phase | Power draw | Safe? |
|-------|-----------|-------|
| Weight load (disk → HBM) | 150–250 W (H200 SXM) | Yes — far below TGP |
| NCCL / topology init | Near-zero SM occupancy | Yes |
| CUDA-graph warm-up | **~300–500 W** (~70% of TGP) | Within rack headroom if `headroom_factor ≤ 0.85` |
| Steady-state serving | Bounded by NVML cap | Safe |

Three structural protections keep the warm-up spike from becoming a PDU event:

1. **Pod count is bounded**: `_apply_power_budget()` sizes replica counts
   assuming caps are in effect. The cluster never starts more pods than
   `total_gpu_power_limit / cap_per_pod`.
2. **Pods are not routed traffic until warm-up finishes**: The engine registers
   with the frontend/router only after CUDA-graph capture completes. A
   "cap-not-yet-applied" pod is also a "pre-traffic" pod.
3. **`headroom_factor ≤ 0.85`**: Sizing the budget to 85% of facility capacity
   leaves room for simultaneous warm-up across all scaling pods on the rack.

---

## Recommended cold-start EMA coefficient values (H200 SXM, dense models)

These are the values recommended in `disagg-conservative-cold-start.yaml`
for Phase 3 closed-loop operation before Phase 5 hardware validation is
available. They are intentionally conservative:

| Coefficient | Recommended | Rationale |
|-------------|------------|-----------|
| `aic_initial_c_ttft` | 1.15 | Compute-bound prefill; 15% over AIC estimate absorbs early queue build-up |
| `aic_initial_c_itl` | 1.15 | Generous for MLA/long-context decode paths |
| `aic_initial_c_power_prefill` | 1.05 | GEMM-heavy prefill tracks AIC closely; small margin |
| `aic_initial_c_power_decode` | 1.15 | HBM-bound decode power can exceed AIC estimate on high-concurrency first tick |
| `aic_initial_c_power_agg` | 1.10 | Aggregated mode average |

All coefficients are EMA-smoothed toward live measurements within 2–3
`aic_reoptimize_interval` cycles. Values above 1.0 make the first few sweeps
**conservatively tighter**; the EMA will loosen them if AIC is accurate.

> **TODO (Phase 5):** Validate and update these values with measured H200 data
> from the `aic_h200_power_data` dataset.

---

## See also

- `disagg-power-aware.yaml` — Phase 1+2 static-cap example
- `disagg-conservative-cold-start.yaml` — Phase 3 AIC optimizer example
- `verify_poweraware.bash` — smoke-test script for a running deployment
- `tools/integrate_aic_power_data.py` — integrate measured power data into AIC
- `components/power_agent/README.md` — Power Agent DaemonSet guide
- `deploy/helm/charts/power-agent/` — Helm chart (production DaemonSet + dev Pod modes)
