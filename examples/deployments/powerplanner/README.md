# Power-Aware Planner Examples

This directory contains example configurations for deploying Dynamo with power-aware scaling.

## Recipes

| File | Phase | Description |
|------|-------|-------------|
| `disagg-power-aware.yaml` | P1+P2 | Power-aware disagg deployment with static per-GPU caps |
| `disagg-conservative-cold-start.yaml` | P3+ | AIC optimizer with conservative H200 cold-start coefficients |

## Prerequisites

1. Deploy the Power Agent DaemonSet on all GPU nodes:
   ```bash
   kubectl apply -f deploy/power_agent/rbac.yaml
   kubectl apply -f deploy/power_agent/daemonset.yaml
   ```

2. Verify the Power Agent is running:
   ```bash
   kubectl get pods -l app=dynamo-power-agent -o wide
   ```

## Quick start (Phase 1+2 — static caps)

1. Edit `disagg-power-aware.yaml` and set:
   - `total_gpu_power_limit`: `(rack_capacity_W × headroom_factor) − non_gpu_overhead`
     - Example for 8× H200 SXM at 700 W TDP, headroom 0.85: `(8 × 700 × 0.85) − 500 ≈ 4260 W`
   - `power_agent_safe_default_watts`: ~70% of SKU TDP (H200 SXM → 500 W)
   - `prefill_engine_gpu_power_limit` / `decode_engine_gpu_power_limit`: operator-set caps

2. Apply:
   ```bash
   kubectl apply -f disagg-power-aware.yaml
   ```

3. Verify annotations are being set on pods:
   ```bash
   kubectl get pods -l nvidia.com/dynamo-graph-deployment=vllm-disagg-power-aware \
     -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}{"\n"}{end}'
   ```

4. Check Power Agent applied the caps:
   ```bash
   # Prometheus query (if DCGM is configured)
   # dynamo_power_agent_applied_limit_watts{gpu="0"}
   ```

## Budget sizing formula

```
total_gpu_power_limit = (rack_capacity_W × headroom_factor) − non_gpu_overhead
```

- `headroom_factor` ≈ 0.85–0.9 (leave headroom for warm-up spikes)
- `non_gpu_overhead` includes CPU, DRAM, NICs, PSU losses, cooling
- **Not included** in `total_gpu_power_limit`: CPU/DRAM power, NIC/PCIe, PSU losses, cooling

## Prometheus alerts

```promql
# Conflicting annotations on a GPU (multi-DGD or MIG topology issue)
increase(dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"}[5m]) > 0

# Safe-default fallback (annotation parse failure or conflict)
increase(dynamo_power_agent_safe_default_applied_total[15m]) > 0

# Budget utilization (planner projected power vs declared budget)
dynamo_planner_power_projected_watts > dynamo_planner_power_budget_total_watts
```

## Multi-DGD notes

- Each planner enforces power only for the workers it manages.
- Multiple DGDs on disjoint GPUs: `Σ total_gpu_power_limit ≤ facility_capacity`.
- Multiple DGDs sharing one physical GPU (MIG/MPS): **not supported in v1**.
  See `design-docs/powerplanner-design.md` §3.3.
