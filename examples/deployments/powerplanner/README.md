# Power-Aware Autoscaling for Dynamo

Power-aware autoscaling for the Dynamo AI inference platform with GPU power budget enforcement. The planner monitors workload metrics and scales workers while respecting power constraints, with the Power Agent enforcing GPU power limits via NVML.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Architecture](#architecture)
4. [Verification & Monitoring](#verification--monitoring)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced-topics)
7. [Additional Resources](#additional-resources)

---

## Quick Start

### Requirements

Before running the deployment scripts, ensure you have:

**Binaries** (must be in PATH or `${DEV_REPO}/bin_bin/`):
- `kubectl` - Kubernetes command-line tool
- `minikube` - Local Kubernetes cluster
- `helm` - Kubernetes package manager
- `docker` - Container runtime

**Environment Variables**:
```bash
export HF_TOKEN=hf_your_token_here  # Get from https://huggingface.co/settings/tokens
```

**System**:
- Docker installed and running
- NVIDIA GPUs with drivers installed (for power enforcement)
- 500GB+ memory recommended for Minikube

> **Note**: The deployment scripts will check for these prerequisites and exit with clear error messages if anything is missing.

### Deploy in Two Steps

```bash
# Step 1: Deploy base infrastructure (Prometheus, Dynamo platform)
cd examples/deployments/powerplanner
bash deploy_poweraware_baseinfra.bash 1

# Step 2: Deploy power-aware features (100% automated)
bash deploy_poweraware.bash
```

**Expected time**: ~10-12 minutes total
- Base infrastructure: ~5-7 minutes
- Power-aware features: ~3-5 minutes (includes profiling)

**Automation**: The scripts automatically handle profiling data generation, RBAC configuration, Prometheus setup, and power limit application.

### Verify Deployment

```bash
# Run comprehensive verification
bash verify_poweraware.bash

# Quick check for power limit annotations
kubectl get pods -n dynamo-system -o custom-columns=\
NAME:.metadata.name,\
POWER-LIMIT:.metadata.annotations.dynamo\\.nvidia\\.com/gpu-power-limit
```

**Expected result**: All verification tests pass, and worker pods show power limit annotations (e.g., `250`).

---

## Configuration

### Power Budget Settings

Default configuration:
- **Total GPU power budget**: 1000W
- **Prefill GPU power limit**: 250W per GPU
- **Decode GPU power limit**: 250W per GPU
- **Planner adjustment interval**: 30 seconds

### Customizing Power Limits

Edit `examples/deployments/powerplanner/deploy_poweraware.bash` and modify the planner arguments:

```yaml
args:
  - --enable-power-awareness
  - --total-gpu-power-limit=1000        # Change total budget
  - --prefill-engine-gpu-power-limit=250  # Change prefill limit
  - --decode-engine-gpu-power-limit=250   # Change decode limit
```

Then redeploy:

```bash
bash examples/deployments/powerplanner/deploy_poweraware.bash
```

### Profiling Data

The planner uses profiling data from the `planner-profile-data` ConfigMap to calculate required replicas. The deployment script automatically runs profiling if the ConfigMap is missing.

To use custom profiling data:
1. Run your own profiling job
2. Create a ConfigMap named `planner-profile-data` containing:
   - `prefill_raw_data.json`
   - `decode_raw_data.json`

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Power-Aware Planner                       │
│  • Monitors load via Prometheus metrics                     │
│  • Calculates required replicas based on SLA targets        │
│  • Checks power budget constraints                          │
│  • Scales down if power budget exceeded                     │
│  • Sets pod annotations with power limits                   │
└────────────┬────────────────────────────────────────────────┘
             │ Annotations: dynamo.nvidia.com/gpu-power-limit
             ↓
┌─────────────────────────────────────────────────────────────┐
│                     Power Agent (DaemonSet)                  │
│  • Watches pod annotations continuously                      │
│  • Maps PIDs to pods via cgroups                             │
│  • Enforces GPU power limits via NVML                        │
└─────────────────────────────────────────────────────────────┘
             ↓
         GPU Hardware
```

### How It Works

1. **Planner monitors load** via Prometheus metrics (TTFT, ITL, request rate)
2. **Calculates required replicas** based on SLA targets and profiling data
3. **Power budget check**:
   ```
   required_power = (num_prefill × prefill_limit) + (num_decode × decode_limit)
   ```
4. **Enforcement**:
   - If `required_power ≤ total_budget`: Deploy all replicas
   - If `required_power > total_budget`: Scale down proportionally
5. **Annotation**: Sets `dynamo.nvidia.com/gpu-power-limit` on worker pods
6. **Power Agent**: Enforces GPU power limits via NVML

<details>
<summary><strong>Power Budget Enforcement Algorithm</strong> (click to expand)</summary>

```python
def apply_power_limits(self, prefill_replicas, decode_replicas):
    # Calculate required power
    required_power = (
        prefill_replicas * self.prefill_power_limit +
        decode_replicas * self.decode_power_limit
    )

    # Check budget
    if required_power <= self.total_power_budget:
        # Under budget - deploy all replicas
        self.set_power_annotations(prefill_replicas, decode_replicas)
        return prefill_replicas, decode_replicas
    else:
        # Over budget - scale down proportionally
        scale_factor = self.total_power_budget / required_power
        scaled_prefill = int(prefill_replicas * scale_factor)
        scaled_decode = int(decode_replicas * scale_factor)

        logger.warning(f"Power budget exceeded: {required_power}W > {self.total_power_budget}W")
        logger.info(f"Scaling down: prefill {prefill_replicas}→{scaled_prefill}, "
                   f"decode {decode_replicas}→{scaled_decode}")

        self.set_power_annotations(scaled_prefill, scaled_decode)
        return scaled_prefill, scaled_decode
```

</details>

### Components

**Planner** (`components/src/dynamo/planner/`):
- Monitors Prometheus metrics
- Calculates replica requirements
- Enforces power budget constraints
- Annotates pods with power limits

**Power Agent** (`components/power_agent/`):
- Runs as DaemonSet on GPU nodes
- Watches pod annotations
- Maps container PIDs to GPUs via cgroups
- Sets GPU power limits via NVML
- Reconciles every 15 seconds

---

## Verification & Monitoring

### Automated Verification

Run the comprehensive verification suite:

```bash
cd examples/deployments/powerplanner
bash verify_poweraware.bash
```

This tests:
- Infrastructure (Minikube, namespace, pods, PodMonitors)
- Automation (RBAC, profiling data, power annotations)
- Prometheus integration (connectivity, metrics, labels)
- Functionality (model detection, traffic processing, metric observation)
- Hardware enforcement (GPU power limits via nvidia-smi)

### Manual Inspection

#### Check Planner Logs

```bash
PLANNER_POD=$(kubectl get pods -n dynamo-system -l nvidia.com/dynamo-component=Planner | grep Running | awk 'NR==1 {print $1}')
kubectl logs -n dynamo-system ${PLANNER_POD} --tail=100
```

Look for:
```
INFO: Detected model name from deployment: Qwen/Qwen3-0.6B
INFO: Observed num_req: 20.40 isl: 11.65 osl: 5.50
INFO: Observed ttft: 7.48ms itl: 2.40ms
INFO: Applied power limits: 1 prefill @ 250W, 1 decode @ 250W
```

#### Check Power Limit Annotations

```bash
kubectl get pods -n dynamo-system -o custom-columns=\
NAME:.metadata.name,\
POWER-LIMIT:.metadata.annotations.dynamo\\.nvidia\\.com/gpu-power-limit
```

Expected output:
```
NAME                                    POWER-LIMIT
vllm-disagg-vllmprefillworker-xxx       250
vllm-disagg-vllmdecodeworker-xxx        250
```

#### Verify GPU Power Limits (Hardware)

```bash
nvidia-smi --query-gpu=index,power.limit --format=csv
```

Expected output (for GPUs with workloads):
```
index, power.limit [W]
0, 250.00
2, 250.00
```

#### Send Test Traffic

```bash
# Port forward
kubectl port-forward svc/vllm-disagg-frontend 8000:8000 -n dynamo-system &

# Send test request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'
```

#### Check Prometheus Metrics

```bash
PLANNER_POD=$(kubectl get pods -n dynamo-system -l nvidia.com/dynamo-component=Planner -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n dynamo-system ${PLANNER_POD} -- python3 -c "
from prometheus_api_client import PrometheusConnect
prom = PrometheusConnect(url='http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090', disable_ssl=True)
result = prom.custom_query(query='vllm:time_to_first_token_seconds_sum')
if result:
    m = result[0]['metric']
    print(f\"model_name: {m.get('model_name', 'MISSING')}\")
    print(f\"dynamo_namespace: {m.get('dynamo_namespace', 'MISSING')}\")
"
```

### Real-Time Monitoring

Watch planner logs:
```bash
kubectl logs -f -n dynamo-system ${PLANNER_POD}
```

Monitor pod power limits:
```bash
watch -n 2 'kubectl get pods -n dynamo-system -o custom-columns=NAME:.metadata.name,POWER:.metadata.annotations.dynamo\\.nvidia\\.com/gpu-power-limit'
```

Check Prometheus targets:
```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
# Open http://localhost:9090/targets in browser
```

View Power Agent status:
```bash
kubectl get daemonset power-agent -n dynamo-system
kubectl logs -n dynamo-system -l app=power-agent
```

---

## Troubleshooting

### Power Agent Can't Enforce Limits in Minikube

**Symptom**:
- Power Agent is running
- Pod annotations are set correctly (250W)
- But GPU power limits remain at default (700W)

**Cause**: Minikube with `--driver=docker` creates nested containerization that prevents the Power Agent from accessing host PIDs.

**Resolution**: The deployment scripts handle this automatically:
- Minikube is started with `--mount --mount-string="/proc:/host/proc"`
- Power Agent DaemonSet mounts `/host/proc` and uses it for PID mapping
- GPU power limits are enforced successfully

**Verification**: Run test 13 in the verification suite or check with `nvidia-smi --query-gpu=power.limit --format=csv`.

### Planner Shows "No Prometheus Metric Data"

**Symptom**:
```
WARN: No prometheus metric data available for vllm:time_to_first_token_seconds
```

**Cause**: No traffic in the last 30 seconds (planner uses `increase()[30s]`).

**Resolution**: Send test traffic to generate metrics:
```bash
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}' > /dev/null
  sleep 2
done
```

### Profiling ConfigMap Not Found

**Symptom**:
```
✗ ConfigMap planner-profile-data not found
```

**Resolution**: The deployment script automatically runs profiling if the ConfigMap is missing. Wait for completion (~5-10 minutes).

To manually trigger profiling:
```bash
kubectl apply -f examples/deployments/powerplanner/profile_sla_aic_dgdr.yaml -n dynamo-system
kubectl wait --for=condition=complete dynamographdeploymentrequest/sla-aic -n dynamo-system --timeout=600s
```

### RBAC Permission Errors

**Symptom**:
```
ERROR: pods is forbidden: User "system:serviceaccount:dynamo-system:planner-serviceaccount" cannot list resource "pods"
```

**Resolution**: The deployment script automatically patches the ClusterRole. If it fails, manually apply:
```bash
kubectl apply -f examples/deployments/powerplanner/planner-clusterrole-patch.yaml
```

### dynamo_namespace Label Missing

**Symptom**: Prometheus metrics don't have `dynamo_namespace` label.

**Resolution**: The deployment script automatically configures PodMonitor relabeling. If it fails, manually apply:
```bash
kubectl apply -f examples/deployments/powerplanner/dynamo-worker-podmonitor.yaml
```

Wait 30 seconds for Prometheus to reload, then send fresh traffic.

### Debug Commands

Check planner image:
```bash
kubectl get pod ${PLANNER_POD} -n dynamo-system -o jsonpath='{.spec.containers[0].image}'
# Should show: dynamo/planner-power-aware:dev
```

Check planner arguments:
```bash
kubectl get pod ${PLANNER_POD} -n dynamo-system -o jsonpath='{.spec.containers[*].args}' | jq
```

Check if power awareness is enabled:
```bash
kubectl logs -n dynamo-system ${PLANNER_POD} | grep -i "power"
```

Check profiling data mount:
```bash
kubectl exec -n dynamo-system ${PLANNER_POD} -- ls -la /workspace/profiling_results/
```

Check PodMonitor configuration:
```bash
kubectl get podmonitor dynamo-worker -n dynamo-system -o yaml
```

---

## Advanced Topics

### DynamoGraphDeployment Rollout Restart

The operator supports controlled restarts of graph deployments with customizable strategies:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  restart:
    id: "restart-2026-01-09"  # Change this value to trigger a restart
    strategy:
      type: Sequential  # or Parallel
      order:  # Optional: specify restart order
        - Frontend
        - VLLMPrefillWorker
        - VLLMDecodeWorker
```

**Benefits for power-aware deployments**:
- Controlled restarts when updating power limits
- Sequential restart strategy prevents power spikes
- Ordered restart ensures frontend comes up last

**Status tracking**:
```bash
kubectl get dgd vllm-disagg -n dynamo-system -o jsonpath='{.status.restart}'
```

See the [API Reference](../../../docs/kubernetes/api_reference.md#restart) for more details.

### Multinode Deployments

For multinode power-aware deployments:
- Use `--host 0.0.0.0` to expose SGLang bootstrap server on all interfaces
- Configure `--disaggregation-bootstrap-port` for cross-node communication
- Ensure network ports are accessible between nodes

See [examples/basics/multinode/README.md](../../basics/multinode/README.md) for details.

### Production Deployment Checklist

Before deploying to production:

**Infrastructure**:
- [ ] Kubernetes cluster with GPU nodes
- [ ] NVIDIA GPU Operator or device plugin installed
- [ ] Prometheus with DCGM exporter configured
- [ ] kube-state-metrics deployed
- [ ] Persistent storage for profiling data

**Configuration**:
- [ ] Measure actual GPU power consumption in your datacenter
- [ ] Set realistic power budgets based on measurements
- [ ] Run performance profiling for your specific models
- [ ] Configure appropriate SLA targets (TTFT/ITL)
- [ ] Tune power limits based on workload patterns

**Testing**:
- [ ] Test power budget enforcement with various workloads
- [ ] Verify Power Agent enforces limits on real GPUs
- [ ] Load test with traffic patterns matching production
- [ ] Verify SLA compliance under power constraints
- [ ] Test failover and recovery scenarios

**Monitoring**:
- [ ] Set up Grafana dashboards for power metrics
- [ ] Configure alerts for power budget violations
- [ ] Monitor GPU power consumption trends
- [ ] Track SLA compliance metrics
- [ ] Set up logging aggregation

---

## Additional Resources

### Documentation
- [Kubernetes API Reference](../../../docs/kubernetes/api_reference.md) - DynamoGraphDeployment restart mechanism
- [Multinode Deployment Guide](../../basics/multinode/README.md) - Multi-node setup with KV routing
- [KV Cache Routing](../../../docs/router/kv_cache_routing.md) - KV-aware routing architecture
- [Disaggregated Serving](../../../docs/design_docs/disagg_serving.md) - Disaggregation design

### Scripts

**Deployment**:
- `deploy_poweraware_baseinfra.bash` - Base infrastructure deployment
- `deploy_poweraware.bash` - Power-aware features deployment
- `verify_poweraware.bash` - Comprehensive verification suite
- `full_clean_test.bash` - Complete clean test (all phases)
- `monitor_poweraware.bash` - Real-time monitoring dashboard

**Configuration Files**:
- `planner-clusterrole-patch.yaml` - RBAC permissions for planner
- `dynamo-worker-podmonitor.yaml` - Prometheus metric relabeling
- `profile_sla_aic_dgdr.yaml` - Profiling job configuration
- `prometheus-values.yaml` - Prometheus Helm values
- `agg.yaml` / `disagg.yaml` - Local deployment configurations

### Quick Reference

```bash
# Full clean deployment test
cd examples/deployments/powerplanner
bash full_clean_test.bash

# Two-step deployment
bash deploy_poweraware_baseinfra.bash 1  # Base
bash deploy_poweraware.bash              # Power-aware

# Verification
bash verify_poweraware.bash

# Monitoring
bash monitor_poweraware.bash

# Cleanup
bash deploy_poweraware_baseinfra.bash 0
```

---

**Ready to deploy!** For questions or issues, refer to the [Troubleshooting](#troubleshooting) section or check the deployment logs.

**For implementation details and verification test results, see the [CHANGELOG.md](CHANGELOG.md).**
