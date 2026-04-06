# DeepSeek-V3.1-NVFP4 GAIE deployment (v1 bundle)

Config directory: `deployments/nvfp4-gaie-v1/` (aggregated + disaggregated GAIE, Gateway routes, policies, benchmarks).

Disaggregated (1 Prefill + 1 Decode) deployment of **nvidia/DeepSeek-V3.1-NVFP4** using the
NVIDIA AI Dynamo platform with GAIE EPP scheduling, routed through the GKE Inference Gateway.

## Architecture


| Component | Nodes | GPUs    | Role                                                             |
| --------- | ----- | ------- | ---------------------------------------------------------------- |
| Prefill   | 1     | 8x B200 | Prompt processing (DP=8, EP=8, TP=8)                             |
| Decode    | 1     | 8x B200 | Token generation (DP=8, EP=8, TP=8) + EAGLE speculative decoding |
| EPP       | 1     | —       | Endpoint Picker (disagg-aware prefill/decode scoring)            |


**Total: 16 GPUs across 2 nodes**, interconnected via 8x RoCE RDMA interfaces (mlx5_0..mlx5_7)
for KV cache transfer using NIXL.

### Key Optimizations

- **NVFP4 quantization** (`--quantization modelopt_fp4`) — 4-bit floating-point weights
- **JIT DeepGEMM** — dynamically compiled GEMM kernels for varying batch sizes
- **EAGLE speculative decoding** — `num-steps=2, eagle-topk=1, num-draft-tokens=3`
- **CUDA graphs** on decode — `--cuda-graph-max-bs 512` for reduced kernel launch overhead
- **FlashInfer allreduce fusion** — fused communication for expert parallelism
- **Disaggregated KV transfer** via NIXL over UCX/RDMA (RoCE)

## Files


| File                         | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| `dgd-disagg-gaie-nvfp4.yaml` | DynamoGraphDeployment — NVFP4 disagg (prefill, decode, EPP)                   |
| `dgd-agg-gaie.yaml`          | DynamoGraphDeployment — NVFP4 aggregated GAIE (EPP + worker sidecar)         |
| `dgd-agg-native.yaml`        | DynamoGraphDeployment — NVFP4 aggregated native (no Gateway / no EPP)        |
| `httproute.yaml`             | HTTPRoute — `x-pool: gaie-agg`, `gaie-disagg-nvfp4`, default → agg pool     |
| `health-check-policy.yaml`   | HealthCheckPolicy — `:8000/health` for **agg** and **disagg** pool Services |
| `backend-policy.yaml`        | GCPBackendPolicy — 600s timeout for **agg** and **disagg** pool Services    |
| `benchmark-aiperf.sh`        | Parameterized `aiperf profile` (edit `CONCURRENCY`, `X_POOL`, `GATEWAY_IP`) |


## Deployment Steps

### 1. Deploy the DynamoGraphDeployment

```bash
kubectl apply -f dgd-disagg-gaie-nvfp4.yaml -n dynamo-system
```

Wait for all pods to become ready (~5-10 min for model loading):

```bash
kubectl get pods -n dynamo-system -l nvidia.com/dynamo-graph-deployment-name=sglang-dsv31-nvfp4-disagg-gaie -w
```

### 2. Identify the auto-generated pool-ips Service names

Each InferencePool gets a headless `*-pool-ips-<hash>` Service. Both
`health-check-policy.yaml` and `backend-policy.yaml` must reference the **current**
names:

```bash
kubectl get svc -n dynamo-system | grep -E 'nvfp4-agg-gaie-pool-ips|nvfp4-disagg-gaie-pool-ips'
```

Example:

```
sglang-dsv31-nvfp4-agg-gaie-pool-ips-e00ef715      ClusterIP   None   <none>   54321/TCP   2h
sglang-dsv31-nvfp4-disagg-gaie-pool-ips-c01ab801   ClusterIP   None   <none>   54321/TCP   2h
```

If a hash differs from the YAML, update **both** `targetRef.name` entries for that pool
in `health-check-policy.yaml` and `backend-policy.yaml` before applying.

### 3. Apply the HTTPRoute

```bash
kubectl apply -f httproute.yaml -n dynamo-system
```

### 4. Apply the HealthCheckPolicy

```bash
kubectl apply -f health-check-policy.yaml -n dynamo-system
```

### 5. Apply the GCPBackendPolicy (timeout fix)

```bash
kubectl apply -f backend-policy.yaml -n dynamo-system
```

Verify both policies attached:

```bash
kubectl get gcpbackendpolicy -n dynamo-system
for p in nvfp4-agg-gaie-backend-policy nvfp4-disagg-gaie-backend-policy; do
  kubectl get gcpbackendpolicy "$p" -n dynamo-system -o jsonpath='{.metadata.name}: {.status.conditions[0].reason} {.status.conditions[0].status}{"\n"}'
done
```

You should see `Attached` / `True` for each.

## Fixing the 504 Gateway Timeout (Backend Service Timeout)

### The Problem

By default, GKE's backend service timeout is **30 seconds**. LLM inference requests —
especially at high concurrency or with long output sequences — regularly exceed this.
When they do, the GKE load balancer terminates the connection and returns a
**504 Gateway Timeout**, even though the backend is still processing normally.

Symptoms:

- `aiperf` or `curl` requests fail with HTTP 504 after exactly 30s
- Backend pod logs show the request was still being processed (no errors)
- The issue is more pronounced at higher concurrency or with streaming responses

### The Fix

The `GCPBackendPolicy` resource lets you override the backend service timeout at the
Kubernetes level, without going through the GCP Console.

The repo ships **two** `GCPBackendPolicy` objects (one per pool); see `backend-policy.yaml`.
Example `targetRef` for the disaggregated pool:

```yaml
apiVersion: networking.gke.io/v1
kind: GCPBackendPolicy
metadata:
  name: nvfp4-disagg-gaie-backend-policy
  namespace: dynamo-system
spec:
  default:
    timeoutSec: 600          # 10 minutes (up from default 30s)
    connectionDraining:
      drainingTimeoutSec: 600  # graceful drain on pod termination
  targetRef:
    group: ""
    kind: Service
    name: sglang-dsv31-nvfp4-disagg-gaie-pool-ips-c01ab801  # <-- auto-generated pool-ips Service
```

**Key details:**

- The `targetRef` must point to the **headless pool-ips Service** created by the InferencePool
controller, not the EPP service or the worker services
- `timeoutSec: 600` sets the backend service timeout to 10 minutes
- `connectionDraining.drainingTimeoutSec: 600` allows in-flight requests to finish during
rolling updates or pod eviction
- After applying, it takes ~1-2 min for the GKE load balancer to reconcile the change
- Verify with: `kubectl get gcpbackendpolicy -n dynamo-system` — status should show `Attached`

### How to find the right Service name

The pool-ips Service name includes a hash suffix that changes if the InferencePool is
recreated. Always verify before applying:

```bash
kubectl get svc -n dynamo-system | grep -E 'nvfp4-.*-pool-ips'
```

## Benchmarking

The repo includes **`benchmark-aiperf.sh`** — same flags as below, with **`CONCURRENCY`**, **`X_POOL`**, and **`GATEWAY_IP`** called out at the top of the script so you can sweep load without editing long command lines.

From the `perf-gaie-fp8` pod (or any pod with `aiperf`):

```bash
GATEWAY_IP=192.168.0.75   # kubectl get gateway inference-gateway -n dynamo-system -o jsonpath='{.status.addresses[0].value}'

aiperf profile \
  --url "http://${GATEWAY_IP}" \
  --header 'x-pool:gaie-disagg-nvfp4' \
  --artifact-dir '/workspace/results/nvfp4/gaie-disagg-nvfp4-c500' \
  --model 'nvidia/DeepSeek-V3.1-NVFP4' \
  --tokenizer '/opt/model-cache/hub/models--nvidia--DeepSeek-V3.1-NVFP4/snapshots/68b4a17cce1482e94030ca00dacda3dec4c6359d' \
  --endpoint-type 'chat' --endpoint '/v1/chat/completions' --streaming \
  --synthetic-input-tokens-mean 1000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 250 --output-tokens-stddev 0 \
  --extra-inputs 'max_tokens:250' --extra-inputs 'min_tokens:250' \
  --extra-inputs 'ignore_eos:true' --extra-inputs 'repetition_penalty:1.0' \
  --extra-inputs 'temperature:0.0' \
  --use-server-token-count \
  --concurrency 500 --request-count 500 --warmup-request-count 20 \
  --num-dataset-entries 12800 --random-seed 100 --workers-max 50 \
  --record-processors 16
```

### Disaggregated GAIE Results (2026-04-05, ISL=1000, OSL=250, 16 GPUs — 1P+1D)

| Concurrency | Output TPS | Req/s | TPOT p50 (ms) | TPOT avg (ms) | TPS/User p50 | TPS/User avg | TTFT p50 (ms) | ITL p50 (ms) | TPS/GPU |
|:-----------:|:----------:|:-----:|:-------------:|:-------------:|:------------:|:------------:|:-------------:|:------------:|:-------:|
| 10          | 855        | 3.42  | 9.67          | 9.71          | 103.42       | 103.16       | 378           | 9.67         | 53.5    |
| 100         | 5,271      | 21.08 | 13.45         | 14.28         | 74.34        | 71.73        | 996           | 13.45        | 329.4   |
| 200         | 7,938      | 31.75 | 16.04         | 16.48         | 62.35        | 62.05        | 1,301         | 16.04        | 496.2   |
| 500         | 10,935     | 43.74 | 19.28         | 19.20         | 51.87        | 52.86        | 4,714         | 19.28        | 683.4   |

### Aggregated GAIE Results (2026-04-05, ISL=1000, OSL=250, 8 GPUs — 1 node)

| Concurrency | Output TPS | Req/s | TPOT p50 (ms) | TPOT avg (ms) | TPS/User p50 | TPS/User avg | TTFT p50 (ms) | ITL p50 (ms) | TPS/GPU | Errors |
|:-----------:|:----------:|:-----:|:-------------:|:-------------:|:------------:|:------------:|:-------------:|:------------:|:-------:|:------:|
| 10          | 1,051      | 4.20  | 7.98          | 7.99          | 125.28       | 125.15       | 293           | 7.98         | 131.4   | 0      |
| 100         | 1,985      | 7.94  | 11.49         | 11.35         | 87.07        | 88.26        | 9,161         | 11.49        | 248.1   | 0      |
| 200         | 2,005      | 8.02  | 11.48         | 11.21         | 87.12        | 89.43        | 20,610        | 11.48        | 250.6   | 0      |
| 500*        | 1,996      | 8.02  | 11.49         | 11.27         | 87.00        | 88.99        | 13,367        | 11.49        | 249.4   | 260    |

> *C=500 agg: 260/500 requests failed with `ClientPayloadError` (transfer encoding). Only 240 completed.
> The agg deployment has `--max-running-requests 24` — it saturates and queues heavily beyond C=10.

### Head-to-Head Comparison (Disagg vs Agg)

| Metric                    | C=10 Disagg | C=10 Agg | C=100 Disagg | C=100 Agg | C=200 Disagg | C=200 Agg | C=500 Disagg | C=500 Agg* |
|:--------------------------|:-----------:|:--------:|:------------:|:---------:|:------------:|:---------:|:------------:|:----------:|
| **Output TPS**            | 855         | 1,051    | 5,271        | 1,985     | 7,938        | 2,005     | 10,935       | 1,996      |
| **TPOT p50 (ms)**         | 9.67        | 7.98     | 13.45        | 11.49     | 16.04        | 11.48     | 19.28        | 11.49      |
| **TPS/User p50**          | 103.42      | 125.28   | 74.34        | 87.07     | 62.35        | 87.12     | 51.87        | 87.00      |
| **TTFT p50 (ms)**         | 378         | 293      | 996          | 9,161     | 1,301        | 20,610    | 4,714        | 13,367     |
| **TPS/GPU**               | 53.5        | 131.4    | 329.4        | 248.1     | 496.2        | 250.6     | 683.4        | 249.4      |
| **Req/s**                 | 3.42        | 4.20     | 21.08        | 7.94      | 31.75        | 8.02      | 43.74        | 8.02       |
| **Errors**                | 0           | 0        | 0            | 0         | 0            | 0         | 0            | 260        |

### Key Takeaways

1. **At low concurrency (C=10), agg wins on per-user latency**: 7.98ms TPOT vs 9.67ms, 125 TPS/user vs 103 — and higher TPS/GPU (131 vs 54) since all 8 GPUs serve both prefill+decode
2. **Disagg dominates at scale (C>=100)**: 5.5x higher total throughput at C=500 (10,935 vs 1,996 tok/s) while agg plateaus at ~2,000 tok/s
3. **TTFT is dramatically better for disagg under load**: 996ms vs 9,161ms at C=100 — the dedicated prefill node keeps TTFT low while agg queues everything behind `max-running-requests=24`
4. **TPS/GPU crossover at ~C=50**: Disagg surpasses agg TPS/GPU efficiency between C=10 and C=100
5. **Agg breaks at C=500**: 52% error rate from connection failures — the single-node agg cannot absorb 500 concurrent streams
6. **Disagg TPOT degrades gracefully**: 9.67ms -> 19.28ms (2x) from C=10 to C=500, while agg stays flat at ~11.5ms but only because it queues most requests (huge TTFT)

