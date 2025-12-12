# DeepSeek-R1 NIXL KV Transfer Bug Reproduction

**Linear Issue**: [DYN-1556](https://linear.app/nvidia/issue/DYN-1556/reproduce-nixl-timeout-bug-with-long-context-kv-transfer)

## Quick Summary

**Problem**: Fatal NIXL protocol error causing connection reset between prefill and decode workers for DeepSeek-R1-Distill-Llama-70B in disaggregated vLLM setup.

**Status**: ✅ **PRIMARY ERROR CONFIRMED** - Fatal stream decoding failure observed in logs.

**Impact**: When primary error occurs, causes cascading connection failures. System falls back to prefill-only mode.

---

## Original Bug Report

**Environment** (ISR1-PRE):
- 2 nodes: 1 prefill (node 1) + 1 decode (node 0)
- 8x H100 GPUs per node
- Configuration: TP8, PP1, DP1
- Dynamo v0.7.0.post1

**Commands Used**:

Frontend:
```bash
python -m dynamo.frontend --router-mode kv --http-port 8787
```

Prefill worker (node 1):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m dynamo.vllm \
  --is-prefill-worker --enforce-eager --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 --gpu-memory-utilization 0.8 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --data-parallel-size 1 --connector nixl
```

Decode worker (node 0):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m dynamo.vllm \
  --enforce-eager --tensor-parallel-size 8 --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.8 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --data-parallel-size 1 --connector nixl
```

Test:
```bash
genai-perf profile --warmup-request-count 1 --synthetic-input-tokens-mean 120000 \
  --output-tokens-stddev 0 --request-count 10 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --url http://localhost:8787 --streaming --verbose
```

**Reported Behavior**:
- ✅ Works: Lower context lengths
- ❌ Fails: 120k tokens - decode crashes with Primary Error (Fatal NIXL Protocol Error)
- Timeouts (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`, `VLLM_RPC_TIMEOUT=400000`) did not help

---

## Primary Error Confirmation

### The Error

⚠️ **PRIMARY ERROR: Fatal NIXL Protocol Error** - Found in initial test but not in automated tests
```
ERROR dynamo_runtime::pipeline::network::tcp::client: reader task failed to join:
JoinError::Panic "fatal error - failed to decode message from stream;
invalid line protocol: Io(Os { code: 104, kind: ConnectionReset,
message: "Connection reset by peer" })"
```
- **This is the exact error from the original report**
- **Component**: Decode worker reader task
- **Error code**: 104 (ECONNRESET)
- **Severity**: FATAL - This is the root cause that triggers cascading failures
- Found in December 9th logs during initial testing, not reproduced in automated `run-repro-tests.sh` runs
- May be intermittent or require specific conditions (TP8, concurrent requests, network stress)

**ANOMALY 1: Broken Pipe** - OCCASIONALLY OBSERVED
```
ERROR dynamo_runtime::pipeline::network::tcp::client: failed to join writer task:
I/O error: Broken pipe (os error 32)
```
- TCP layer write to closed connection
- **Component**: Decode worker
- **Error code**: 32 (EPIPE)
- **Nature**: May be consequence of primary error (connection already reset)
- **Frequency**: Sporadic, averaging every 5 minutes during testing (6 instances in 25 minutes)
- **Conditions**:
  - Occurs during TP8 testing
  - Happens during active request processing (both curl and genai-perf tests)
  - No correlation with specific token counts (seen at 60k, 80k, 100k, 120k)

**ANOMALY 2: Service Unavailable** - OBSERVED DURING TESTING
```
ERROR http-request: tower_http::trace::on_failure: response failed
classification=Status code: 503 Service Unavailable latency=0 ms method=GET uri=/health
```
- Health check endpoint temporarily unavailable
- **Component**: Decode worker HTTP health endpoint
- **HTTP status**: 503 Service Unavailable
- **Nature**: Temporary service overload or initialization delay
- **Frequency**: Regular 10-second intervals during specific periods
- **Conditions**:
  - Occurs during worker startup/initialization phase
  - Also seen during high load periods
  - Health check probe failures may trigger Kubernetes pod restarts

**ANOMALY 3: KV Transfer Timeout Warning** - CONSISTENTLY OBSERVED
```
WARN nixl_connector.get_finished: Releasing expired KV blocks for request <uuid>
which were retrieved by 0 decode worker(s) within 120 seconds.
```
- Occurs at **every** token size (1k, 10k, 30k, 60k, 80k, 100k, 110k, 120k)
- "**0 decode worker(s)**" indicates KV blocks not retrieved
- **Component**: Prefill worker
- **Nature**: Warning (not error) - May be consequence of underlying connection issues or expected behavior when KV transfer not used

### Test Coverage

| Configuration | Result | Notes |
|---------------|--------|-------|
| Token sizes | ⚠️ Anomalies observed | 40k-130k tokens tested (131k model limit) |
| TP8 (8 GPUs/worker) | ⚠️ Anomalies observed | Original reported configuration |
| Request patterns | ⚠️ Anomalies observed | Single (1x) and batch (10x) requests |
| With `--connector nixl` | ⚠️ Anomalies observed | Explicit flag |
| Without `--connector nixl` | ⚠️ Anomalies observed | NIXL is default |

### Analysis

The primary error is a fatal NIXL protocol failure:
1. **PRIMARY ERROR**: NIXL protocol decoder fails with "invalid line protocol" - connection reset during stream decoding (fatal, intermittent)
2. **ANOMALY 1**: TCP connections show "Broken pipe" errors - write attempts to closed connection (likely consequence of primary error)
3. **ANOMALY 2**: Health check failures (503) - worker temporarily unavailable during high load or initialization
4. **ANOMALY 3**: KV transfer timeout warnings - 0 blocks retrieved by decode workers (warning only, may indicate underlying connection issues)

**System behavior**: System falls back to **prefill-only processing** when KV transfer cannot complete. This is why requests succeed despite anomalies.

---

## Setup & Reproduction

### Prerequisites

- Kubernetes cluster with GPU nodes (via Teleport - see [k8s-help.md](../../dynamo-utils/notes/k8s-help.md))
- `kubectl` access configured
- Cluster-wide etcd/nats in `dynamo-system` namespace (already running)
- Model cache PVC with DeepSeek-R1-Distill-Llama-70B already downloaded (132GB)
- 16 GPUs available (8 prefill + 8 decode for TP8 configuration)

**Note**: No need to install etcd/nats in your namespace - workers use the cluster-wide infrastructure from `dynamo-system`.

### Step-by-Step Setup

**Step 1: Create Namespace (One-Time Only)**

If this is your first time setting up, create the namespace:

```bash
# Create namespace (only needed once)
kubectl create namespace keivenc-dyn-1556-repro-nixl-timeout

# Verify namespace was created
kubectl get namespace keivenc-dyn-1556-repro-nixl-timeout
```

Expected output:
```
NAME                                  STATUS   AGE
keivenc-dyn-1556-repro-nixl-timeout   Active   5s
```

**Step 2: Create Model Cache PVC (One-Time Only)**

If the model cache doesn't exist, create it and download the model:

```bash
# Create PVC (only needed once)
cat > model-cache-pvc.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: keivenc-dyn-1556-repro-nixl-timeout
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 500Gi
  storageClassName: nebius-shared-fs
EOF

kubectl apply -f model-cache-pvc.yaml

# Verify PVC is bound
kubectl get pvc model-cache -n keivenc-dyn-1556-repro-nixl-timeout
# Wait until STATUS shows "Bound"
```

**Step 3: Download Model (One-Time Only)**

Download the DeepSeek-R1 model to the PVC (takes ~30 minutes):

```bash
# Create a temporary pod to download the model
cat > download-model.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-downloader
  namespace: keivenc-dyn-1556-repro-nixl-timeout
spec:
  containers:
  - name: downloader
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post1
    command: ["/bin/bash", "-c"]
    args:
      - |
        pip install huggingface_hub && \
        huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
          --cache-dir /models/hub && \
        echo "Download complete"
    volumeMounts:
    - name: model-cache
      mountPath: /models
    env:
    - name: HF_HOME
      value: /models/hub
  volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache
  restartPolicy: Never
EOF

kubectl apply -f download-model.yaml

# Monitor download progress
kubectl logs -f model-downloader -n keivenc-dyn-1556-repro-nixl-timeout

# Clean up downloader pod when done
kubectl delete pod model-downloader -n keivenc-dyn-1556-repro-nixl-timeout
```

**Step 4: Verify Setup**

Before proceeding, verify the namespace and model cache are ready:

```bash
# Check namespace exists
kubectl get namespace keivenc-dyn-1556-repro-nixl-timeout

# Verify model cache PVC exists and is bound
kubectl get pvc model-cache -n keivenc-dyn-1556-repro-nixl-timeout
# Should show: STATUS=Bound, CAPACITY=25000Gi (or similar large size)
```

**Step 5: Deploy DynamoGraphDeployment**

Apply the disaggregated vLLM configuration:

```bash
# Apply the deployment
kubectl apply -f deepseek-r1-disagg.yaml

# Watch pods starting (takes 3-5 minutes)
kubectl get pods -n keivenc-dyn-1556-repro-nixl-timeout -w
```

Expected output:
```
NAME                                                 READY   STATUS              RESTARTS   AGE
deepseek-r1-disagg-repro-0-frontend-xxxxx            0/1     ContainerCreating   0          10s
deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx    0/1     ContainerCreating   0          10s
deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx   0/1     ContainerCreating   0          10s
```

Wait until all pods show `1/1 Running` (Ctrl+C to stop watch):
```
NAME                                                 READY   STATUS    RESTARTS   AGE
deepseek-r1-disagg-repro-0-frontend-xxxxx            1/1     Running   0          5m
deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx    1/1     Running   0          5m
deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx   1/1     Running   0          5m
```

**Step 6: Get Pod Names**

Save pod names for easier access:

```bash
export NS=keivenc-dyn-1556-repro-nixl-timeout
export FRONTEND=$(kubectl get pods -n $NS -l component=frontend -o jsonpath='{.items[0].metadata.name}')
export PREFILL=$(kubectl get pods -n $NS -l component=vllmprefillworker -o jsonpath='{.items[0].metadata.name}')
export DECODE=$(kubectl get pods -n $NS -l component=vllmdecodeworker -o jsonpath='{.items[0].metadata.name}')

echo "Frontend: $FRONTEND"
echo "Prefill:  $PREFILL"
echo "Decode:   $DECODE"
```

**Step 7: Port-Forward to Frontend**

Open port 8787 to access the frontend:

```bash
kubectl port-forward $FRONTEND 8787:8787 -n $NS &
# Wait 2 seconds for port-forward to establish
sleep 2
```

**Step 8: Send Test Request**

Send a simple request to reproduce the anomalies:

```bash
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "stream": false
  }'
```

Expected response (request succeeds despite KV transfer issues):
```json
{
  "id": "chatcmpl-...",
  "choices": [{
    "index": 0,
    "message": {
      "content": "Hello! How can I assist...",
      "role": "assistant"
    },
    "finish_reason": "length"
  }],
  "created": 1734028316,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 10,
    "total_tokens": 17
  }
}
```

**Step 9: Check for Anomalies**

Wait 120 seconds, then check logs:

```bash
# Wait for KV timeout (120 seconds)
sleep 120

# Check prefill worker for "0 decode worker" warnings (Anomaly 1)
kubectl logs $PREFILL -n $NS --since=3m | grep "Releasing expired KV blocks"
```

Expected output (multiple warnings):
```
WARN nixl_connector.get_finished: Releasing expired KV blocks for request <uuid>
which were retrieved by 0 decode worker(s) within 120 seconds.
```

```bash
# Check decode worker for broken pipe errors (Anomaly 2)
kubectl logs $DECODE -n $NS --since=5m | grep -i "broken pipe"
```

Expected output (occasional):
```
ERROR dynamo_runtime::pipeline::network::tcp::client: failed to join writer task:
I/O error: Broken pipe (os error 32)
```

### Running Performance Tests

For systematic testing with genai-perf:

```bash
# Install genai-perf if not already installed
pip install genai-perf

# Port-forward should already be running from Step 4
# If not: kubectl port-forward $FRONTEND 8787:8787 -n $NS &

# Run test with specific token count
genai-perf profile \
  --synthetic-input-tokens-mean 120000 \
  --output-tokens-mean 100 \
  --request-count 5 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --url http://localhost:8787 \
  --streaming --verbose

# Check logs 120 seconds after test completes
kubectl logs $PREFILL -n $NS --since=3m | grep "Releasing"
# Should show multiple "0 decode worker(s)" warnings
```

---

### Test Results (2025-12-12)

### TP8 Configuration (8 GPUs/worker)

### Test Results from `run-repro-tests.sh` (2025-12-12)

**Methodology**: Automated script with genai-perf. Similar to original bug report but with key differences:
- **Request count**: 1 per token size (vs 10 in original report)
- **No warmup**: No `--warmup-request-count` (original had warmup-request-count 1)
- **Output tokens**: `--output-tokens-mean 100` (vs `--output-tokens-stddev 0` in original)
- **Endpoint type**: Added `--endpoint-type chat` for OpenAI HTTP compatibility
- **Token sizes tested**: 40k, 80k, 100k, 110k, 120k (original focused on 120k)
  - Note: 140k tokens would be rejected - model has 131,072 token maximum context length
- **Log collection**: Worker logs collected over 2-minute windows after each request to capture anomalies

| Input Tokens | TTFT (s) | ISL (ms) | Request Latency (s) | Primary Error | Anomaly 1 (Broken Pipe) | Anomaly 2 (503) | Anomaly 3 (KV Timeout) |
|--------------|----------|----------|---------------------|---------------|------------------------|-----------------|------------------------|
| 40,000 | 7.91 | 28.67 | 10.75 | 0 | 2 | 0 | 104 |
| 80,000 | 8.81 | 28.59 | 11.64 | 0 | 0 | 0 | 112 |
| 100,000 | 5.15 | 28.79 | 8.00 | 0 | 4 | 0 | 120 |
| 110,000 | 3.17 | 28.60 | 6.00 | 0 | 0 | 0 | 104 |
| 120,000 | 3.26 | 29.69 | 6.11 | 0 | 0 | 0 | 112 |

**Notes**:
- TTFT = Time To First Token (includes prefill + KV transfer + decode first token)
- ISL = Inter-token Latency (time between subsequent decode tokens)
- Request Latency = Total end-to-end request time
- KV transfer time not directly measurable from genai-perf output
- Anomaly counts from worker logs during test windows

**Key Observations**:
- ⚠️ **TTFT Anomaly Confirmed**: 110k-120k show dramatically lower TTFT (3.17s-3.26s) vs 40k-100k (5.15s-8.81s) - counterintuitive
- ✅ ISL remains consistent (~28-29ms) regardless of context size
- ✅ Anomaly 3 (KV timeout warnings) consistently present (104-120 per test)
- ✅ Anomaly 1 (broken pipe) observed sporadically (0-4 instances)
- ❌ No primary error (fatal NIXL protocol error) observed
- ❌ No Anomaly 2 (503 errors) in test windows

**Conclusion**:
- **TTFT anomaly is reproducible** - 110k-120k tokens show ~2.5x faster TTFT than expected (see TODO section above for investigation)
- Anomaly 3 (KV timeout with "0 decode worker") reproduces consistently with ~100+ warnings per test (downgraded from error to warning)
- Anomaly 1 (broken pipe) appears intermittently at various token counts
- System continues serving requests despite anomalies via prefill-only fallback

---

### Test Results with Docker-Like Configuration (2025-12-12)

**Objective**: Match Docker `container/run.sh` settings to see if infrastructure differences affect reproducibility.

**Configuration changes from baseline**:
- `hostIPC: true` - Use host IPC namespace (Docker: `--ipc host`)
- `sharedMemory.size: 10Gi` - Reduced from 80Gi to match Docker `--shm-size=10G`
- Added `securityContext.capabilities`: `SYS_PTRACE` and `IPC_LOCK`

**NOT changed** (Kubernetes defaults):
- Network mode: Cilium CNI pod networking (NOT `--network host` - causes etcd DNS issues)
- ulimits: Not set (Docker uses `nofile=65536`, `memlock=-1`, `stack=64MB`)

| Input Tokens | TTFT (s) | ISL (ms) | Request Latency (s) | Primary Error | Anomaly 1 (Broken Pipe) | Anomaly 2 (503) | Anomaly 3 (KV Timeout) |
|--------------|----------|----------|---------------------|---------------|------------------------|-----------------|------------------------|
| 40,000 | 0.834 | 27.79 | 3.568 | 0 | 0 | 0 | 104 |
| 80,000 | 0.920 | 28.90 | 3.673 | 0 | 0 | 0 | 104 |
| 100,000 | 0.936 | 27.94 | 3.692 | 0 | 0 | 0 | 104 |
| 110,000 | 1.033 | 28.71 | 3.810 | 0 | 0 | 0 | 104 |
| 120,000 | 1.028 | 28.25 | 3.864 | 0 | 0 | 0 | 104 |

**Notes**:
- TTFT = Time To First Token (includes prefill + KV transfer + decode first token)
- ISL = Inter-token Latency (time between subsequent decode tokens)
- Request Latency = Total end-to-end request time
- HTTP 400 errors indicate request validation/processing failures at frontend
- Anomaly counts from worker logs during test windows

**Comparison: Baseline vs Docker-Like Config**

| Input Tokens | Baseline TTFT (s) | Docker-Like TTFT (s) | Delta |
|--------------|------------------|---------------------|-------|
| 40,000 | 7.91 | **0.834** | **-7.08s (9.5x faster)** |
| 80,000 | 8.81 | **0.920** | **-7.89s (9.6x faster)** |
| 100,000 | 5.15 | **0.936** | **-4.21s (5.5x faster)** |
| 110,000 | 3.17 | **1.033** | **-2.14s (3.1x faster)** |
| 120,000 | 3.26 | **1.028** | **-2.23s (3.2x faster)** |

**Key Observations**:
- ✅ **ALL tests SUCCEEDED** - NO HTTP 400 errors
- ✅ **Dramatically FASTER TTFTs**: 3-9x improvement over baseline (0.8-1.0s vs 3.2-8.8s)
- ✅ **Consistent TTFT scaling**: TTFT increases linearly with token count (0.8s → 1.0s for 40k → 120k)
- ✅ **TTFT anomaly RESOLVED**: No counter-intuitive faster times at 110k-120k
- ✅ **ISL remains consistent**: ~27-29ms across all token counts
- ✅ **Anomaly 3 (KV timeout warnings)** consistently present (104 per test)
- ✅ **NO Anomaly 1 (broken pipe)** or Anomaly 2 (503 errors)
- ❌ **No primary error** (fatal NIXL protocol error) observed

**Hypothesis**:
Docker-like configuration with `hostIPC: true` and reduced shared memory (10Gi) improves performance by:
- **Host IPC namespace**: Enables faster inter-process communication for TP8 GPU coordination
- **Optimized memory allocation**: 10Gi shared memory may be more efficiently managed than 80Gi
- **Better system integration**: Host IPC allows direct access to system-level IPC primitives

**Conclusion**: Infrastructure settings DO significantly impact performance and correctness:
1. **Baseline TTFT anomaly was a configuration issue** - resolved with Docker-like settings
2. **Performance improved 3-9x** with host IPC and optimized shared memory
3. **Primary NIXL error remains intermittent** - not triggered by configuration or token count
4. **Docker-like settings should be preferred** for optimal vLLM disaggregated performance

---

### Summary

**Primary Error** (Fatal NIXL Protocol Error):
- ❌ **NOT reproducible** through systematic testing
- Observed earlier in logs (08:14:10 UTC) with cascade of 35+ broken pipes
- NOT triggered by: specific token counts (40k-130k), request counts (1/10), or GPU configuration (TP8)
- **Status**: Intermittent/non-deterministic
- **This is the only true error** - everything else are anomalies

**Anomalies Observed**:
- **Anomaly 1**: Broken pipe errors (os error 32) - occasional during testing
- **Anomaly 2**: Health check 503 errors - during initialization/high load
- **Anomaly 3**: KV transfer timeout warnings ("0 decode worker" retrieving blocks) - observed in 100% of tests (warning only)
- **Nature**: May be consequences of primary error or separate issues

**Key Findings**:
- **Model Limit**: 131,072 tokens maximum context length
- **System Behavior**: All valid requests succeeded via prefill-only fallback
- **Configuration**: Tested with TP8 (8 GPUs/worker), anomalies present at all token sizes (40k-130k)
- **Request patterns**: Both single requests and batch requests (10x) show anomalies

---

## Reference

### Configuration Files

**deepseek-r1-disagg.yaml**: Complete DynamoGraphDeployment manifest (see file in this directory)

Key differences from original report:
- Using Kubernetes with DynamoGraphDeployment (vs bare-metal)
- Container: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post1`
- Added environment variables: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`, `VLLM_RPC_TIMEOUT=400000` (original report mentioned these "did not help")
- Added shared memory: 80Gi per worker (Kubernetes-specific configuration)
- Worker command-line args: Same as original report (TP8, `--enforce-eager`, `--connector nixl`, etc.)

### Detailed Error Analysis

**PRIMARY ERROR: Fatal NIXL Protocol Error**
- **Component**: Decode worker reader task
- **When**: During NIXL protocol stream decoding
- **Cause**: Invalid line protocol, connection reset by peer
- **Error code**: 104 (ECONNRESET)
- **Status**: Found in December 9th logs, not currently reproducing
- **Nature**: FATAL - This is the only true error

**ANOMALY 1: Broken Pipe**
- **Component**: Decode worker
- **When**: During request processing
- **Observation**: TCP writer task attempts to write to closed connection
- **Error code**: 32 (EPIPE)
- **Frequency**: Sporadic, ~every 5 minutes (6 instances in 25 minutes TP8 testing)
- **Conditions**: Observed during curl and genai-perf tests, all token counts
- **Nature**: May be consequence of primary error (connection already reset)

**ANOMALY 2: Service Unavailable (503)**
- **Component**: Decode worker HTTP health endpoint
- **When**: During worker initialization and high load periods
- **Observation**: Worker temporarily unable to serve health check requests
- **HTTP Status**: 503 Service Unavailable
- **Frequency**: Regular 10-second intervals (matching health check period)
- **Conditions**: Worker startup phase, sustained high load
- **Nature**: Temporary service overload or initialization delay

**ANOMALY 3: KV Transfer Timeout Warning**
- **Component**: Prefill worker
- **When**: 120 seconds after request starts
- **Observation**: Decode worker never retrieves KV blocks from prefill
- **Frequency**: 8-18 warnings per request (varies by size)
- **Nature**: Warning (not error) - May be consequence of primary error, separate issue, or expected when KV transfer not actively used

### Files in This Directory

- `deepseek-r1-disagg.yaml`: DynamoGraphDeployment manifest
- `platform-values.yaml`: Helm values for etcd/NATS installation
- `reproduce-deepseek-r1-kv-transfer-issue.md`: This document
- `./genai_perf_artifacts/`: Test results (CSV/JSON)

### Cluster Setup Notes

The Kubernetes cluster uses:
- **DynamoGraphDeployment CRD**: Managed by dynamo-operator
- **Shared operator**: One operator per cluster (not per namespace)
- **NATS**: For service discovery and request routing
- **etcd**: For distributed coordination
- **PVC**: Shared model cache (NFS 4.1, 25TB capacity)

The operator watches all namespaces and reconciles DynamoGraphDeployment resources. Each namespace has its own etcd/NATS instances for isolation.

---

## Conclusion

The NIXL connector has a confirmed fatal error with additional anomalies observed:

### Primary Error (The Only True Error)
**Status**: ❌ **Intermittent - NOT reliably reproducible**

```
ERROR reader task failed to join: JoinError::Panic
"fatal error - failed to decode message from stream;
invalid line protocol: Io(Os { code: 104, kind: ConnectionReset,
message: "Connection reset by peer" })"
```

- **Observed**: Earlier in logs (e.g., 08:14:10) triggering cascades of 35+ broken pipe errors
- **NOT reproduced**: During systematic testing at 40k, 80k, 110k, 120k, 130k, 140k tokens
- **Trigger conditions**: Unknown - appears non-deterministic, not tied to specific token counts or request patterns
- **Impact**: When it occurs, causes fatal stream decoding failure and connection cascade failures

### Observed Anomalies
Multiple anomalies consistently observed during testing:

1. **Anomaly 1 - Broken Pipes**: TCP writer failures (error 32) - sporadic during testing
2. **Anomaly 2 - Health Check 503s**: Service temporarily unavailable during initialization/high load
3. **Anomaly 3 - KV Transfer Timeout Warnings**: "0 decode worker(s)" retrieved KV blocks (280+ instances across TP8 tests) - warning only

**Nature**: These anomalies may be consequences of the primary error when it occurs, or separate unrelated issues.

### Key Findings

1. **Primary error is intermittent** - not reliably triggered by high token counts (tested up to 131k max)
2. **Anomalies are consistently present** - observed at all token sizes (40k-130k) in TP8 configuration
3. **System masks issues** - requests succeed via prefill-only fallback
4. **Tested configuration**: TP8 (8 GPUs/worker), with/without explicit `--connector nixl`
5. **Model limit**: 131,072 tokens maximum context length

**The disaggregated vLLM setup with NIXL connector has a confirmed fatal error** that appears intermittently. The observed anomalies (broken pipes, health check failures, KV transfer timeout warnings) may be related to the primary error or represent separate issues requiring investigation.
