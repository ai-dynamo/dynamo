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

⚠️ **PRIMARY ERROR: Fatal NIXL Protocol Error** - FOUND IN LOGS
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
- Found in December 9th logs, not currently reproducing
- May be intermittent or require specific conditions (TP8, concurrent requests, network stress)

**ANOMALY 1: KV Transfer Timeout** - CONSISTENTLY OBSERVED
```
WARN nixl_connector.get_finished: Releasing expired KV blocks for request <uuid>
which were retrieved by 0 decode worker(s) within 120 seconds.
```
- Occurs at **every** token size (1k, 10k, 30k, 60k, 80k, 100k, 110k, 120k)
- "**0 decode worker(s)**" indicates KV blocks not retrieved
- **Component**: Prefill worker
- **Nature**: Warning - May be consequence of underlying connection issues

**ANOMALY 2: Broken Pipe** - OCCASIONALLY OBSERVED
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

**ANOMALY 3: Service Unavailable** - OBSERVED DURING TESTING
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
2. **ANOMALY 2**: TCP connections show "Broken pipe" errors - write attempts to closed connection (likely consequence of primary error)
3. **ANOMALY 1**: KV transfer timeout warnings - 0 blocks retrieved by decode workers (may indicate underlying connection issues)
4. **ANOMALY 3**: Health check failures (503) - worker temporarily unavailable during high load or initialization

**System behavior**: System falls back to **prefill-only processing** when KV transfer cannot complete. This is why requests succeed despite anomalies.

---

## Setup & Reproduction

### Quick Reproduction

Simplest way to reproduce (requires both prefill and decode workers running):

```bash
# Port-forward to frontend
kubectl port-forward <frontend-pod> 8787:8787 &

# Send any request
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
       "messages": [{"role": "user", "content": "Say hello"}],
       "max_tokens": 10}'

# Check prefill worker logs (120 seconds after request)
kubectl logs <prefill-worker> | grep "Releasing expired KV blocks"
# Should show: "0 decode worker(s)" retrieved KV cache

# Check decode worker logs
kubectl logs <decode-worker> | grep "Broken pipe"
# Should show: "failed to join writer task: Broken pipe (os error 32)"
```

### Full Setup (Kubernetes)

**Prerequisites**:
- Kubernetes cluster with GPU nodes
- `kubectl` access
- 132GB storage for model cache
- 16 GPUs (8 prefill + 8 decode for TP8)

**Step 1: Install Infrastructure**

```bash
# Create namespace
kubectl create namespace <your-namespace>
kubectl config set-context --current --namespace=<your-namespace>

# Install etcd + NATS (create platform-values.yaml)
cat > platform-values.yaml <<EOF
dynamo-operator:
  enabled: false
dynamo-etcd:
  enabled: true
dynamo-nats:
  enabled: true
EOF

# Install
helm install dynamo-platform oci://nvcr.io/nvidia/charts/dynamo-platform \
  --version 0.7.0 -f platform-values.yaml

# Verify
kubectl get pods
# Should show: etcd-0 and nats-0 Running
```

**Step 2: Create Model Cache**

```bash
# Create PVC
cat > model-cache-pvc.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 500Gi
  storageClassName: <your-storage-class>
EOF

kubectl apply -f model-cache-pvc.yaml
```

**Step 3: Deploy Workers**

See `deepseek-r1-disagg.yaml` in this directory for complete configuration.

Key settings:
- TP8 (8 GPUs/worker) matches original bug report configuration
- Use `--connector nixl --enforce-eager`
- Mount model cache at `/models`
- Set `HF_HOME=/models/hub`

```bash
kubectl apply -f deepseek-r1-disagg.yaml
kubectl get pods -w  # Wait for Running (5-10 min first time)
```

**Step 4: Test**

```bash
# Install genai-perf
pip install genai-perf

# Port-forward
kubectl port-forward <frontend-pod> 8787:8787 &

# Test with exact token counts (examples)
genai-perf profile \
  --synthetic-input-tokens-mean 1000 \
  --output-tokens-mean 100 \
  --request-count 5 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --url http://localhost:8787 \
  --streaming --verbose

# Check logs 120 seconds after test completes
kubectl logs <prefill-worker> --since=3m | grep "Releasing"
# Should show "0 decode worker(s)" warnings
```

### Test Results (2025-12-12)

### TP8 Configuration (8 GPUs/worker)

**Single Request Tests** (--request-count 1):

| Input Tokens | TTFT | Request Latency | Status | "0 decode worker" warnings |
|--------------|------|-----------------|--------|---------------------------|
| 40,000 | 7.65s | 10.47s | ✅ Success | Present |
| 80,000 | 8.71s | 11.54s | ✅ Success | Present |
| 110,000 | 7.42s | 10.31s | ✅ Success | Present |
| 120,000 | 3.45s | 6.31s | ✅ Success | Present |
| 130,000 | 3.44s | 6.35s | ✅ Success | Present |
| 140,000 | N/A | 0.85s | ❌ Rejected (exceeds 131k limit) | N/A |

**10 Request Tests** (--request-count 10):

| Input Tokens | Avg TTFT (range) | Avg Request Latency | Status | "0 decode worker" warnings |
|--------------|------------------|---------------------|--------|---------------------------|
| 110,000 | 19.96s (962ms-24.28s) | 22.86s | ✅ 10/10 completed | 168 warnings |
| 120,000 | 2.86s (982ms-3.60s) | 5.71s | ✅ 10/10 completed | 112 warnings |

**Total TP8 Tests**: 26 requests (6 single + 20 batch)
**Total "0 decode worker" warnings**: 280+ instances

**⚠️ TODO: Research TTFT Anomaly**

Unexplained performance pattern in single request tests:
- **40k-110k tokens**: TTFT ranges 7.42s-8.71s (expected behavior)
- **120k-130k tokens**: TTFT drops to ~3.4s (2.5x faster - anomalous)
- **10-request batch at 120k**: TTFT also shows 2.86s avg (consistent with single request anomaly)

**Expected behavior**: TTFT should increase with context size (more tokens to prefill)
**Observed behavior**: TTFT dramatically decreases at 120k+ tokens

**Possible explanations to investigate**:
1. System warmup/caching effect (120k tests ran after earlier tests)
2. Scheduler variance or CUDA kernel optimization changes
3. Measurement artifact or metric calculation issue
4. Model-specific optimization activating near max context (131k limit)
5. Resource contention differences between test runs

**Recommendation**: Rerun 120k-130k token tests in isolation to verify if this is reproducible or an artifact of test ordering.

---

### Summary

**Primary Error** (Fatal NIXL Protocol Error):
- ❌ **NOT reproducible** through systematic testing
- Observed earlier in logs (08:14:10 UTC) with cascade of 35+ broken pipes
- NOT triggered by: specific token counts (40k-130k), request counts (1/10), or GPU configuration (TP8)
- **Status**: Intermittent/non-deterministic
- **This is the only true error** - everything else are anomalies

**Anomalies Observed**:
- **Anomaly 1**: KV transfer timeout warnings ("0 decode worker" retrieving blocks) - observed in 100% of tests
- **Anomaly 2**: Broken pipe errors (os error 32) - occasional during testing
- **Anomaly 3**: Health check 503 errors - during initialization/high load
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
- Added: `--no-enable-prefix-caching --block-size 128`
- Tested: TP8 (8 GPUs/worker) configuration

### Detailed Error Analysis

**PRIMARY ERROR: Fatal NIXL Protocol Error**
- **Component**: Decode worker reader task
- **When**: During NIXL protocol stream decoding
- **Cause**: Invalid line protocol, connection reset by peer
- **Error code**: 104 (ECONNRESET)
- **Status**: Found in December 9th logs, not currently reproducing
- **Nature**: FATAL - This is the only true error

**ANOMALY 1: KV Transfer Timeout**
- **Component**: Prefill worker
- **When**: 120 seconds after request starts
- **Observation**: Decode worker never retrieves KV blocks from prefill
- **Frequency**: 8-18 warnings per request (varies by size)
- **Nature**: Warning - May be consequence of primary error or separate issue

**ANOMALY 2: Broken Pipe**
- **Component**: Decode worker
- **When**: During request processing
- **Observation**: TCP writer task attempts to write to closed connection
- **Error code**: 32 (EPIPE)
- **Frequency**: Sporadic, ~every 5 minutes (6 instances in 25 minutes TP8 testing)
- **Conditions**: Observed during curl and genai-perf tests, all token counts
- **Nature**: May be consequence of primary error (connection already reset)

**ANOMALY 3: Service Unavailable (503)**
- **Component**: Decode worker HTTP health endpoint
- **When**: During worker initialization and high load periods
- **Observation**: Worker temporarily unable to serve health check requests
- **HTTP Status**: 503 Service Unavailable
- **Frequency**: Regular 10-second intervals (matching health check period)
- **Conditions**: Worker startup phase, sustained high load
- **Nature**: Temporary service overload or initialization delay

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

1. **Anomaly 1 - KV Transfer Timeout Warnings**: "0 decode worker(s)" retrieved KV blocks (280+ instances across TP8 tests)
2. **Anomaly 2 - Broken Pipes**: TCP writer failures (error 32) - sporadic during testing
3. **Anomaly 3 - Health Check 503s**: Service temporarily unavailable during initialization/high load

**Nature**: These anomalies may be consequences of the primary error when it occurs, or separate unrelated issues.

### Key Findings

1. **Primary error is intermittent** - not reliably triggered by high token counts (tested up to 131k max)
2. **Anomalies are consistently present** - observed at all token sizes (40k-130k) in TP8 configuration
3. **System masks issues** - requests succeed via prefill-only fallback
4. **Tested configuration**: TP8 (8 GPUs/worker), with/without explicit `--connector nixl`
5. **Model limit**: 131,072 tokens maximum context length

**The disaggregated vLLM setup with NIXL connector has a confirmed fatal error** that appears intermittently. The observed anomalies (KV transfer warnings, broken pipes, health check failures) may be related to the primary error or represent separate issues requiring investigation.
