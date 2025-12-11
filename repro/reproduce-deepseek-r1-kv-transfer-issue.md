# DeepSeek-R1 Long-Context KV Transfer Issue

## Table of Contents
1. [Issue Summary](#issue-summary)
2. [Original Environment](#original-environment)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Reproduction Steps](#reproduction-steps)
6. [Troubleshooting](#troubleshooting)
7. [Debugging](#debugging)
8. [Reference](#reference)

---

## Issue Summary

**Problem**: Dynamo vLLM disaggregated setup with DeepSeek-R1-Distill-Llama-70B fails for long-context requests (120k tokens) during KV cache transfer between prefill and decode workers.

**Error**: `Connection reset by peer` / `fatal error - failed to decode message from stream; invalid line protocol: Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })`

**Behavior**:
- ✅ **Works**: Lower context lengths (e.g., 1k tokens)
- ❌ **Fails**: Long context (120k synthetic input tokens)

**Status** (as of December 11, 2025):
- ✅ Infrastructure setup complete (etcd, NATS, model cache PVC)
- ✅ Model verified as publicly accessible and cached (132GB downloaded to shared PVC)
- ✅ Model loading works correctly (8 seconds load time per GPU, cached from NFS)
- ❌ **BLOCKER**: Disaggregated vLLM setup unstable with vllm-runtime:0.7.0.post1
  - **Tried**: Example pattern with `subComponentType` + `--is-decode-worker` → workers crash loop
  - **Tried**: Recipe pattern (llama-3-70b) without `subComponentType`, no `--is-decode-worker` → workers still unstable (restart loop)
  - **Root cause**: Workers register as `component=backend` regardless of configuration
  - Health checks fail: "instance_id not found for endpoint .../backend/generate"
  - Pods initialize successfully but crash ~60 seconds later due to failed health checks
- ❌ **Bug reproduction blocked** - cannot test KV transfer with unstable workers

**Next Steps to Unblock**:
1. **Try aggregated mode** (single combined worker) - won't test disagg-specific issues but may still show problems
2. **Test with newer runtime version** if available (check for vllm-runtime:0.7.1 or later)
3. **Report runtime bug** to Dynamo team - disaggregated vLLM worker registration is broken
4. **Alternative**: Test with TRTLLM disaggregated setup (subComponentType works correctly for TRTLLM)

---

## Original Environment

- **Environment**: ISR1-PRE
- **Nodes**: 2 nodes (1 prefill, 1 decode)
- **GPUs**: 8x H100 per node
- **Configuration**: TP8, PP1, DP1
- **Dynamo Tag**: v0.7.0.post1
- **Container**: 39428499-vllm-amd64
- **Model**: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

### Component Commands (Original Setup)

**Frontend**:
```bash
python -m dynamo.frontend --router-mode kv --http-port 8787
```

**Decode worker**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m dynamo.vllm \
  --enforce-eager \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --data-parallel-size 1 \
  --connector nixl \
  > /cloudai_run_results/dynamo_decode_0_0.log 2>&1
```

**Prefill worker**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m dynamo.vllm \
  --is-prefill-worker \
  --enforce-eager \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --data-parallel-size 1 \
  --connector nixl \
  > /cloudai_run_results/dynamo_prefill_1_0.log 2>&1
```

### Environment Variables Tested

```bash
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
VLLM_RPC_TIMEOUT=400000
```

**Note**: Increasing these timeouts did NOT resolve the issue.

---

## Prerequisites

### Required Access
- Teleport access with Kubernetes permissions
- Access to one of these GPU clusters:
  - **dynamo-nebius-1** (Recommended) - 16 nodes × 8 H200 GPUs
  - **dynamo-aks-dev** - 8 nodes × 8 A100 80GB GPUs

### Required Software
- `kubectl` - Kubernetes CLI
- `helm` - Kubernetes package manager
- `tsh` - Teleport client

### Model Information
- **Model**: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- **Size**: ~140GB (17 safetensors files)
- **Authentication**: None required (publicly accessible)
- **Verification**:
  ```bash
  curl -I "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/config.json"
  # Returns: HTTP 200 (no auth needed)
  ```

### Known Issues

**HuggingFace Rate Limiting**: The cluster's shared egress IP is rate limited by HuggingFace (5000 requests per 5 minutes). Without a shared model cache, pods will fail with `429 Too Many Requests`. Solution is provided in setup steps below.

---

## Setup Instructions

### Step 1: Connect to Cluster

```bash
# Log into the cluster with GPUs
tsh kube login dynamo-nebius-1

# Verify connection
kubectl config current-context
# Should show: nv-prd-dgxc.teleport.sh-dynamo-nebius-1
```

### Step 2: Create Namespace

```bash
# Create namespace for this ticket
kubectl create namespace keivenc-dyn-1556-repro-nixl-timeout

# Set as default namespace
kubectl config set-context --current --namespace=keivenc-dyn-1556-repro-nixl-timeout
```

**Note**: Kubernetes namespace names must be lowercase with hyphens (not underscores or slashes).

### Step 3: Install Infrastructure (etcd and NATS)

DynamoGraphDeployment requires etcd (key-value store) and NATS (messaging system).

```bash
# 1. Install Helm if not available
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 2. Navigate to dynamo2 directory
cd /path/to/dynamo2

# 3. Build Helm chart dependencies
cd deploy/cloud/helm/platform && helm dependency build && cd -

# 4. Create custom values file (disable operator since cluster-wide operator exists)
cat > platform-values.yaml <<'EOF'
# Install only etcd and NATS (operator already exists cluster-wide)
dynamo-operator:
  enabled: false

etcd:
  enabled: true
  replicaCount: 1
  persistence:
    enabled: true
    size: 1Gi
  auth:
    rbac:
      create: false

nats:
  enabled: true
  config:
    cluster:
      enabled: false
    jetstream:
      enabled: true
      fileStore:
        enabled: true
        pvc:
          enabled: true
          size: 10Gi
EOF

# 5. Install platform (etcd + NATS only)
helm install dynamo-platform ./deploy/cloud/helm/platform \
  --namespace keivenc-dyn-1556-repro-nixl-timeout \
  --values platform-values.yaml \
  --wait --timeout 10m

# 6. Verify etcd and NATS are running
kubectl get pods
# Should see:
#   dynamo-platform-etcd-0    1/1     Running
#   dynamo-platform-nats-0    2/2     Running
```

### Step 4: Create Shared Model Cache PVC

To avoid HuggingFace rate limiting, create a shared 25TB PVC for model caching:

```bash
# Create PVC configuration file
cat > model-cache-pvc.yaml <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: keivenc-dyn-1556-repro-nixl-timeout
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nebius-shared-fs
  resources:
    requests:
      storage: 25000Gi
EOF

# Create the PVC
kubectl apply -f model-cache-pvc.yaml

# Verify PVC is bound
kubectl get pvc model-cache
# Should show: STATUS = Bound
```

**Why this is needed**: The cluster's shared IP is rate limited by HuggingFace. Once one pod downloads the model to the PVC, all subsequent deployments use the cached copy (no downloads, instant startup).

### Step 5: Verify Configuration Files

The configuration file `deepseek-r1-disagg.yaml` should already exist in the `dynamo2/` directory with:

- **Frontend**: Routes requests (router-mode=kv, port 8787)
- **VllmDecodeWorker**: 8 GPUs, receives KV cache, generates tokens
- **VllmPrefillWorker**: 8 GPUs, processes input, sends KV cache to decode
- **Image**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post1`
- **Configuration**: TP8, PP1, DP1, NIXL connector
- **Model Cache**: Mounted at `/models`, `HF_HOME=/models/hub`

**Verify the container image** (optional):
```bash
docker manifest inspect nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post1
# Should show manifest details
```

**Key Configuration** (both workers must have):
```yaml
envs:
  - name: HF_HOME
    value: "/models/hub"
volumeMounts:
  - mountPoint: /models
    name: model-cache
    useAsCompilationCache: false
```

---

## Reproduction Steps

### Step 1: Deploy

```bash
# Apply the configuration
kubectl apply -f deepseek-r1-disagg.yaml

# Watch pods starting up
kubectl get pods -w
```

**Expected pods** (press Ctrl+C to stop watching):
- `deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx`
- `deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx`
- `deepseek-r1-disagg-repro-0-frontend-xxxxx`

Wait until all show `Running` status (5-10 minutes for model download on first run, instant on subsequent runs if model is cached).

### Step 2: Verify Model is Loaded

```bash
# List your pods
kubectl get pods

# Check decode worker (replace xxxxx with actual hash)
kubectl logs deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx | grep "added model"

# Check prefill worker
kubectl logs deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx | grep "added model"

# Should see: "added model model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
```

### Step 3: Test with Short Context (Baseline - Should Work)

```bash
# Forward port to access frontend from your machine
kubectl port-forward deepseek-r1-disagg-repro-0-frontend-xxxxx 8787:8787 &

# Test with ~1k tokens (should succeed)
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role": "user", "content": "'"$(python3 -c "print('test ' * 250)")"'"}],
    "max_tokens": 100,
    "stream": true
  }'
```

**Expected**: Request completes successfully.

### Step 4: Test with Long Context (Reproduce Bug - Should Fail)

```bash
# Generate ~120k token prompt
python3 -c "print('test ' * 30000)" > long_prompt.txt

# This should trigger the KV cache transfer failure
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role": "user", "content": "'"$(cat long_prompt.txt)"'"}],
    "max_tokens": 100,
    "stream": true
  }'
```

### Step 5: Observe the Failure

In another terminal, watch decode worker logs:

```bash
kubectl logs -f deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx
```

**Expected error in logs**:
```
reader task failed to join
fatal error - failed to decode message from stream; invalid line protocol: Io(Os { code: 104, kind: ConnectionReset, message: "Connection reset by peer" })
```

**Timing**: During or immediately after KV cache transfer from prefill to decode for 120k token input

**Failure Pattern**: Consistently fails at long context (120k tokens), works at short context

### Step 6: Collect Debug Data

Save logs from all components:

```bash
# Decode worker logs
kubectl logs deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx > decode-worker.log

# Prefill worker logs
kubectl logs deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx > prefill-worker.log

# Frontend logs
kubectl logs deepseek-r1-disagg-repro-0-frontend-xxxxx > frontend.log

# Pod details
kubectl describe pod deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx > decode-worker-describe.txt
kubectl describe pod deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx > prefill-worker-describe.txt
```

---

## Troubleshooting

### Model Download Failures (429 Rate Limiting)

**Symptom**:
```
Exception: Failed to download file 'model-00005-of-000017.safetensors' from model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': request error: HTTP status client error (429 Too Many Requests)
```

**Root Cause**: Cluster's shared egress IP hits HuggingFace rate limit (5000 requests per 5 minutes)

**Solution**: Already implemented in Step 4 (Shared Model Cache PVC). If still seeing this error:

1. Check PVC exists and is bound:
   ```bash
   kubectl get pvc model-cache
   ```

2. Verify volumeMounts in YAML:
   ```bash
   kubectl get dynamographdeployment deepseek-r1-disagg-repro -o yaml | grep -A 5 volumeMounts
   ```

3. Check HF_HOME environment variable:
   ```bash
   kubectl get pod <pod-name> -o yaml | grep HF_HOME
   # Should show: HF_HOME=/models/hub
   ```

4. Check if model is already cached:
   ```bash
   # Exec into any worker pod
   kubectl exec -it deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx -- /bin/bash

   # Inside pod, check cache
   ls -lh /models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/
   ```

### Pods Crash on Startup

**Symptom**: Pods enter `CrashLoopBackOff` with connection errors

**Common Causes**:

1. **Missing etcd/NATS**: Verify infrastructure pods are running:
   ```bash
   kubectl get pods | grep dynamo-platform
   # Should show etcd-0 and nats-0 in Running state
   ```

2. **Wrong namespace**: Ensure namespace is correct in YAML:
   ```yaml
   metadata:
     namespace: keivenc-dyn-1556-repro-nixl-timeout
   ```

3. **Image pull issues**: Check pod events:
   ```bash
   kubectl describe pod <pod-name> | grep -A 10 Events
   ```

### Component Registration Failure (Known Issue - vllm-runtime:0.7.0.post1)

**Symptom**: Workers crash loop with health check failures:
```
ERROR: Health check request failed for generate: instance_id=... not found for endpoint
"v1/instances/.../backend/generate"
```

**Root Cause**: The runtime doesn't respect `subComponentType: decode` for service discovery registration. Workers always register as `component=backend` instead of `component=decode` or `component=prefill`.

**Diagnosis**:
```bash
# Check worker logs for registration
kubectl logs <decode-worker-pod> | grep "Registering NATS endpoint"
# Shows: component=backend (WRONG - should be component=decode)

# Check health check errors
kubectl logs <decode-worker-pod> | grep "Health check request failed"
# Shows: looking for /backend/generate but should be /decode/generate

# Verify --is-decode-worker flag is present
kubectl describe pod <decode-worker-pod> | grep -A 15 "Args:"
# Should show: --is-decode-worker (flag is present but ignored by runtime)

# Check subComponentType in deployment
kubectl get dynamographdeployment <name> -o yaml | grep -A 3 "subComponentType:"
# Shows: subComponentType: decode (correctly set in YAML)
```

**Impact**: Disaggregated prefill/decode setup cannot start - workers crash loop after ~60 seconds

**Configuration Evolution** (what was tried):

1. **Attempt 1: Example Pattern**
   - Added `subComponentType: decode` and `subComponentType: prefill`
   - Added `--is-decode-worker` flag to decode worker args
   - Added `--is-prefill-worker` flag to prefill worker args (was already present)
   - **Result**: Workers crash loop, register as `component=backend`

2. **Attempt 2: Recipe Pattern**
   - Compared with `recipes/llama-3-70b/vllm/disagg-multi-node/deploy.yaml`
   - Found recipe does NOT use `subComponentType` at all
   - Removed `subComponentType` from both workers
   - Removed `--is-decode-worker` flag (decode worker has NO special flag in recipe)
   - Kept `--is-prefill-worker` flag on prefill worker only
   - **Result**: Workers still crash loop, still register as `component=backend`

**Key Discovery**: Neither configuration pattern works with vllm-runtime:0.7.0.post1. The runtime ignores both:
- The `subComponentType` field in DynamoGraphDeployment spec
- The `--is-decode-worker` command-line flag

The runtime's NATS registration logic always uses `component=backend` regardless of vLLM's internal disaggregated mode configuration.

**Comparison with TRTLLM**: The `subComponentType` field works correctly for TRTLLM deployments, indicating this is specific to the vLLM runtime integration.

**Workarounds**:
1. **Use aggregated mode** (single worker, no prefill/decode separation) - won't reproduce disagg-specific KV transfer issues
2. **Wait for runtime fix** - requires update to vllm-runtime image
3. **Patch operator** - modify operator to pass subComponentType as environment variable that runtime can use
4. **Try TRTLLM** - disaggregated setup works correctly with TRTLLM runtime

**Status**: This is a known limitation in vllm-runtime:0.7.0.post1. The `--is-decode-worker` flag configures vLLM internally but doesn't affect Dynamo runtime's service discovery registration logic. Bug needs to be reported to Dynamo team.

---

## Debugging

### Access Pod Shell

```bash
# Wait for pods to be Running first
kubectl get pods -w  # Press Ctrl+C to stop watching

# Access decode worker shell
kubectl exec -it deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx -- /bin/bash

# Access prefill worker shell
kubectl exec -it deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx -- /bin/bash

# Exit the shell
exit  # or press Ctrl+D
```

### View Logs

```bash
# Follow logs in real-time
kubectl logs -f deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx

# View last 50 lines
kubectl logs --tail=50 deepseek-r1-disagg-repro-0-vllmdecodeworker-xxxxx

# View previous container logs (if crashed)
kubectl logs deepseek-r1-disagg-repro-0-vllmprefillworker-xxxxx --previous
```

### Inspect Model Cache Storage

Check PVC ownership, filesystem details, and usage:

```bash
# 1. Check directory ownership and permissions
kubectl exec <pod-name> -n <namespace> -- ls -ld /models /models/hub /models/hub/hub
# Output shows:
#   drwxrwsr-x root:1000      /models          (setgid bit set)
#   drwxrwsr-x dynamo:1000    /models/hub
#   drwxrwsr-x dynamo:1000    /models/hub/hub

# 2. Check what user the container runs as
kubectl exec <pod-name> -n <namespace> -- id
# Output: uid=1000(dynamo) gid=0(root) groups=0(root),1000

# 3. Get PVC details
kubectl get pvc model-cache -n <namespace> -o yaml
# Shows: storage class, access mode, bound PV name

# 4. Get PersistentVolume details
kubectl get pv <pv-name> -o yaml
# Shows: NFS mount options, CSI driver, filesystem type

# 5. Get StorageClass details
kubectl get storageclass <storage-class-name> -o yaml
# Shows: provisioner, reclaim policy, mount options

# 6. Check filesystem usage
kubectl exec <pod-name> -n <namespace> -- df -h /models
# Shows: 25TB total, 21TB used (84%), 4.1TB available

# 7. List cached models
kubectl exec <pod-name> -n <namespace> -- ls -lh /models/hub/hub/
# Shows: models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B

# 8. Count models cached
kubectl exec <pod-name> -n <namespace> -- find /models/hub/hub -maxdepth 1 -type d -name "models--*" | wc -l
```

**Key findings from inspection**:
- **Storage**: NFS 4.1 (Nebius `nebius-shared-fs` with `mounted-fs-path.csi.nebius.ai`)
- **Filesystem**: ext4 over NFS
- **Size**: 25TB provisioned, 21TB used (shared across namespaces), 4.1TB available
- **Access Mode**: ReadWriteMany (multiple pods can read/write simultaneously)
- **Reclaim Policy**: Retain (data persists after PVC deletion)
- **Mount Options**: `nfsvers=4.1, hard, timeo=600, retrans=2, rsize=1048576, wsize=1048576`
- **Container User**: `uid=1000(dynamo)` with group `1000` write access
- **Model Location**: `/models/hub/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/` (132GB)

### Debug Data to Collect

When investigating the KV transfer failure:

1. **Full logs** from all three components (prefill, decode, frontend)
2. **Timestamp analysis**: When does the connection reset occur relative to KV transfer start?
3. **Network metrics**: Packet loss, bandwidth usage during transfer
4. **GPU memory usage**: Are we hitting OOM during KV transfer?
5. **NIXL configuration**: What are the NIXL buffer sizes, timeouts?
6. **Message size**: How large is the KV cache being transferred?
   - Formula: 120k tokens × num_layers × hidden_size × 2 (for K and V) × dtype_size
7. **Network trace**: tcpdump/Wireshark capture during failed transfer
8. **NIXL debug logs**: Enable verbose logging in NIXL connector

### Investigation Steps

#### 1. Find KV Cache Transfer Threshold
Test with intermediate context lengths to find exact failure point:
- 30k tokens
- 60k tokens
- 90k tokens
- 120k tokens

#### 2. Measure KV Cache Size
Calculate exact transfer size:
- DeepSeek-R1-Distill-Llama-70B model specs
- Number of layers, hidden size, attention heads
- Token count × layer count × size per token

#### 3. Check NIXL Configuration
- NIXL buffer size limits
- NIXL timeout settings (separate from VLLM_RPC_TIMEOUT)
- NIXL line protocol max message size
- NIXL connection pooling settings

#### 4. Try Alternative Solutions
- Use TCP connector instead of NIXL: `--connector tcp`
- Implement KV cache chunking/streaming
- Increase NIXL-specific buffer sizes
- Enable NIXL connection keepalive

#### 5. Enable Debug Logging
```bash
# Enable NIXL debug logs (if available)
export NIXL_LOG_LEVEL=debug
export NIXL_TRACE=1

# Enable vLLM debug logs
export VLLM_LOGGING_LEVEL=DEBUG
```

#### 6. Capture Network Traffic
```bash
# On decode node (requires exec into pod)
tcpdump -i any -w kv_transfer.pcap port <nixl-port>
```

### Potential Root Causes

1. **NIXL timeout**: KV transfer for 120k tokens exceeds NIXL connection timeout
   - VLLM_RPC_TIMEOUT may not apply to NIXL layer
   - NIXL may have separate, lower timeout

2. **Buffer overflow**: KV cache message size exceeds NIXL buffer limits
   - NIXL line protocol may have max message size
   - Need to check NIXL buffer configuration

3. **Network congestion**: Large KV transfer saturates network between nodes
   - 120k tokens = very large KV cache
   - May exceed network throughput or trigger packet loss

4. **Memory pressure**: Decode worker runs out of memory during KV reception
   - Check GPU memory before/during/after transfer
   - Check system memory for buffer allocation

5. **Protocol mismatch**: NIXL line protocol can't handle very large messages
   - "invalid line protocol" suggests NIXL parsing failure
   - May need chunking or streaming protocol

6. **TCP connection limits**: Underlying TCP connection may timeout or break
   - Check TCP keepalive settings
   - Check firewall/network rules

### Questions for Investigation

- What is the exact size of the KV cache being transferred for 120k tokens?
- What are the NIXL buffer size limits?
- Is there a NIXL-specific timeout separate from VLLM_RPC_TIMEOUT?
- Does the NIXL line protocol have a maximum message size?
- Can we split KV cache transfer into multiple chunks?
- Can we reproduce with a different model of similar size?
- Does TCP connector work where NIXL fails?
- What is the network bandwidth between nodes?
- Are there any firewall rules or network policies affecting large transfers?

---

## Reference

### Cleanup Commands

```bash
# Delete the deployment
kubectl delete -f deepseek-r1-disagg.yaml

# Uninstall Helm chart (etcd, NATS)
helm uninstall dynamo-platform --namespace keivenc-dyn-1556-repro-nixl-timeout

# Delete PVC (WARNING: deletes all cached models)
kubectl delete pvc model-cache

# Or delete everything in your namespace (removes all resources)
kubectl delete namespace keivenc-dyn-1556-repro-nixl-timeout
```

### Related Code Locations

Key areas to investigate in Dynamo codebase:
- NIXL connector implementation
- KV cache transfer logic in vLLM disagg mode
- Message serialization/deserialization for KV cache
- Connection timeout configuration
- Buffer allocation for large messages

### Cluster Resources

**Available via Teleport** (`tsh kube login <cluster-name>`):

1. **dynamo-nebius-1** (Recommended)
   - 16 nodes × 8 H200 SXM GPUs (141GB each) = 128 GPUs
   - Storage class: `nebius-shared-fs`

2. **dynamo-aks-dev**
   - 8 nodes × 8 A100 80GB GPUs = 64 GPUs
   - Storage class: `csi-mounted-fs-path-sc`

3. **dynamo-aws-dev-gb200**
   - CPU-only nodes (no GPUs)

4. **dynamo-aks-ci**
   - Access restricted

### Files Created

All files in `/home/keivenc/nvidia/dynamo2/`:

1. **platform-values.yaml** - Helm values for etcd/NATS only
2. **model-cache-pvc.yaml** - 25TB shared model storage PVC
3. **deepseek-r1-disagg.yaml** - DynamoGraphDeployment with model caching

### Useful kubectl Commands

```bash
# List all resources in namespace
kubectl get all

# Watch pod status
kubectl get pods -w

# Get deployment details
kubectl get dynamographdeployment

# Check PVC status
kubectl get pvc

# View all events in namespace
kubectl get events --sort-by='.lastTimestamp'

# Check node GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```
