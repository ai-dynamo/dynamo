# K8s Failover End-to-End Testing

End-to-end validation of GPU failover on Kubernetes with DRA, GMS shadow mode,
and etcd-based multinode coordination.

## Branch: `failover/k8s-test`

This branch contains all operator + engine changes for failover, validated on
the `dynamo-exp` AKS cluster.

### Commits

| Commit | Description |
|--------|-------------|
| `e4a9ae368` | PR #6818: GMS shadow mode engine patches |
| `4a0618ccd` | Epoch 3: per-container K8s discovery (Pod reflector) |
| `e2c84474a` | Epoch 4: single-node failover operator (CRD, DRA, webhook) |
| `44dbe19f6` | Epoch 5: multinode failover operator (numberOfNodes, role) |
| `0b8cb8fc1` | Epoch 6: etcd-based coordinated restart harness |
| `edee68e7e` | Operator fixes found during K8s testing |

### Operator fixes (`edee68e7e`)

Three bugs found during e2e testing:

1. **RCT sync in DCD controller**: `ResourceClaimTemplate` was only synced in the
   Grove path (`reconcileGroveResources`). Single-node failover uses the DCD
   controller path, which never created the RCT. Pods were stuck with
   `FailedResourceClaimCreation`.

2. **Master-port injection**: `staggerMasterPort()` only offsets an existing
   `--master-port` flag. When users don't specify it, both engines default to
   29500 causing NCCL TCP store collisions. Fix: if `--master-port` isn't in
   args, engine-1 gets `--master-port 29600` appended.

3. **Shadow mode env var**: Changed `DYN_VLLM_GMS_MODE=shadow` to
   `DYN_VLLM_GMS_SHADOW_MODE=true` to match the engine code from PR #6818.

## Images

Built from commit `0b8cb8fc1` on this branch:

| Image | Tag |
|-------|-----|
| Engine | `dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-0b8cb8fc1-vllm-runtime` |
| Operator | `dynamoci.azurecr.io/ai-dynamo/kubernetes-operator:multinode-failover-0b8cb8fc1` |

### Building images

```bash
# Engine (~30 min, Rust compilation)
python3 container/render.py --framework vllm --target runtime
docker build -f container/vllm-runtime-cuda12.9-amd64-rendered.Dockerfile \
  -t dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-$(git rev-parse --short HEAD)-vllm-runtime .
docker push dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-$(git rev-parse --short HEAD)-vllm-runtime

# Operator (~2 min)
cd deploy/operator
make manifests
make docker-build IMG=dynamoci.azurecr.io/ai-dynamo/kubernetes-operator:multinode-failover-$(git rev-parse --short HEAD)
docker push dynamoci.azurecr.io/ai-dynamo/kubernetes-operator:multinode-failover-$(git rev-parse --short HEAD)
```

## Prerequisites

### Cluster

- `dynamo-exp` AKS cluster (context: `dynamo-exp-6f8d9a`)
- DRA enabled (GPU ResourceSlices present on A100 nodes)
- Docker authenticated with `dynamoci.azurecr.io`
- Grove and KAI Scheduler installed cluster-wide (see
  [velonix](https://github.com/ai-dynamo/velonix) flux-apps/dynamo-dependencies-apps
  for installation patterns). Required for multinode tests.

### Namespace setup

```bash
NAMESPACE="failover-e2e-test"
kubectl create namespace ${NAMESPACE}
```

### etcd

```bash
kubectl apply -n ${NAMESPACE} -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: etcd
spec:
  replicas: 1
  selector:
    matchLabels:
      app: etcd
  template:
    metadata:
      labels:
        app: etcd
    spec:
      containers:
      - name: etcd
        image: quay.io/coreos/etcd:v3.5.21
        command:
        - etcd
        - --listen-client-urls=http://0.0.0.0:2379
        - --advertise-client-urls=http://etcd.${NAMESPACE}.svc.cluster.local:2379
        ports:
        - containerPort: 2379
---
apiVersion: v1
kind: Service
metadata:
  name: etcd
spec:
  selector:
    app: etcd
  ports:
  - port: 2379
    targetPort: 2379
EOF
kubectl wait --for=condition=available deployment/etcd -n ${NAMESPACE} --timeout=60s
```

### HF token secret

```bash
kubectl create secret generic hf-token-secret -n ${NAMESPACE} \
  --from-literal=HF_TOKEN=<your-token>
```

### CRDs

Apply CRDs from this branch (includes `FailoverSpec` fields):

```bash
kubectl apply --server-side --force-conflicts -f deploy/operator/config/crd/bases/
```

### Platform (dynamo-operator)

```bash
OPERATOR_TAG="multinode-failover-0b8cb8fc1"

helm dependency build deploy/helm/charts/platform

helm upgrade dynamo-platform deploy/helm/charts/platform \
  -n ${NAMESPACE} --install \
  --set dynamo-operator.controllerManager.manager.image.repository=dynamoci.azurecr.io/ai-dynamo/kubernetes-operator \
  --set dynamo-operator.controllerManager.manager.image.tag=${OPERATOR_TAG} \
  --set dynamo-operator.controllerManager.manager.image.pullPolicy=Always \
  --set dynamo-operator.etcdAddr=http://etcd.${NAMESPACE}.svc.cluster.local:2379 \
  --set dynamo-operator.namespaceRestriction.enabled=true \
  --wait --timeout 5m
```

## Single-Node Failover Test

### Deploy

```bash
kubectl apply -f test/k8s-failover/single-node-failover.yaml -n ${NAMESPACE}
```

### What to expect

- 1 worker pod with 3 containers: `engine-0`, `engine-1`, `gms-weights`
- `ResourceClaimTemplate` for DRA shared GPU access
- Both engines init in shadow mode (GMS RW for engine-0, RO for engine-1)
- Both engines sleep after init, one acquires flock and wakes
- Active engine allocates KV cache and registers `generate` endpoint
- Standby engine remains sleeping, waiting for lock

### Automated test

```bash
# Full deploy + test + cleanup
bash test/k8s-failover/test_k8s_failover.sh \
  --engine-image dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-0b8cb8fc1-vllm-runtime \
  --namespace ${NAMESPACE} \
  --node aks-a100exp-11297970-vmss000001

# Test existing deployment
bash test/k8s-failover/test_k8s_failover.sh --skip-deploy --namespace ${NAMESPACE}
```

The script validates: operator resources (RCT, containers, env vars), engine
lifecycle (shadow mode, GMS, lock, KV cache), inference, failover (kill active,
standby takes over), and recovery (killed engine restarts as standby).

### Manual failover test

```bash
POD=$(kubectl get pod -n ${NAMESPACE} -l nvidia.com/dynamo-component=VllmWorker -o jsonpath='{.items[0].metadata.name}')

# Identify active engine (check which one acquired the lock)
kubectl logs ${POD} -c engine-0 -n ${NAMESPACE} | grep "Lock acquired"
kubectl logs ${POD} -c engine-1 -n ${NAMESPACE} | grep "Lock acquired"

# Kill active engine (e.g. engine-1)
kubectl exec ${POD} -c engine-1 -n ${NAMESPACE} -- kill 1

# Watch standby wake (~5-12s)
kubectl logs ${POD} -c engine-0 -n ${NAMESPACE} -f | grep -E "Lock acquired|KV cache|Registered.*generate"

# Test inference
FRONTEND=$(kubectl get svc -n ${NAMESPACE} | grep frontend | awk '{print $1}')
kubectl run curl-test --rm -i --restart=Never -n ${NAMESPACE} --image=curlimages/curl -- \
  curl -sf http://${FRONTEND}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'

# Verify killed engine recovers as standby (~60s for full reinit)
kubectl logs ${POD} -c engine-1 -n ${NAMESPACE} | grep "Engine sleeping.*waiting for lock"
```

## Multinode Failover Test

### Deploy

```bash
kubectl apply -f test/k8s-failover/multinode-failover.yaml -n ${NAMESPACE}
```

### What to expect

- 2 worker pods: `*-ldr-*` (leader) and `*-wkr-*` (worker)
- Each pod has 3 containers: `engine-0`, `engine-1`, `gms-weights`
- Harness `ConfigMap` with `harness_leader.sh` + `harness_worker.sh`
- Engine entrypoints wrapped: `bash /harness/harness_leader.sh <original-cmd>`
- Leader publishes etcd key, worker joins, go signal sent (~1s formation)
- Both pods: one engine acquires flock and serves, other sleeps

### Validate harness formation

```bash
LDR=$(kubectl get pods -n ${NAMESPACE} -o name | grep worker.*ldr | head -1)
WKR=$(kubectl get pods -n ${NAMESPACE} -o name | grep worker.*wkr | head -1)

# Leader harness logs
kubectl logs ${LDR} -c engine-0 -n ${NAMESPACE} | grep "^\[" | head -12

# Worker harness logs
kubectl logs ${WKR} -c engine-0 -n ${NAMESPACE} | grep "^\[" | head -12
```

Expected leader sequence:
```
[leader/engine-0] Starting (hash=..., nnodes=2)
[leader/engine-0] Lease created: ... (TTL=5s)
[leader/engine-0] Published leader key
[leader/engine-0] Keepalive PID: ...
[leader/engine-0] Waiting for 1 worker(s) to join...
[leader/engine-0] All workers joined
[leader/engine-0] Sent go signal
[leader/engine-0] Starting engine: python3 -m dynamo.vllm ...
[leader/engine-0] Monitoring workers...
```

Expected worker sequence:
```
[worker/engine-0/rank-1] Starting (uuid=...)
[worker/engine-0/rank-1] Waiting for leader...
[worker/engine-0/rank-1] Found leader (hash=...)
[worker/engine-0/rank-1] Registered under leader hash
[worker/engine-0/rank-1] Waiting for go signal...
[worker/engine-0/rank-1] Go signal received
[worker/engine-0/rank-1] Starting engine: ...
[worker/engine-0/rank-1] Monitoring leader...
```

### Inference test

```bash
FRONTEND=$(kubectl get svc -n ${NAMESPACE} | grep frontend | awk '{print $1}')
kubectl run curl-mn --rm -i --restart=Never -n ${NAMESPACE} --image=curlimages/curl -- \
  curl -sf http://${FRONTEND}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

## Known Issues

1. **Wake error doesn't block endpoint registration**: `handlers.wake_up()` catches
   all exceptions and returns `{"status": "error"}`. The caller in
   `worker_factory.py` doesn't check the return value, so a broken engine can
   register the `generate` endpoint and cause 500s. This is an engine-side bug,
   not an operator bug.

2. **Engine registers then unregisters before sleep**: In shadow mode, the startup
   path runs `register_vllm_model` (publishes MDC + registers endpoint) then
   immediately calls `sleep()` which unregisters the endpoint. The endpoint is
   briefly visible in discovery before being removed. Functionally harmless.

3. **Harness lease TTL sensitivity**: The etcd lease TTL defaults to 5s. If
   `etcdctl lease keep-alive` can't reach etcd for >5s (e.g. during heavy init),
   the lease expires and the harness kills the engine. This was observed once
   during testing and did not reproduce on retry.

## Cleanup

```bash
kubectl delete dgd --all -n ${NAMESPACE}

# Full namespace teardown
kubectl delete namespace ${NAMESPACE}
```
