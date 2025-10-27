# Fault Injection Service - Quick Start

## Setup (One-Time)

### 1. Build and Push Images

**Recommended: Push to ACR (dynamoci.azurecr.io)**

```bash
cd dynamo/tests/fault_tolerance/hardware/fault-injection-service

# Login to ACR
az acr login --name dynamoci

# Build and push API
docker buildx build --platform linux/amd64 \
  -t dynamoci.azurecr.io/fault-injection-api:latest \
  --push -f api-service/Dockerfile api-service/

# Build and push network injector (optional, only if modified)
docker buildx build --platform linux/amd64 \
  -t dynamoci.azurecr.io/network-fault-injector:latest \
  --push -f agents/network-fault-injector/Dockerfile agents/network-fault-injector/

# Update deployment to use ACR images
kubectl set image deployment/fault-injection-api \
  api=dynamoci.azurecr.io/fault-injection-api:latest \
  -n fault-injection-system

# Wait for rollout
kubectl rollout status deployment/fault-injection-api -n fault-injection-system
```

**Alternative: Push to NGC (if cluster has NGC access)**

```bash
# Login to NGC
docker login nvcr.io  # Username: $oauthtoken, Password: <NGC-API-key>

# Build and push
export NGC_REGISTRY="nvcr.io/nvidian/dynamo-dev"
docker buildx build --platform linux/amd64 \
  -t $NGC_REGISTRY/fault-injection-api:latest \
  --push -f api-service/Dockerfile api-service/

# Restart pod to pull new image
kubectl delete pod -n fault-injection-system -l app=fault-injection-api
kubectl wait --for=condition=ready pod -l app=fault-injection-api -n fault-injection-system --timeout=120s
```

### 2. Deploy Infrastructure

**For Network Tests (NetworkPolicy + ChaosMesh):**

```bash
cd deploy/

# Deploy fault injection service
kubectl apply -f namespace.yaml
kubectl apply -f api-service.yaml

# Wait for API to be ready
kubectl wait --for=condition=ready pod -l app=fault-injection-api -n fault-injection-system --timeout=300s
```

**Install ChaosMesh for advanced faults (packet loss, delay) (already done on AKS dynamo-dev cluster)**

```bash
kubectl create ns chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock
```

Components:
- ✅ Fault Injection API (required)
- ✅ ChaosMesh (optional, for packet loss/delay tests)

### 3. Run Tests

**Recommended:** Use the in-cluster script (no port-forwarding needed):

```bash
# NetworkPolicy tests (complete blocking)
python3 scripts/run_test_incluster.py examples/test_partition_worker_to_nats.py
python3 scripts/run_test_incluster.py examples/test_specific_pod_to_pod_blocking.py

# ChaosMesh tests (packet loss, delay) - requires ChaosMesh installed
python3 scripts/run_test_incluster.py examples/test_nats_packet_loss_50_percent.py
```

> **Note:** Update `FRONTEND_URL` in `scripts/run_test_incluster.py` (line 99) to match your deployment (e.g. `vllm-agg-frontend` or `vllm-disagg-frontend`). Verify with: `kubectl get svc -n {NAMESPACE}`

**Available Tests:**

| Test | Type | Description |
|------|------|-------------|
| `test_partition_worker_to_nats.py` | NetworkPolicy | Complete NATS isolation |
| `test_specific_pod_to_pod_blocking.py` | NetworkPolicy | Precise pod-to-pod blocking |
| `test_disagg_pod_to_pod.py` | NetworkPolicy | Multi-worker fault tolerance with latency measurement (vllm-disagg) |
| `test_nats_packet_loss_50_percent.py` | ChaosMesh | 50% packet loss + delay tests |

**Alternative:** Local testing with port-forwarding:

```bash
# Terminal 1: Port-forward API
kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080

# Terminal 2: Run tests
pip install -e client/
cd examples && python3 test_nats_packet_loss_50_percent.py
```

---

## Write Your Own Test

See [API_USER_GUIDE.md](API_USER_GUIDE.md) for detailed examples and instructions.

---

## Cleanup

```bash
# Remove all infrastructure
kubectl delete namespace fault-injection-system
```

---

## Documentation

- [API User Guide](API_USER_GUIDE.md) - Complete API reference and usage examples
