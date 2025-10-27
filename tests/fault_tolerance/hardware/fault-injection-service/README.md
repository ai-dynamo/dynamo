# Fault Injection Service - Quick Start

## Setup (One-Time)

### 1. Build and Push Images

**For Network Tests:**

```bash
cd dynamo/tests/fault_tolerance/hardware/fault-injection-service

# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-NGC-API-key from https://ngc.nvidia.com/setup/api-key>

# Set your registry
export NGC_REGISTRY="nvcr.io/nvidian/dynamo-dev"

# Build only what's needed for network partition tests
docker buildx build --platform linux/amd64 -t $NGC_REGISTRY/fault-injection-api:latest --push -f api-service/Dockerfile api-service/

docker buildx build --platform linux/amd64 -t $NGC_REGISTRY/network-fault-injector:latest --push -f agents/network-fault-injector/Dockerfile agents/network-fault-injector/
```


**Quick rebuild after code changes (API service only):**
```bash
cd dynamo/tests/fault_tolerance/hardware/fault-injection-service
export NGC_REGISTRY="nvcr.io/nvidian/dynamo-dev"

# Rebuild and push API
docker buildx build --platform linux/amd64 -t $NGC_REGISTRY/fault-injection-api:latest --push -f api-service/Dockerfile api-service/

# Restart the API pod to pick up changes
kubectl delete pod -n fault-injection-system -l app=fault-injection-api

# Wait for new pod to be ready
kubectl wait --for=condition=ready pod -l app=fault-injection-api -n fault-injection-system --timeout=120s
```

### 2. Update Manifests (if using custom registry)

If you're using a custom registry, update the image references:

```bash
export NGC_REGISTRY="nvcr.io/nvidian/dynamo-dev"

# For network tests only
sed -i '' "s|image: .*fault-injection-api.*|image: $NGC_REGISTRY/fault-injection-api:latest|g" deploy/api-service.yaml
sed -i '' "s|image: .*network-fault-injector.*|image: $NGC_REGISTRY/network-fault-injector:latest|g" deploy/network-fault-injector.yaml

# Or update all manifests
find deploy -name "*.yaml" -exec sed -i '' \
  "s|image: fault-injection-api.*|image: $NGC_REGISTRY/fault-injection-api:latest|g" {} \;
find deploy -name "*.yaml" -exec sed -i '' \
  "s|image: network-fault-injector.*|image: $NGC_REGISTRY/network-fault-injector:latest|g" {} \;
```

### 3. Deploy Infrastructure

**For Network Tests:**

Only deploy what's needed for network partition tests:

```bash
cd deploy/

# Deploy namespace, API, and network injector
kubectl apply -f namespace.yaml
kubectl apply -f api-service.yaml
kubectl apply -f network-fault-injector.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=fault-injection-api -n fault-injection-system --timeout=300s
kubectl wait --for=condition=ready pod -l app=network-fault-injector -n fault-injection-system --timeout=300s
```

Components deployed:
- Fault Injection API (required)
- Network Fault Injector (required for network tests)

### 4. Run Tests

**Recommended:** Use the in-cluster script (no port-forwarding needed):

```bash
python scripts/run_test_incluster.py examples/test_partition_worker_to_nats.py
```

**Alternative:** For local testing, port-forward the API and install the client:

```bash
# Terminal 1: Port-forward API (keep running)
kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080

# Terminal 2: Install client and run tests
pip install -e client/
cd examples && python test_partition_worker_to_nats.py
```

**Available Tests:**
- `test_partition_worker_to_nats.py` - Worker→NATS partition (system stays operational)
- `test_partition_worker_to_frontend.py` - Worker→Frontend partition (validates redundancy)
- `test_partition_frontend_to_nats.py` - Frontend→NATS partition (critical failure & recovery)

---

## Write Your Own Test

See [API_USER_GUIDE.md](API_USER_GUIDE.md) for detailed examples and instructions.

---

## Cleanup

```bash
# Remove all infrastructure
kubectl delete namespace fault-injection-system
```

