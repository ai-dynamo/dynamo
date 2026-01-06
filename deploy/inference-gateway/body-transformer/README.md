# Body Transformer

An Envoy ext_proc service that injects `nvext` routing information into request bodies for Dynamo inference backends.

## Why This Exists

Dynamo backends require routing information in the request body as an `nvext` field. The GAIE EPP sets routing headers (`x-worker-instance-id`, etc.), but doesn't modify the request body.

This service bridges that gap by:
1. Reading the routing headers set by the GAIE EPP
2. Injecting the `nvext` field into the request body

## Supported Gateways

This ext_proc service works with **any Envoy-based gateway**:

| Gateway | Configuration |
|---------|---------------|
| **kGateway** | `TrafficPolicy.extProc` (see `deploy/kgateway-config.yaml`) |
| **Istio** | `EnvoyFilter` with ext_proc filter |
| **Envoy Gateway** | `EnvoyExtensionPolicy` or `BackendTrafficPolicy` |
| **Plain Envoy** | ext_proc filter in listener config |

**Note:** For Istio and Envoy Gateway, you can alternatively use Lua filters (see `config/lua-filter/`) which don't require deploying an extra service.

## Architecture

```
Client → kGateway → GAIE EPP → Body Transformer → Dynamo Backend
              ↓           ↓              ↓
         routing     sets headers    injects nvext
         decision    in response     into body
```

## Body Transformation

**Headers read:**
- `x-worker-instance-id`: Primary worker ID (decode worker in disagg mode)
- `x-prefiller-host-port`: Prefill worker ID (disaggregated mode only)
- `x-dynamo-routing-mode`: "aggregated" or "disaggregated"

**Aggregated Mode (default):**
```json
{
  "model": "llama",
  "messages": [...],
  "nvext": {
    "backend_instance_id": 42
  }
}
```

**Disaggregated Mode:**
```json
{
  "model": "llama",
  "messages": [...],
  "nvext": {
    "prefill_worker_id": 10,
    "decode_worker_id": 42
  }
}
```

## Quick Start

### 1. Build the Image

```bash
cd body-transformer
make image-build

# For minikube:
make image-load
```

### 2. Deploy

```bash
# Deploy the service (update namespace as needed)
kubectl apply -f deploy/deployment.yaml -n my-model

# Configure kGateway to use it
kubectl apply -f deploy/kgateway-config.yaml
```

### 3. Verify

```bash
# Check pod is running
kubectl get pods -l app=body-transformer -n my-model

# Check logs
make logs
```

## Configuration

### Namespace Configuration

Update these files if your namespaces differ:

1. `deploy/deployment.yaml` - Deploy to your workload namespace i.e. my-model
2. `deploy/kgateway-config.yaml`:
   - `GatewayExtension.spec.extProc.grpcService.backendRef.namespace` - Where body-transformer runs
   - `TrafficPolicy.metadata.namespace` - Where Gateway is deployed
   - `ReferenceGrant.metadata.namespace` - Where body-transformer runs

## Development

```bash
# Build locally
make build

# Run locally
./bin/body-transformer -port 9003

# Run tests
make test
```

## Troubleshooting

### No nvext in body

1. Check body-transformer logs: `make logs`
2. Verify headers are being set by EPP:
   ```bash
   kubectl logs -l app=qwen-epp -n my-model | grep "x-worker-instance-id"
   ```
3. Check TrafficPolicy status:
   ```bash
   kubectl get trafficpolicy nvext-body-injector -o yaml
   ```

### Body not being sent to transformer

Ensure `requestBodyMode: BUFFERED` is set in TrafficPolicy.

### Cross-namespace reference errors

Ensure ReferenceGrant is applied in the body-transformer's namespace.

