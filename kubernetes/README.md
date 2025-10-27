# Kubernetes Testing Setup for EndpointSlice Discovery

This directory contains Kubernetes manifests and test utilities for testing the EndpointSlice-based service discovery implementation.

## Files

- **deployment.yaml** - Sample nginx deployment with 3 replicas
- **service.yaml** - Service with proper Dynamo labels for discovery
- **endpoint-slice-test.yaml** - Manual EndpointSlice for testing without a full deployment

## Setup for Testing

### 1. Configure Kubernetes Connection

Make sure your cluster is accessible via kubectl:

```bash
# Check cluster access
kubectl cluster-info

# Set KUBECONFIG if needed
export KUBECONFIG=/path/to/your/kubeconfig

# Set the namespace where you'll deploy (defaults to "default")
export POD_NAMESPACE=default
```

### 2. Deploy Test Resources

You have two options:

#### Option A: Deploy Full Stack (Deployment + Service)

This will create actual pods and let Kubernetes automatically create EndpointSlices:

```bash
# Deploy the nginx deployment and service
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=nginx-test --timeout=60s

# Verify EndpointSlices were created
kubectl get endpointslices -l dynamo.namespace=test,dynamo.component=worker
```

#### Option B: Use Manual EndpointSlice (No Pods)

For quick testing without deploying actual pods:

```bash
# Create a manual EndpointSlice
kubectl apply -f kubernetes/endpoint-slice-test.yaml

# Verify it was created
kubectl get endpointslices test-worker-endpointslice -o yaml
```

### 3. Run the Integration Tests

```bash
# Quick connection test
cargo test --package dynamo-runtime test_kubernetes_connection -- --ignored --nocapture

# Full integration test (lists instances and watches for changes)
cargo test --package dynamo-runtime test_kubernetes_discovery -- --ignored --nocapture
```

## Testing the Watch Functionality

When running the full integration test, you'll have 30 seconds to make changes and observe events:

### With Full Deployment:

```bash
# In another terminal, scale up
kubectl scale deployment nginx-test --replicas=5

# Scale down
kubectl scale deployment nginx-test --replicas=2

# Delete a pod (it will be recreated by the deployment)
kubectl delete pod -l app=nginx-test --field-selector status.phase=Running | head -1
```

### With Manual EndpointSlice:

```bash
# Edit the EndpointSlice to add/remove endpoints or change ready status
kubectl edit endpointslice test-worker-endpointslice

# Or delete and recreate
kubectl delete -f kubernetes/endpoint-slice-test.yaml
kubectl apply -f kubernetes/endpoint-slice-test.yaml
```

## Expected Test Output

### list_instances:
Should show all ready pods that match the labels:
```
Found 3 instances:
  - worker-pod-1
  - worker-pod-2
```

### watch:
Should show events as pods become ready or are removed:
```
Watching for changes (will wait 30 seconds)...
  [ADDED] Instance: worker-pod-4
  [REMOVED] Instance: worker-pod-1
  [ADDED] Instance: worker-pod-5
```

## Label Requirements

For the discovery system to find your EndpointSlices, they must have these labels:

```yaml
labels:
  dynamo.namespace: test
  dynamo.component: worker
```

## Cleanup

```bash
# Remove all test resources
kubectl delete -f kubernetes/

# Or individually
kubectl delete deployment nginx-test
kubectl delete service nginx-test-service
kubectl delete endpointslice test-worker-endpointslice
```

## Troubleshooting

### "Failed to create Kubernetes discovery client"
- Check that KUBECONFIG is set correctly
- Verify kubectl can access the cluster: `kubectl cluster-info`

### "Failed to list instances"
- Verify EndpointSlices exist: `kubectl get endpointslices -A`
- Check labels: `kubectl get endpointslices -l dynamo.namespace=test,dynamo.component=worker`
- Verify POD_NAMESPACE is set correctly

### No instances found
- Check that endpoints have `ready: true` condition
- Verify endpoints have `targetRef.kind: Pod` and `targetRef.name`
- Use `kubectl describe endpointslice <name>` to inspect

### Watch doesn't receive events
- The watch starts after the test begins, so existing instances won't trigger Added events
- Make changes (scale, delete pods) during the 30-second watch window
- Check for errors in the test output

