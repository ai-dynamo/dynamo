# Quick Start - Testing Kubernetes Service Discovery

## TL;DR - Commands to Run

```bash
# 1. Point to your cluster
export KUBECONFIG=/path/to/your/kubeconfig
export POD_NAMESPACE=default

# 2. Deploy test resources to your cluster
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# 3. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=nginx-test --timeout=60s

# 4. Run the quick connection test
cargo test --package dynamo-runtime test_kubernetes_connection -- --ignored --nocapture

# 5. Run the full test (list + watch for 30 seconds)
cargo test --package dynamo-runtime test_kubernetes_discovery -- --ignored --nocapture
```

## During the Watch Test (30 second window)

In another terminal, trigger events:

```bash
# Scale up to see ADDED events
kubectl scale deployment nginx-test --replicas=5

# Scale down to see REMOVED events
kubectl scale deployment nginx-test --replicas=2
```

## What the Test Checks

✅ **list_instances("test", "worker")** - Lists all ready pods with matching labels  
✅ **watch("test", "worker")** - Streams ADDED/REMOVED events when pods change

## Expected Labels on EndpointSlices

The implementation looks for EndpointSlices with:
```yaml
labels:
  dynamo.namespace: test
  dynamo.component: worker
```

Kubernetes automatically creates these when you apply the service (it inherits the service's labels).

## Cleanup

```bash
kubectl delete -f kubernetes/
```

