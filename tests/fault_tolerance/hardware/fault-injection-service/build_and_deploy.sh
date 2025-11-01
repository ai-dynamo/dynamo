#!/bin/bash
# Build and deploy fault injection services for AMD64 (Azure AKS)

set -e

echo "=== Building and Pushing Docker Images for AMD64 ==="

cd "$(dirname "$0")"

# Login to ACR (if needed)
# az acr login --name dynamoci

# 1. Build and push GPU fault injector (CRITICAL - fixes hang issue)
echo "Building GPU fault injector..."
docker buildx build \
  --platform linux/amd64 \
  -t dynamoci.azurecr.io/gpu-fault-injector:latest \
  -f agents/gpu-fault-injector/Dockerfile \
  agents/gpu-fault-injector/ \
  --push

# 2. Build and push API service (for debug logs)
echo "Building API service..."
docker buildx build \
  --platform linux/amd64 \
  -t dynamoci.azurecr.io/fault-injection-api:latest \
  -f api-service/Dockerfile \
  api-service/ \
  --push

echo "=== Restarting Kubernetes Deployments ==="

# 3. Restart daemonset to pick up new GPU agent image
echo "Restarting GPU fault injector daemonset..."
kubectl rollout restart daemonset/gpu-fault-injector-kernel -n fault-injection-system

# 4. Restart API service deployment
echo "Restarting API service..."
kubectl rollout restart deployment/fault-injection-api -n fault-injection-system

echo "=== Waiting for Rollouts to Complete ==="

# 5. Wait for rollouts
kubectl rollout status daemonset/gpu-fault-injector-kernel -n fault-injection-system --timeout=300s
kubectl rollout status deployment/fault-injection-api -n fault-injection-system --timeout=300s

echo "=== Checking Pod Status ==="
kubectl get pods -n fault-injection-system -l app=gpu-fault-injector-kernel -o wide
kubectl get pods -n fault-injection-system -l app=fault-injection-api

echo ""
echo "✅ Build and deployment complete!"
echo ""
echo "To verify the new images are running:"
echo "  kubectl get pods -n fault-injection-system -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u"
