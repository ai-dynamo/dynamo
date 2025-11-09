#!/bin/bash
# Deploy test resources to Kubernetes cluster

set -e

# Parse namespace argument (default to "default")
NAMESPACE="${1:-default}"

echo "🚀 Deploying Dynamo test resources to namespace: $NAMESPACE"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "📦 Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
else
    echo "✅ Namespace $NAMESPACE already exists"
fi

echo ""
echo "Applying manifests..."

# Apply manifests with namespace override
kubectl apply -f manifests/test-deployment.yaml --namespace="$NAMESPACE"

echo ""
echo "✅ Resources deployed!"
echo ""
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=dynamo-test --namespace="$NAMESPACE" --timeout=60s

echo ""
echo "📊 Current status in namespace $NAMESPACE:"
kubectl get deployment dynamo-test-worker --namespace="$NAMESPACE"
kubectl get service dynamo-test-service --namespace="$NAMESPACE"
kubectl get pods -l app=dynamo-test --namespace="$NAMESPACE"
kubectl get endpointslices -l kubernetes.io/service-name=dynamo-test-service --namespace="$NAMESPACE"

echo ""
echo "✅ Test environment is ready in namespace: $NAMESPACE!"
echo ""
echo "To run tests against this namespace, set POD_NAMESPACE=$NAMESPACE in your test client"

