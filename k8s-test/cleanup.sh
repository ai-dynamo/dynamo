#!/bin/bash
# Clean up test resources from Kubernetes cluster

set -e

# Parse namespace argument (default to "default")
NAMESPACE="${1:-default}"

echo "🧹 Cleaning up Dynamo test resources from namespace: $NAMESPACE"

# Delete manifests
kubectl delete -f manifests/test-deployment.yaml --namespace="$NAMESPACE" --ignore-not-found=true

echo ""
echo "✅ Test resources cleaned up from namespace: $NAMESPACE!"
echo ""
echo "Note: The namespace itself was not deleted. To delete it, run:"
echo "  kubectl delete namespace $NAMESPACE"

