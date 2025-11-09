#!/bin/bash
# Run integration tests for Kubernetes discovery client

set -e

echo "🧪 Running Kubernetes Discovery Integration Tests"
echo ""

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ kubectl is not configured or cluster is not accessible"
    echo "   Please ensure you have access to a Kubernetes cluster"
    exit 1
fi

echo "✅ kubectl is configured"
echo "   Cluster: $(kubectl config current-context)"
echo ""

# Parse command line arguments
TEST_SUITE="${1:-kube_client}"
TEST_NAME="${2:-}"
NAMESPACE="${3:-default}"

echo "🔍 Checking for test resources in namespace: $NAMESPACE"

# Check if test resources are deployed
PODS=$(kubectl get pods -l app=dynamo-test --namespace="$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [ "$PODS" -eq 0 ]; then
    echo "⚠️  Test resources not deployed in namespace: $NAMESPACE"
    echo "   Run ./deploy.sh $NAMESPACE to create test resources"
    echo "   (Tests will still run but may not find any endpoints)"
    echo ""
else
    echo "✅ Found $PODS test pods in namespace: $NAMESPACE"
    echo ""
fi

case "$TEST_SUITE" in
    "client"|"kube_client")
        echo "Running KubeDiscoveryClient tests..."
        if [ -n "$TEST_NAME" ]; then
            cargo test --test kube_client_integration "$TEST_NAME" -- --ignored --nocapture --test-threads=1
        else
            cargo test --test kube_client_integration -- --ignored --nocapture --test-threads=1
        fi
        ;;
    "raw"|"kube_api")
        echo "Running raw Kubernetes API tests..."
        if [ -n "$TEST_NAME" ]; then
            cargo test --test kube_discovery_integration "$TEST_NAME" -- --ignored --nocapture --test-threads=1
        else
            cargo test --test kube_discovery_integration -- --ignored --nocapture --test-threads=1
        fi
        ;;
    "all")
        echo "Running all integration tests..."
        cargo test --test kube_client_integration -- --ignored --nocapture --test-threads=1
        echo ""
        echo "---"
        echo ""
        cargo test --test kube_discovery_integration -- --ignored --nocapture --test-threads=1
        ;;
    *)
        echo "Usage: $0 [client|raw|all] [test_name] [namespace]"
        echo ""
        echo "Arguments:"
        echo "  test_suite  - Which test suite to run (default: client)"
        echo "  test_name   - Specific test to run (optional)"
        echo "  namespace   - Kubernetes namespace to check (default: default)"
        echo ""
        echo "Test suites:"
        echo "  client (default) - Run KubeDiscoveryClient tests (recommended)"
        echo "  raw              - Run raw Kubernetes API tests"
        echo "  all              - Run all integration tests"
        echo ""
        echo "Examples:"
        echo "  $0                                              # Run client tests (default namespace)"
        echo "  $0 client test_list_all_endpoints               # Run specific client test"
        echo "  $0 client test_list_all_endpoints my-namespace  # Run test, check my-namespace"
        echo "  $0 raw test_list_endpointslices                 # Run specific raw API test"
        echo "  $0 all \"\" my-namespace                          # Run all tests, check my-namespace"
        exit 1
        ;;
esac

echo ""
echo "✅ Tests completed"
echo ""
echo "Note: Tests check for resources in namespace: $NAMESPACE"
echo "      The actual KubeDiscoveryClient namespace is determined by POD_NAMESPACE env var in test code"

