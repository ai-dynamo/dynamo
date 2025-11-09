#!/bin/bash
# Create a pod and service for local testing with DYN_LOCAL_KUBE_TEST
# The pod name will be in format: dynamo-test-worker-<port>
# This allows the discovery client to connect to localhost:<port> for the metadata server

set -e

# Parse arguments
PORT="${1:-9000}"
K8S_NAMESPACE="${2:-discovery}"
DYNAMO_NAMESPACE="${3:-hello_world}"
DYNAMO_COMPONENT="${4:-backend}"

if [ -z "$PORT" ]; then
    echo "Usage: $0 <port> [k8s-namespace] [dynamo-namespace] [dynamo-component]"
    echo ""
    echo "Creates a pod and service that will be discovered by the Kubernetes client."
    echo "When DYN_LOCAL_KUBE_TEST is set, the client will connect to localhost:<port>"
    echo "for the metadata endpoint instead of the pod IP."
    echo ""
    echo "Arguments:"
    echo "  port              - Port number to use (required)"
    echo "  k8s-namespace     - Kubernetes namespace (default: discovery)"
    echo "  dynamo-namespace  - Dynamo namespace label (default: hello_world)"
    echo "  dynamo-component  - Dynamo component label (default: backend)"
    echo ""
    echo "Examples:"
    echo "  $0 8080                                    # backend component (default)"
    echo "  $0 8081 discovery                          # backend in discovery namespace"
    echo "  $0 8082 discovery hello_world backend      # Explicit backend component"
    echo "  $0 8083 discovery hello_world prefill      # prefill component"
    echo "  $0 8084 discovery dynamo frontend          # frontend component"
    echo ""
    echo "After creating the pod, run your metadata server locally:"
    echo "  # In one terminal:"
    echo "  your-metadata-server --port $PORT"
    echo ""
    echo "  # In another terminal:"
    echo "  export DYN_LOCAL_KUBE_TEST=1"
    echo "  cargo test --test kube_client_integration test_watch_all_endpoints -- --ignored --nocapture"
    exit 1
fi

POD_NAME="dynamo-test-worker-${PORT}"
SERVICE_NAME="dynamo-test-${DYNAMO_COMPONENT}"

echo "🚀 Creating local test resources in K8s namespace: $K8S_NAMESPACE"
echo "   Pod name: $POD_NAME"
echo "   Service name: $SERVICE_NAME (component: $DYNAMO_COMPONENT)"
echo "   Port: $PORT"
echo "   Dynamo namespace: $DYNAMO_NAMESPACE"
echo "   Dynamo component: $DYNAMO_COMPONENT"
echo ""

# Create namespace if it doesn't exist
if ! kubectl get namespace "$K8S_NAMESPACE" &> /dev/null; then
    echo "📦 Creating Kubernetes namespace: $K8S_NAMESPACE"
    kubectl create namespace "$K8S_NAMESPACE"
fi

# Create the pod and service using kubectl
cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $K8S_NAMESPACE
  labels:
    app: dynamo-local-test
    local-test-port: "$PORT"
    dynamo.nvidia.com/namespace: "$DYNAMO_NAMESPACE"
    dynamo.nvidia.com/component: "$DYNAMO_COMPONENT"
spec:
  containers:
  - name: worker
    image: nginx:alpine
    ports:
    - containerPort: 80
      name: http
      protocol: TCP
    readinessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 2
      periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: $SERVICE_NAME
  namespace: $K8S_NAMESPACE
  labels:
    app: dynamo-local-test
    dynamo.nvidia.com/namespace: "$DYNAMO_NAMESPACE"
    dynamo.nvidia.com/component: "$DYNAMO_COMPONENT"
spec:
  selector:
    app: dynamo-local-test
    dynamo.nvidia.com/namespace: "$DYNAMO_NAMESPACE"
    dynamo.nvidia.com/component: "$DYNAMO_COMPONENT"
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
    name: http
  type: ClusterIP
EOF

echo ""
echo "✅ Resources created: $POD_NAME and $SERVICE_NAME"
echo ""
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod/$POD_NAME --namespace="$K8S_NAMESPACE" --timeout=60s

echo ""
echo "📊 Resource status:"
kubectl get pod/$POD_NAME --namespace="$K8S_NAMESPACE"
kubectl get service/$SERVICE_NAME --namespace="$K8S_NAMESPACE"
kubectl get endpointslices -l kubernetes.io/service-name=$SERVICE_NAME --namespace="$K8S_NAMESPACE"

echo ""
echo "✅ Resources are ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Local Testing Instructions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Start your metadata server locally on port $PORT:"
echo "   export PORT=$PORT"
echo "   export DYN_SYSTEM_PORT=\$PORT"
echo "   export POD_NAME=$POD_NAME"
echo "   export POD_NAMESPACE=$K8S_NAMESPACE"
echo "   export DYN_DISCOVERY_BACKEND=kubernetes"
echo "   python3 -m your_app"
echo ""
echo "2. In another terminal, run your client with DYN_LOCAL_KUBE_TEST:"
echo "   export DYN_LOCAL_KUBE_TEST=1"
echo "   export POD_NAMESPACE=$K8S_NAMESPACE"
echo "   export DYN_DISCOVERY_BACKEND=kubernetes"
echo "   python3 -m your_client"
echo ""
echo "3. The client will discover the pod and connect to localhost:$PORT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To delete these resources:"
echo "  kubectl delete pod/$POD_NAME --namespace=$K8S_NAMESPACE"
echo "  kubectl delete service/$SERVICE_NAME --namespace=$K8S_NAMESPACE"
echo ""
echo "Or delete all local test resources:"
echo "  kubectl delete pods,services -l app=dynamo-local-test --namespace=$K8S_NAMESPACE"
echo ""

