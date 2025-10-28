#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy the Dynamo scheduler and DynamoGraphDeployments

set -e

NAMESPACE="${NAMESPACE:-default}"

echo "======================================"
echo "Deploying Dynamo Scheduler"
echo "======================================"
echo "Namespace: ${NAMESPACE}"
echo ""
echo "Note: Deploying to namespace '${NAMESPACE}'"
echo "To deploy to a different namespace: NAMESPACE=my-namespace ./deploy.sh"
echo ""

# Check if deployment.yaml exists
if [ ! -f "deployment.yaml" ]; then
    echo "ERROR: deployment.yaml not found!"
    exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found. Please install kubectl."
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster."
    exit 1
fi

# Check for HuggingFace secret
if ! kubectl get secret hf-token-secret -n ${NAMESPACE} &> /dev/null; then
    echo "WARNING: hf-token-secret not found in namespace ${NAMESPACE}"
    echo "Create it with:"
    echo "  kubectl create secret generic hf-token-secret \\"
    echo "    --from-literal=token=your_huggingface_token \\"
    echo "    -n ${NAMESPACE}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create ConfigMap from local scheduler directory
echo ""
echo "Creating ConfigMap from local scheduler code..."
kubectl create configmap dynamo-scheduler-code \
  --from-file=scheduler/ \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f - -n ${NAMESPACE}

echo "ConfigMap created successfully"

# Apply deployment
echo ""
echo "Applying deployment to namespace ${NAMESPACE}..."
kubectl apply -f deployment.yaml -n ${NAMESPACE}

# Wait for scheduler deployment
echo ""
echo "Waiting for scheduler deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/dynamo-scheduler -n ${NAMESPACE} || true

# Wait for DynamoGraphDeployments
echo ""
echo "Waiting for DynamoGraphDeployments to be ready..."
echo "(This may take several minutes as models are downloaded and loaded)"
kubectl wait --for=jsonpath='{.status.state}'=Ready \
  dynamographdeployment/llama-deployment-a \
  dynamographdeployment/llama-deployment-b \
  -n ${NAMESPACE} \
  --timeout=1200s || true

# Get status
echo ""
echo "======================================"
echo "Deployment Status"
echo "======================================"
echo ""
kubectl get deployment dynamo-scheduler -n ${NAMESPACE}
echo ""
kubectl get dynamographdeployment -n ${NAMESPACE}
echo ""
kubectl get svc dynamo-scheduler -n ${NAMESPACE}

# Get scheduler URL
echo ""
echo "======================================"
echo "Access Information"
echo "======================================"
echo ""

# Check if Ingress is deployed
INGRESS_HOST=$(kubectl get ingress dynamo-scheduler-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}' 2>/dev/null)
if [ -n "$INGRESS_HOST" ]; then
    echo "Scheduler accessible via Ingress:"
    echo "  Host: ${INGRESS_HOST}"
    echo ""
    echo "Get Ingress IP:"
    INGRESS_IP=$(kubectl get ingress dynamo-scheduler-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -n "$INGRESS_IP" ]; then
        echo "  Ingress IP: ${INGRESS_IP}"
        echo ""
        echo "Add to /etc/hosts:"
        echo "  ${INGRESS_IP} ${INGRESS_HOST}"
        echo ""
        echo "Test with:"
        echo "  curl http://${INGRESS_HOST}/health"
        echo "  curl http://${INGRESS_HOST}/frontends"
    else
        echo "  Ingress IP not yet assigned. Check with:"
        echo "  kubectl get ingress dynamo-scheduler-ingress -n ${NAMESPACE}"
    fi
fi

# Check if NodePort is deployed
NODEPORT=$(kubectl get svc dynamo-scheduler-nodeport -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null)
if [ -n "$NODEPORT" ]; then
    echo ""
    echo "Scheduler accessible via NodePort:"
    echo "  Port: ${NODEPORT}"
    echo ""
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null)
    if [ -z "$NODE_IP" ]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    fi
    if [ -n "$NODE_IP" ]; then
        echo "  Node IP: ${NODE_IP}"
        echo ""
        echo "Test with:"
        echo "  curl http://${NODE_IP}:${NODEPORT}/health"
        echo "  curl http://${NODE_IP}:${NODEPORT}/frontends"
    else
        echo "Get node IP with: kubectl get nodes -o wide"
    fi
fi

# Fallback to port-forward
if [ -z "$INGRESS_HOST" ] && [ -z "$NODEPORT" ]; then
    echo "Using ClusterIP - Access via port-forward:"
    echo "  kubectl port-forward svc/dynamo-scheduler 8080:80 -n ${NAMESPACE}"
    echo ""
    echo "Then test with:"
    echo "  curl http://localhost:8080/health"
    echo "  curl http://localhost:8080/frontends"
fi

echo ""
echo "View logs:"
echo "  kubectl logs -l app=dynamo-scheduler -n ${NAMESPACE} --tail=50 -f"
echo ""

