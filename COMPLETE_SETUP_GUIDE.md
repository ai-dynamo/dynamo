# Complete Dynamo Operator Setup Guide

A complete guide to deploy Dynamo operator and test inference models from git clone to working deployment.

## 🚀 Quick Start

This guide will take you from git clone to a working Dynamo operator with test inference models in ~10 minutes.

## 📋 Prerequisites

- Kubernetes cluster (k3d, minikube, or cloud)
- `kubectl` configured and connected
- `git` installed

## 🔧 Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

# Verify kubectl connection
kubectl cluster-info
```

## 🏗️ Step 2: Install Dynamo CRDs

```bash
# Install Custom Resource Definitions
kubectl apply -f deploy/cloud/operator/config/crd/bases/

# Verify CRDs are installed
kubectl get crd | grep nvidia.com
# Expected: dynamocomponentdeployments.nvidia.com, dynamocomponents.nvidia.com, dynamographdeployments.nvidia.com
```

## 🤖 Step 3: Deploy Dynamo Operator

```bash
# Deploy the operator infrastructure
kubectl apply -f dynamo-operator-deployment.yaml

# Wait for operator to be ready
kubectl wait --for=condition=ready pod -l control-plane=controller-manager -n dynamo-system --timeout=60s

# Verify operator is running
kubectl get pods -n dynamo-system
```

## 🧪 Step 4: Deploy Test HTTP Server

```bash
# Create namespace for test deployments
kubectl create namespace dynamo-cloud

# Deploy test HTTP server (for endpoint testing)
kubectl apply -f test-http-server.yaml

# Wait for HTTP server to be ready
kubectl wait --for=condition=ready pod -l app=dynamo-http-frontend -n dynamo-cloud --timeout=60s
```

## 📝 Step 5: Deploy Test Models

### Option A: Simple Aggregated Model
```bash
# Deploy simple aggregated inference model
kubectl apply -f simple-operator-deployment.yaml

# Verify deployment
kubectl get dynamographdeployments -n dynamo-cloud
```

### Option B: Disaggregated Model (Separate Prefill/Decode)
```bash
# Deploy disaggregated inference model with separate workers
kubectl apply -f dynamo-disagg-simple.yaml

# Deploy mock operator to create actual pods
kubectl apply -f mock-operator-job.yaml

# Wait for deployments to be created
sleep 30
```

## 🔍 Step 6: Verify Deployments

```bash
# Check all resources
kubectl get all -n dynamo-cloud

# Check disaggregated pods (if deployed)
kubectl get pods -n dynamo-cloud | grep disagg-simple
# Expected:
# disagg-simple-llm-frontend-*      1/1 Running  # Frontend API
# disagg-simple-llm-prefillworker-* 1/1 Running  # Prefill worker
# disagg-simple-llm-vllmworker-*    1/1 Running  # Decode worker (2 replicas)
# disagg-simple-llm-processor-*     1/1 Running  # Request processor
```

## 🧪 Step 7: Test the Deployment

### HTTP Endpoint Testing
```bash
# Start port-forward for HTTP testing
kubectl port-forward -n dynamo-cloud svc/dynamo-http-frontend-service 8000:8000 &

# Test health endpoint
curl http://localhost:8000/health
# Expected: {"healthy": true}

# Test status endpoint
curl http://localhost:8000/
# Expected: {"status": "ok", "service": "dynamo-frontend", ...}
```

### Chat Completions Testing
```bash
# Test OpenAI-compatible chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Expected response:
# {
#   "id": "test-123",
#   "object": "chat.completion",
#   "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#   "choices": [{"message": {"role": "assistant", "content": "Hello! This is a test response from Dynamo frontend."}}]
# }
```

### Automated Testing
```bash
# Run comprehensive HTTP tests
./test-curl-script.sh
```

## 📊 Step 8: Verify Architecture

### Check Disaggregated Architecture
```bash
# Verify separate prefill and decode workers
kubectl get deployments -n dynamo-cloud | grep disagg-simple
# Expected scaling:
# disagg-simple-llm-prefillworker   1/1  # 1 prefill worker
# disagg-simple-llm-vllmworker      2/2  # 2 decode workers

# Check worker logs
kubectl logs -n dynamo-cloud -l service=prefillworker --tail=3
# Expected: "🚀 PREFILL WORKER: Processing prompt inputs and initial tokens"

kubectl logs -n dynamo-cloud -l service=vllmworker --tail=3  
# Expected: "🔄 DECODE WORKER: Generating tokens sequentially"
```

## 🎯 What You've Achieved

✅ **Dynamo Operator Deployed**: Custom resource controller managing inference graphs  
✅ **CRDs Installed**: DynamoGraphDeployment, DynamoComponent definitions  
✅ **Test Models Running**: Both aggregated and disaggregated architectures  
✅ **HTTP API Working**: OpenAI-compatible chat completions endpoint  
✅ **Architecture Verified**: Separate prefill and decode workers  

## 📁 Key Files

| File | Purpose |
|------|---------|
| `dynamo-operator-deployment.yaml` | Operator infrastructure |
| `simple-operator-deployment.yaml` | Simple aggregated model |
| `dynamo-disagg-simple.yaml` | Disaggregated model (prefill/decode) |
| `test-http-server.yaml` | HTTP test server |
| `mock-operator-job.yaml` | Mock operator for testing |
| `test-curl-script.sh` | Automated HTTP testing |

## 🔧 Production Migration

To move from test to production:

1. **Replace Test Images**: Build actual Dynamo runtime images
   ```bash
   ./container/build.sh --framework vllm
   ```

2. **Deploy Real Operator**: Use production operator image
   ```bash
   # Update dynamo-operator-deployment.yaml with real operator image
   # image: nvcr.io/nvidia/dynamo-operator:latest
   ```

3. **Configure GPU Resources**: Add GPU resource requests
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: "1"
   ```

4. **Set up Model Storage**: Configure PVCs for model weights
   ```yaml
   volumes:
   - name: model-storage
     persistentVolumeClaim:
       claimName: model-pvc
   ```

## 🐛 Troubleshooting

### Common Issues

1. **CRDs Not Found**
   ```bash
   kubectl get crd | grep nvidia.com
   # If empty, re-run: kubectl apply -f deploy/cloud/operator/config/crd/bases/
   ```

2. **Operator Pod Not Running**
   ```bash
   kubectl logs -n dynamo-system deployment/dynamo-controller-manager
   kubectl describe pod -n dynamo-system -l control-plane=controller-manager
   ```

3. **Port-Forward Connection Refused**
   ```bash
   # Check if pod is running
   kubectl get pods -n dynamo-cloud
   # Use working HTTP server instead of mock pods
   kubectl port-forward -n dynamo-cloud svc/dynamo-http-frontend-service 8000:8000
   ```

4. **No Pods Created from DynamoGraphDeployment**
   ```bash
   # Check if mock operator job ran
   kubectl logs -n dynamo-system job/mock-dynamo-operator
   # Re-run if needed: kubectl apply -f mock-operator-job.yaml
   ```

### Cleanup Commands
```bash
# Clean up test deployments
kubectl delete namespace dynamo-cloud
kubectl delete namespace dynamo-system

# Remove CRDs
kubectl delete -f deploy/cloud/operator/config/crd/bases/
```

## 🚀 Next Steps

- **Scale Workers**: Adjust replicas in DynamoGraphDeployment
- **Add Monitoring**: Deploy Prometheus/Grafana for metrics
- **Configure Ingress**: Expose services externally
- **Production Models**: Replace with real LLM inference engines
- **GPU Scheduling**: Configure GPU resource allocation

## 📚 Reference

- [Dynamo Documentation](https://docs.nvidia.com/dynamo/latest/index.html)
- [Kubernetes Operators](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)

---

**🎉 Congratulations!** You now have a working Dynamo operator setup with disaggregated inference architecture and HTTP API endpoints.