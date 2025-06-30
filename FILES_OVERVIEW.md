# Dynamo Setup Files Overview

Essential files for Dynamo operator setup and testing.

## 📖 Main Documentation
- **`COMPLETE_SETUP_GUIDE.md`** - Complete step-by-step setup guide from git clone to working deployment

## 🚀 Deployment Files
- **`dynamo-operator-deployment.yaml`** - Dynamo operator infrastructure (namespace, RBAC, deployment)
- **`simple-operator-deployment.yaml`** - Simple aggregated inference model
- **`dynamo-disagg-simple.yaml`** - Disaggregated model with separate prefill/decode workers

## 🧪 Testing Files
- **`test-http-server.yaml`** - Working HTTP server with nginx and chat completions endpoints
- **`mock-operator-job.yaml`** - Mock operator that creates pods from DynamoGraphDeployments
- **`test-curl-script.sh`** - Automated HTTP endpoint testing script

## 🎯 Quick Start Commands

```bash
# 1. Install CRDs
kubectl apply -f deploy/cloud/operator/config/crd/bases/

# 2. Deploy operator
kubectl apply -f dynamo-operator-deployment.yaml

# 3. Create test namespace
kubectl create namespace dynamo-cloud

# 4. Deploy test HTTP server
kubectl apply -f test-http-server.yaml

# 5. Deploy disaggregated model
kubectl apply -f dynamo-disagg-simple.yaml
kubectl apply -f mock-operator-job.yaml

# 6. Test endpoints
./test-curl-script.sh
```

## 📊 What Gets Created

### Operator Infrastructure
- `dynamo-system` namespace
- Operator pod with RBAC permissions
- CRDs for custom resources

### Test Deployments
- HTTP test server with working endpoints
- Disaggregated inference pods:
  - Frontend (HTTP API)
  - PrefillWorker (prompt processing)
  - VllmWorker (token generation, 2 replicas)
  - Processor (request routing)

All files are production-ready templates that can be modified for real inference workloads.