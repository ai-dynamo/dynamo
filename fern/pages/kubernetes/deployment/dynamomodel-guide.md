---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Managing Models with DynamoModel"
---

## Overview

`DynamoModel` is a Kubernetes Custom Resource that represents a machine learning model deployed on Dynamo. It enables you to:

- **Deploy LoRA adapters** on top of running base models
- **Track model endpoints** and their readiness across your cluster
- **Manage model lifecycle** declaratively with Kubernetes

DynamoModel works alongside `DynamoGraphDeployment` (DGD) or `DynamoComponentDeployment` (DCD) resources. While DGD/DCD deploy the inference infrastructure (pods, services), DynamoModel handles model-specific operations like loading LoRA adapters.

## Quick Start

### Prerequisites

Before creating a DynamoModel, you need:

1. A running `DynamoGraphDeployment` or `DynamoComponentDeployment`
2. Components configured with `modelRef` pointing to your base model
3. Pods are ready and serving your base model

For complete setup including DGD configuration, see [Integration with DynamoGraphDeployment](#integration-with-dynamographdeployment).

### Deploy a LoRA Adapter

**1. Create your DynamoModel:**

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: my-lora
  namespace: dynamo-system
spec:
  modelName: my-custom-lora
  baseModelName: Qwen/Qwen3-0.6B  # Must match modelRef.name in your DGD
  modelType: lora
  source:
    uri: s3://my-bucket/loras/my-lora
```

**2. Apply and verify:**

```bash
# Apply the DynamoModel
kubectl apply -f my-lora.yaml

# Check status
kubectl get dynamomodel my-lora
```

**Expected output:**
```
NAME      TOTAL   READY   AGE
my-lora   2       2       30s
```

That's it! The operator automatically discovers endpoints and loads the LoRA.

For detailed status monitoring, see [Monitoring & Operations](#monitoring--operations).

## Understanding DynamoModel

### Model Types

DynamoModel supports three model types:

| Type | Description | Use Case |
|------|-------------|----------|
| **`base`** | Reference to an existing base model | Tracking endpoints for a base model (default) |
| **`lora`** | LoRA adapter that extends a base model | Deploy fine-tuned adapters on existing models |
| **`adapter`** | Generic model adapter | Future extensibility for other adapter types |

Most users will use **`lora`** to deploy fine-tuned models on top of their base model deployments.

### How It Works

When you create a DynamoModel, the operator:

1. **Discovers endpoints**: Finds all pods running your `baseModelName` (by matching `modelRef.name` in DGD/DCD)
2. **Creates service**: Automatically creates a Kubernetes Service to track these pods
3. **Loads LoRA**: Calls the LoRA load API on each endpoint (for `lora` type)
4. **Updates status**: Reports which endpoints are ready

**Key linkage:**
```yaml
# DGD modelRef.name ↔ DynamoModel baseModelName must match
Worker:
  modelRef:
    name: Qwen/Qwen3-0.6B
---
spec:
  baseModelName: Qwen/Qwen3-0.6B
```

## Configuration Overview

DynamoModel requires just a few key fields to deploy a model or adapter:

| Field | Required | Purpose | Example |
|-------|----------|---------|---------|
| `modelName` | Yes | Model identifier | `my-custom-lora` |
| `baseModelName` | Yes | Links to DGD modelRef | `Qwen/Qwen3-0.6B` |
| `modelType` | No | Type: base/lora/adapter | `lora` (default: `base`) |
| `source.uri` | For LoRA | Model location | `s3://bucket/path` or `hf://org/model` |

**Example minimal LoRA configuration:**
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: my-lora
spec:
  modelName: my-custom-lora
  baseModelName: Qwen/Qwen3-0.6B
  modelType: lora
  source:
    uri: s3://my-bucket/my-lora
```

**For complete field specifications, validation rules, and all options, see:**
📖 [DynamoModel API Reference](../api-reference.md#dynamomodel)

### Status Summary

The status shows discovered endpoints and their readiness:

```bash
kubectl get dynamomodel my-lora
```

**Key status fields:**
- `totalEndpoints` / `readyEndpoints`: Counts of discovered vs ready endpoints
- `endpoints[]`: List with addresses, pod names, and ready status
- `conditions`: Standard Kubernetes conditions (EndpointsReady, ServicesFound)

For detailed status usage, see the [Monitoring & Operations](#monitoring--operations) section below

## Common Use Cases

### Use Case 1: S3-Hosted LoRA Adapter

Deploy a LoRA adapter stored in an S3 bucket.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: customer-support-lora
  namespace: production
spec:
  modelName: customer-support-adapter-v1
  baseModelName: meta-llama/Llama-3.3-70B-Instruct
  modelType: lora
  source:
    uri: s3://my-models-bucket/loras/customer-support/v1
```

**Prerequisites:**
- S3 bucket accessible from your pods (IAM role or credentials)
- Base model `meta-llama/Llama-3.3-70B-Instruct` running via DGD/DCD

**Verification:**
```bash
# Check LoRA is loaded
kubectl get dynamomodel customer-support-lora -o jsonpath='{.status.readyEndpoints}'
# Should output: 2 (or your number of replicas)

# View which pods are serving
kubectl get dynamomodel customer-support-lora -o jsonpath='{.status.endpoints[*].podName}'
```

### Use Case 2: HuggingFace-Hosted LoRA

Deploy a LoRA adapter from HuggingFace Hub.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: multilingual-lora
  namespace: dynamo-system
spec:
  modelName: multilingual-adapter
  baseModelName: Qwen/Qwen3-0.6B
  modelType: lora
  source:
    uri: hf://myorg/qwen-multilingual-lora@v1.0.0  # Optional: @revision
```

**Prerequisites:**
- HuggingFace Hub accessible from your pods
- If private repo: HF token configured as secret and mounted in pods
- Base model `Qwen/Qwen3-0.6B` running via DGD/DCD

**With HuggingFace token:**
```yaml
# In your DGD/DCD
spec:
  services:
    worker:
      envFromSecret: hf-token-secret  # Provides HF_TOKEN env var
      modelRef:
        name: Qwen/Qwen3-0.6B
      # ... rest of config
```

### Use Case 3: Multiple LoRAs on Same Base Model

Deploy multiple LoRA adapters on the same base model deployment.

```yaml
---
# LoRA for customer support
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: support-lora
spec:
  modelName: support-adapter
  baseModelName: Qwen/Qwen3-0.6B
  modelType: lora
  source:
    uri: s3://models/support-lora

---
# LoRA for code generation
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: code-lora
spec:
  modelName: code-adapter
  baseModelName: Qwen/Qwen3-0.6B  # Same base model
  modelType: lora
  source:
    uri: s3://models/code-lora
```

Both LoRAs will be loaded on all pods serving `Qwen/Qwen3-0.6B`. Your application can then route requests to the appropriate adapter.

## Monitoring & Operations

### Checking Status

**Quick status check:**
```bash
kubectl get dynamomodel
```

**Example output:**
```
NAME              TOTAL   READY   AGE
my-lora           2       2       5m
customer-lora     4       3       2h
```

**Detailed status:**
```bash
kubectl describe dynamomodel my-lora
```

**Example output:**
```
Name:         my-lora
Namespace:    dynamo-system
Spec:
  Model Name:       my-custom-lora
  Base Model Name:  Qwen/Qwen3-0.6B
  Model Type:       lora
  Source:
    Uri:  s3://my-bucket/my-lora
Status:
  Ready Endpoints:  2
  Total Endpoints:  2
  Endpoints:
    Address:   http://10.0.1.5:9090
    Pod Name:  worker-0
    Ready:     true
    Address:   http://10.0.1.6:9090
    Pod Name:  worker-1
    Ready:     true
  Conditions:
    Type:     EndpointsReady
    Status:   True
    Reason:   EndpointsDiscovered
Events:
  Type    Reason              Message
  ----    ------              -------
  Normal  EndpointsReady      Discovered 2 ready endpoints for base model Qwen/Qwen3-0.6B
```

### Understanding Readiness

An endpoint is **ready** when:
1. The pod is running and healthy
2. The LoRA load API call succeeded

**Condition states:**
- `EndpointsReady=True`: All endpoints are ready (full availability)
- `EndpointsReady=False, Reason=NotReady`: Not all endpoints ready (check message for counts)
- `EndpointsReady=False, Reason=NoEndpoints`: No endpoints found

When `readyEndpoints < totalEndpoints`, the operator automatically retries loading every 30 seconds.

### Viewing Endpoints

**Get endpoint addresses:**
```bash
kubectl get dynamomodel my-lora -o jsonpath='{.status.endpoints[*].address}' | tr ' ' '\n'
```

**Output:**
```
http://10.0.1.5:9090
http://10.0.1.6:9090
```

**Get endpoint pod names:**
```bash
kubectl get dynamomodel my-lora -o jsonpath='{.status.endpoints[*].podName}' | tr ' ' '\n'
```

**Check readiness of each endpoint:**
```bash
kubectl get dynamomodel my-lora -o json | jq '.status.endpoints[] | {podName, ready}'
```

**Output:**
```json
{
  "podName": "worker-0",
  "ready": true
}
{
  "podName": "worker-1",
  "ready": true
}
```

### Updating a Model

To update a LoRA (e.g., deploy a new version):

```bash
# Edit the source URI
kubectl edit dynamomodel my-lora

# Or apply an updated YAML
kubectl apply -f my-lora-v2.yaml
```

The operator will detect the change and reload the LoRA on all endpoints.

### Deleting a Model

```bash
kubectl delete dynamomodel my-lora
```

For LoRA models, the operator will:
1. Unload the LoRA from all endpoints
2. Clean up associated resources
3. Remove the DynamoModel CR

The base model deployment (DGD/DCD) continues running normally.

## Troubleshooting

### No Endpoints Found

**Symptom:**
```yaml
status:
  totalEndpoints: 0
  readyEndpoints: 0
  conditions:
  - type: EndpointsReady
    status: "False"
    reason: NoEndpoints
    message: "No endpoint slices found for base model Qwen/Qwen3-0.6B"
```

**Common Causes:**

1. **Base model deployment not running**
   ```bash
   # Check if pods exist
   kubectl get pods -l nvidia.com/dynamo-component-type=worker
   ```
   **Solution:** Deploy your DGD/DCD first, wait for pods to be ready.

2. **`baseModelName` mismatch**
   ```bash
   # Check modelRef in your DGD
   kubectl get dynamographdeployment my-deployment -o yaml | grep -A2 modelRef
   ```
   **Solution:** Ensure `baseModelName` in DynamoModel exactly matches `modelRef.name` in DGD.

3. **Pods not ready**
   ```bash
   # Check pod status
   kubectl get pods -l nvidia.com/dynamo-component-type=worker
   ```
   **Solution:** Wait for pods to reach `Running` and `Ready` state.

4. **Wrong namespace**
   **Solution:** Ensure DynamoModel is in the same namespace as your DGD/DCD.

### LoRA Load Failures

**Symptom:**
```yaml
status:
  totalEndpoints: 2
  readyEndpoints: 0  # ← No endpoints ready despite pods existing
  conditions:
  - type: EndpointsReady
    status: "False"
    reason: NoReadyEndpoints
```

**Common Causes:**

1. **Source URI not accessible**
   ```bash
   # Check operator logs
   kubectl logs -n dynamo-system deployment/dynamo-operator-controller-manager -f | grep "Failed to load"
   ```
   **Solution:**
   - For S3: Verify bucket permissions, IAM role, credentials
   - For HuggingFace: Verify token is valid, repo exists and is accessible

2. **Invalid LoRA format**
   **Solution:** Ensure your LoRA weights are in the format expected by your backend framework (vLLM, SGLang, etc.)

3. **Endpoint API errors**
   ```bash
   # Check operator logs for HTTP errors
   kubectl logs -n dynamo-system deployment/dynamo-operator-controller-manager | grep "error"
   ```
   **Solution:** Check the backend framework's logs in the worker pods:
   ```bash
   kubectl logs worker-0
   ```

4. **Out of memory**
   **Solution:** LoRA adapters require additional memory. Increase memory limits in your DGD:
   ```yaml
   resources:
     limits:
       memory: "32Gi"  # Increase if needed
   ```

### Status Shows Not Ready

**Symptom:**
Some endpoints remain not ready for extended periods.

**Diagnosis:**
```bash
# Check which endpoints are not ready
kubectl get dynamomodel my-lora -o json | jq '.status.endpoints[] | select(.ready == false)'

# View operator logs for that specific pod
kubectl logs -n dynamo-system deployment/dynamo-operator-controller-manager | grep "worker-0"

# Check the worker pod logs
kubectl logs worker-0 | tail -50
```

**Common Causes:**

1. **Network issues**: Pod can't reach S3/HuggingFace
2. **Resource constraints**: Pod is OOMing or being throttled
3. **API endpoint not responding**: Backend framework isn't serving the LoRA API

**When to wait vs investigate:**
- **Wait**: If readyEndpoints is increasing over time (LoRAs loading progressively)
- **Investigate**: If stuck at same readyEndpoints for >5 minutes

### Viewing Events and Logs

**Check events:**
```bash
kubectl describe dynamomodel my-lora | tail -20
```

**View operator logs:**
```bash
# Follow logs
kubectl logs -n dynamo-system deployment/dynamo-operator-controller-manager -f

# Filter for specific model
kubectl logs -n dynamo-system deployment/dynamo-operator-controller-manager | grep "my-lora"
```

**Common events and messages:**

| Event/Message | Meaning | Action |
|---------------|---------|--------|
| `EndpointsReady` | All endpoints are ready | ✅ Good - full service availability |
| `NotReady` | Not all endpoints ready | ⚠️ Check readyEndpoints count - operator will retry |
| `PartialEndpointFailure` | Some endpoints failed to load | Check logs for errors |
| `NoEndpointsFound` | No pods discovered | Verify DGD running and modelRef matches |
| `EndpointDiscoveryFailed` | Can't query endpoints | Check operator RBAC permissions |
| `Successfully reconciled` | Reconciliation complete | ✅ Good |

## Integration with DynamoGraphDeployment

This section shows the complete end-to-end workflow for deploying base models and LoRA adapters together.

DynamoModel and DynamoGraphDeployment work together to provide complete model deployment:

- **DGD**: Deploys the infrastructure (pods, services, resources)
- **DynamoModel**: Manages model-specific operations (LoRA loading)

### Linking Models to Components

The connection is established through the `modelRef` field in your DGD:

**Complete example:**

```yaml
---
# 1. Deploy the base model infrastructure
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  backendFramework: vllm
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      dynamoNamespace: my-app
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest

    Worker:
      # This modelRef creates the link to DynamoModel
      modelRef:
        name: Qwen/Qwen3-0.6B  # ← Key linking field

      componentType: worker
      replicas: 2
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --tensor-parallel-size
            - "1"

---
# 2. Deploy LoRA adapters on top
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: my-lora
spec:
  modelName: my-custom-lora
  baseModelName: Qwen/Qwen3-0.6B  # ← Must match modelRef.name above
  modelType: lora
  source:
    uri: s3://my-bucket/loras/my-lora
```

### Deployment Workflow

**Recommended order:**

```bash
# 1. Deploy base model infrastructure
kubectl apply -f my-deployment.yaml

# 2. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-component-type=worker --timeout=5m

# 3. Deploy LoRA adapters
kubectl apply -f my-lora.yaml

# 4. Verify LoRA is loaded
kubectl get dynamomodel my-lora
```

**What happens behind the scenes:**

| Step | DGD | DynamoModel |
|------|-----|-------------|
| 1 | Creates pods with modelRef | - |
| 2 | Pods become running and ready | - |
| 3 | - | CR created, discovers endpoints via auto-created Service |
| 4 | - | Calls LoRA load API on each endpoint |
| 5 | - | All endpoints ready ✓ |

The operator automatically handles all service discovery - you don't configure services, labels, or selectors manually.

## API Reference

For complete field specifications, validation rules, and detailed type definitions, see:

**📖 [Dynamo CRD API Reference](../api-reference.md#dynamomodel)**

## Summary

DynamoModel provides declarative model management for Dynamo deployments:

✅ **Simple**: 2-step deployment of LoRA adapters
✅ **Automatic**: Endpoint discovery and loading handled by operator
✅ **Observable**: Rich status reporting and conditions
✅ **Integrated**: Works seamlessly with DynamoGraphDeployment

**Next Steps:**
- Try the [Quick Start](#quick-start) example
- Explore [Common Use Cases](#common-use-cases)
- Check the [API Reference](../api-reference.md#dynamomodel) for advanced configuration

