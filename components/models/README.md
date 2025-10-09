# DynamoModel Definitions

This directory contains pre-configured `DynamoModel` resources for commonly used models.

## Available Models

### Qwen 3 0.6B
**File:** `qwen3-0.6b.yaml`
- **Size:** ~2GB
- **Use Case:** Testing, development, lightweight inference
- **Public:** Yes (no authentication required)

```bash
kubectl apply -f qwen3-0.6b.yaml -n your-namespace
```

### Llama 3.3 70B Instruct
**File:** `llama-3-70b.yaml`
- **Size:** ~140GB
- **Use Case:** Production inference, high-quality responses
- **Public:** Gated (requires HuggingFace token)

```bash
# Create secret first
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n your-namespace

kubectl apply -f llama-3-70b.yaml -n your-namespace
```

## Usage

### 1. Deploy Model

```bash
kubectl apply -f <model-file>.yaml -n your-namespace
```

### 2. Check Status

```bash
# Watch model download progress
kubectl get dynamomodel -n your-namespace -w

# Check detailed status
kubectl describe dynamomodel qwen3-0.6b -n your-namespace

# View download logs
kubectl logs job/qwen3-0.6b-download -n your-namespace -f
```

### 3. Reference in Deployment

Once the model state is "Ready", reference it in your `DynamoGraphDeployment`:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  modelRef: qwen3-0.6b  # Reference the model by name
  backendFramework: vllm
  services:
    VllmWorker:
      replicas: 1
      resources:
        limits:
          nvidia.com/gpu: "1"
```

## Customization

Update the following fields based on your cluster:

- **`storageClass`**: Use your cluster's available storage class
- **`size`**: Adjust based on model requirements
- **`version`**: Pin to specific commit SHA for production
- **`secretRef`**: Add if model requires authentication

## Adding New Models

Create a new YAML file following this template:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: my-model
spec:
  name: organization/model-name
  version: commit-sha  # Optional
  sourceURL: hf://organization/model-name
  secretRef: secret-name  # Optional
  pvc:
    create: true
    storageClass: your-storage-class
    size: XXGi
    volumeAccessMode: ReadWriteMany
```

