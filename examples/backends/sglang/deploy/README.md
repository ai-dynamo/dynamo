# SGLang Kubernetes Deployment Configurations

This directory contains Kubernetes Custom Resource Definition (CRD) templates for deploying SGLang inference graphs using the **DynamoGraphDeployment** resource.

## Available Deployment Patterns

### 1. **Aggregated Deployment** (`agg.yaml`)
Basic deployment pattern with frontend and a single decode worker.

**Architecture:**
- `Frontend`: OpenAI-compatible API server
- `ModelExpress`: Shared model caching service across workers
- `SGLangDecodeWorker`: Single worker handling both prefill and decode

### 2. **Aggregated Router Deployment** (`agg_router.yaml`)
Enhanced aggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: OpenAI-compatible API server with router mode enabled (`--router-mode kv`)
- `ModelExpress`: Shared model caching service across workers
- `SGLangDecodeWorker`: Single worker handling both prefill and decode

### 3. **Disaggregated Deployment** (`disagg.yaml`)
High-performance deployment with separated prefill and decode workers.

**Architecture:**
- `Frontend`: HTTP API server coordinating between workers
- `ModelExpress`: Shared model caching service across workers
- `SGLangDecodeWorker`: Specialized decode-only worker (`--disaggregation-mode decode`)
- `SGLangPrefillWorker`: Specialized prefill-only worker (`--disaggregation-mode prefill`)
- Communication via NIXL transfer backend (`--disaggregation-transfer-backend nixl`)

## CRD Structure

All templates use the **DynamoGraphDeployment** CRD:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: <deployment-name>
spec:
  services:
    <ServiceName>:
      # Service configuration
```

### Key Configuration Options

**Resource Management:**
```yaml
resources:
  requests:
    cpu: "10"
    memory: "20Gi"
    gpu: "1"
  limits:
    cpu: "10"
    memory: "20Gi"
    gpu: "1"
```

**Container Configuration:**
```yaml
extraPodSpec:
  mainContainer:
    image: my-registry/sglang-runtime:my-tag
    workingDir: /workspace/examples/backends/sglang
    args:
      - "python3"
      - "-m"
      - "dynamo.sglang"
      # Model-specific arguments
```

## Prerequisites

Before using these templates, ensure you have:

1. **Dynamo Cloud Platform installed** - See [Installing Dynamo Cloud](../../../../docs/kubernetes/installation_guide.md)
2. **Kubernetes cluster with GPU support**
3. **Container registry access** for SGLang runtime images
4. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)

### Persistent Volume Claim (PVC)

All templates expect a pre-created PVC named `model-cache-pvc` for the shared model cache used by ModelExpress and SGLang workers.

Apply the shared PVC once per namespace before deploying any graph:

```bash
kubectl apply -f model_cache_pvc.yaml -n $NAMESPACE
```

Note: If your cluster requires a specific storage class, edit `model_cache_pvc.yaml` to set `storageClassName` accordingly.

### Container Images

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd prefer to use your own registry, build and push your own image:

```bash
./container/build.sh --framework SGLANG
# Tag and push to your container registry
# Update the image references in the YAML files
```

## Usage

### 1. Choose Your Template
Select the deployment pattern that matches your requirements:
- Use `agg.yaml` for development/testing
- Use `agg_router.yaml` for production with load balancing
- Use `disagg.yaml` for maximum performance

### 2. Customize Configuration
Edit the template to match your environment:

```yaml
# Update image registry and tag
image: my-registry/sglang-runtime:my-tag

# Configure your model
args:
  - "--model-path"
  - "your-org/your-model"
  - "--served-model-name"
  - "your-org/your-model"
```

### 3. Deploy

Use the following command to deploy the deployment file.

First, create a secret for the HuggingFace token.
```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Then, deploy the model using the deployment file.

```bash
export DEPLOYMENT_FILE=agg.yaml

# Create the shared model cache PVC (run once per namespace)
kubectl apply -f model_cache_pvc.yaml -n $NAMESPACE

# Apply the SGLang deployment
kubectl apply -f $DEPLOYMENT_FILE -n ${NAMESPACE}
```

### 4. Using Custom Dynamo Frameworks Image for SGLang

To use a custom dynamo frameworks image for SGLang, you can update the deployment file using yq:

```bash
export DEPLOYMENT_FILE=agg.yaml
export FRAMEWORK_RUNTIME_IMAGE=<sglang-image>

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(FRAMEWORK_RUNTIME_IMAGE)' $DEPLOYMENT_FILE  > $DEPLOYMENT_FILE.generated
kubectl apply -f $DEPLOYMENT_FILE.generated -n $NAMESPACE
```

## Model Configuration

All templates use **Qwen/Qwen3-0.6B** as the default model. You can use any SGLang arguments and configuration. Key parameters include `--model-path`, `--served-model-name`, and disaggregation flags (see YAMLs for examples).

## Monitoring and Health

- **Frontend health endpoint**: `http://<frontend-service>:8000/health`
- **Liveness probes**: Check process health every 60s

## Further Reading

- **Deployment Guide**: [Creating Kubernetes Deployments](../../../../docs/kubernetes/deployment/create_deployment.md)
- **Quickstart**: [Deployment Quickstart](../../../../docs/kubernetes/README.md)
- **Platform Setup**: [Dynamo Cloud Installation](../../../../docs/kubernetes/installation_guide.md)
- **Examples**: [Deployment Examples](../../../../docs/examples/README.md)
- **Kubernetes CRDs**: [Custom Resources Documentation](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)

## Troubleshooting

Common issues and solutions:

1. **Pod fails to start**: Check image registry access and HuggingFace token secret
2. **GPU not allocated**: Verify cluster has GPU nodes and proper resource limits
3. **Health check failures**: Review model loading logs and increase `initialDelaySeconds`
4. **Out of memory**: Increase memory limits or reduce model batch size

For additional support, refer to the [deployment guide](../../../../docs/kubernetes/README.md).
