# SGLang Kubernetes Deployment Configurations

This directory contains Kubernetes Custom Resource Definition (CRD) templates for deploying SGLang inference graphs using the **DynamoGraphDeployment** resource.

## Available Deployment Patterns

### 1. **Aggregated Deployment** (`agg.yaml`)
Basic deployment pattern with frontend and a single decode worker.

**Architecture:**
- `Frontend`: OpenAI-compatible API server
- `SGLangDecodeWorker`: Single worker handling both prefill and decode

### 2. **Aggregated Router Deployment** (`agg_router.yaml`)
Enhanced aggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: OpenAI-compatible API server with router mode enabled (`--router-mode kv`)
- `SGLangDecodeWorker`: Single worker handling both prefill and decode

### 3. **Disaggregated Deployment** (`disagg.yaml`)**
High-performance deployment with separated prefill and decode workers.

**Architecture:**
- `Frontend`: HTTP API server coordinating between workers
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
    workingDir: /workspace/components/backends/sglang
    args:
      - "python3"
      - "-m"
      - "dynamo.sglang.worker"
      # Model-specific arguments
```

## Prerequisites

Before using these templates, ensure you have:

1. **Dynamo Cloud Platform installed** - See [Installing Dynamo Cloud](../../docs/guides/dynamo_deploy/dynamo_cloud.md)
2. **Kubernetes cluster with GPU support**
3. **Container registry access** for SGLang runtime images
4. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)

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
image: your-registry/sglang-runtime:your-tag

# Configure your model
args:
  - "--model-path"
  - "your-org/your-model"
  - "--served-model-name"
  - "your-org/your-model"
```

### 3. Deploy
```bash
kubectl apply -f <your-template>.yaml
```

## Model Configuration

All templates use **DeepSeek-R1-Distill-Llama-8B** as the default model. But you can use any sglang argument and configuration. Key parameters:

## Monitoring and Health

- **Frontend health endpoint**: `http://<frontend-service>:8000/health`
- **Liveness probes**: Check process health every 60s

## Further Reading

- **Deployment Guide**: [Creating Kubernetes Deployments](../../../../docs/guides/dynamo_deploy/create_deployment.md)
- **Quickstart**: [Deployment Quickstart](../../../../docs/guides/dynamo_deploy/quickstart.md)
- **Platform Setup**: [Dynamo Cloud Installation](../../../../docs/guides/dynamo_deploy/dynamo_cloud.md)
- **Examples**: [Deployment Examples](../../../../docs/examples/README.md)
- **Kubernetes CRDs**: [Custom Resources Documentation](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)

## Troubleshooting

Common issues and solutions:

1. **Pod fails to start**: Check image registry access and HuggingFace token secret
2. **GPU not allocated**: Verify cluster has GPU nodes and proper resource limits
3. **Health check failures**: Review model loading logs and increase `initialDelaySeconds`
4. **Out of memory**: Increase memory limits or reduce model batch size

For additional support, refer to the [deployment troubleshooting guide](../../docs/guides/dynamo_deploy/quickstart.md#troubleshooting).
