# vLLM Kubernetes Deployment Configurations

This directory contains Kubernetes Custom Resource Definition (CRD) templates for deploying vLLM inference graphs using the **DynamoGraphDeployment** resource.

## Available Deployment Patterns

### 1. **Aggregated Deployment** (`agg.yaml`)
Basic deployment pattern with frontend and a single decode worker.

**Architecture:**
- `Frontend`: OpenAI-compatible API server (with kv router mode disabled)
- `VLLMDecodeWorker`: Single worker handling both prefill and decode

### 2. **Aggregated Router Deployment** (`agg_router.yaml`)
Enhanced aggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: OpenAI-compatible API server (with kv router mode enabled)
- `VLLMDecodeWorker`: Single worker handling both prefill and decode

### 3. **Disaggregated Deployment** (`disagg.yaml`)
High-performance deployment with separated prefill and decode workers.

**Architecture:**
- `Frontend`: HTTP API server coordinating between workers
- `VLLMDecodeWorker`: Specialized decode-only worker
- `VLLMPrefillWorker`: Specialized prefill-only worker (`--is-prefill-worker`)
- Communication via NIXL transfer backend

### 4. **Disaggregated Router Deployment** (`disagg_router.yaml`)
Advanced disaggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: HTTP API server with KV-aware routing
- `VLLMDecodeWorker`: Specialized decode-only worker
- `VLLMPrefillWorker`: Specialized prefill-only worker (`--is-prefill-worker`)

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
    image: my-registry/vllm-runtime:my-tag
    workingDir: /workspace/examples/backends/vllm
    args:
      - "python3"
      - "-m"
      - "dynamo.vllm"
      - "--model"
      - "Qwen/Qwen3-0.6B"
      # Optional: Enable prompt embeddings feature
      # - "--enable-prompt-embeds"
      # Other model-specific arguments
```

**Common vLLM Flags:**
- `--enable-prompt-embeds`: Enable prompt embeddings feature
- `--enable-multimodal`: Enable multimodal (vision) support
- `--is-prefill-worker`: Prefill-only mode for disaggregated serving
- `--connector [nixl|lmcache|kvbm|none]`: KV transfer backend selection

## Prerequisites

Before using these templates, ensure you have:

1. **Dynamo Kubernetes Platform installed** - See [Quickstart Guide](../../../../docs/kubernetes/README.md)
2. **Kubernetes cluster with GPU support**
3. **Container registry access** for vLLM runtime images
4. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)

### Container Images

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd prefer to use your own registry, build and push your own image:

```bash
./container/build.sh --framework VLLM
# Tag and push to your container registry
# Update the image references in the YAML files
```

### Pre-Deployment Profiling (SLA Planner Only)

If using the SLA Planner deployment (`disagg_planner.yaml`), follow the [pre-deployment profiling guide](../../../../docs/components/profiler/profiler_guide.md) to run pre-deployment profiling.

## Usage

### 1. Choose Your Template
Select the deployment pattern that matches your requirements:
- Use `agg.yaml` for simple testing
- Use `agg_router.yaml` for production with load balancing
- Use `disagg.yaml` for maximum performance
- Use `disagg_router.yaml` for high-performance with KV cache routing
- Use `disagg_planner.yaml` for SLA-optimized performance

### 2. Customize Configuration
Edit the template to match your environment:

```yaml
# Update image registry and tag
image: my-registry/vllm-runtime:my-tag

# Configure your model
args:
  - "--model"
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

Export the NAMESPACE you used in your Dynamo Kubernetes Platform Installation.

```bash
cd <dynamo-source-root>/examples/backends/vllm/deploy
export DEPLOYMENT_FILE=agg.yaml

kubectl apply -f $DEPLOYMENT_FILE -n $NAMESPACE
```

### 4. Using Custom Dynamo Frameworks Image for vLLM

To use a custom dynamo frameworks image for vLLM, you can update the deployment file using yq:

```bash
export DEPLOYMENT_FILE=agg.yaml
export FRAMEWORK_RUNTIME_IMAGE=<vllm-image>

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(FRAMEWORK_RUNTIME_IMAGE)' $DEPLOYMENT_FILE  > $DEPLOYMENT_FILE.generated
kubectl apply -f $DEPLOYMENT_FILE.generated -n $NAMESPACE
```

### 5. Port Forwarding

After deployment, forward the frontend service to access the API:

```bash
kubectl port-forward deployment/vllm-v1-disagg-frontend-<pod-uuid-info> 8000:8000
```

## Configuration Options

### Environment Variables

To change `DYN_LOG` level, edit the yaml file by adding:

```yaml
...
spec:
  envs:
    - name: DYN_LOG
      value: "debug" # or other log levels
  ...
```

### vLLM Worker Configuration

vLLM workers are configured through command-line arguments. Key parameters include:

- `--model`: Model to serve (e.g., `Qwen/Qwen3-0.6B`)
- `--is-prefill-worker`: Enable prefill-only mode for disaggregated serving
- `--metrics-endpoint-port`: Port for publishing KV metrics to Dynamo

See the [vLLM CLI documentation](https://docs.vllm.ai/en/v0.9.2/configuration/serve_args.html?h=serve+arg) for the full list of configuration options.

## Testing the Deployment

Send a test request to verify your deployment:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

## Model Configuration

All templates use **Qwen/Qwen3-0.6B** as the default model, but you can use any vLLM-supported LLM model and configuration arguments.

## Monitoring and Health

- **Frontend health endpoint**: `http://<frontend-service>:8000/health`
- **Liveness probes**: Check process health regularly
- **KV metrics**: Published via metrics endpoint port

## Request Migration

You can enable [request migration](../../../../docs/fault_tolerance/request_migration.md) to handle worker failures gracefully by adding the migration limit argument to worker configurations:

```yaml
args:
  - "--migration-limit"
  - "3"
```

## Further Reading

- **Deployment Guide**: [Creating Kubernetes Deployments](../../../../docs/kubernetes/deployment/create_deployment.md)
- **Quickstart**: [Deployment Quickstart](../../../../docs/kubernetes/README.md)
- **Platform Setup**: [Dynamo Kubernetes Platform Installation](../../../../docs/kubernetes/installation_guide.md)
- **SLA Planner**: [SLA Planner Quickstart Guide](../../../../docs/components/planner/planner_guide.md)
- **Examples**: [Deployment Examples](../../../../docs/examples/README.md)
- **Architecture Docs**: [Disaggregated Serving](../../../../docs/design_docs/disagg_serving.md), [KV-Aware Routing](../../../../docs/components/router/README.md)

## Troubleshooting

Common issues and solutions:

1. **Pod fails to start**: Check image registry access and HuggingFace token secret
2. **GPU not allocated**: Verify cluster has GPU nodes and proper resource limits
3. **Health check failures**: Review model loading logs and increase `initialDelaySeconds`
4. **Out of memory**: Increase memory limits or reduce model batch size
5. **Port forwarding issues**: Ensure correct pod UUID in port-forward command

For additional support, refer to the [deployment troubleshooting guide](../../../../docs/kubernetes/README.md).
