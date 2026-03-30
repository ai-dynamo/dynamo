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
- `VLLMPrefillWorker`: Specialized prefill-only worker (`--disaggregation-mode prefill`)
- Communication via NIXL transfer backend

### 4. **Disaggregated Router Deployment** (`disagg_router.yaml`)
Advanced disaggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: HTTP API server with KV-aware routing
- `VLLMDecodeWorker`: Specialized decode-only worker
- `VLLMPrefillWorker`: Specialized prefill-only worker (`--disaggregation-mode prefill`)

### 5. **Global Planner Deployments** (see [`examples/global_planner/`](../../../global_planner/))
Centralized scaling across multiple DGDs via GlobalPlanner. Examples include single-endpoint multi-pool and multi-model GPU budget patterns. See the [global planner examples](../../../global_planner/) for details.

### 6. **Deployments with Intel XPU (Optional)** (`agg_xpu_dra.yaml`, `disagg_xpu_dra.yaml`, `disagg_xpu_cuda.yaml`, or `disagg_xpu_cuda_rdma.yaml`)
Hardware-specific aggregated/disaggregated deployments using Kubernetes Dynamic Resource Allocation (DRA) or mixed-device configurations.

**Aggregated Architecture** (`agg_xpu_dra.yaml`):
- `Frontend`: OpenAI-compatible API server
- `VllmDecodeWorker`: Single worker with XPU target (`VLLM_TARGET_DEVICE=xpu`)
- GPU allocation via `ResourceClaimTemplate` and pod-level `resourceClaims`

**Disaggregated Architecture — XPU prefill + XPU decode** (`disagg_xpu_dra.yaml`):
- `Frontend`: HTTP API server coordinating between workers
- `VllmDecodeWorker`: Specialized decode-only worker with XPU target
- `VllmPrefillWorker`: Specialized prefill-only worker with XPU target
- GPU allocation via `ResourceClaimTemplate` and pod-level `resourceClaims`
- Communication via NIXL transfer backend with XPU buffer

**Mixed Disaggregated Architecture — XPU prefill + CUDA decode** (`disagg_xpu_cuda.yaml`):
- `Frontend`: HTTP API server with KV-aware routing (`DYN_ROUTER_MODE=kv`)
- `DecodeWorker`: CUDA GPU decode worker; KV cache via CPU buffer (`kv_buffer_device: cpu`), UCX over TCP
- `PrefillWorker`: Intel XPU prefill worker (`VLLM_TARGET_DEVICE=xpu`); GPU via `ResourceClaimTemplate` (`gpu.intel.com`), KV cache via CPU buffer
- KV events published over ZMQ for cache-aware routing

**Mixed Disaggregated Architecture with RDMA — XPU prefill + CUDA decode** (`disagg_xpu_cuda_rdma.yaml`):
- `Frontend`: HTTP API server
- `DecodeWorker`: CUDA GPU decode worker; KV cache transferred directly over RDMA (`kv_buffer_device: cuda`), UCX over InfiniBand (`ib,rc,cuda_copy`)
- `PrefillWorker`: Intel XPU prefill worker (`VLLM_TARGET_DEVICE=xpu`); KV cache via XPU buffer (`kv_buffer_device: xpu`), UCX over `ib,rc,ud,ze_copy`
- Both workers claim an RDMA NIC via `ResourceClaimTemplate` (`rdma-dranet`)
- High-performance GPU-to-GPU KV transfer via NIXL over RDMA fabric

> **Note:** Only `IPC_LOCK` is required — it allows the kernel's IB verbs layer (`ib_umem_get`) to bypass `RLIMIT_MEMLOCK` when pinning RDMA memory regions.

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
- `--disaggregation-mode prefill`: Prefill-only mode for disaggregated serving
- `--kv-transfer-config '<json>'`: KV transfer backend configuration (e.g., `'{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`)

## Prerequisites

Before using these templates, ensure you have:

1. **Dynamo Kubernetes Platform installed** - See [Quickstart Guide](../../../../docs/kubernetes/README.md)
2. **Kubernetes cluster with GPU support**
3. **Container registry access** for vLLM runtime images (optional for default NGC CUDA images - `nvcr.io/nvidia/ai-dynamo/*` images are publicly accessible; Intel XPU users should build custom images with `--device xpu`)
4. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)

### Container Images

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd prefer to use your own registry, build and push your own image:

```bash
python container/render.py --framework=vllm --output-short-filename
docker build -f container/rendered.Dockerfile .
# Tag and push to your container registry
# Update the image references in the YAML files
```

### Pre-Deployment Profiling (SLA Planner Only)

If using the SLA Planner deployment (`disagg_planner.yaml`), follow the [pre-deployment profiling guide](../../../../docs/components/profiler/profiler-guide.md) to run pre-deployment profiling.

## Usage

### 1. Choose Your Template
Select the deployment pattern that matches your requirements:
- Use `agg.yaml` for simple testing
- Use `agg_router.yaml` for production with load balancing
- Use `disagg.yaml` for maximum performance
- Use `disagg_router.yaml` for high-performance with KV cache routing
- Use `disagg_planner.yaml` for SLA-optimized performance
- Use `agg_xpu_dra.yaml` for aggregated deployment on Intel XPU clusters using Kubernetes DRA
- Use `disagg_xpu_dra.yaml` for disaggregated deployment on Intel XPU clusters using Kubernetes DRA
- Use `disagg_xpu_cuda.yaml` for mixed disaggregated deployment with XPU prefill and CUDA decode (no RDMA)
- Use `disagg_xpu_cuda_rdma.yaml` for mixed disaggregated deployment with XPU prefill and CUDA decode over RDMA
- Use [global planner examples](../../../global_planner/) for centralized scaling across multiple DGDs

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

#### Deploy with Intel XPU  (Optional)
If your cluster uses Intel GPU devices via Kubernetes Dynamic Resource Allocation (DRA), ensure:
- Your Kubernetes cluster is **v1.34+** (required for DRA API v1), and
- The [Intel XPU Resource Driver](https://github.com/intel/intel-resource-drivers-for-kubernetes) is installed — see the [XPU DRA usage guide](https://github.com/intel/intel-resource-drivers-for-kubernetes/blob/main/doc/gpu/USAGE.md) for device class setup and claim examples.

Deploy the XPU template (includes the ResourceClaimTemplate):
```bash
cd <dynamo-source-root>/examples/backends/vllm/deploy

# For aggregated deployment
kubectl apply -f agg_xpu_dra.yaml -n $NAMESPACE

# OR for disaggregated deployment
kubectl apply -f disagg_xpu_dra.yaml -n $NAMESPACE
```

Verify claim allocation:

```bash
kubectl get resourceclaim -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
```

`agg_xpu_dra.yaml` and `disagg_xpu_dra.yaml` are optional hardware-specific templates and do not change the default deployment paths defined by `agg.yaml` and `disagg.yaml`.

##### Mixed XPU + CUDA Disaggregated (no RDMA)

Use `disagg_xpu_cuda.yaml` when your cluster has Intel XPU nodes for prefill and NVIDIA CUDA nodes for decode, without an RDMA fabric:
```bash
cd <dynamo-source-root>/examples/backends/vllm/deploy
kubectl apply -f disagg_xpu_cuda.yaml -n $NAMESPACE
```

Verify the deployment and XPU resource claims:
```bash
kubectl get resourceclaim -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE
```

##### Mixed XPU + CUDA Disaggregated with RDMA

Use `disagg_xpu_cuda_rdma.yaml` when your cluster has an InfiniBand/RoCE RDMA fabric for high-throughput KV cache transfer between the Intel XPU prefill node and NVIDIA CUDA decode node:

Additional prerequisites:
- An RDMA-capable network device per node (InfiniBand or RoCE)
- [DRANet](https://github.com/kubernetes-sigs/dranet?tab=readme-ov-file#installation) installed — provides the `rdma-dranet` device class that exposes RDMA NICs via Kubernetes DRA

```bash
cd <dynamo-source-root>/examples/backends/vllm/deploy
kubectl apply -f disagg_xpu_cuda_rdma.yaml -n $NAMESPACE
```

Verify both XPU and RDMA resource claims:
```bash
kubectl get resourceclaim -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE
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
- `--disaggregation-mode prefill`: Enable prefill-only mode for disaggregated serving
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

You can enable [request migration](../../../../docs/fault-tolerance/request-migration.md) to handle worker failures gracefully by adding the migration limit argument to worker configurations:

```yaml
args:
  - "--migration-limit"
  - "3"
```

## Further Reading

- **Deployment Guide**: [Creating Kubernetes Deployments](../../../../docs/kubernetes/deployment/create-deployment.md)
- **Quickstart**: [Deployment Quickstart](../../../../docs/kubernetes/README.md)
- **Platform Setup**: [Dynamo Kubernetes Platform Installation](../../../../docs/kubernetes/installation-guide.md)
- **SLA Planner**: [SLA Planner Quickstart Guide](../../../../docs/components/planner/planner-guide.md)
- **Global Planner**: [Global Planner Deployment Guide](../../../../docs/components/planner/global-planner.md)
- **Examples**: [Deployment Examples](../../../../docs/getting-started/examples.md)
- **Architecture Docs**: [Disaggregated Serving](../../../../docs/design-docs/disagg-serving.md), [KV-Aware Routing](../../../../docs/components/router/README.md)

## Troubleshooting

Common issues and solutions:

1. **Pod fails to start**: Check image registry access and HuggingFace token secret
2. **GPU not allocated**: Verify cluster has GPU nodes and proper resource limits
3. **Health check failures**: Review model loading logs and increase `initialDelaySeconds`
4. **Out of memory**: Increase memory limits or reduce model batch size
5. **Port forwarding issues**: Ensure correct pod UUID in port-forward command

For additional support, refer to the [deployment troubleshooting guide](../../../../docs/kubernetes/README.md).
