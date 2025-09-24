# TensorRT-LLM Kubernetes Deployment Configurations

This directory contains Kubernetes Custom Resource Definition (CRD) templates for deploying TensorRT-LLM inference graphs using the **DynamoGraphDeployment** resource.

## Available Deployment Patterns

### 1. **Aggregated Deployment** (`agg.yaml`)
Basic deployment pattern with frontend and a single worker.

**Architecture:**
- `Frontend`: OpenAI-compatible API server (with kv router mode disabled)
- `TRTLLMWorker`: Single worker handling both prefill and decode

### 2. **Aggregated Router Deployment** (`agg_router.yaml`)
Enhanced aggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: OpenAI-compatible API server (with kv router mode enabled)
- `TRTLLMWorker`: Multiple workers handling both prefill and decode (2 replicas for load balancing)

### 3. **Disaggregated Deployment** (`disagg.yaml`)
High-performance deployment with separated prefill and decode workers.

**Architecture:**
- `Frontend`: HTTP API server coordinating between workers
- `TRTLLMDecodeWorker`: Specialized decode-only worker
- `TRTLLMPrefillWorker`: Specialized prefill-only worker

### 4. **Disaggregated Router Deployment** (`disagg_router.yaml`)
Advanced disaggregated deployment with KV cache routing capabilities.

**Architecture:**
- `Frontend`: HTTP API server (with kv router mode enabled)
- `TRTLLMDecodeWorker`: Specialized decode-only worker
- `TRTLLMPrefillWorker`: Specialized prefill-only worker (2 replicas for load balancing)

### 5. **Aggregated Deployment with Config** (`agg-with-config.yaml`)
Aggregated deployment with custom configuration.

**Architecture:**
- `nvidia-config`: ConfigMap containing a custom trtllm configuration
- `Frontend`: OpenAI-compatible API server (with kv router mode disabled)
- `TRTLLMWorker`: Single worker handling both prefill and decode with custom configuration mounted from the configmap

### 6. **Disaggregated Planner Deployment** (`disagg_planner.yaml`)
Advanced disaggregated deployment with SLA-based automatic scaling.

**Architecture:**
- `Frontend`: HTTP API server coordinating between workers
- `Planner`: SLA-based planner that monitors performance and scales workers automatically
- `Prometheus`: Metrics collection and monitoring
- `TRTLLMDecodeWorker`: Specialized decode-only worker
- `TRTLLMPrefillWorker`: Specialized prefill-only worker

> [!NOTE]
> This deployment requires pre-deployment profiling to be completed first. See [Pre-Deployment Profiling](../../../../docs/benchmarks/pre_deployment_profiling.md) for detailed instructions.

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
    image: my-registry/trtllm-runtime:my-tag
    workingDir: /workspace/components/backends/trtllm
    args:
      - "python3"
      - "-m"
      - "dynamo.trtllm"
      # Model-specific arguments
```

## Prerequisites

Before using these templates, ensure you have:

1. **Dynamo Cloud Platform installed** - See [Quickstart Guide](../../../../docs/kubernetes/README.md)
2. **Kubernetes cluster with GPU support**
3. **Container registry access** for TensorRT-LLM runtime images
4. **HuggingFace token secret** (referenced as `envFromSecret: hf-token-secret`)

### Container Images

The deployment files currently require access to `my-registry/trtllm-runtime`. If you don't have access, build and push your own image:

```bash
./container/build.sh --framework tensorrtllm
# Tag and push to your container registry
# Update the image references in the YAML files
```

**Note:** TensorRT-LLM uses git-lfs, which needs to be installed in advance:
```bash
apt-get update && apt-get -y install git git-lfs
```

For ARM machines, use:
```bash
./container/build.sh --framework tensorrtllm --platform linux/arm64
```

## Usage

### 1. Choose Your Template
Select the deployment pattern that matches your requirements:
- Use `agg.yaml` for simple testing
- Use `agg_router.yaml` for production with KV cache routing and load balancing
- Use `disagg.yaml` for maximum performance with separated workers
- Use `disagg_router.yaml` for high-performance with KV cache routing and disaggregation

### 2. Customize Configuration
Edit the template to match your environment:

```yaml
# Update image registry and tag
image: my-registry/trtllm-runtime:my-tag

# Configure your model and deployment settings
args:
  - "python3"
  - "-m"
  - "dynamo.trtllm"
  # Add your model-specific arguments
```

### 3. Deploy

See the [Create Deployment Guide](../../../../docs/kubernetes/create_deployment.md) to learn how to deploy the deployment file.

First, create a secret for the HuggingFace token.
```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Then, deploy the model using the deployment file.

Export the NAMESPACE you used in your Dynamo Cloud Installation.

```bash
cd dynamo/components/backends/trtllm/deploy
export DEPLOYMENT_FILE=agg.yaml
kubectl apply -f $DEPLOYMENT_FILE -n $NAMESPACE
```

### 4. Using Custom Dynamo Frameworks Image for TensorRT-LLM

To use a custom dynamo frameworks image for TensorRT-LLM, you can update the deployment file using yq:

```bash
export DEPLOYMENT_FILE=agg.yaml
export FRAMEWORK_RUNTIME_IMAGE=<trtllm-image>

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(FRAMEWORK_RUNTIME_IMAGE)' $DEPLOYMENT_FILE  > $DEPLOYMENT_FILE.generated
kubectl apply -f $DEPLOYMENT_FILE.generated -n $NAMESPACE
```

### 5. Port Forwarding

After deployment, forward the frontend service to access the API:

```bash
kubectl port-forward deployment/trtllm-v1-disagg-frontend-<pod-uuid-info> 8000:8000
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

### TensorRT-LLM Worker Configuration

TensorRT-LLM workers are configured through command-line arguments in the deployment YAML. Key configuration areas include:

- **Disaggregation Strategy**: Control request flow with `DISAGGREGATION_STRATEGY` environment variable
- **KV Cache Transfer**: Choose between UCX (default) or NIXL for disaggregated serving
- **Request Migration**: Enable graceful failure handling with `--migration-limit`

### Disaggregation Strategy

The disaggregation strategy controls how requests are distributed between prefill and decode workers:

- **`decode_first`** (default): Requests routed to decode worker first, then forwarded to prefill worker
- **`prefill_first`**: Requests routed directly to prefill worker (used with KV routing)

Set via environment variable:
```yaml
envs:
  - name: DISAGGREGATION_STRATEGY
    value: "prefill_first"
```

## Testing the Deployment

Send a test request to verify your deployment. See the [client section](../../../../components/backends/vllm/README.md#client) for detailed instructions.

**Note:** For multi-node deployments, target the node running `python3 -m dynamo.frontend <args>`.

## Model Configuration

The deployment templates support various TensorRT-LLM models and configurations. You can customize model-specific arguments in the worker configuration sections of the YAML files.

## Monitoring and Health

- **Frontend health endpoint**: `http://<frontend-service>:8000/health`
- **Worker health endpoints**: `http://<worker-service>:9090/health`
- **Liveness probes**: Check process health every 5 seconds
- **Readiness probes**: Check service readiness with configurable delays

## KV Cache Transfer Methods

TensorRT-LLM supports two methods for KV cache transfer in disaggregated serving:

- **UCX** (default): Standard method for KV cache transfer
- **NIXL** (experimental): Alternative transfer method

For detailed configuration instructions, see the [KV cache transfer guide](../kv-cache-transfer.md).

## Request Migration

You can enable [request migration](../../../../docs/architecture/request_migration.md) to handle worker failures gracefully by adding the migration limit argument to worker configurations:

```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.trtllm"
  - "--migration-limit"
  - "3"
```

## Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script: [perf.sh](../../../../benchmarks/llm/perf.sh)

Configure the `model` name and `host` based on your deployment.

## Further Reading

- **Deployment Guide**: [Creating Kubernetes Deployments](../../../../docs/kubernetes/create_deployment.md)
- **Quickstart**: [Deployment Quickstart](../../../../docs/kubernetes/README.md)
- **Platform Setup**: [Dynamo Cloud Installation](../../../../docs/kubernetes/installation_guide.md)
- **Examples**: [Deployment Examples](../../../../docs/examples/README.md)
- **Architecture Docs**: [Disaggregated Serving](../../../../docs/architecture/disagg_serving.md), [KV-Aware Routing](../../../../docs/architecture/kv_cache_routing.md)
- **Multinode Deployment**: [Multinode Examples](../multinode/multinode-examples.md)
- **Speculative Decoding**: [Llama 4 + Eagle Guide](../llama4_plus_eagle.md)
- **Kubernetes CRDs**: [Custom Resources Documentation](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)

## Troubleshooting

Common issues and solutions:

1. **Pod fails to start**: Check image registry access and HuggingFace token secret
2. **GPU not allocated**: Verify cluster has GPU nodes and proper resource limits
3. **Health check failures**: Review model loading logs and increase `initialDelaySeconds`
4. **Out of memory**: Increase memory limits or reduce model batch size
5. **Port forwarding issues**: Ensure correct pod UUID in port-forward command
6. **Git LFS issues**: Ensure git-lfs is installed before building containers
7. **ARM deployment**: Use `--platform linux/arm64` when building on ARM machines

For additional support, refer to the [deployment troubleshooting guide](../../../../docs/kubernetes/README.md).
