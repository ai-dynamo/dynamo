# Creating Kubernetes Deployments

The scripts in the `examples/<backend>/launch` folder like [agg.sh](../../../examples/backends/vllm/launch/agg.sh) demonstrate how you can serve your models locally.
The corresponding YAML files like [agg.yaml](../../../examples/backends/vllm/deploy/agg.yaml) show you how you could create a Kubernetes deployment for your inference graph.

This guide explains how to create your own deployment files.

## Step 1: Choose Your Architecture Pattern

Before choosing a template, understand the different architecture patterns:

### Aggregated Serving (agg.yaml)

**Pattern**: Prefill and decode on the same GPU in a single process.

**Suggested to use for**:
- Small to medium models (under 70B parameters)
- Development and testing
- Low to moderate traffic
- Simplicity is prioritized over maximum throughput

**Tradeoffs**:
- Simpler setup and debugging
- Lower operational complexity
- GPU utilization may not be optimal (prefill and decode compete for resources)
- Lower throughput ceiling compared to disaggregated

**Example**: [`agg.yaml`](../../../examples/backends/vllm/deploy/agg.yaml)

### Aggregated + Router (agg_router.yaml)

**Pattern**: Load balancer routing across multiple aggregated worker instances.

**Suggested to use for**:
- Medium traffic requiring high availability
- Need horizontal scaling
- Want some load balancing without disaggregation complexity

**Tradeoffs**:
- Better scalability than plain aggregated
- High availability through multiple replicas
- Still has GPU underutilization issues of aggregated serving
- More complex than plain aggregated but simpler than disaggregated

**Example**: [`agg_router.yaml`](../../../examples/backends/vllm/deploy/agg_router.yaml)

### Disaggregated Serving (disagg_router.yaml)

**Pattern**: Separate prefill and decode workers with specialized optimization.

**Suggested to use for**:
- Production-style deployments
- High throughput requirements
- Large models (70B+ parameters)
- Maximum GPU utilization needed

**Tradeoffs**:
- Maximum performance and throughput
- Better GPU utilization (prefill and decode specialized)
- Independent scaling of prefill and decode
- More complex setup and debugging
- Requires understanding of prefill/decode separation

**Example**: [`disagg_router.yaml`](../../../examples/backends/vllm/deploy/disagg_router.yaml)

### Quick Selection Guide

Select the architecture pattern as your template that best fits your use case.

For example, when using the `vLLM` backend:

- **Development / Testing**: Use [`agg.yaml`](../../../examples/backends/vllm/deploy/agg.yaml) as the base configuration.

- **Production with Load Balancing**: Use [`agg_router.yaml`](../../../examples/backends/vllm/deploy/agg_router.yaml) to enable scalable, load-balanced inference.

- **High Performance / Disaggregated Deployment**: Use [`disagg_router.yaml`](../../../examples/backends/vllm/deploy/disagg_router.yaml) for maximum throughput and modular scalability.


## Step 2: Customize the Template

You can run the Frontend on one machine, for example a CPU node, and the worker on a different machine (a GPU node).
The Frontend serves as a framework-agnostic HTTP entry point and is likely not to need many changes.

It serves the following roles:
1. OpenAI-Compatible HTTP Server
  * Provides `/v1/chat/completions` endpoint
  * Handles HTTP request/response formatting
  * Supports streaming responses
  * Validates incoming requests

2. Service Discovery and Routing
  * Auto-discovers backend workers via etcd
  * Routes requests to the appropriate Processor/Worker components
  * Handles load balancing between multiple workers

3. Request Preprocessing
  * Initial request validation
  * Model name verification
  * Request format standardization

You should then pick a worker and specialize the config. For example,

```yaml
VllmWorker:         # vLLM-specific config
  enforce-eager: true
  enable-prefix-caching: true

SglangWorker:       # SGLang-specific config
  router-mode: kv
  disagg-mode: true

TrtllmWorker:       # TensorRT-LLM-specific config
  engine-config: ./engine.yaml
  kv-cache-transfer: ucx
```

Here's a template structure based on the examples:

```yaml
    YourWorker:
      dynamoNamespace: your-namespace
      componentType: worker
      replicas: N
      envFromSecret: your-secrets  # e.g., hf-token-secret
      # Health checks for worker initialization
      readinessProbe:
        exec:
          command: ["/bin/sh", "-c", 'grep "Worker.*initialized" /tmp/worker.log']
      resources:
        requests:
          gpu: "1"  # GPU allocation
      extraPodSpec:
        mainContainer:
          image: your-image
          command:
            - /bin/sh
            - -c
          args:
            - python -m dynamo.YOUR_INFERENCE_ENGINE --model YOUR_MODEL --your-flags
```

Consult the corresponding sh file. Each of the python commands to launch a component will go into your yaml spec under the
`extraPodSpec: -> mainContainer: -> args:`

The front end is launched with "python3 -m dynamo.frontend [--http-port 8000] [--router-mode kv]"
Each worker will launch `python -m dynamo.YOUR_INFERENCE_BACKEND --model YOUR_MODEL --your-flags `command.
If you are a Dynamo contributor the [dynamo run guide](../../reference/cli.md) for details on how to run this command.


## Step 3: Key Customization Points

### Model Configuration

```yaml
   args:
     - "python -m dynamo.YOUR_INFERENCE_BACKEND --model YOUR_MODEL --your-flag"
```

### Resource Allocation

```yaml
   resources:
     requests:
       cpu: "N"
       memory: "NGi"
       gpu: "N"
```

### Scaling

```yaml
   replicas: N  # Number of worker instances
```

### Routing Mode
```yaml
   args:
     - --router-mode
     - kv  # Enable KV-cache routing
```

### Worker Specialization

```yaml
   args:
     - --is-prefill-worker  # For disaggregated prefill workers
```

### Image Pull Secret Configuration

#### Automatic Discovery and Injection

By default, the Dynamo operator automatically discovers and injects image pull secrets based on container registry host matching. The operator scans Docker config secrets within the same namespace and matches their registry hostnames to the container image URLs, automatically injecting the appropriate secrets into the pod's `imagePullSecrets`.

**Disabling Automatic Discovery:**
To disable this behavior for a component and manually control image pull secrets:

```yaml
    YourWorker:
      dynamoNamespace: your-namespace
      componentType: worker
      annotations:
        nvidia.com/disable-image-pull-secret-discovery: "true"
```

When disabled, you can manually specify secrets as you would for a normal pod spec via:
```yaml
    YourWorker:
      dynamoNamespace: your-namespace
      componentType: worker
      annotations:
        nvidia.com/disable-image-pull-secret-discovery: "true"
      extraPodSpec:
        imagePullSecrets:
          - name: my-registry-secret
          - name: another-secret
        mainContainer:
          image: your-image
```

This automatic discovery eliminates the need to manually configure image pull secrets for each deployment.