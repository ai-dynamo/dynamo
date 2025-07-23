# Creating Kubernetes Deployments

The scripts in the `components/<backend>/launch` folder like [agg.sh](../../../components/backends/vllm/launch/agg.sh) demonstrate how you can serve your models locally.
The corresponding YAML files like [agg.yaml](../../../components/backends/vllm/deploy/agg.yaml) show you how you could create a kubernetes deployment for your inference graph.


This guide explains how to create your own deployment files.

## Step 1: Choose Your Architecture Pattern

Select the architecture pattern as your template that best fits your use case.

For example, when using the `VLLM` inference backend:

- **Development / Testing**
  Use [`agg.yaml`](../../../components/backends/vllm/deploy/agg.yaml) as the base configuration.

- **Production with Load Balancing**
  Use [`agg_router.yaml`](../../../components/backends/vllm/deploy/agg_router.yaml) to enable scalable, load-balanced inference.

- **High Performance / Disaggregated Deployment**
  Use [`disagg_router.yaml`](../../../components/backends/vllm/deploy/disagg_router.yaml) for maximum throughput and modular scalability.


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
If you are a Dynamo contributor the [dynamo run guide](../dynamo_run.md) for details on how to run this command.


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