# Creating Kubernetes Deployments

The scripts in the `launch` folder like [agg.sh](../../../examples/vllm/launch/agg.sh) demonstrate how you can serve your models locally.
The corresponding YAML files like [agg.yaml](../../../examples/vllm/deploy/agg.yaml) show you how you could create a kubernetes deployment for your inference graph.


This guide explains how to create your own deployment files.

## Step 1: Choose Your Architecture Pattern

Select the architecture pattern as your template that best fits your use case.

For example, when using the `vLLM` inference backend:

- **Development / Testing**
  Use [`agg.yaml`](../../../examples/vllm/deploy/agg.yaml) as the base configuration.

- **Production with Load Balancing**
  Use [`agg_router.yaml`](../../../examples/vllm/deploy/agg_router.yaml) to enable scalable, load-balanced inference.

- **High Performance / Disaggregated Deployment**
  Use [`disagg_router.yaml`](../../../examples/vllm/deploy/disagg_router.yaml) for maximum throughput and modular scalability.


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
          args:
            - "python3 components/main.py --model YOUR_MODEL --your-flags"
```

Consult the corresponding sh file. Each of the python commands to launch a component will go into your yaml spec under the
`extraPodSpec: -> mainContainer: -> args:`

The front end will launch `dynamo run in=http out=dyn &` or its python counterpart `python -m dynamo.frontend`
Each worker will launch `dynamo run in=dyn//name out=sglang <model>`command or its python counterparts `"python3 components/*.py`
See the [dynamo run guide](../dynamo_run.md) for details on how to run this command.


## Step 3: Key Customization Points

### Model Configuration

```yaml
   args:
     - "python3 components/main.py --model YOUR_MODEL_PATH --your-custom-flags"
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