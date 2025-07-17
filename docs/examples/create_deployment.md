# Creating Kubernetes Deployments

The scripts in the `launch` folder like [agg.sh](../../examples/vllm/launch/agg.sh) demonstrate how you can serve your models locally.
The corresponding yaml files like [agg.yaml](../../examples/vllm/deploy/agg.yaml) show you how you could create a kubernetes deployment for your inference graph.


This guide explains how to create your own deployment files.

## Step 1: Choose Your Architecture Pattern

Select the architecture pattern that best fits your use case.

For example, when using the `vLLM` inference backend:

- **Development / Testing**
  Use [`agg.yaml`](./../examples/vllm/deploy/agg.yaml) as the base configuration.

- **Production with Load Balancing**
  Use [`agg_router.yaml`](./../examples/vllm/deploy/agg_router.yaml) to enable scalable, load-balanced inference.

- **High Performance / Disaggregated Deployment**
  Use [`disagg_router.yaml`](./../examples/vllm/deploy/disagg_router.yaml) for maximum throughput and modular scalability.


## Step 2: Customize the Base Template

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