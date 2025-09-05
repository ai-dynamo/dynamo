# Health Check Payload Configuration Guide

This guide explains how to configure health check payloads for vLLM backends in Dynamo.

## Overview

Health check payloads are small inference requests sent periodically to verify that endpoints can process requests correctly. This goes beyond simple connectivity checks to ensure the model is actually functioning.

## Configuration Priority

Health check payloads can be configured at multiple levels, with the following priority (highest to lowest):

1. **Runtime Parameter** - Explicitly passed to `serve_endpoint()`
2. **Environment Variable** - `DYN_HEALTH_CHECK_PAYLOAD`
3. **Config File** - `health_check_payload` field
4. **Model-specific Defaults** - Based on model type
5. **Generic Defaults** - Minimal payload

## Default Payloads

### Minimal (Default)
Fastest health checks with minimal processing:
```python
{
    "prompt": "1",        # Single character
    "max_tokens": 1,      # Generate only 1 token
    "temperature": 0.0,   # Deterministic
    "stream": False,
    "_health_check": True
}
```

### Standard
More realistic but still lightweight:
```python
{
    "prompt": "The sky is",
    "max_tokens": 5,
    "temperature": 0.7,
    "stream": False,
    "_health_check": True
}
```

### Comprehensive
Tests more advanced features:
```python
{
    "prompt": "Write a haiku about artificial intelligence:",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stream": False,
    "_health_check": True
}
```

## Configuration Methods

### 1. Environment Variables

#### Set Health Check Level
```bash
# Choose: minimal, standard, or comprehensive
export DYN_HEALTH_CHECK_LEVEL=standard
```

#### Custom Payload via JSON
```bash
export DYN_HEALTH_CHECK_PAYLOAD='{"prompt": "test", "max_tokens": 1, "temperature": 0}'
```

#### Custom Payload via File
```bash
# Create a JSON file with your payload
cat > /tmp/health_check.json << EOF
{
    "prompt": "Hello world",
    "max_tokens": 3,
    "temperature": 0.5,
    "top_p": 0.9
}
EOF

# Reference the file with @ prefix
export DYN_HEALTH_CHECK_PAYLOAD='@/tmp/health_check.json'
```

#### Disable Health Check Payloads
```bash
export DYN_HEALTH_CHECK_PAYLOAD_DISABLED=true
```

### 2. Python Code (Runtime)

#### Basic Usage
```python
from dynamo.vllm.health_check import get_health_check_payload

# Use defaults based on model type
health_check_payload = get_health_check_payload(
    config=config,
    model_type="generate"
)

# Serve endpoint with health check
await endpoint.serve_endpoint(
    handler=handler.generate,
    health_check_payload=health_check_payload
)
```

#### Custom Payload at Runtime
```python
# Define custom payload
custom_payload = {
    "prompt": "What is 1+1?",
    "max_tokens": 2,
    "temperature": 0.0
}

# Pass directly to serve_endpoint
await endpoint.serve_endpoint(
    handler=handler.generate,
    health_check_payload=custom_payload  # Highest priority
)
```

#### Set Global Custom Default
```python
from dynamo.vllm.health_check import set_custom_health_check_payload

# Set a custom default for all endpoints
set_custom_health_check_payload({
    "prompt": "ping",
    "max_tokens": 1,
    "temperature": 0.0
})
```

### 3. Config File

Add to your vLLM configuration:
```yaml
# config.yaml
model: "gpt2"
health_check_level: "standard"
health_check_payload:
  prompt: "The weather is"
  max_tokens: 3
  temperature: 0.5
```

Or in Python config:
```python
class Config:
    model = "gpt2"
    health_check_level = "standard"
    health_check_payload = {
        "prompt": "The weather is",
        "max_tokens": 3,
        "temperature": 0.5
    }
```

## Model-Specific Payloads

### Chat Models
```python
# Automatically selected for chat models
{
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
    "_health_check": True
}
```

### Embedding Models
```python
# For embedding endpoints
{
    "input": "test",
    "_health_check": True
}
```

## Best Practices

### 1. Production Environments
Use minimal payloads to reduce overhead:
```bash
export DYN_HEALTH_CHECK_LEVEL=minimal
```

### 2. Staging/Testing
Use standard or comprehensive for better coverage:
```bash
export DYN_HEALTH_CHECK_LEVEL=standard
```

### 3. Model-Specific Tuning
Different models may need different prompts:
```python
# For code models
health_payload = {
    "prompt": "def hello():",
    "max_tokens": 5,
    "temperature": 0.0
}

# For chat models
health_payload = {
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 2,
    "temperature": 0.0
}
```

### 4. Monitoring Impact
Monitor the impact of health checks:
- Check latency metrics
- Monitor token usage
- Adjust payload size if needed

## Complete Example

```python
import os
from dynamo import DistributedRuntime
from dynamo.vllm.health_check import get_health_check_payload

async def setup_vllm_with_health_checks():
    # Configure via environment
    os.environ["DYN_HEALTH_CHECK_ENABLED"] = "true"
    os.environ["DYN_HEALTH_CHECK_LEVEL"] = "minimal"
    os.environ["DYN_HEALTH_CHECK_INTERVAL"] = "30"  # seconds

    # Initialize runtime
    runtime = await DistributedRuntime.from_settings()
    namespace = runtime.namespace("vllm")
    component = namespace.component("inference")

    # Create endpoints
    generate_endpoint = component.endpoint("generate")

    # Get appropriate health check payload
    health_check_payload = get_health_check_payload(
        config=config,
        model_type="generate"
    )

    # Or use a custom payload
    custom_health_payload = {
        "prompt": "Test",
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 42  # For reproducibility
    }

    # Serve with health checks
    await generate_endpoint.serve_endpoint(
        handler=handler.generate,
        graceful_shutdown=True,
        metrics_labels=[("model", "gpt2")],
        health_check_payload=health_check_payload  # or custom_health_payload
    )
```

## Troubleshooting

### Health Checks Not Running
1. Check if enabled: `echo $DYN_HEALTH_CHECK_ENABLED`
2. Verify payload is set: Check logs for "Health check payload" messages
3. Check system status endpoint: `curl http://localhost:8082/health`

### Health Checks Failing
1. Test payload manually with the model
2. Reduce payload complexity (smaller max_tokens)
3. Check model logs for errors
4. Verify the payload format matches model expectations

### Performance Impact
1. Use minimal payloads in production
2. Increase check interval: `export DYN_HEALTH_CHECK_INTERVAL=60`
3. Monitor token usage and adjust accordingly

## Summary

The health check payload system provides:
- **Flexibility**: Configure at multiple levels
- **Defaults**: Sensible defaults for quick setup
- **Customization**: Full control when needed
- **Safety**: Mark requests with `_health_check` flag
- **Monitoring**: Verify endpoint functionality beyond connectivity

Choose the configuration method that best fits your deployment workflow and adjust payloads based on your specific requirements and model characteristics.
