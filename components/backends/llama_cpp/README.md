# llama.cpp engine for Dynamo

Usage:
```
# Install ai-dynamo llama.cpp backend (CPU Mode)
pip install "ai-dynamo[llama_cpp]"

# [Optional] To build llama.cpp for CUDA (needs a recent pip)
pip install -r --force-reinstall requirements.gpu.txt

python -m dynamo.llama_cpp --model-path /data/models/Qwen3-0.6B-Q8_0.gguf [args]
```

## Request Migration

In a Distributed System, a request may fail due to connectivity issues between the Frontend and the Backend.

The Frontend will automatically track which Backends are having connectivity issues with it and avoid routing new requests to the Backends with known connectivity issues.

For ongoing requests, there is a `--migration-limit` flag which can be set on the Backend that tells the Frontend how many times a request can be migrated to another Backend should there be a loss of connectivity to the current Backend.

For example,
```bash
python3 -m dynamo.llama_cpp ... --migration-limit=3
```
indicates a request to this model may be migrated up to 3 times to another Backend, before failing the request, should the Frontend detects a connectivity issue to the current Backend.

The migrated request will continue responding to the original request, allowing for a seamless transition between Backends, and a reduced overall request failure rate at the Frontend for enhanced user experience.
