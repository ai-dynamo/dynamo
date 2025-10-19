# Dynamo Companion Server

A lightweight companion server for CUDA IPC weight sharing in vLLM workloads. The companion server loads model weights once and shares them with multiple vLLM worker processes via CUDA Inter-Process Communication (IPC), eliminating redundant weight loading and reducing memory usage.

## Overview

When running multiple vLLM workers on the same GPU, each worker typically loads the full model weights independently. The companion server solves this by:
1. Loading model weights once in a separate process
2. Sharing weights with workers via CUDA IPC (zero-copy memory sharing)
3. Enabling fast worker startup and memory savings

## How It Works

1. Companion server loads model weights using vLLM's standard model loader
2. Server extracts the complete module tree (parameters, buffers, submodules) as CUDA IPC handles
3. vLLM workers request weights via Dynamo RPC
4. Workers reconstruct tensors from IPC handles (zero-copy memory sharing)

## Protocol

The companion server exposes a `load_model` endpoint that:
- Accepts a request with pickled VllmConfig and rank information
- Returns a module tree containing CUDA IPC handles for all tensors (parameters, buffers, tensor attributes)
- Each tensor is represented as a tuple from `torch.multiprocessing.reductions.reduce_tensor`
- Workers reconstruct tensors using `rebuild_cuda_tensor` for zero-copy access

## Usage

### Starting a Companion Server

```bash
# Start companion server for GPUs 0 and 1
python3 -m dynamo.companion --device-id 0 &
python3 -m dynamo.companion --devivce-id 1 &

# Optional: specify custom master port
python3 -m dynamo.companion --device-id 0 --companion-master-port 29600 &
python3 -m dynamo.companion --device-id 1 --companion-master-port 29600 &
```

The server creates a Dynamo component named `companion-gpu{device_id}` (or custom name) and exposes a `load_model` endpoint.

### Using from vLLM

Use `load_format="dynamo_companion"` in your vLLM configuration to load weights from the companion server via CUDA IPC.

## Key Features

- **Memory efficient**: Single copy of weights per GPU, shared across workers via CUDA IPC
- **Fast startup**: Workers skip weight loading and use pre-loaded weights
- **Zero copy**: Direct memory sharing without serialization overhead
- **Configuration caching**: Models are cached by configuration hash (model config, parallel config, ranks)

## Limitations

- Each companion server can only serve one model configuration
- Workers must run on the same physical GPU as the companion server
