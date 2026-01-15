# Triton Server Backend for Dynamo

> **⚠️ Work in Progress / Proof of Concept**
>
> This example demonstrates integrating NVIDIA Triton Inference Server as a backend for Dynamo.
> It is currently a proof-of-concept and may require additional work for production use.

## Overview

This example shows how to run Triton Server models through Dynamo's distributed runtime, exposing them via the KServe gRPC protocol. The integration allows Triton models to benefit from Dynamo's service discovery, routing, and infrastructure.

**Architecture:**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────────┐
│  Triton Client  │────▶│  Dynamo Frontend│────▶│       Dynamo Worker         │
│  (KServe gRPC)  │     │  (port 8787)    │     │  ┌───────────────────────┐  │
└─────────────────┘     └─────────────────┘     │  │    Triton Server      │  │
                              │                 │  │  (Python bindings)    │  │
                              ▼                 │  └───────────────────────┘  │
                    ┌─────────────────┐         └─────────────────────────────┘
                    │    KV Store     │
                    └─────────────────┘
```

## Prerequisites

- NVIDIA GPU with CUDA support
- For local development: Python 3.10+ with Dynamo installed
- For container deployment: Docker with NVIDIA Container Toolkit

## Quick Start

### Option 1: Container Deployment

#### Step 1: Build Container Images

From the Dynamo repository root:

```bash
# Build the base Dynamo image
./container/build.sh

# Build the Triton worker image
cd examples/backends/tritonserver
docker build -t dynamo-triton:latest .
```

#### Step 2: Run the Container

```bash
docker run --rm -it --gpus all --network host \
  dynamo-triton:latest \
  ./examples/backends/tritonserver/launch/identity.sh
```

#### Step 3: Test the Deployment

In another terminal:

```bash
# Install client dependencies
pip install tritonclient[grpc]

# Test with the client
cd examples/backends/tritonserver
python src/client.py --port 8000
```

### Option 2: Local Development

This requires Dynamo to be installed locally.

```bash
# From the dynamo repo root
cd examples/backends/tritonserver

# Build Triton Server (first time only, ~30 minutes)
make all

# Install Python dependencies
pip install wheelhouse/tritonserver-*.whl
pip install tritonclient[grpc]

# Launch the server
./launch/identity.sh

# In another terminal, test with the client
python src/client.py
```

## Directory Structure

```
tritonserver/
├── launch/
│   └── identity.sh      # Launch script (frontend + worker)
├── src/
│   ├── tritonworker.py  # Main Dynamo worker implementation
│   └── client.py        # Test client (KServe gRPC)
├── model_repo/
│   └── identity/        # Sample identity model
│       ├── config.pbtxt
│       └── 1/
├── backends/            # Triton backends (built by `make all`)
├── lib/                 # Triton libraries (built by `make all`)
├── wheelhouse/          # Python wheels (built by `make all`)
├── Dockerfile           # Triton worker container
└── Makefile             # Build Triton from source
```

## Configuration

### Launch Script Options

```bash
./launch/identity.sh --help

Options:
  --model-name <name>         Model name to load (default: identity)
  --model-repository <path>   Path to model repository
  --backend-directory <path>  Path to Triton backends
  --log-verbose <level>       Triton log verbosity 0-6 (default: 1)
  --store-kv <backend>        KV store backend: file, etcd, mem (default: file)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_STORE_KV` | KV store backend: `file`, `etcd`, or `mem` | `file` |
| `DYN_LOG` | Log level (debug, info, warn, error) | `info` |
| `DYN_HTTP_PORT` | Frontend HTTP port | `8000` |
| `ETCD_ENDPOINTS` | etcd connection URL (only when `--store-kv etcd`) | `http://localhost:2379` |
| `NATS_SERVER` | NATS connection URL (only for distributed mode) | `nats://localhost:4222` |

## Adding Your Own Models

1. Create a model directory in `model_repo/`:

   ```text
   model_repo/
   └── my_model/
       ├── config.pbtxt
       └── 1/
           └── model.plan  # or other model file
   ```

2. Define the model config (`config.pbtxt`):

   ```protobuf
   name: "my_model"
   backend: "tensorrt"  # or onnxruntime, python, etc.
   max_batch_size: 8

   input [
     {
       name: "input"
       data_type: TYPE_FP32
       dims: [3, 224, 224]
     }
   ]
   output [
     {
       name: "output"
       data_type: TYPE_FP32
       dims: [1000]
     }
   ]
   ```

3. Launch with your model:

   ```bash
   ./launch/identity.sh --model-name my_model
   ```

## Known Limitations

- **Single model**: Currently loads one model at a time
- **Identity backend only**: The Makefile builds the identity backend by default; other backends require modifying the build configuration

## Building Triton from Source

Required for local development. The Makefile builds Triton Server and the identity backend.

```bash
cd examples/backends/tritonserver

# Build Triton Server (~30 minutes, clones and builds from source)
make all

# Check build status
make status

# This produces:
#   lib/libtritonserver.so     - Core library
#   bin/tritonserver           - Server binary
#   backends/identity/         - Identity backend
#   wheelhouse/*.whl           - Python bindings

# Clean up build artifacts
make clean      # Remove installed artifacts
make distclean  # Remove everything including build cache
```

To add other backends (TensorRT, ONNX, Python, etc.), edit the Makefile's `build.py` invocation to include additional `--backend=<name>` flags.

## Troubleshooting

### "Model not found" error

- Verify the model exists in `model_repo/<model_name>/`
- Check that `config.pbtxt` is valid
- Ensure the backend is available in `backends/`

### Worker fails to start

- Check `LD_LIBRARY_PATH` includes Triton libraries
- Verify GPU is available: `nvidia-smi`
- Increase log verbosity: `--log-verbose 6`

## Related Documentation

- [Dynamo Backend Guide](../../../docs/development/backend-guide.md)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [KServe Protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/)
