<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (installation and one example).
Do not edit it unless the user explicitly asks you to.
-->

# Sidecars

Rust sidecars connect Dynamo to vLLM, SGLang, and TensorRT-LLM engines over
their native gRPC APIs. The engine runs in its own process; the sidecar
registers it with Dynamo and serves its requests.

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

Engine-specific guides: [vLLM](vllm/README.md), [SGLang](sglang/README.md),
[TensorRT-LLM](trtllm/README.md).

## Installation

### pip

The sidecars ship in the `ai-dynamo` wheel:

```bash
pip install ai-dynamo
python -m dynamo.vllm.sidecar --help    # also dynamo.sglang.sidecar, dynamo.trtllm.sidecar
```

The `launch/` scripts run the standalone executables instead. Build them and
put them on `PATH`:

```bash
cargo build --release -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar -p dynamo-trtllm-sidecar
export PATH="$PWD/target/release:$PATH"
```

### Docker

The CPU-only `dynamo-sidecar` image contains all three sidecars and is
published to NGC:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version>
docker run --rm nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version> vllm --help
```

To build it from the repository root instead:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/Dockerfile -t <your-registry>/dynamo-sidecar:1.6.0-dev --push .
```
