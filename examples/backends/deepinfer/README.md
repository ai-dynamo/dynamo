<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepInfer Experiments

This checkout contains both Dynamo integration approaches. Both use the real
Dynamo frontend for rendering, tokenization, reasoning parsing and tool parsing.
CPU mock modes let you experiment without loading model weights.

| Approach | Backend boundary | Guide |
| --- | --- | --- |
| Python wrapper | Dynamo Common Worker calls the Python DeepInfer adapter | [Wrapper setup and native/mock serving](README-wrapper.md) |
| OpenEngine sidecar | Dynamo sidecar calls a Python-hosted Rust gRPC server through OpenEngine; Rust calls Python through PyO3 | [Sidecar setup and native/mock serving](README-sidecar.md) |

## Run the Wrapper Mock

With Dynamo built and installed from this checkout:

```bash
bash examples/backends/deepinfer/launch/agg.sh --mock
```

In another terminal with the same environment:

```bash
python examples/backends/deepinfer/smoke_test.py
```

The wrapper defaults to native DeepInfer; `--mock` selects scripted CPU output.
Its `prepare_wrapper_model.py` helper downloads pinned model configuration and
tokenizer assets without weights.

## Run the Sidecar Mock

After completing the sidecar guide's Rust/Python prerequisites:

```bash
DYN_HTTP_PORT=8100 DEEPINFER_GRPC_PORT=30101 \
  bash examples/backends/deepinfer/launch.sh --verify
```

This builds and installs the Rust extension and binaries, checks the full serving
chain, and stops its processes. These ports also allow the wrapper mock to remain
on port 8000. The sidecar defaults to mock execution; `DEEPINFER_ENGINE=deepinfer`
selects native execution. Its `prepare_model.py` helper prepares the pinned Recipe
tokenizer with mock model metadata in a separate directory.

## Components for a Combined Container

Install Dynamo's Python runtime/package and the shared example requirements in
the same Python 3.12 environment:

```bash
uv pip install -r examples/backends/deepinfer/requirements.txt
```

The shared requirements cover both approaches and pin `tokenizers==0.22.2`,
which supports Dynamo's Transformers dependency. Build and install
`_deepinfer_openengine` as described in the sidecar guide. Include `dynamo-trtllm-sidecar`,
`deepinfer-example-frontend`, and the `deepinfer-openengine` probe in the image.
The sidecar client binary does not require the TensorRT-LLM Python package.

The wrapper entrypoint is `python -m dynamo.deepinfer`; the sidecar backend
entrypoint is `python examples/backends/deepinfer/server.py`. Each guide documents
its frontend and transport setup. Native experiments additionally require the
DeepInfer accelerator dependencies and a mounted checkpoint. Native GPU execution
remains unqualified by the CPU checks.
