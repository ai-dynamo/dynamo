<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X K2 NVFP4 vLLM Runtime Image

A.X K2 requires the
[`SKT-AI/vllm`](https://github.com/SKT-AI/vllm/tree/axk2-v0.23.0) fork until its
model implementation is available in upstream vLLM. The Dockerfile pins commit
`9cca6f06f4b9d0b4247f9446252ed81cf197aaf4` from the `axk2-v0.23.0` branch and
overlays its Python package on
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`. Both use vLLM 0.23.0, so the
image retains the base image's compiled CUDA extensions, Dynamo integration,
and NIXL libraries without rebuilding vLLM.

This integration is experimental. The build checks the vLLM version and the
presence of the compiled extension and A.X K2 implementation without loading
architecture-specific CUDA libraries. Verify Dynamo imports, NIXL, parser
registration, and model execution with a 4-GPU B200 smoke test.

## Build

Build the A.X K2 runtime from the repository root:

```bash
export AXK2_IMAGE=<your-registry>/dynamo/vllm-runtime:axk2-v0.23.0

docker build \
  -f recipes/a.x-k2/container/Dockerfile.axk2.vllm.b200 \
  -t "${AXK2_IMAGE}" \
  .

docker push "${AXK2_IMAGE}"
```

To test another fork revision, pass
`--build-arg AXK2_VLLM_REF=<full-commit-sha>`. Pin a full commit rather than the
branch name.

To use another compatible Dynamo image, pass
`--build-arg DYNAMO_VLLM_IMAGE=<image>`. The image must contain vLLM 0.23.0;
the Dockerfile rejects another version during the build.

## Verify

Run import checks on an AMD64 host with an NVIDIA driver:

```bash
docker run --rm --gpus all --entrypoint python3 "${AXK2_IMAGE}" -c \
  "import dynamo.vllm; from vllm.transformers_utils.configs.axk2 import AXK2Config; print(AXK2Config.model_type)"
```

The command must print `axk2`. A complete verification still requires starting
the aggregated recipe on a 4-GPU B200 node.
