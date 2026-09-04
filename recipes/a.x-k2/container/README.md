<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 vLLM Runtime Image

The A.X-K2 model implementation is not in upstream vLLM yet. Until it lands, the
workers in this recipe need a custom image that overlays the
[SKT-AI vLLM fork](https://github.com/SKT-AI/vllm) on a Dynamo vLLM runtime
image. The frontend runs on the stock `nvcr.io/nvidia/ai-dynamo/vllm-runtime`
image and does not need the fork.

## What the image must contain

| Requirement | Why |
| --- | --- |
| `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2` as the base | Supplies `dynamo.vllm`, vLLM 0.28, NIXL 1.3.2, UCX, and the compiled CUDA extensions the recipe was measured with. The `runtimeVersionOverride: 1.4.2` on the worker components tells the operator this version is inside the image. |
| The A.X-K2 fork's `vllm` Python package overlaid on the base image's installed `vllm` | Registers the `axk2` config and model classes. The fork changes Python files only, so the base image's compiled extensions stay in place. |
| Fork revision matched to the base image's vLLM version | A Python overlay from a different vLLM minor version breaks against the base image's compiled extensions and dependency set. Pin a full commit, never a branch name. |
| `ninja` and `nvcc` on `PATH` | DeepGEMM and FlashInfer kernels are JIT-compiled on first use. The Dynamo runtime image provides them. |

## Build pattern

The pattern below is the same "compiled base + Python overlay" build the
measured image used. Replace `AXK2_VLLM_REF` with the full commit SHA of a
fork revision whose vLLM version matches the base image. Do not build with an
unmatched revision.

```dockerfile
ARG DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2
ARG AXK2_VLLM_REPOSITORY=https://github.com/SKT-AI/vllm.git
ARG AXK2_VLLM_REF=<full-commit-sha-matching-the-base-vllm-version>

FROM ${DYNAMO_VLLM_IMAGE} AS axk2_source
ARG AXK2_VLLM_REPOSITORY
ARG AXK2_VLLM_REF
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${AXK2_VLLM_REPOSITORY}" /opt/axk2-vllm \
    && git -C /opt/axk2-vllm fetch --depth=1 origin "${AXK2_VLLM_REF}" \
    && git -C /opt/axk2-vllm checkout --detach "${AXK2_VLLM_REF}"

FROM ${DYNAMO_VLLM_IMAGE} AS runtime
USER root
COPY --from=axk2_source /opt/axk2-vllm/vllm /opt/axk2-vllm/vllm
# Overlay Python files only; keep the base image's compiled *.so files.
RUN VLLM_PACKAGE="$(python3 -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent)")" \
    && ( cd /opt/axk2-vllm/vllm && tar cf - --exclude='*.so' --exclude='__pycache__' . ) \
       | ( cd "${VLLM_PACKAGE}" && tar xf - ) \
    && rm -rf /opt/axk2-vllm \
    && find "${VLLM_PACKAGE}" -iname '*axk2*' | grep -q . \
    && set -- "${VLLM_PACKAGE}"/_C*.so && test -e "$1"
USER dynamo
WORKDIR /workspace
```

```bash
export AXK2_IMAGE=<your-registry>/dynamo/vllm-runtime:1.4.2-axk2
docker build -f Dockerfile \
  --build-arg AXK2_VLLM_REF=<full-commit-sha> \
  -t "${AXK2_IMAGE}" .
docker push "${AXK2_IMAGE}"
```

Then replace the worker `image:` in the recipe manifests with `${AXK2_IMAGE}`:

```bash
yq -i \
  '(.spec.components[] | select(.type != "frontend") | .podTemplate.spec.containers[] | select(.name == "main").image) = strenv(AXK2_IMAGE)' \
  vllm/disagg-h200/deploy.yaml
```

## Verify

On a host with an NVIDIA driver:

```bash
docker run --rm --gpus all --entrypoint python3 "${AXK2_IMAGE}" -c \
  "import dynamo.vllm, vllm; print(vllm.__version__); from vllm.model_executor.models.registry import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if 'AXK2' in a])"
```

The command must print the vLLM version of the base image and a non-empty list
of `AXK2*` architectures. Full verification (NIXL handshake, KV events, parser
registration) requires starting the aggregated recipe on one 8-GPU H200 node.

## Validation status

- The recipes were measured with an image built this way from an A.X-K2 fork
  revision at vLLM `0.28.1rc1` on the Dynamo `1.4.2` runtime (NIXL 1.3.2). That
  revision is not on the public fork yet.
- The public fork branch `axk2-v0.23.0` targets vLLM 0.23.0 and pairs with
  `vllm-runtime:1.3.0`, not `1.4.2`. The model publisher states that newer bases
  will be published as additional branches. Two flags in the manifests are
  spelled for the 0.28 CLI and need checking against an older fork:
  `--kernel-config.enable_flashinfer_autotune=False` (vLLM 0.23 uses
  `--no-enable-flashinfer-autotune`) and `--dyn-default-thinking-mode`.
- The A.X-K2 model card also publishes `skt/A.X-K2-NVFP4` for Blackwell. The
  `*-h200` targets serve the FP8 checkpoint on Hopper.
