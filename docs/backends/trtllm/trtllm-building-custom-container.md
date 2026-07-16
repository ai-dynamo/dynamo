---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Building a Custom TensorRT-LLM Container
---

For the prebuilt container, see the [TensorRT-LLM Quick Start](README.md#quick-start).

## How the Image Is Composed

The Dynamo TensorRT-LLM image layers Dynamo on top of the upstream NVIDIA TensorRT-LLM release container — it does **not** build TensorRT-LLM from source. The base used by the rendered Dockerfile is set in [`container/context.yaml`](../../../container/context.yaml):

```yaml
trtllm:
  cuda13.1:
    runtime_image: nvcr.io/nvidia/tensorrt-llm/release
    runtime_image_tag: 1.3.0rc14
```

For `--target=runtime` (the focus of this guide), `container/render.py` emits a stage that:

1. Starts from `${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG}` (the upstream TRT-LLM image),
2. Builds the Dynamo wheels (`ai-dynamo`, `ai-dynamo-runtime`, optionally `kvbm`) in a separate `wheel_builder` manylinux stage, and
3. Installs those wheels into a `/opt/dynamo/venv` virtual environment created with `--system-site-packages` so the upstream Python solve stays importable.

The `dev` and `local-dev` targets create `/opt/dynamo/venv` in `container/templates/dev.Dockerfile` and lay down the build toolchain (`maturin`, `uv`) so the user can run `maturin develop` against a mounted workspace at runtime for an editable install — the Dynamo wheels in `/opt/dynamo/wheelhouse/` are left uninstalled in these targets.

The upstream `nvcr.io/nvidia/tensorrt-llm/release` image ships a multi-arch manifest (`linux/amd64` and `linux/arm64`), so the Dynamo TensorRT-LLM image can be built for either architecture.

## Building the Default Image

```bash
# On an x86_64 host:
python container/render.py --framework=trtllm --target=runtime --output-short-filename --cuda-version=13.1
docker build -t dynamo:trtllm-latest -f container/rendered.Dockerfile .

# On an arm64 host (Grace, arm64 EC2, etc.):
python container/render.py --framework=trtllm --target=runtime --platform=linux/arm64 --output-short-filename --cuda-version=13.1
docker buildx build --platform=linux/arm64 -t dynamo:trtllm-latest -f container/rendered.Dockerfile .
```

Run the resulting image:

```bash
./container/run.sh --image dynamo:trtllm-latest -it
```

## Pinning a Different Upstream TensorRT-LLM Release

To pick up a different upstream TRT-LLM release (newer rc tag, a hotfix tag, etc.) without editing `context.yaml`, override the runtime image ARGs at `docker build` time:

```bash
python container/render.py --framework=trtllm --target=runtime --output-short-filename --cuda-version=13.1
docker build --pull \
  --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt-llm/release \
  --build-arg RUNTIME_IMAGE_TAG=<your-tag> \
  -t dynamo:trtllm-latest -f container/rendered.Dockerfile .
```

`--pull` is recommended when changing only the upstream tag: without it, Docker may reuse a previously-cached layer that resolved against the old tag's manifest, producing a half-stale image that boots but breaks at NIXL init if upstream moved bundled libraries between tags.

If a tag move changes where the upstream image installs `libnixl.so` or the NIXL plugin directory, the runtime stage's `test -f` / `test -d` guards fail the build instead of producing a silently broken image. Update the `LD_PRELOAD` / `NIXL_PLUGIN_DIR` paths in [`container/templates/trtllm_runtime.Dockerfile`](../../../container/templates/trtllm_runtime.Dockerfile) and re-run `render.py` (otherwise `container/rendered.Dockerfile` is stale and the build silently uses the old paths) if that happens.

## Building TensorRT-LLM From Source

Dynamo no longer builds TensorRT-LLM itself. If you need a custom TRT-LLM build (your own patch, a non-released commit, etc.), the supported path is to produce an upstream-equivalent image yourself and point Dynamo at it via the same `RUNTIME_IMAGE` / `RUNTIME_IMAGE_TAG` build args.

1. Build a TensorRT-LLM container following the upstream instructions: [TensorRT-LLM — Build from Source on Linux](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html). Use the "Building a TensorRT LLM Docker Image" section (specifically `make -C docker release_build`) — this produces a container with `tensorrt_llm` in the system site-packages **and** the bundled `libnixl.so` at the canonical path. A bare `build_wheel.py` invocation only produces a wheel and won't have the NIXL bits.

   Three constraints the resulting image must satisfy or the Dynamo build will fail its sanity guards:
   - **Python 3.12** in the system Python (`/usr/local/lib/python3.12/dist-packages/...`) — the `LD_PRELOAD` and `NIXL_PLUGIN_DIR` paths in the runtime Dockerfile are hardcoded to 3.12. If your custom build switches to a different Python minor version, edit those env vars in [`container/templates/trtllm_runtime.Dockerfile`](../../../container/templates/trtllm_runtime.Dockerfile) and re-render.
   - `tensorrt_llm` installed into system site-packages (not a venv), matching upstream layout.
   - `libnixl.so` present at `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so` and plugins under `nixl/plugins/`.
2. Tag it locally, e.g. `nvcr.io/nvidia/ai-dynamo/tensorrt-llm:<commit-sha>` (use the source commit you built so the tag carries provenance — pasting a literal `:custom` makes the image untraceable later).
3. Render and build the Dynamo image against your custom base:

   ```bash
   python container/render.py --framework=trtllm --target=runtime --output-short-filename --cuda-version=13.1
   docker build \
     --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/tensorrt-llm \
     --build-arg RUNTIME_IMAGE_TAG=<commit-sha> \
     -t dynamo:trtllm-<commit-sha> -f container/rendered.Dockerfile .
   ```

   Do **not** add `--pull` here — your custom image only exists locally, and `--pull` will make Docker try to fetch it from docker.io and fail. `--pull` is only useful when the base image lives in a remote registry (see the previous section on pinning an upstream tag).

If your custom build places TRT-LLM's bundled NIXL at a different path (or uses a non-3.12 Python), edit the `LD_PRELOAD` and `NIXL_PLUGIN_DIR` env vars in [`container/templates/trtllm_runtime.Dockerfile`](../../../container/templates/trtllm_runtime.Dockerfile) (and the matching `test -f`/`test -d` guards), **then re-run `python container/render.py ...` to regenerate `container/rendered.Dockerfile` before `docker build`** — otherwise the build silently uses the previously-rendered file. Those env vars exist to work around [`ai-dynamo/nixl#1668`](https://github.com/ai-dynamo/nixl/issues/1668) — `nixl-cu13`'s bundled UCX 1.20.0 hangs under multi-agent init — by forcing every process in the image to load TRT-LLM's 0.9.0 `libnixl.so` instead.
