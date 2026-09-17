---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Additional Media Decoders
subtitle: Explicitly install software decoders for input formats the image cannot decode
---

Dynamo's runtime images ship a deliberately small media stack. The in-tree FFmpeg is built for VP8/VP9 video, and the wider backend decode packages are not pre-installed, which keeps the distributed images small and their input-format surface narrow by default.

Two input classes are already covered without installing anything:

- **VP8 and VP9 video** decode through the in-tree FFmpeg.
- **H.264 and H.265 video** decode on the GPU through NVDEC, which every backend uses by default. NVDEC needs a GPU with a video decode engine and a container granted the `video` driver capability. See [Video Decode GPU Requirements](video-decode-gpu-requirements.md) for the hardware and capability matrix.

Installing an additional decoder package covers what remains:

- **AAC and other compressed audio**, which NVDEC does not decode at all.
- **H.264 and H.265 on hosts where NVDEC is unavailable** — no video decode engine on the GPU, or a container without the `video` capability.

Each backend decodes such input through a specific Python package whose wheel bundles its own FFmpeg, so the support is added with a plain `pip install` — no image rebuild. **Nothing installs automatically.** There is no environment switch and no startup hook: an operator runs the install as a deliberate, visible step, so a deployment that broadens the image's codec surface says so in its Dockerfile, pod spec, or runbook.

> [!NOTE]
> This affects **input decoding** only. Generated video **output** is unaffected and always uses VP9.

## What each backend needs

| Backend | Input | Package (validated version bounds) | Import |
|---------|-------|------------------------------------|--------|
| vLLM | video | `opencv-python-headless`, matching the installed version | `cv2` |
| vLLM | audio | `av>=18.0.0,<19` | `av` |
| SGLang | video | `decord2>=3.4.0,<4` | `decord` |
| TensorRT-LLM | video | `opencv-python-headless>=4.13.0.92,<5` | `cv2` |

For vLLM video, the installer replaces codec-free OpenCV with a binary wheel of the installed version. If OpenCV is absent, it uses `opencv-python-headless>=4.13.0.92,<5`. For the bounded specs, the lower bound is the version validated against Dynamo's multimodal test suite; the upper bound excludes the next major release so an install cannot silently pick up an unvalidated version. PyNvVideoCodec is not in this list because every CUDA runtime image already ships it — it is the NVDEC path, not a fallback, so there is nothing to install. torchcodec is left out because no Dynamo decode path imports it.

## Install with pip

The table above is the contract; these commands are its direct translation, and work with the installer of your choice (`pip`, `uv pip`, ...):

```bash
# vLLM: audio input
pip install --no-deps 'av>=18.0.0,<19'

# SGLang: video input
pip install --no-deps 'decord2>=3.4.0,<4'

# TensorRT-LLM: video input
pip install --no-deps 'opencv-python-headless>=4.13.0.92,<5'
```

`--no-deps` keeps the install from changing the image's pinned dependency stack (for example numpy under PyTorch/vLLM); the carriers only need numpy, which the backend already provides. To pin different versions or install a custom subset, adjust these commands directly — that is the intended customization path.

## Install with the bundled installer

Every runtime image ships an installer for the packages above as part of the `ai-dynamo` package. Run it in the worker container (not the frontend) for the backend you deploy:

```bash
python -m dynamo.common.utils.install_media_decoders vllm
```

It skips usable decoders, serializes concurrent installs, and verifies imports in a fresh interpreter. For OpenCV, it also checks for a video backend. On vLLM, it replaces codec-free OpenCV with a same-version binary wheel using `--force-reinstall --only-binary` and installs missing PyAV. Other packages use the validated bounds. It exits non-zero if verification fails. To see what it would do first:

```bash
python -m dynamo.common.utils.install_media_decoders vllm --dry-run
```

## Kubernetes

Prefer baking the install into an image layer, so the deployment manifest deploys exactly what was reviewed and scanned:

```dockerfile
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:x.y.z
RUN python -m dynamo.common.utils.install_media_decoders vllm
```

For a quick evaluation without an image build, run the installer ahead of the worker in the pod command, where it is visible in the manifest:

```yaml
command: ["sh", "-c"]
args:
  - python -m dynamo.common.utils.install_media_decoders vllm &&
    exec python -m dynamo.vllm --model $MODEL
```

## Air-gapped and allowlisted networks

The install needs a package index or a local wheelhouse. Point pip at one with `--pip-args` (use the `=` form, so a value starting with a dash is not mistaken for an option):

```bash
python -m dynamo.common.utils.install_media_decoders vllm \
  --pip-args="--no-index --find-links /opt/wheels"
```

The default pip timeout is 600 seconds (`--timeout-s` overrides it; `0` disables it), so a stalled index fails the run rather than hanging it.

## Notes and limits

- For H.264 and H.265, prefer NVDEC. Granting the container the `video` driver capability decodes those formats on the GPU with no extra package. Install a software decoder when that is not an option, or when the input is audio.
- Installing a decoder package brings in that wheel's bundled media libraries. The runtime images are scanned for media components at build time; a package installed afterwards is not covered by that scan. Review what your deployment ships — a baked image layer keeps the change visible and reviewable.
- On TensorRT-LLM, the install puts back `opencv-python-headless`, which those images deliberately do not ship. H.264 and H.265 already decode there through NVDEC, so install it only for a host where NVDEC is unavailable.
- The vLLM images ship OpenCV already, rebuilt from source with every video backend disabled. It covers still images — which multimodal Mistral models need, because `mistral_common` resizes every image through `cv2` — and decodes no video at all. Video input on vLLM goes through NVDEC. To decode video in software instead, swap that build for the PyPI wheel of the same version:

```bash
VERSION=$(pip show opencv-python-headless | awk '/^Version:/{print $2}')
if [ -n "$VERSION" ]; then
  SPEC="opencv-python-headless==${VERSION}"
# Without metadata, match the library version plus its packaging revision.
elif VERSION=$(python -c 'import cv2; print(cv2.__version__)') && [ -n "$VERSION" ]; then
  SPEC="opencv-python-headless==${VERSION}.*"
else
  SPEC='opencv-python-headless>=4.13.0.92,<5'
fi
pip install --no-deps --force-reinstall --only-binary opencv-python-headless "$SPEC"
```

`--force-reinstall --only-binary` replaces the source build even when its version already satisfies pip. Pinning preserves the image's OpenCV version; with neither metadata nor an importable `cv2`, the command falls back to the bounded range, which is what the bundled installer does too. The wheel restores bundled FFmpeg and codecs, so review it against your distribution policy. The bundled vLLM installer performs this replacement and installs missing PyAV.

- The optional Rust frontend decoder (`--frontend-decoding`) links FFmpeg's compiled-in decoders and always decodes VP8/VP9 regardless of installed Python packages; backend decoding is what an install extends. Re-encoding an input to VP9 (`ffmpeg -i input.mp4 -c:v libvpx-vp9 -an output.webm`) is an alternative that needs no additional packages.
