---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Decoding and GPU Requirements
subtitle: Decode video input with NVDEC or the frontend decoder, and add software decoders when needed
---

Dynamo provides two video-input decode paths. CUDA runtime images can use both;
vLLM CPU and XPU runtimes can use the frontend software path:

- H.264 and H.265 (HEVC) decode on the GPU using NVDEC, NVIDIA's dedicated
  hardware video decoder, through
  [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/).
- VP8 and VP9 decode on the CPU through Dynamo's codec-limited, in-tree FFmpeg
  when frontend decoding is enabled on vLLM, or SGLang on CUDA.

The in-tree FFmpeg does not include H.264, H.265, or AV1 decoders. Without
frontend decoding, video input remains owned by the backend and requires its
Python decode carrier (OpenCV, PyAV, or decord). The images deliberately omit
those wider software-decode carriers by default; vLLM does ship OpenCV, but it
is built without a video backend for still-image work only.

This page covers which GPUs provide NVDEC, what the container must expose, and how
Dynamo behaves when hardware decode is unavailable.

## GPU support

NVDEC is a fixed-function decode engine, separate from the SMs used for inference, so
decoding adds negligible load to the GPU beyond a small YUV-to-RGB conversion.

A common misconception is that datacenter GPUs have no video engines. That applies to
**NVENC**, the hardware *encoder*, which NVIDIA omits from datacenter parts. The
*decoder* is present:

| GPU | Architecture | NVDEC (decode) | NVENC (encode) | H.264 | HEVC |
|-----|--------------|----------------|----------------|-------|------|
| A100 | Ampere | 5 engines | none | Yes | Yes |
| H100, H200 | Hopper | 7 engines | none | Yes | Yes (Main, Main 10) |
| B200, GB200 | Blackwell | 7 engines | none | Yes | Yes (Main, Main 10, 422 10/12) |
| L4 | Ada Lovelace | 4 engines | 2 | Yes | Yes |
| L40, L40S | Ada Lovelace | 3 engines | 3 | Yes | Yes |
| RTX 6000 Ada | Ada Lovelace | 3 engines | 3 | Yes (8/10-bit) | Yes (8/10/12-bit, up to 4:4:4) |

Every GPU above decodes both codecs Dynamo routes to hardware, so H.264 and H.265 video
input works across the lineup. Hopper's NVDEC matches Turing's feature set and
does **not** decode AV1; Blackwell adds AV1 decode. The Ada parts (L4, L40S, RTX 6000 Ada)
decode AV1 as well, and unlike A100, H100, H200, B200, and GB200 they also carry NVENC.

> [!NOTE]
> Under Multi-Instance GPU (MIG), NVDEC engines are divided across instances. A given MIG
> profile may expose fewer decoders than the full GPU, and some profiles expose none.
> Verify decode works in the exact profile you deploy.


## Container requirements

NVDEC links `libnvcuvid` at runtime, which the NVIDIA container runtime only mounts when
the **`video` driver capability** is requested. Without it, `import PyNvVideoCodec` fails
and Dynamo falls back to software decode.

Dynamo's runtime images already declare it:

```dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
```

With Docker that is usually enough, because the toolkit reads the capability from the
image. To be explicit, or when overriding the variable for other reasons:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility ...
```

### Kubernetes

> [!IMPORTANT]
> On Kubernetes the image's `ENV` is **not** reliably sufficient. Set the variable on the
> container spec as well. Dynamo's own GPU test runners shipped images carrying the `ENV`
> and still had no hardware decode until the pod spec set it explicitly.

Add it to the container that runs the worker:

```yaml
spec:
  containers:
    - name: worker
      env:
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: video,compute,utility
      resources:
        limits:
          nvidia.com/gpu: "1"
```

If it still does not take effect, the capability is being dropped below the pod. Check
`supported-driver-capabilities` in `/etc/nvidia-container-runtime/config.toml` on the
node, and — if the cluster runs in CDI mode — whether the generated device spec includes
the video libraries, since in that mode capabilities come from the spec rather than the
environment variable.

> [!WARNING]
> Missing the `video` capability is the most common cause of hardware decode being
> silently unavailable. The GPU itself is fine; the container simply cannot see the
> decoder.

## Verifying hardware decode

```bash
python3 -c "
from dynamo.common.multimodal.nvdec_decoder import nvdec_available
print('NVDEC available:', nvdec_available())
"
```

`False` means Dynamo will not use hardware decode in that container. Check, in order: the
`video` driver capability, that `PyNvVideoCodec` is installed, and that
`DYN_DISABLE_NVDEC` is unset.

To distinguish "capability missing" from every other cause, look for the decode library
itself. It is mounted by the container runtime, not installed by the image, so its absence
points squarely at the capability:

```bash
ldconfig -p | grep -i nvcuvid
```

Expect `libnvcuvid.so.1`. Nothing means the `video` capability did not reach this
container. If it is present on the node but not inside, the capability is being dropped
between the two.


## Behavior when NVDEC is unavailable

Hardware decode is additive and never blocks a request on its own: routing falls through
to a software decode path where one exists. VP8 and VP9 frontend decoding on vLLM, or
SGLang on CUDA, does not depend on NVDEC.

> [!IMPORTANT]
> The shipped images do not include a software H.264, H.265, or AV1 decoder.
> Therefore, if NVDEC is unavailable, H.264 and H.265 fail with an
> unsupported-codec error unless the backend's wider Python decode carrier is
> installed. AV1 also requires an additional carrier because Dynamo does not
> route it through NVDEC and the in-tree FFmpeg excludes it.
>
> Grant the container the `video` driver capability so NVDEC can serve H.264 and H.265.
> For VP8 and VP9, use frontend decoding on vLLM, or SGLang on CUDA. For
> other software-decoded cases, [install an additional decoder](#install-additional-decoders)
> or transcode the input before sending it.


## Install Additional Decoders

Dynamo's runtime images ship a deliberately small media stack. The in-tree FFmpeg is built for VP8/VP9 video, and the wider backend decode packages are not pre-installed, which keeps the distributed images small and their input-format surface narrow by default.

Two input classes are already covered without installing anything:

- **VP8 and VP9 video** decode through the in-tree FFmpeg when frontend
  decoding is enabled on vLLM, or SGLang on CUDA.
- **H.264 and H.265 video** decode on the GPU through NVDEC on supported backend paths. NVDEC needs a GPU with a video decode engine and a container granted the `video` driver capability. See [GPU Support](#gpu-support) for the hardware and capability matrix.

Without frontend decoding, the backend worker owns VP8/VP9 decoding. Follow the
[same-version replacement procedure](#notes-and-limits) for vLLM video; other
backends need the corresponding package from the table below. Installing an
additional decoder package also covers:

- **AAC and other compressed audio**, which NVDEC does not decode at all.
- **H.264 and H.265 on hosts where NVDEC is unavailable** — no video decode engine on the GPU, or a container without the `video` capability.

Each backend decodes such input through a specific Python package whose wheel bundles its own FFmpeg, so the support is added with a plain `pip install` — no image rebuild. **Nothing installs automatically.** There is no environment switch and no startup hook: an operator runs the install as a deliberate, visible step, so a deployment that broadens the image's codec surface says so in its Dockerfile, pod spec, or runbook.

> [!NOTE]
> This affects **input decoding** only. Generated video **output** is unaffected and always uses VP9.

### What Each Backend Needs

| Backend | Input | Package (validated version bounds) | Import |
|---------|-------|------------------------------------|--------|
| vLLM | video | `opencv-python-headless`, matching the installed version | `cv2` |
| vLLM | audio | `av>=18.0.0,<19` | `av` |
| SGLang | video | `decord2>=3.4.0,<4` | `decord` |
| TensorRT-LLM | video | `opencv-python-headless>=4.13.0.92,<5` | `cv2` |

For vLLM video, the installer replaces codec-free OpenCV with a binary wheel of the installed version. If OpenCV is absent, it uses `opencv-python-headless>=4.13.0.92,<5`. For the bounded specs, the lower bound is the version validated against Dynamo's multimodal test suite; the upper bound excludes the next major release so an install cannot silently pick up an unvalidated version. PyNvVideoCodec is not in this list because every CUDA runtime image already ships it — it is the NVDEC path, not a fallback, so there is nothing to install. torchcodec is left out because no Dynamo decode path imports it.

### Install with pip

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

### Install with the bundled installer

Every runtime image ships an installer for the packages above as part of the `ai-dynamo` package. Run it in the worker container (not the frontend) for the backend you deploy:

```bash
python -m dynamo.common.utils.install_media_decoders vllm
```

It skips usable decoders, serializes concurrent installs, and verifies imports in a fresh interpreter. For OpenCV, it also checks for a video backend. On vLLM, it replaces codec-free OpenCV with a same-version binary wheel using `--force-reinstall --only-binary` and installs missing PyAV. Other packages use the validated bounds. It exits non-zero if verification fails. To see what it would do first:

```bash
python -m dynamo.common.utils.install_media_decoders vllm --dry-run
```

### Kubernetes

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

### Air-gapped and allowlisted networks

The install needs a package index or a local wheelhouse. Point pip at one with `--pip-args` (use the `=` form, so a value starting with a dash is not mistaken for an option):

```bash
python -m dynamo.common.utils.install_media_decoders vllm \
  --pip-args="--no-index --find-links /opt/wheels"
```

The default pip timeout is 600 seconds (`--timeout-s` overrides it; `0` disables it), so a stalled index fails the run rather than hanging it.


### Notes and Limits

- For H.264 and H.265, prefer NVDEC. Granting the container the `video` driver capability decodes those formats on the GPU with no extra package. Install a software decoder when that is not an option, or when the input is audio.
- Installing a decoder package brings in that wheel's bundled media libraries. The runtime images are scanned for media components at build time; a package installed afterwards is not covered by that scan. Review what your deployment ships — a baked image layer keeps the change visible and reviewable.
- On TensorRT-LLM, the install puts back `opencv-python-headless`, which those images deliberately do not ship. H.264 and H.265 already decode there through NVDEC, so install it only for a host where NVDEC is unavailable.
- The vLLM images ship OpenCV already, rebuilt from source with every video backend disabled. It covers still images — which multimodal Mistral models need, because `mistral_common` resizes every image through `cv2` — and decodes no video at all. Backend video input on vLLM goes through NVDEC. To decode video in software instead, swap that build for the PyPI wheel of the same version:

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

## Video Output and NVENC

There is nothing to enable. Dynamo does not use NVENC on any path.

Video **output** — the generation path — encodes VP9 on the CPU with the in-tree FFmpeg.
That is deliberate and works everywhere: A100, H100, H200, B200, and GB200 have no
hardware encoder, so a hardware encode path would be unavailable on the GPUs most
deployments run. On GPUs that do have NVENC (L4, L40S, RTX 6000 Ada) it stays unused.

The same `video` driver capability governs both engines, so a container configured for
NVDEC as above needs no additional change. Encode performance does not depend on it.


## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_DISABLE_NVDEC` | unset | Set to `1` to skip hardware decode. In a shipped image that leaves H.264 and H.265 input with no decoder, so it is a debugging switch rather than a fallback for those codecs. VP8 and VP9 frontend decoding is unaffected. Read as a boolean: `1`/`true`/`yes` disable, anything else does not. |
| `DYN_NVDEC_GPU_ID` | `0` | GPU ordinal used for decode. |
| `DYN_MM_VIDEO_NUM_FRAMES` | `32` | Frames sampled uniformly from each clip. |
| `DYN_MM_MAX_FILE_SIZE_MB` | `64` | Maximum size in MiB for each remote image, audio, or video download. TensorRT-LLM uses its backend-specific `--max-file-size-mb` option (`DYN_TRTLLM_MAX_FILE_SIZE_MB`) instead. |


## Sources

- [NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
- [NVDEC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-application-note/index.html)

