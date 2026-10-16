---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Decode GPU Requirements
---

Dynamo provides two video-input decode paths. CUDA runtime images can use both;
vLLM CPU and XPU runtimes can use the frontend software path:

- H.264 and H.265 (HEVC) decode on the GPU using NVDEC, NVIDIA's dedicated
  hardware video decoder, through
  [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/) when explicitly
  enabled with `DYN_DISABLE_NVDEC=0`. See [Configuration](#configuration).
- VP8 and VP9 decode on the CPU through Dynamo's codec-limited, in-tree FFmpeg
  when frontend decoding is enabled on vLLM, or SGLang on CUDA.

The in-tree FFmpeg does not include H.264, H.265, or AV1 decoders. Without
frontend decoding, video input remains owned by the backend and requires its
Python decode carrier (OpenCV, PyAV, or decord). The images deliberately omit
those wider software-decode carriers by default; vLLM does ship OpenCV, but it
is built without a video backend for still-image work only.

This page covers which GPUs provide NVDEC, what the container must expose, and how
Dynamo behaves when hardware decode is unavailable.

## Enabling NVDEC

NVDEC video decoding is disabled by default on the SGLang, vLLM, and TensorRT-LLM
workers. Set `DYN_DISABLE_NVDEC=0` before starting a worker to enable it for
trusted H.264/H.265 inputs. The shipped images do not include a software decoder
for these codecs, so requests using them fail while NVDEC is disabled.

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

Every GPU above decodes both codecs Dynamo routes to hardware when NVDEC is enabled.
Hopper's NVDEC matches Turing's feature set and does **not** decode AV1; Blackwell adds
AV1 decode. The Ada parts (L4, L40S, RTX 6000 Ada) decode AV1 as well, and unlike the
datacenter accelerators they also carry NVENC.

Because no datacenter GPU ships NVENC, Dynamo's video *generation* path encodes with a
CPU VP9 encoder rather than a hardware H.264 encoder.

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
> Missing the `video` capability also makes hardware decode unavailable after
> `DYN_DISABLE_NVDEC=0` is set. The GPU itself is fine; the container simply
> cannot see the decoder.

## Verifying hardware decode

```bash
python3 -c "
from dynamo.common.multimodal.nvdec_decoder import nvdec_available
print('NVDEC available:', nvdec_available())
"
```

`False` means Dynamo will not use hardware decode in that container. Check, in order:
`DYN_DISABLE_NVDEC=0`, the `video` driver capability, and that
`PyNvVideoCodec` is installed. Set the flag before starting the worker.

To distinguish "capability missing" from every other cause, look for the decode library
itself. It is mounted by the container runtime, not installed by the image, so its absence
points squarely at the capability:

```bash
ldconfig -p | grep -i nvcuvid
```

Expect `libnvcuvid.so.1`. Nothing means the `video` capability did not reach this
container. If it is present on the node but not inside, the capability is being dropped
between the two.

## Behavior when NVDEC is disabled or unavailable

Hardware decode is additive and never blocks a request on its own: routing falls through
to a software decode path where one exists. VP8 and VP9 frontend decoding on vLLM, or
SGLang on CUDA, does not depend on NVDEC.

> [!IMPORTANT]
> The shipped images do not include a software H.264, H.265, or AV1 decoder.
> Therefore, if NVDEC is disabled or unavailable, H.264 and H.265 fail with a
> missing-decoder error unless the backend's wider Python decode carrier is
> installed. AV1 also requires an additional carrier because Dynamo does not
> route it through NVDEC and the in-tree FFmpeg excludes it.
>
> To use NVDEC for H.264 and H.265, set `DYN_DISABLE_NVDEC=0` and grant the
> container the `video` driver capability. See [Configuration](#configuration).
> For VP8 and VP9, use frontend decoding on vLLM, or SGLang on CUDA. For
> other software-decoded cases, install a decode carrier alongside or transcode
> the input before sending it.

### Installing a software decoder

To decode a format NVDEC does not cover — or H.264/H.265 when NVDEC is disabled
or unavailable — explicitly install the backend's decode package at the
validated version bounds:

```bash
# vLLM: audio input
pip install --no-deps 'av>=18.0.0,<19'

# SGLang: video input
pip install --no-deps 'decord2>=3.4.0,<4'

# TensorRT-LLM: video input
pip install --no-deps 'opencv-python-headless>=4.13.0.92,<5'
```

vLLM **video** input is the exception, because those images already ship OpenCV built
without a video backend. Adding video decode there means replacing that build with the
binary wheel of the same version, not installing a range:

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

With neither pip metadata nor an importable `cv2`, this falls back to the bounded range instead of
an unresolvable `==.*` pin.

Nothing installs automatically — this is a deliberate operator step. The images also ship
an installer with the same bounds plus idempotency and air-gap support
(`python -m dynamo.common.utils.install_media_decoders <backend>`). For vLLM, it replaces
codec-free OpenCV with a same-version binary wheel and installs missing PyAV. See
[Additional Media Decoders](additional-media-decoders.md) for the full workflow,
including baking the install into an image layer for Kubernetes.

## Hardware encode (NVENC)

There is nothing to enable. Dynamo does not use NVENC on any path.

Video **output** — the generation path — encodes VP9 on the CPU with the in-tree FFmpeg.
That is deliberate and works everywhere: no datacenter GPU ships an encoder, so a
hardware encode path would be unavailable on exactly the parts most deployments run. On
workstation parts that do have NVENC (L4, L40S, RTX 6000 Ada) it simply stays unused.

The same `video` driver capability governs both engines, so a container configured for
NVDEC as above needs no additional change. Encode performance does not depend on it.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_DISABLE_NVDEC` | unset (NVDEC disabled) | Set to `0` to enable NVDEC for H.264/H.265 input. `false` and `no` also enable it; any other value leaves it disabled. Without a software decoder, H.264/H.265 requests fail while NVDEC is disabled. VP8/VP9 are unaffected. |
| `DYN_NVDEC_GPU_ID` | `0` | GPU ordinal used for decode. |
| `DYN_MM_VIDEO_NUM_FRAMES` | `32` | Frames sampled uniformly from each clip. |
| `DYN_MM_MAX_FILE_SIZE_MB` | `64` | Maximum size in MiB for each remote image, audio, or video download. TensorRT-LLM uses its backend-specific `--max-file-size-mb` option (`DYN_TRTLLM_MAX_FILE_SIZE_MB`) instead. |

## Sources

- [NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
- [NVDEC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-application-note/index.html)
