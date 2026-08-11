---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Decode GPU Requirements
---

Dynamo decodes H.264 and H.265 (HEVC) video input on the GPU using NVDEC, NVIDIA's
dedicated hardware video decoder, through
[PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/).

Other formats — VP8, VP9 and AV1 — have **no video-input decoder** in the shipped images.
The in-tree VP8/VP9 FFmpeg serves the video *output* (generation) path; it is not wired to
video input, and the Rust `media-ffmpeg` decoder is not built into these images. Video
input decodes through Python carriers (OpenCV, PyAV, decord) that the images deliberately
omit, so a VP8/VP9/AV1 clip fails with an unsupported-codec error unless one of those
packages is installed alongside.

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
input works across the datacenter lineup. Hopper's NVDEC matches Turing's feature set and
does **not** decode AV1; Blackwell adds AV1 decode. The Ada parts (L4, L40S, RTX 6000 Ada)
decode AV1 as well, and unlike the datacenter accelerators they also carry NVENC.

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
to the software decode path where one exists.

> [!IMPORTANT]
> In the shipped images there is no software decode path for video input, for any format.
> The Python carriers that decode video input (OpenCV, PyAV, decord) are deliberately not
> installed, and the in-tree VP8/VP9 FFmpeg serves the video *output* path rather than
> input. So if NVDEC is unavailable, H.264 and H.265 fail with an unsupported-codec error
> — and VP8, VP9 and AV1 fail the same way whether NVDEC is available or not, since NVDEC
> does not decode them either.
>
> Grant the container the `video` driver capability so NVDEC can serve H.264 and H.265.
> For the other formats, install a decode carrier alongside, or transcode the input to
> H.264/H.265 before sending it.

### Installing a software decoder

To decode a format NVDEC does not cover — or H.264/H.265 on a host with no NVDEC —
explicitly install the backend's decode package at the validated version bounds:

```bash
# vLLM: video + audio input
pip install --no-deps 'opencv-python-headless>=4.13.0.92,<5' 'av>=18.0.0,<19'

# SGLang: video input
pip install --no-deps 'decord2>=3.4.0,<4'

# TensorRT-LLM: video input
pip install --no-deps 'opencv-python-headless>=4.13.0.92,<5'
```

Nothing installs automatically — this is a deliberate operator step. The images also ship
an installer with the same bounds plus idempotency and air-gap support
(`python -m dynamo.common.utils.install_media_decoders <backend>`); see
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
| `DYN_DISABLE_NVDEC` | unset | Set to `1` to skip hardware decode. In a shipped image that leaves video input with no decoder at all, so it is a debugging switch rather than a fallback. Read as a boolean: `1`/`true`/`yes` disable, anything else does not. |
| `DYN_NVDEC_GPU_ID` | `0` | GPU ordinal used for decode. |
| `DYN_MM_VIDEO_NUM_FRAMES` | `32` | Frames sampled uniformly from each clip. |

## Sources

- [NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
- [NVDEC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-application-note/index.html)
