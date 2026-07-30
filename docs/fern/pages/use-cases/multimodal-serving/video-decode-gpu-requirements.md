---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Decode GPU Requirements
---

Dynamo decodes H.264 and H.265 (HEVC) video input on the GPU using NVDEC, NVIDIA's
dedicated hardware video decoder, through
[PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/). Other formats (VP8, VP9)
continue to use the software decode path.

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

Every GPU above decodes both codecs Dynamo routes to hardware, so H.264 and H.265 video
input works across the datacenter lineup. Hopper's NVDEC matches Turing's feature set and
does **not** decode AV1; Blackwell adds AV1 decode.

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

Dynamo's runtime images already set this:

```dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
```

When running the image yourself, request the capability explicitly:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video ...
```

On Kubernetes, confirm your device plugin or `runtimeClass` does not drop `video` from
the capability list.

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

## Behavior when NVDEC is unavailable

Hardware decode is additive and never blocks a request on its own: routing falls through
to the software decode path.

> [!IMPORTANT]
> Dynamo's runtime images ship a VP8/VP9-only in-tree FFmpeg, so the software path cannot
> decode H.264 or H.265. If NVDEC is unavailable in one of these images, those formats
> have no decoder and the request fails with an unsupported-codec error. Install the
> optional software decoders (`DYN_ENABLE_MEDIA_DECODERS`) or fix the `video` capability.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_DISABLE_NVDEC` | unset | Set to `1` to force the software decode path. |
| `DYN_NVDEC_GPU_ID` | `0` | GPU ordinal used for decode. |
| `DYN_MM_VIDEO_NUM_FRAMES` | `32` | Frames sampled uniformly from each clip. |

## Sources

- [NVIDIA Video Encode and Decode GPU Support Matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
- [NVDEC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-application-note/index.html)
