---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Additional Media Decoders
subtitle: Opt in to broader video and audio input formats at runtime
---

## Overview

Dynamo's runtime images ship a deliberately small media stack. The in-tree FFmpeg is built for VP8/VP9 video, and the wider backend decode packages are not pre-installed, which keeps the distributed images small and their input-format surface narrow by default. As a result, sending input that uses another format (for example H.264, H.265, or AAC) fails with an actionable error that names the missing decoder.

Each backend decodes compressed input through a specific Python package whose wheel bundles its own FFmpeg, so support for additional input formats can be added with a plain `pip install` — no image rebuild. The `DYN_ENABLE_MEDIA_DECODERS` switch (off by default) performs that install automatically at worker startup, for the package(s) that sit on the running backend's decode path.

> [!NOTE]
> This affects **input decoding** only. Generated video **output** is unaffected and always uses VP9.

## Enable it

Set the environment variable on the worker (not the frontend):

```bash
export DYN_ENABLE_MEDIA_DECODERS=1
```

When enabled, Dynamo installs the on-path decoder package(s) for the backend at startup, if they are not already present:

| Backend | Input | Package installed | Import |
|---------|-------|-------------------|--------|
| vLLM | video | `opencv-python-headless` | `cv2` |
| vLLM | audio | `av` | `av` |
| SGLang | video | `decord2` | `decord` |

TensorRT-LLM workers decode no compressed media input, so the switch is a no-op there. Packages that some base images bundle but Dynamo never imports are intentionally left out, so only the decoder actually on the request path is installed.

The install is idempotent: if the package already imports (for example in an image that pre-bakes it), startup skips it. Concurrent workers sharing one environment are serialized so only one installs.

> [!WARNING]
> Installing a decoder package brings in that wheel's bundled media libraries. Review what your deployment ships before enabling this in production.

## Air-gapped and pinned installs

For offline clusters or version pinning, two optional overrides are available:

- `DYN_MEDIA_DECODER_PACKAGES` — a whitespace-separated pip spec that replaces the per-backend defaults, for example `opencv-python-headless==4.10.0.84 av==12.0.0`. This install resolves dependencies, so pin transitive versions if that matters.
- `DYN_MEDIA_DECODER_PIP_ARGS` — extra arguments appended to `pip install`, for example `--find-links /wheels --no-index` to install from a local wheelhouse.
- `DYN_MEDIA_DECODER_TIMEOUT_S` — pip timeout in seconds (default `600`; set `0` to disable). Bounds startup so a stalled index cannot hang the worker.

```bash
export DYN_ENABLE_MEDIA_DECODERS=1
export DYN_MEDIA_DECODER_PIP_ARGS="--find-links /opt/wheels --no-index"
```

A launch script can run the install before the worker starts, instead of relying on the in-process startup hook:

```bash
python -m dynamo.common.utils.media_decoders vllm
```

## Notes and limits

- The worker needs access to a package index (or a local wheelhouse) at startup, plus a writable environment. If the install fails, the worker still starts and requests that need the missing format fail with an actionable error at request time.
- The default install runs with `--no-deps`, so it cannot change the image's pinned dependency stack (for example numpy under PyTorch/vLLM); the carriers only need numpy, which the backend already provides.
- Enabling this adds cold-start latency while the package installs.
- The optional Rust frontend decoder (`--frontend-decoding`) links FFmpeg's compiled-in decoders and always decodes VP8/VP9 regardless of this switch; backend decoding is what gets extended. Re-encoding an input to VP9 (`ffmpeg -i input.mp4 -c:v libvpx-vp9 -an output.webm`) is an alternative that needs no additional packages.
