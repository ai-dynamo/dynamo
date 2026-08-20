# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Opt-in development/qualification overlay for MiniMax-H3. The standard Dynamo
# image intentionally remains on its royalty-free VP9-only media stack.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
    && ln -sf /usr/bin/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install \
        --python /opt/dynamo/venv/bin/python \
        --no-deps \
        av==18.0.0 \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264rgb \
    && /opt/dynamo/venv/bin/python -c \
        'import av; av.codec.Codec("h264", "w"); av.codec.Codec("aac", "w")'

USER dynamo
