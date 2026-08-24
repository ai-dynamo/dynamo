# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Opt-in development and qualification overlay for vLLM-Omni models that
# generate joint video and audio. The standard Dynamo image intentionally
# remains on its royalty-free VP9-only media stack.
ARG BASE_IMAGE=dynamo:latest-vllm-runtime
FROM ${BASE_IMAGE}

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
    && ln -sf /usr/bin/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install \
        --system \
        --no-deps \
        av==18.0.0 \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -Eq '(^| )libx264( |$)' \
    && python3 -c \
        'import av; av.codec.Codec("h264", "w"); av.codec.Codec("aac", "w")'

RUN python3 <<'PY'
import io

import av
import numpy as np
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

fps = 24
sample_rate = 32000
frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
waveform = np.zeros((2, sample_rate // 4), dtype=np.float32)
payload = mux_video_audio_bytes(
    frames,
    waveform,
    fps=fps,
    audio_sample_rate=sample_rate,
)
with av.open(io.BytesIO(payload), mode="r") as container:
    video = container.streams.video[0]
    audio = container.streams.audio[0]
    assert video.codec_context.name == "h264"
    assert int(video.average_rate) == fps
    assert audio.codec_context.name == "aac"
    assert audio.codec_context.sample_rate == sample_rate
PY

USER dynamo
