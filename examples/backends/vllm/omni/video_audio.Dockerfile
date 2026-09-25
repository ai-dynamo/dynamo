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
# generate joint video and audio, including FastH3 VSA. The standard Dynamo
# image intentionally remains on its royalty-free VP9-only media stack and
# does not acquire FastVideo's optional CUDA kernel.
ARG BASE_IMAGE=dynamo:latest-vllm-runtime
FROM ${BASE_IMAGE}

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
    && ln -sf /usr/bin/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /var/lib/apt/lists/*

# fastvideo-kernel 0.3.5's optional native extension targets torch 2.12,
# while the vLLM 0.29 image uses torch 2.13. FastH3 VSA uses the portable
# Triton path, so install without dependencies and do not require the native
# extension. Avoid importing the package while building because that initializes
# Triton, which requires a CUDA driver even though the image build does not.
RUN uv pip install \
        --system \
        --no-deps \
        av==18.0.0 \
        fastvideo-kernel==0.3.5 \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -Eq '(^| )libx264( |$)' \
    && python3 -c \
        'from importlib.util import find_spec; import av; assert find_spec("fastvideo_kernel") is not None; av.codec.Codec("h264", "w"); av.codec.Codec("aac", "w")'

RUN python3 <<'PY'
import io

import av
import numpy as np
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

fps = 24
sample_rate = 32000
frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
waveform = np.zeros(
    (2, round(sample_rate * len(frames) / fps)), dtype=np.float32
)
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
    video_duration = float(video.duration * video.time_base)
    audio_duration = float(audio.duration * audio.time_base)
    assert abs(video_duration - audio_duration) <= 1 / fps
PY

USER dynamo
