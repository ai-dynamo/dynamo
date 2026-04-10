# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.multimodal import (
    MultimodalModelProfile,
    TopologyConfig,
    make_audio_payload,
    make_image_payload,
    make_video_payload,
)

VLLM_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg": "agg_multimodal.sh",
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "p_d": "disagg_multimodal_p_d.sh",
}

VLLM_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=220,
                profiled_vram_gib=9.6,
            ),
            "e_pd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=340,
                single_gpu=True,
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
                single_gpu=True,
            ),
            "p_d": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
                single_gpu=True,
            ),
        },
        request_payloads=[make_image_payload(["green"])],
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b-video",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
                single_gpu=True,
            ),
        },
        request_payloads=[make_video_payload(["red", "static", "still"])],
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen2.5-VL-7B-Instruct",
        short_name="qwen2.5-vl-7b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=360,
                profiled_vram_gib=19.9,
                requested_vllm_kv_cache_bytes=922_354_000,
            ),
        },
        request_payloads=[make_image_payload(["purple"])],
    ),
    # Audio: uses agg topology with DYN_CHAT_PROCESSOR=vllm because the Rust
    # Jinja engine cannot render multimodal content arrays (audio_url).
    MultimodalModelProfile(
        name="Qwen/Qwen2-Audio-7B-Instruct",
        short_name="qwen2-audio-7b",
        topologies={
            "agg": TopologyConfig(
                marks=[
                    pytest.mark.skip(
                        reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2604"
                    ),
                    pytest.mark.post_merge,
                ],
                timeout_s=600,
                env={"DYN_CHAT_PROCESSOR": "vllm"},
            ),
        },
        request_payloads=[make_audio_payload(["Hester", "Pynne"])],
    ),
    MultimodalModelProfile(
        name="google/gemma-3-4b-it",
        short_name="gemma3-4b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                profiled_vram_gib=12.0,
            ),
        },
        request_payloads=[make_image_payload(["green"])],
        extra_vllm_args=["--dtype", "bfloat16"],
        gated=True,
    ),
]
