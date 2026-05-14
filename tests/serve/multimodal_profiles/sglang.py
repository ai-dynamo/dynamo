# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_image_payload,
    make_image_payload_cached_tokens,
)

# agg_router only for now; disagg/EPD variants when their scripts exist.
SGLANG_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg_router": "agg_multimodal_router.sh",
}

# VLM coverage mirrors the vLLM profile registry. SINGLE_GPU=true packs both
# workers on GPU 0 for the gpu_1 CI box. requested_sglang_kv_tokens caps
# `--max-total-tokens`.
SGLANG_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=400,
                profiled_vram_gib=18.7,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen2.5-VL-3B-Instruct",
        short_name="qwen2.5-vl-3b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=500,
                profiled_vram_gib=19.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen2-VL-2B-Instruct",
        short_name="qwen2-vl-2b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=500,
                profiled_vram_gib=16.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="microsoft/Phi-3-vision-128k-instruct",
        short_name="phi3-vision",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=500,
                profiled_vram_gib=22.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
    # LLaVA-1.5 and LLaVA-NeXT cover both lightseek processor specs (plain
    # LlavaProcessor vs LlavaNextProcessor's anyres multi-crop). Two 7B
    # workers won't fit on one GPU, so no SINGLE_GPU here.
    MultimodalModelProfile(
        name="llava-hf/llava-1.5-7b-hf",
        short_name="llava-1.5-7b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                gpu_marker="gpu_2",
                profiled_vram_gib=19.2,
                requested_sglang_kv_tokens=4096,
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="llava-hf/llava-v1.6-mistral-7b-hf",
        short_name="llava-next-mistral-7b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                gpu_marker="gpu_2",
                profiled_vram_gib=19.2,
                requested_sglang_kv_tokens=4096,
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            min_cached_tokens=0,
                            require_lightseek_init=True,
                            min_routing_total_blocks=10,
                        )
                    )
                ],
            ),
        },
    ),
]
