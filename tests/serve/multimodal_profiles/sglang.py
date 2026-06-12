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

# pd_no_encoder = disaggregated prefill+decode, single GPU, no encode worker.
SGLANG_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg_router": "agg_multimodal_router.sh",
    "pd_no_encoder": "disagg_same_gpu.sh",
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
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
                        )
                    )
                ],
            ),
            # Plain color-check payload; disagg KV semantics make the
            # cached-tokens hit-rate assertions inapplicable here.
            "pd_no_encoder": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=500,
                profiled_vram_gib=22.0,
                requested_sglang_kv_tokens=8192,
                delayed_start=10,
                tests=[MmCase(payload=make_image_payload(["green"]))],
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
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
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
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
                        )
                    )
                ],
            ),
        },
    ),
]
