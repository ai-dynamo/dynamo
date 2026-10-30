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

# pd = multimodal prefill+decode on one GPU, no separate encode worker.
SGLANG_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg_router": "agg_multimodal_router.sh",
    "pd": "disagg_same_gpu.sh",
}

# VLM coverage mirrors the vLLM profile registry. SINGLE_GPU=true packs both
# workers on GPU 0 for the gpu_1 CI box. requested_sglang_kv_tokens caps
# `--max-total-tokens`.
SGLANG_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3.5-0.8B",
        short_name="qwen3.5-0.8b",
        topologies={
            # Filler 120 produces about six routing blocks, so the expected
            # cache-hit-rate ceiling is approximately 0.83; threshold 0.7.
            "agg_router": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=400,
                profiled_vram_gib=9.8,  # actual peak with the mamba cap below
                requested_sglang_kv_tokens=8192,
                # Qwen3.5 is a hybrid (linear-attention) model. SGLang sizes its
                # mamba state cache from free VRAM, not --max-total-tokens, so
                # without a cap each worker takes a share of the whole card and
                # starves co-scheduled tests in the GPU-parallel stage.
                env={
                    "SINGLE_GPU": "true",
                    "SGLANG_EXTRA_ARGS": "--max-mamba-cache-size 64",
                },
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.7,
                            prompt_filler_repeats=120,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
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
            "pd": TopologyConfig(
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
