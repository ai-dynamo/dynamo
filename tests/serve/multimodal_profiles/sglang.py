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

# SGLang topology → launch script mapping. Currently only the
# MM-aware aggregated router is exercised; disagg/EPD MM-routing
# variants can be added once their scripts exist.
SGLANG_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg_router": "agg_multimodal_router.sh",
}

# Lightseek-supported VLM coverage on SGLang `agg_router` (Rust-frontend
# MM-aware routing path). The model list mirrors the vLLM profile registry
# (Qwen3-VL-2B on pre_merge baseline, Qwen2.5-VL / Qwen2-VL / Phi-3-vision /
# LLaVA-1.5 / LLaVA-NeXT on post_merge) so the same matrix is exercised on
# both backends.
#
# SINGLE_GPU=true packs both SGLang workers onto GPU 0 to fit the gpu_1
# single-GPU CI box. KV cap is set via requested_sglang_kv_tokens (translates
# to `--max-total-tokens`); 2048 tokens is enough for prompt + max_tokens=100
# plus scheduler reserve on every model tested here.
#
# The cached_tokens-asserting payload (`make_image_payload_cached_tokens`)
# sends two identical MM requests and asserts the second sees
# `cached_tokens >= 1` — proves the MM-aware router routed both requests to
# the same worker AND the worker reused its KV cache for the image-pad-value
# block. A silent regression to text-prefix-only routing would still return
# "green" but `cached_tokens` would be 0 and this case would fail.
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
        },
    ),
    # Phi-3-vision is the largest of the post_merge additions for SGLang
    # (~8.6 GB weights × 2 workers + KV). The launch script already passes
    # --trust-remote-code unconditionally, so no extra args are needed here.
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
        },
    ),
    # LLaVA-1.5 and LLaVA-NeXT-mistral: 7B each, two workers, one per GPU on
    # gpu_2 (no SINGLE_GPU packing — 7B × 2 would exceed 24 GiB on a single
    # card). Different lightseek processor specs (LlavaProcessor vs
    # LlavaNextProcessor's anyres multi-crop), so both are covered.
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
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
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
        },
    ),
]
