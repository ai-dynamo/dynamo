# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import platform

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


def _is_cuda12() -> bool:
    v = os.environ.get("CUDA_VERSION", "")
    # handles "12", "12.9", etc.
    return v.startswith("12")


def _is_aarch64() -> bool:
    arch = os.environ.get("TARGETARCH") or os.environ.get("ARCH") or platform.machine()
    return arch in ("aarch64", "arm64")


def _xfail_cuda12_upstream_nixl_disagg():
    return pytest.mark.xfail(
        _is_cuda12() and not _is_aarch64(),
        reason=(
            "Upstream vllm/vllm-openai CUDA 12.9 image has an unstable "
            "multi-GPU NIXL/UCX transfer path in CI; CUDA 13 covers the "
            "upstream-vLLM NIXL path for this PR"
        ),
        strict=False,
    )


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
                marks=[
                    pytest.mark.skip(
                        reason="vLLM engine core init fails on disagg e_pd. "
                        "https://linear.app/nvidia/issue/OPS-4445"
                    ),
                    pytest.mark.pre_merge,
                ],
                timeout_s=340,
                single_gpu=True,
            ),
            "epd": TopologyConfig(
                marks=[
                    pytest.mark.skip(
                        reason="vLLM engine core init fails on disagg epd. "
                        "https://linear.app/nvidia/issue/OPS-4445"
                    ),
                    pytest.mark.pre_merge,
                ],
                timeout_s=300,
                single_gpu=True,
            ),
            "p_d": TopologyConfig(
                # p_d still exercises NIXL/UCX even when packed into gpu_1 CI.
                marks=[_xfail_cuda12_upstream_nixl_disagg(), pytest.mark.pre_merge],
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
                marks=[
                    pytest.mark.skip(
                        reason="vLLM engine core init fails on disagg epd. "
                        "https://linear.app/nvidia/issue/OPS-4445"
                    ),
                    pytest.mark.pre_merge,
                ],
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
        request_payloads=[make_image_payload(["green"])],
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
                        reason="vLLM engine core init fails on amd64 post-merge. "
                        "OPS-4445"
                    ),
                    pytest.mark.post_merge,
                ],
                timeout_s=600,
                env={"DYN_CHAT_PROCESSOR": "vllm"},
            ),
        },
        request_payloads=[make_audio_payload(["Hester", "Pynne"])],
        extra_vllm_args=["--max-model-len", "7232"],
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
    # [gluo NOTE] LLaVA 1.5 7B is big model and require at least 3 GPUs to run.
    # We may use less GPUs by squeezing the model onto 2 GPUs.
    MultimodalModelProfile(
        name="llava-hf/llava-1.5-7b-hf",
        short_name="llava-1.5-7b",
        topologies={
            "e_pd": TopologyConfig(
                marks=[_xfail_cuda12_upstream_nixl_disagg(), pytest.mark.pre_merge],
                timeout_s=340,
                gpu_marker="gpu_4",
            ),
            "epd": TopologyConfig(
                marks=[_xfail_cuda12_upstream_nixl_disagg(), pytest.mark.pre_merge],
                timeout_s=300,
                gpu_marker="gpu_4",
            ),
        },
        request_payloads=[make_image_payload(["green"])],
    ),
]
