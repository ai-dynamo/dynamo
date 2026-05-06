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
                # TODO: re-enable GPU-parallel scheduling with
                # profiled_vram_gib=9.6 once this has a bounded --kv-bytes profile.
                timeout_s=220,
            ),
            "e_pd": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=340,
                single_gpu=True,
                profiled_vram_gib=15.0,
                requested_vllm_kv_cache_bytes=4_096_361_000,
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                single_gpu=True,
                requested_vllm_kv_cache_bytes=1_714_881_000,
            ),
            "p_d": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                single_gpu=True,
                profiled_vram_gib=15.7,
                requested_vllm_kv_cache_bytes=1_714_881_000,
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
                timeout_s=720,
                delayed_start=60,
                profiled_vram_gib=8.2,
                requested_vllm_kv_cache_bytes=1_719_075_000,
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                delayed_start=60,
                single_gpu=True,
                profiled_vram_gib=19.7,
                requested_vllm_kv_cache_bytes=1_714_881_000,
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
    # Non-Qwen VLM coverage
    MultimodalModelProfile(
        name="google/gemma-4-E2B-it",
        short_name="gemma4-e2b-it",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                # TODO: re-enable GPU-parallel scheduling with
                # profiled_vram_gib=12.0 once this has a bounded --kv-bytes profile.
                timeout_s=300,
            ),
        },
        request_payloads=[make_image_payload(["green"])],
        extra_vllm_args=["--dtype", "bfloat16"],
    ),
    # [gluo NOTE] LLaVA 1.5 7B is big model and require at least 3 GPUs to run.
    # We may use less GPUs by squeezing the model onto 2 GPUs.
    #
    # Encoder VRAM is STATIC (~13.5 GB peak on a 48 GB RTX 6000 Ada,
    # independent of --gpu-memory-utilization): the dynamo encode worker
    # falls through to AutoModel.from_pretrained(..., torch_dtype=fp16) for
    # non-Qwen-VL models and loads the full LLaVA-1.5-7b weights before
    # extracting .visual. See disagg_multimodal_e_pd.sh and
    # components/src/dynamo/vllm/multimodal_utils/model.py:load_vision_model.
    # PD VRAM is bounded by --kv-cache-memory-bytes (set via
    # requested_vllm_kv_cache_bytes marker → _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES);
    # without that marker, the PD fraction $DYN_PD_GPU_MEM applies.
    MultimodalModelProfile(
        name="llava-hf/llava-1.5-7b-hf",
        short_name="llava-1.5-7b",
        topologies={
            "e_pd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                gpu_marker="gpu_2",
                # Profiled with `tests/utils/profile_pytest.py --gpus 0,1` on
                # 2x RTX 6000 Ada (48 GB each). Encoder GPU peaked ~13.5 GB
                # (static, full model fp16 load); PD GPU peaked ~19 GB
                # (weights + KV @ 4 GB cap + activations). 2x safety on KV.
                profiled_vram_gib=19.0,
                requested_vllm_kv_cache_bytes=4_308_848_000,
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                gpu_marker="gpu_4",
                # Default 3-GPU layout: encode → GPU 0, prefill → GPU 1,
                # decode → GPU 2. Each worker has its own GPU; per-GPU peak
                # ~14-18 GB (single worker + weights + KV). Fits on 24 GB L4
                # tier.
                #
                # The launch script also supports a 2-GPU pack via --two-gpu
                # (TopologyConfig.two_gpu=True): encode + prefill on GPU 0,
                # decode on GPU 1. NIXL still transfers KV from prefill→decode
                # across the GPU boundary, so it preserves the disagg semantic.
                # Profiled values for the 2-GPU layout (measured on RTX 6000 Ada
                # 48 GB; not yet enabled because GPU 0 peaks at 30 GB which
                # doesn't fit on 24 GB L4 cards):
                #   gpu_marker="gpu_2"
                #   two_gpu=True
                #   profiled_vram_gib=32.2  # GPU 0 peak, encoder + prefill
                #   requested_vllm_kv_cache_bytes=4_297_773_000  # 2× safety over min 2_148_886_016
                # Re-enable when CI runner pool has ≥32 GB cards on at least
                # one of the 2 slots.
            ),
        },
        # LLaVA 1.5 color naming varies across CUDA backends under vLLM 0.20;
        # keep this as a multimodal serving smoke check, not a color oracle.
        # The model also occasionally degenerates into newline-padded output
        # (observed in CI: '\n\nWhat\n\n...' and '\n\n1') even with
        # temperature=0; this is a known LLaVA-1.5-on-vLLM flake, so the
        # payload retries the validation in-process (server stays up) before
        # CI fails. See tests/README.md "Flaky Tests" — In-Process Query Retry.
        request_payloads=[
            make_image_payload(
                [
                    "green",
                    "white",
                    "black",
                    "purple",
                    "red",
                    "pink",
                    "yellow",
                    "blue",
                    "orange",
                ],
                max_attempts=5,
            )
        ],
    ),
]
