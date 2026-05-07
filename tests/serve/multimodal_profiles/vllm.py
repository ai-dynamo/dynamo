# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_audio_payload,
    make_image_payload,
    make_image_payload_b64,
    make_video_payload,
)
from tests.utils.payload_builder import chat_payload, chat_payload_default

# LLaVA 1.5 color-identification reference set: the model legitimately
# emits these colors (though the order/subset varies across CUDA backends
# under vLLM 0.20). Reused by every LLaVA topology that runs the standard
# color-identification payload.
_LLAVA_EXPECTED_COLORS = [
    "green",
    "white",
    "black",
    "purple",
    "red",
    "pink",
    "yellow",
    "blue",
    "orange",
]

# Multiple topology keys can map to the same shell script — the topology
# key is what makes the EngineConfig name unique; the script is just the
# launcher. ``agg_video`` and ``epd_video`` reuse ``agg_multimodal.sh`` /
# ``disagg_multimodal_epd.sh`` so video-specific TopologyConfigs (longer
# timeout, delayed_start, video-only VRAM bounds) can live alongside the
# image variants on the same model profile.
VLLM_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg": "agg_multimodal.sh",
    "agg_video": "agg_multimodal.sh",
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "epd_video": "disagg_multimodal_epd.sh",
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
                tests=[
                    # Vanilla / baseline single-GPU multimodal smoke
                    # (HTTP-URL image, no frontend decoding).
                    MmCase(payload=make_image_payload(["green"])),
                    # Vanilla inline base64 path (no Rust frontend decode).
                    MmCase(
                        suffix="b64",
                        payload=make_image_payload_b64(["green"]),
                    ),
                    # --frontend-decoding (HTTP-URL): exercises strip_inline_data_urls
                    # + NIXL RDMA. post_merge-only — local pre-merge builds outside
                    # docker can pick up NIXL stubs that don't support this path;
                    # CI post_merge runs in a container with real NIXL.
                    MmCase(
                        suffix="frontend_decoding",
                        payload=make_image_payload(["green"]),
                        extra_script_args=["--frontend-decoding"],
                    ),
                    # --frontend-decoding (inline base64). Same NIXL stub
                    # caveat as the HTTP-URL variant above.
                    MmCase(
                        suffix="b64_frontend_decoding",
                        payload=make_image_payload_b64(["green"]),
                        extra_script_args=["--frontend-decoding"],
                    ),
                ],
            ),
            "agg_video": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
                profiled_vram_gib=8.2,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                tests=[MmCase(payload=make_video_payload(["red", "static", "still"]))],
            ),
            "e_pd": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=340,
                single_gpu=True,
                profiled_vram_gib=15.0,
                requested_vllm_kv_cache_bytes=4_096_361_000,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                single_gpu=True,
                requested_vllm_kv_cache_bytes=1_714_881_000,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
            "epd_video": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                delayed_start=60,
                single_gpu=True,
                profiled_vram_gib=19.7,
                requested_vllm_kv_cache_bytes=1_714_881_000,
                tests=[MmCase(payload=make_video_payload(["red", "static", "still"]))],
            ),
            "p_d": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                single_gpu=True,
                profiled_vram_gib=15.7,
                requested_vllm_kv_cache_bytes=1_714_881_000,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3.5-0.8B",
        short_name="qwen3.5-0.8b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                profiled_vram_gib=4.0,
                tests=[
                    # HTTP-URL color test on hybrid Mamba/full-attention VL.
                    # post_merge — qwen3-vl-2b carries the pre_merge baseline.
                    MmCase(payload=make_image_payload(["green"])),
                    # Inline-base64 + --frontend-decoding (NIXL RDMA path) on
                    # the hybrid Mamba/full-attention VL. post_merge for the
                    # same NIXL-stub reason as qwen3-vl-2b's frontend_decoding
                    # cases — see that topology for the rationale.
                    MmCase(
                        suffix="b64_frontend_decoding",
                        payload=make_image_payload_b64(["green"]),
                        extra_script_args=["--frontend-decoding"],
                        marks=[pytest.mark.post_merge],
                    ),
                ],
            ),
        },
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
                tests=[MmCase(payload=make_audio_payload(["Hester", "Pynne"]))],
            ),
        },
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
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
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
    #
    # LLaVA 1.5 color naming varies across CUDA backends under vLLM 0.20;
    # keep this as a multimodal serving smoke check, not a color oracle.
    # The model also occasionally degenerates into newline-padded output
    # (observed in CI: '\n\nWhat\n\n...' and '\n\n1') even with
    # temperature=0; this is a known LLaVA-1.5-on-vLLM flake, so the
    # 9-color payload below uses max_attempts=5 to retry validation
    # in-process before CI fails. See tests/README.md "Flaky Tests" —
    # In-Process Query Retry.
    MultimodalModelProfile(
        name="llava-hf/llava-1.5-7b-hf",
        short_name="llava-1.5-7b",
        topologies={
            "agg": TopologyConfig(
                # nightly-only: 7B 1-GPU footprint is tight (vram=19.2 GiB).
                # Exercises a different image (coco bus) + a string-content
                # smoke check that the multimodal templating handles.
                marks=[pytest.mark.nightly],
                timeout_s=360,
                gpu_marker="gpu_1",
                profiled_vram_gib=19.2,
                requested_vllm_kv_cache_bytes=4_318_854_000,  # 2x safety over min=2_159_426_560
                tests=[
                    MmCase(
                        payload=chat_payload(
                            [
                                {"type": "text", "text": "What is in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                                    },
                                },
                            ],
                            repeat_count=1,
                            expected_response=["bus"],
                            temperature=0.0,
                        ),
                    ),
                    # String content (not array) — verifies string → array
                    # conversion for multimodal templates. Just validate no error.
                    MmCase(
                        suffix="default",
                        payload=chat_payload_default(
                            repeat_count=1,
                            expected_response=[],
                        ),
                    ),
                ],
            ),
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
                tests=[
                    MmCase(
                        payload=make_image_payload(
                            _LLAVA_EXPECTED_COLORS,
                            max_attempts=5,
                        )
                    )
                ],
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
                tests=[
                    MmCase(
                        payload=make_image_payload(
                            _LLAVA_EXPECTED_COLORS,
                            max_attempts=5,
                        )
                    )
                ],
            ),
        },
    ),
]
