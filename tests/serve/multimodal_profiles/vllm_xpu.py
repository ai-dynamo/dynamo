# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.serve.common import WORKSPACE_DIR
from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_audio_payload,
    make_image_payload,
    make_image_payload_b64,
    make_image_payload_cached_tokens,
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
    "agg": "xpu/agg_multimodal_xpu.sh",
    "agg_video": "xpu/agg_multimodal_xpu.sh",
    # Aggregated MM-aware router. Default uses the Rust frontend with the
    # `lightseek-mm` feature; the `_chat_processor` variant uses the vLLM
    # Python preprocessor (`--dyn-chat-processor=vllm`) to enable the
    # DYNAMO_MM_TRANSFER shm/NIXL pre-rendered mm_kwargs delivery channel.
    "agg_router": "xpu/agg_multimodal_router_xpu.sh",
    "agg_router_chat_processor": "xpu/agg_multimodal_router_chat_processor_xpu.sh",
    # Frontend-decode reuses the same script as `agg_router`; the topology
    # config below toggles `--frontend-decoding` on the worker via env.
    "agg_router_frontend_decode": "xpu/agg_multimodal_router_xpu.sh",
}


VLLM_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        gpu_marker="xpu_1",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge, pytest.mark.xpu_1],
                # TODO: re-enable XPU-parallel scheduling with
                # profiled_vram_gib=9.6 once this has a bounded --kv-bytes profile.
                timeout_s=300,
                tests=[
                    # Vanilla / baseline single-XPU multimodal smoke
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
                marks=[
                    pytest.mark.skip(
                        reason="flaky test, local video file downloading may fail due to network issues"
                    ),
                    pytest.mark.nightly,
                    pytest.mark.multimodal,
                ],
                timeout_s=600,
                delayed_start=60,
                profiled_vram_gib=8.2,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                tests=[
                    MmCase(
                        payload=make_video_payload(["red", "static", "still"]),
                        env={"DYN_MM_LOCAL_PATH": WORKSPACE_DIR},
                    )
                ],
            ),
            # Pre_merge gater for the lightseek MM-routing path. Fine-grained
            # assertions live in tests/mm_router/test_router_rust_mm_router_e2e.py
            # (post_merge).
            "agg_router": TopologyConfig(
                marks=[pytest.mark.pre_merge, pytest.mark.xpu_2],
                gpu_marker="xpu_2",
                timeout_s=400,
                profiled_vram_gib=18.7,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                env={"NUM_WORKERS": "2"},
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
            # The chat-processor variant of the MM-aware router: same routing
            # architecture, but the frontend uses --dyn-chat-processor=vllm
            # (Python preprocessor) instead of the Rust+lightseek path. Kept
            # on post_merge — the lightseek variant above (`agg_router`) is
            # the pre_merge gate; adding chat_processor doubles the XPU0
            # queue time at worker scale without catching distinct bugs
            # (both paths share the kv_router downstream).
            # SINGLE_GPU=true packs both workers onto GPU 0 to match the
            # single-GPU CI environment.
            "agg_router_chat_processor": TopologyConfig(
                marks=[pytest.mark.post_merge, pytest.mark.xpu_2],
                gpu_marker="xpu_2",
                timeout_s=400,
                profiled_vram_gib=18.7,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                env={"NUM_WORKERS": "2"},
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
            # Frontend-decode variant: same `agg_multimodal_router.sh` as
            # `agg_router`, but `VLLM_EXTRA_ARGS=--frontend-decoding` so
            # the worker registers a `media_decoder` on its model card and
            # the frontend's MediaLoader runs in-process. mm_hash becomes
            # content-addressed (xxh3 over decoded RGB), enabling
            # cross-URL cache reuse. Smoke-level gate so regressions to
            # the decoded branch in `gather_multi_modal_data` surface in
            # CI; the content-hash correctness assertion lives in
            # tests/mm_router/test_router_rust_mm_frontend_decode_e2e.py.
            "agg_router_frontend_decode": TopologyConfig(
                marks=[pytest.mark.post_merge, pytest.mark.xpu_2],
                gpu_marker="xpu_2",
                timeout_s=400,
                profiled_vram_gib=18.7,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                env={
                    "NUM_WORKERS": "2",
                    "VLLM_EXTRA_ARGS": "--frontend-decoding",
                },
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
        extra_vllm_args=["--block-size", "16"],
    ),
    # Lightseek-supported VLM coverage on `agg_router` (Rust-frontend
    # MM-aware routing path). Each profile below adds the same smoke test
    # as Qwen3-VL-2B's agg_router (pre_merge), but on post_merge with the
    # corresponding family (Qwen2.5-VL, Qwen2-VL) — Qwen-family coverage,
    # not the full lightseek FAMILIES list. SINGLE_GPU=true packs both workers onto GPU 0 to match
    # the gpu_1 single-GPU box. Initial VRAM profiles are estimates; the
    # first post_merge run will surface real peaks and we'll tighten.
    MultimodalModelProfile(
        name="Qwen/Qwen2.5-VL-3B-Instruct",
        short_name="qwen2.5-vl-3b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=500,
                gpu_marker="xpu_2",
                profiled_vram_gib=19.0,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                env={"NUM_WORKERS": "2"},
                tests=[MmCase(payload=make_image_payload(["green"]))],
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
                gpu_marker="xpu_2",
                profiled_vram_gib=16.0,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                env={"NUM_WORKERS": "2"},
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
                gpu_marker="xpu_1",
                profiled_vram_gib=4.7,
                requested_vllm_kv_cache_bytes=920_126_000,  # 2x safety over min=460_062_720
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
                        timeout_s=900,
                    ),
                ],
            ),
        },
        extra_vllm_args=["--block-size", "64"],
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
                gpu_marker="xpu_1",
                env={"DYN_CHAT_PROCESSOR": "vllm"},
                tests=[MmCase(payload=make_audio_payload(["Hester", "Pynne"]))],
            ),
        },
        extra_vllm_args=["--max-model-len", "7232", "--block-size", "64"],
    ),
    # Non-Qwen VLM coverage
    MultimodalModelProfile(
        name="google/gemma-4-E2B-it",
        short_name="gemma4-e2b-it",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=500,
                gpu_marker="xpu_1",
                profiled_vram_gib=12.0,
                requested_vllm_kv_cache_bytes=922_354_000,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
        extra_vllm_args=["--dtype", "bfloat16", "--block-size", "64"],
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
            # Rust-frontend MM-aware routing for the LLaVA family. Two
            # workers, one per GPU on the gpu_2 runner (no SINGLE_GPU
            # packing — 7B × 2 would exceed 24 GiB on a single card). Each
            # worker peaks ~19 GiB; the 24 GiB L4 tier has headroom.
            # Skipped: LLaVA-1.5 on vLLM 0.20 is flaky — the model
            # degenerates into "No." / newline-padded output even at
            # temperature=0 (see PR #9336 for the e_pd manifestation
            # and the comment block above on color-naming variance).
            # The agg_router routing path is already covered by the
            # Qwen3-VL-2B / Qwen2.5-VL-3B / Qwen2-VL-2B
            # profiles above without the LLaVA flake.
            "agg_router": TopologyConfig(
                marks=[
                    pytest.mark.skip(
                        reason="LLaVA-1.5 flake on vLLM 0.20 (see PR #9336); "
                        "agg_router routing path is covered by Qwen profiles"
                    ),
                    pytest.mark.post_merge,
                ],
                timeout_s=600,
                gpu_marker="xpu_2",
                profiled_vram_gib=19.2,
                requested_vllm_kv_cache_bytes=4_318_854_000,
                # cached_tokens-asserting payload proves MM-aware routing
                # engaged for LLaVA-1.5 (placeholder-template `<image>` path).
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
            "agg": TopologyConfig(
                # nightly-only: 7B 1-GPU footprint is tight (vram=19.2 GiB).
                # Exercises a different image (coco bus) + a string-content
                # smoke check that the multimodal templating handles.
                marks=[pytest.mark.nightly],
                timeout_s=360,
                gpu_marker="xpu_1",
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
        },
    ),
    # LLaVA-NeXT covers a separate lightseek processor (LlavaNextProcessor,
    # anyres multi-crop) vs LLaVA-1.5's plain LlavaProcessor. Same gpu_2
    # multi-GPU layout as LLaVA-1.5 agg_router above; ~14 GiB / GPU.
    #
    # Skipped: LLaVA-NeXT inherits the same LLaVA-on-vLLM-0.20 output
    # flake as LLaVA-1.5 (see PR #9336); the agg_router routing path is
    # covered by the Qwen profiles above.
    MultimodalModelProfile(
        name="llava-hf/llava-v1.6-mistral-7b-hf",
        short_name="llava-next-mistral-7b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[
                    pytest.mark.skip(
                        reason="LLaVA-NeXT inherits LLaVA-1.5 flake on vLLM 0.20 "
                        "(see PR #9336); agg_router routing path is covered by "
                        "Qwen profiles"
                    ),
                    pytest.mark.post_merge,
                ],
                timeout_s=600,
                gpu_marker="xpu_2",
                profiled_vram_gib=19.2,
                requested_vllm_kv_cache_bytes=4_318_854_000,
                # cached_tokens-asserting payload proves MM-aware routing
                # engaged for LLaVA-NeXT (anyres multi-crop processor).
                tests=[MmCase(payload=make_image_payload_cached_tokens(["green"]))],
            ),
        },
    ),
]
