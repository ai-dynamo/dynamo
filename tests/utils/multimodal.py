# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import pytest

from tests.serve.conftest import MULTIMODAL_IMG_URL
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload
from tests.utils.payloads import BasePayload, ChatPayload

WORKSPACE_DIR = str(Path(__file__).resolve().parents[2])

LOCAL_VIDEO_TEST_PATH = Path(
    WORKSPACE_DIR, "lib/llm/tests/data/media/240p_10.mp4"
).resolve()
LOCAL_VIDEO_TEST_URI = LOCAL_VIDEO_TEST_PATH.as_uri()

AUDIO_TEST_URL = (
    "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client"
    "/main/datasets/mini_en/wav/1221-135766-0002.wav"
)


# ---------------------------------------------------------------------------
# Topology scripts
# ---------------------------------------------------------------------------

TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg": "agg_multimodal.sh",
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "p_d": "disagg_multimodal_p_d.sh",
    "audio_agg": "audio_agg.sh",
    "audio_disagg": "audio_disagg.sh",
}


# ---------------------------------------------------------------------------
# Payload factories
# ---------------------------------------------------------------------------


def make_image_payload(expected_response: list[str]) -> ChatPayload:
    """Standard image color-identification payload using MULTIMODAL_IMG_URL."""
    return chat_payload(
        [
            {
                "type": "text",
                "text": "What colors are in the following image? "
                "Respond only with the colors.",
            },
            {
                "type": "image_url",
                "image_url": {"url": MULTIMODAL_IMG_URL},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
    )


def make_video_payload(expected_response: list[str]) -> ChatPayload:
    """Standard video description payload using the local test video."""
    return chat_payload(
        [
            {"type": "text", "text": "Describe the video in detail"},
            {
                "type": "video_url",
                "video_url": {"url": LOCAL_VIDEO_TEST_URI},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
    )


def make_audio_payload(expected_response: list[str]) -> ChatPayload:
    """Standard audio transcription payload using the remote test WAV."""
    return chat_payload(
        [
            {"type": "text", "text": "What is recited in the audio?"},
            {
                "type": "audio_url",
                "audio_url": {"url": AUDIO_TEST_URL},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
    )


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TopologyConfig:
    """Per-topology overrides for marks, timeout, and VRAM profiling."""

    marks: list[Any] = field(default_factory=list)
    timeout_s: int = 300
    profiled_vram_gib: Optional[float] = None
    requested_vllm_kv_cache_bytes: Optional[int] = None
    delayed_start: int = 0
    directory: Optional[str] = None  # override profile-level directory
    gpu_marker: Optional[str] = None  # override profile-level gpu_marker


@dataclass
class MultimodalModelProfile:
    """Describes a multimodal model's test-relevant properties.

    Each profile generates one config per topology in ``topologies``
    via :func:`make_multimodal_configs`.
    """

    name: str  # HuggingFace model ID
    short_name: str  # kebab-case slug for config key
    topologies: dict[str, TopologyConfig]
    request_payloads: list[BasePayload]
    gpu_marker: str = "gpu_1"
    extra_vllm_args: list[str] = field(default_factory=list)
    marks: list[Any] = field(default_factory=list)  # shared across all topologies
    gated: bool = False  # if True, skip unless DYN_HF_GATED_MODELS_ENABLED=1


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


def make_multimodal_configs(
    profile: MultimodalModelProfile,
    config_cls: Type[EngineConfig],
    directory: str,
) -> dict[str, EngineConfig]:
    """Generate config entries for each topology in *profile*.

    Parameters
    ----------
    config_cls:
        The concrete config class to instantiate (e.g. ``VLLMConfig``).
    directory:
        Default directory; overridden by ``TopologyConfig.directory`` if set.
    """
    configs: dict[str, EngineConfig] = {}
    for topology, topo_cfg in profile.topologies.items():
        script_name = TOPOLOGY_SCRIPTS[topology]
        script_args = ["--model", profile.name] + profile.extra_vllm_args
        if topology not in ("agg", "audio_agg"):
            script_args.append("--single-gpu")

        gpu = topo_cfg.gpu_marker or profile.gpu_marker
        marks: list[Any] = [
            getattr(pytest.mark, gpu),
            pytest.mark.timeout(topo_cfg.timeout_s),
        ]
        marks.extend(topo_cfg.marks)
        if topo_cfg.profiled_vram_gib is not None:
            marks.append(pytest.mark.profiled_vram_gib(topo_cfg.profiled_vram_gib))
        if topo_cfg.requested_vllm_kv_cache_bytes is not None:
            marks.append(
                pytest.mark.requested_vllm_kv_cache_bytes(
                    topo_cfg.requested_vllm_kv_cache_bytes
                )
            )
        if profile.gated:
            marks.append(
                pytest.mark.skipif(
                    not os.environ.get("DYN_HF_GATED_MODELS_ENABLED"),
                    reason=(
                        f"{profile.name} is gated; set DYN_HF_GATED_MODELS_ENABLED=1 "
                        "with an HF_TOKEN that has accepted the license"
                    ),
                )
            )
        marks.extend(profile.marks)

        key = f"mm_{topology}_{profile.short_name}"
        configs[key] = config_cls(
            name=key,
            directory=topo_cfg.directory or directory,
            script_name=script_name,
            model=profile.name,
            script_args=script_args,
            marks=marks,
            delayed_start=topo_cfg.delayed_start,
            request_payloads=profile.request_payloads,
        )
    return configs


# ---------------------------------------------------------------------------
# Multimodal model profiles — add new models here
# ---------------------------------------------------------------------------
# Generated config keys use the "mm_" prefix. Do not use this prefix for
# hand-written configs to avoid collisions.

_AUDIO_DIR = os.path.join(WORKSPACE_DIR, "examples/multimodal")

MULTIMODAL_MODEL_PROFILES: list[MultimodalModelProfile] = [
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
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
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
    MultimodalModelProfile(
        name="Qwen/Qwen2-Audio-7B-Instruct",
        short_name="qwen2-audio-7b",
        topologies={
            "audio_agg": TopologyConfig(
                marks=[pytest.mark.nightly],
                timeout_s=600,
                directory=_AUDIO_DIR,
            ),
            "audio_disagg": TopologyConfig(
                marks=[pytest.mark.nightly],
                timeout_s=600,
                directory=_AUDIO_DIR,
                gpu_marker="gpu_4",
            ),
        },
        gpu_marker="gpu_2",
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
