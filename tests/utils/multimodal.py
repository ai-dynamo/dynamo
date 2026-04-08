# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import pytest

from tests.serve.conftest import MULTIMODAL_IMG_URL
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload
from tests.utils.payloads import ChatPayload


@dataclass
class TopologyConfig:
    """Per-topology overrides for marks, timeout, and VRAM profiling."""

    marks: list[Any] = field(default_factory=list)
    timeout_s: int = 300
    profiled_vram_gib: Optional[float] = None
    requested_vllm_kv_cache_bytes: Optional[int] = None


@dataclass
class MultimodalModelProfile:
    """Describes a multimodal model's test-relevant properties.

    Each profile generates one config per topology in ``topologies``
    via :func:`make_multimodal_configs`.
    """

    name: str  # HuggingFace model ID
    short_name: str  # kebab-case slug for config key (e.g. "qwen3-vl-2b")
    topologies: dict[str, TopologyConfig]
    image_expected_response: list[str]  # OR-substring keywords (any match = pass)
    gpu_marker: str = "gpu_1"
    extra_vllm_args: list[str] = field(default_factory=list)
    marks: list[Any] = field(default_factory=list)  # shared across all topologies
    gated: bool = False  # if True, skip unless DYN_HF_GATED_MODELS_ENABLED=1


TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg": "agg_multimodal.sh",
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "p_d": "disagg_multimodal_p_d.sh",
}


def make_image_payload(expected_response: list[str]) -> ChatPayload:
    """Standard multimodal image test payload.

    Uses the shared test image (MULTIMODAL_IMG_URL) with a color-identification
    prompt. Response validation uses case-insensitive OR-substring matching:
    the check passes if ANY keyword in expected_response appears in the response.
    """
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
        Base directory passed as ``directory`` to the config constructor.
    """
    configs: dict[str, EngineConfig] = {}
    for topology, topo_cfg in profile.topologies.items():
        script_name = TOPOLOGY_SCRIPTS[topology]  # KeyError if invalid
        script_args = ["--model", profile.name] + profile.extra_vllm_args
        if topology != "agg":
            script_args.append("--single-gpu")

        marks: list[Any] = [
            getattr(pytest.mark, profile.gpu_marker),
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
            directory=directory,
            script_name=script_name,
            model=profile.name,
            script_args=script_args,
            marks=marks,
            request_payloads=[make_image_payload(profile.image_expected_response)],
        )
    return configs


# ---------------------------------------------------------------------------
# Multimodal model profiles — add new models here
# ---------------------------------------------------------------------------
# Generated config keys use the "mm_" prefix. Do not use this prefix for
# hand-written configs to avoid collisions.

MULTIMODAL_MODEL_PROFILES: list[MultimodalModelProfile] = [
    # Replaces: multimodal_e_pd_qwen, multimodal_disagg_qwen, mm_agg_qwen3-vl-2b
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
        image_expected_response=["green"],
    ),
    # Replaces: multimodal_agg_qwen (Qwen2.5-VL-7B)
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
        image_expected_response=["purple"],
    ),
    # New model coverage
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
        image_expected_response=["green"],
        extra_vllm_args=["--dtype", "bfloat16"],
        gated=True,
    ),
]
