# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own native profile eligibility and capability publication."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from dynamo.llm import ModelInput, ModelType, WorkerType

from .decision_dispatch import DecisionDispatcher
from .decision_execution import decision_worker_eligible

logger = logging.getLogger(__name__)

D3_PROFILE = "vllm_d3_decision_v1"
CLEF_FLASH_PROFILE = "vllm_clef_flash_joint_decision_v1"
CLEF_OMNI_PROFILE = "vllm_clef_omni_joint_decision_v1"
LIQUID_PROFILE = "vllm_liquid_decision_v1"
DIFFUSION_PROFILE = "vllm_diffusion_canvas_decision_v1"
NATIVE_QUALIFICATION_FAILURE = "vllm_native_decision_qualification_failure_v1"


@dataclass(frozen=True)
class NativeProfile:
    name: str
    kind: Literal["head", "read"]
    architecture: str
    task: str
    strategy: str
    max_candidates: int = 26


@dataclass(frozen=True)
class NativeReady:
    profile: NativeProfile


@dataclass(frozen=True)
class NativeFailed:
    profile: NativeProfile


NativePublication = NativeReady | NativeFailed


PROFILES = MappingProxyType(
    {
        D3_PROFILE: NativeProfile(
            D3_PROFILE,
            "head",
            "D3ForSequenceClassification",
            "classify",
            "D3Strategy",
            255,
        ),
        CLEF_FLASH_PROFILE: NativeProfile(
            CLEF_FLASH_PROFILE,
            "head",
            "ClefFlashForJointSchema",
            "token_classify",
            "ClefFlashStrategy",
        ),
        CLEF_OMNI_PROFILE: NativeProfile(
            CLEF_OMNI_PROFILE,
            "head",
            "ClefOmniForJointSchema",
            "token_classify",
            "JointSchemaStrategy",
        ),
        LIQUID_PROFILE: NativeProfile(
            LIQUID_PROFILE,
            "read",
            "Lfm2VlForConditionalGeneration",
            "generate",
            "LiquidStrategy",
        ),
        DIFFUSION_PROFILE: NativeProfile(
            DIFFUSION_PROFILE,
            "read",
            "DiffusionGemmaForBlockDiffusion",
            "generate",
            "DiffusionGemmaCanvasStrategy",
        ),
    }
)


def profile_for_architecture(
    architecture: str | None, *, kind: Literal["head", "read"]
) -> NativeProfile | None:
    return next(
        (
            profile
            for profile in PROFILES.values()
            if profile.kind == kind and profile.architecture == architecture
        ),
        None,
    )


def supports_decision_generation(model_type: ModelType) -> bool:
    """Match native read-profile cards' Chat-or-Completions model flags."""
    return model_type.supports_chat() or model_type.supports_completions()


async def initialize_native_decisions(
    kind, engine, model_config, config, *, abort_monitor, on_engine_dead=None
) -> DecisionDispatcher:
    """Gate optional native imports before initializing one recognized profile."""
    profile = profile_for_architecture(
        getattr(model_config, "architecture", None), kind=kind
    )
    if profile is None:
        return DecisionDispatcher(kind)
    if not decision_worker_eligible(config):
        logger.info("vLLM native %s decisions disabled by worker configuration", kind)
        return DecisionDispatcher(kind)
    from .decision_surface import NativeDecisionSurface

    return await NativeDecisionSurface.create(
        engine, profile, abort_monitor=abort_monitor, on_engine_dead=on_engine_dead
    )


def publish_native_decision_capability(
    runtime_config,
    publication: NativePublication | None,
    model_input,
    model_type,
    worker_type,
    needs=None,
) -> NativePublication | None:
    """Publish native readiness or a persistent qualification failure on its worker."""
    if worker_type != WorkerType.Aggregated or any(needs or ()):
        return None
    match publication:
        case NativeReady(selected) | NativeFailed(selected):
            pass
        case None:
            return None
        case _:
            raise TypeError("native publication must be ready, failed or absent")
    canonical = PROFILES.get(selected.name)
    if canonical is None or selected.kind != canonical.kind:
        return None
    if selected.kind == "head":
        if not (
            model_input == ModelInput.Text
            and model_type.supports_classify()
            and model_type.supports_pooling()
        ):
            return None
    elif not supports_decision_generation(model_type):
        return None
    match publication:
        case NativeFailed():
            runtime_config.set_engine_specific(
                NATIVE_QUALIFICATION_FAILURE, json.dumps({"profile": selected.name})
            )
        case NativeReady():
            runtime_config.set_engine_specific(selected.name, json.dumps(True))
            runtime_config.set_engine_specific(
                "vllm_native_decision_max_candidates",
                json.dumps(selected.max_candidates),
            )
    return publication
