# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model, endpoint, and encoder configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dynamo.experimental.workflow.vllm import EncoderStage
from dynamo.vllm.multimodal_utils.custom_encoder import (
    resolve_vision_encoder_backend_class,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.hitchhikers_vision_encoder." "HitchhikersVisionEncoder"
)

MODEL = os.environ.get("DYN_MODEL", DEFAULT_MODEL)
ENCODER_CLASS = os.environ.get("DYN_ENCODER_CLASS", DEFAULT_ENCODER_CLASS)
PUBLIC_MODEL_NAME = os.environ.get("DYN_SERVED_MODEL_NAME", "user-ensemble")
DECODER_MODEL_NAME = os.environ.get("DYN_DECODER_MODEL_NAME", "user-ensemble-decoder")

NAMESPACE = os.environ.get("DYN_NAMESPACE", "workflow-user-ensemble")
GENERATOR_ENDPOINT = f"{NAMESPACE}.generator.generate"
ORCHESTRATOR_ENDPOINT = f"{NAMESPACE}.orchestrator.generate"

REPO_ROOT = Path(__file__).resolve().parents[5]
CHAT_TEMPLATE = Path(
    os.environ.get(
        "DYN_CUSTOM_JINJA_TEMPLATE",
        REPO_ROOT / "examples/custom_encoder/templates/qwen_vl.jinja",
    )
)


def build_encoder_stage() -> EncoderStage:
    """Load the configured author backend through Dynamo's reusable stage."""

    backend_type = resolve_vision_encoder_backend_class(ENCODER_CLASS)
    return EncoderStage.from_backend(backend_type(), model=MODEL)
