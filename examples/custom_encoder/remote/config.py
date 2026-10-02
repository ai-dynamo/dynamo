# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the remote custom encoder example."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend

from examples.custom_encoder.hitchhikers_vision_encoder import HitchhikersVisionEncoder

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SERVICE_NAME = "remote-custom-encoder"


@dataclass(frozen=True)
class RemoteEncoderConfig:
    """Application choices loaded once at worker startup."""

    model: str
    service_name: str
    public_model_name: str | None
    encoder_class: type[VisionEncoderBackend[Any, Any, torch.Tensor]]
    chat_template_path: Path

    @property
    def generator_endpoint(self) -> str:
        return f"{self.service_name}.generator.generate"

    @property
    def generator_model_name(self) -> str:
        return f"{self.service_name}-generator"

    @classmethod
    def from_env(cls) -> "RemoteEncoderConfig":
        """Read deployment names while choosing the encoder class in Python."""

        return cls(
            model=os.environ.get("DYN_MODEL", DEFAULT_MODEL),
            service_name=os.environ.get("DYN_SERVICE_NAME", DEFAULT_SERVICE_NAME),
            public_model_name=os.environ.get("DYN_SERVED_MODEL_NAME"),
            encoder_class=HitchhikersVisionEncoder,
            chat_template_path=Path(__file__).resolve().parents[1]
            / "templates/qwen_vl.jinja",
        )
