# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven configuration for the remote custom encoder example."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder"
)
DEFAULT_PUBLIC_MODEL_NAME = "remote-custom-encoder"
DEFAULT_DECODER_MODEL_NAME = "remote-custom-encoder-decoder"
DEFAULT_NAMESPACE = "remote-custom-encoder"


@dataclass(frozen=True)
class RemoteEncoderConfig:
    """Configuration loaded once at worker startup."""

    model: str
    encoder_class_path: str
    public_model_name: str
    decoder_model_name: str
    namespace: str
    generator_endpoint: str
    orchestrator_endpoint: str
    chat_template_path: Path

    @classmethod
    def from_env(cls) -> "RemoteEncoderConfig":
        """Build a configuration from environment variables."""

        namespace = os.environ.get("DYN_NAMESPACE", DEFAULT_NAMESPACE)
        return cls(
            model=os.environ.get("DYN_MODEL", DEFAULT_MODEL),
            encoder_class_path=os.environ.get(
                "DYN_ENCODER_CLASS", DEFAULT_ENCODER_CLASS
            ),
            public_model_name=os.environ.get(
                "DYN_SERVED_MODEL_NAME", DEFAULT_PUBLIC_MODEL_NAME
            ),
            decoder_model_name=os.environ.get(
                "DYN_DECODER_MODEL_NAME", DEFAULT_DECODER_MODEL_NAME
            ),
            namespace=namespace,
            generator_endpoint=f"{namespace}.generator.generate",
            orchestrator_endpoint=f"{namespace}.orchestrator.generate",
            chat_template_path=Path(__file__).resolve().parents[1]
            / "templates/qwen_vl.jinja",
        )

    def resolve_backend_class(
        self,
    ) -> type[VisionEncoderBackend[Any, Any, torch.Tensor]]:
        """Import and validate the author-provided vision encoder backend."""

        module_path, separator, class_name = self.encoder_class_path.rpartition(".")
        if not separator:
            raise ValueError(
                "DYN_ENCODER_CLASS must be a dotted module.ClassName path"
            )
        backend_class = getattr(importlib.import_module(module_path), class_name)
        if not (
            isinstance(backend_class, type)
            and issubclass(backend_class, VisionEncoderBackend)
        ):
            raise TypeError(
                f"DYN_ENCODER_CLASS must resolve to VisionEncoderBackend; got {backend_class!r}"
            )
        return backend_class
