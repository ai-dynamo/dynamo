#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
AIC (AI Configurator) direct session wrapper for mocker perf model.

Provides a Python class that wraps the AIC InferenceSession and exposes
predict_prefill() and predict_decode() methods callable from Rust via PyO3.
"""

import logging

from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, get_supported_databases

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.12.0",
    "sglang": "0.5.6.post2",
}


class AicSession:
    """Wraps AIC InferenceSession with predict_prefill/predict_decode methods."""

    def __init__(
        self,
        backend_name: str,
        system: str,
        model_path: str,
        tp_size: int,
        backend_version: str | None = None,
    ):
        version = backend_version or DEFAULT_BACKEND_VERSIONS.get(
            backend_name, DEFAULT_BACKEND_VERSIONS["vllm"]
        )

        database = get_database(system=system, backend=backend_name, version=version)
        if database is None:
            supported = get_supported_databases().get(system, {}).get(backend_name, [])
            supported_versions = ", ".join(supported) if supported else "<none>"
            raise RuntimeError(
                "AIC perf database not found for "
                f"system={system!r}, backend={backend_name!r}, version={version!r}. "
                f"Supported versions for this system/backend: {supported_versions}"
            )
        model_config = config.ModelConfig(tp_size=tp_size)
        model = get_model(
            model_path=model_path,
            model_config=model_config,
            backend_name=backend_name,
        )
        backend = get_backend(backend_name)
        self._session = InferenceSession(
            model=model, database=database, backend=backend
        )
        self._config = config
        logger.info(
            "AIC session initialized: backend=%s, system=%s, model=%s, tp=%d",
            backend_name,
            system,
            model_path,
            tp_size,
        )

    def predict_prefill(
        self, batch_size: int, isl: int, prefix: int, osl: int
    ) -> float:
        """Predict prefill latency in ms. Parameters match AIC RuntimeConfig."""
        # AIC requires at least 1 new token (isl > prefix)
        actual_prefix = min(prefix, isl - 1) if isl > 0 else 0
        rt = self._config.RuntimeConfig(
            batch_size=batch_size, isl=isl, osl=osl, prefix=actual_prefix
        )
        summary = self._session.run_static(mode="static_ctx", runtime_config=rt)
        return sum(summary.get_context_latency_dict().values())

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict decode (generation) latency in ms."""
        rt = self._config.RuntimeConfig(batch_size=batch_size, isl=isl, osl=osl)
        summary = self._session.run_static(mode="static_gen", runtime_config=rt)
        return sum(summary.get_generation_latency_dict().values())


def create_session(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    backend_version: str | None = None,
) -> AicSession:
    """Factory function called from Rust via PyO3."""
    return AicSession(backend_name, system, model_path, tp_size, backend_version)
