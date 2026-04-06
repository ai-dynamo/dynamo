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
DEFAULT_STATIC_STRIDE = 32


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
        self._database = database
        self._model = model
        # AIC models consistently expose model_path, but some do not surface model_name.
        self._model_name = getattr(model, "model_name", None) or model_path
        self._config = config
        logger.info(
            "AIC session initialized: backend=%s, system=%s, model=%s, tp=%d",
            backend_name,
            system,
            model_path,
            tp_size,
        )

    def _predict_context_latency(self, batch_size: int, isl: int, prefix: int) -> float:
        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(
                f"isl must be greater than prefix, got isl={isl}, prefix={prefix}"
            )

        total_latency = 0.0
        for op in self._model.context_ops:
            # AIC operations identify kernels via Operation._name; there is no public name accessor.
            op_name = getattr(op, "_name", "")
            x = batch_size if "logits_gemm" in op_name else batch_size * effective_isl
            result = op.query(
                self._database,
                x=x,
                batch_size=batch_size,
                beam_width=1,
                s=effective_isl,
                prefix=prefix,
                model_name=self._model_name,
                seq_imbalance_correction_scale=1.0,
            )
            total_latency += float(result)

        return total_latency

    def _predict_generation_latency(self, batch_size: int, isl: int, osl: int) -> float:
        if osl <= 1:
            return 0.0

        # BaseModel stores speculative decode width on _nextn, which generation_ops scale by.
        effective_batch_size = batch_size * (self._model._nextn + 1)
        total_latency = 0.0

        for step in range(0, osl - 1, DEFAULT_STATIC_STRIDE):
            step_latency = 0.0
            for op in self._model.generation_ops:
                result = op.query(
                    self._database,
                    x=effective_batch_size,
                    batch_size=effective_batch_size,
                    beam_width=1,
                    s=isl + step + 1,
                    model_name=self._model_name,
                    gen_seq_imbalance_correction_scale=1.0,
                )
                step_latency += float(result)

            repeat_count = min(DEFAULT_STATIC_STRIDE, osl - 1 - step)
            total_latency += step_latency * repeat_count

        return total_latency

    def predict_prefill(
        self, batch_size: int, isl: int, prefix: int, osl: int
    ) -> float:
        """Predict prefill latency in ms. Parameters match AIC RuntimeConfig."""
        # AIC requires at least 1 new token (isl > prefix)
        actual_prefix = min(prefix, isl - 1) if isl > 0 else 0
        return self._predict_context_latency(batch_size, isl, actual_prefix)

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict decode (generation) latency in ms."""
        return self._predict_generation_latency(batch_size, isl, osl)


def create_session(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    backend_version: str | None = None,
) -> AicSession:
    """Factory function called from Rust via PyO3."""
    return AicSession(backend_name, system, model_path, tp_size, backend_version)
