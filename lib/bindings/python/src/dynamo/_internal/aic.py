# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.12.0",
    "sglang": "0.5.6.post2",
}
DEFAULT_STATIC_STRIDE = 32


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Return the pinned backend version used for AIC perf lookups."""
    if backend_version is not None:
        return backend_version
    return DEFAULT_BACKEND_VERSIONS.get(backend_name, DEFAULT_BACKEND_VERSIONS["vllm"])


def _load_aiconfigurator():
    try:
        from aiconfigurator.sdk import config
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.inference_session import InferenceSession
        from aiconfigurator.sdk.models import get_model
        from aiconfigurator.sdk.perf_database import (
            get_database,
            get_supported_databases,
        )
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError(
            "aiconfigurator is required for AIC perf modeling but is not installed"
        ) from exc

    return {
        "config": config,
        "get_backend": get_backend,
        "InferenceSession": InferenceSession,
        "get_model": get_model,
        "get_database": get_database,
        "get_supported_databases": get_supported_databases,
    }


class AicSession:
    """Wrap an AIC InferenceSession with direct prefill/decode predictors."""

    def __init__(
        self,
        backend_name: str,
        system: str,
        model_path: str,
        tp_size: int,
        backend_version: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        attention_dp_size: int | None = None,
    ):
        aic = _load_aiconfigurator()
        version = resolve_backend_version(backend_name, backend_version)

        database = aic["get_database"](
            system=system, backend=backend_name, version=version
        )
        if database is None:
            supported = (
                aic["get_supported_databases"]().get(system, {}).get(backend_name, [])
            )
            supported_versions = ", ".join(supported) if supported else "<none>"
            raise RuntimeError(
                "AIC perf database not found for "
                f"system={system!r}, backend={backend_name!r}, version={version!r}. "
                f"Supported versions for this system/backend: {supported_versions}"
            )

        model_config = aic["config"].ModelConfig(
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=attention_dp_size or 1,
        )
        model = aic["get_model"](
            model_path=model_path,
            model_config=model_config,
            backend_name=backend_name,
        )
        backend = aic["get_backend"](backend_name)
        self._session = aic["InferenceSession"](
            model=model, database=database, backend=backend
        )
        self._database = database
        self._model = model
        self._model_name = getattr(model, "model_name", None) or model_path
        logger.info(
            "AIC session initialized: backend=%s, system=%s, model=%s, tp=%d",
            backend_name,
            system,
            model_path,
            tp_size,
        )

    def _predict_context_latency(
        self, batch_size: int, effective_isl: int, prefix: int
    ) -> float:
        if effective_isl <= 0:
            raise ValueError(
                f"effective_isl must be positive, got effective_isl={effective_isl}"
            )

        total_latency = 0.0
        for op in self._model.context_ops:
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
        self, batch_size: int, effective_isl: int, prefix: int
    ) -> float:
        """Predict prefill latency in ms from uncached tokens and cached prefix."""
        return self._predict_context_latency(batch_size, effective_isl, prefix)

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict decode (generation) latency in ms."""
        return self._predict_generation_latency(batch_size, isl, osl)


def create_session(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
) -> AicSession:
    """Factory function called from Rust via PyO3."""
    return AicSession(
        backend_name,
        system,
        model_path,
        tp_size,
        backend_version,
        moe_tp_size,
        moe_ep_size,
        attention_dp_size,
    )
