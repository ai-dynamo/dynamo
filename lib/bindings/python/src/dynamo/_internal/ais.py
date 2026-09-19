# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AISimulate session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)

_NEXTN_ACCEPT_RATES_LEN = 5
# Dynamo's historical default when conditional acceptance rates are omitted.
_DEFAULT_NEXTN_ACCEPT_RATES = [0.85, 0.3, 0.0, 0.0, 0.0]

# Resolve defaults through the queryable slots in the pinned AISimulate perf DB.
DEFAULT_BACKEND_VERSIONS = {
    "vllm": "current",
    "sglang": "current",
    "trtllm": "current",
}
DEFAULT_STATIC_STRIDE = 32


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Preserve explicit versions; otherwise use the release database current slot."""
    if backend_version is not None:
        return backend_version
    return DEFAULT_BACKEND_VERSIONS.get(backend_name, DEFAULT_BACKEND_VERSIONS["vllm"])


def _normalize_quant_mode(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None
    if value == "int4":
        return "int4_wo"
    return value


def _resolve_quant_mode(field: str, value: str | None):
    """Resolve a CLI dtype override to the AIS per-field quant-mode enum.

    The four quant fields accept *different* value sets (e.g. KV cache only
    supports ``bfloat16``/``int8``/``fp8``), so the string -> enum lookup is per
    field. On an unsupported value, raise a clear ``ValueError`` naming the
    field and its allowed values instead of letting an opaque ``KeyError``
    escape from the estimator. ``field`` is one of ``gemm``,
    ``moe``, ``fmha``, ``kvcache``, ``comm``.
    """
    normalized = _normalize_quant_mode(value)
    if normalized is None:
        return None
    from aisimulate_core.sdk import common

    enum_cls = {
        "gemm": common.GEMMQuantMode,
        "moe": common.MoEQuantMode,
        "fmha": common.FMHAQuantMode,
        "kvcache": common.KVCacheQuantMode,
        "comm": common.CommQuantMode,
    }[field]
    try:
        return enum_cls[normalized]
    except KeyError:
        allowed = ", ".join(member.name for member in enum_cls)
        raise ValueError(
            f"unsupported AIS {field} quant mode {value!r} "
            f"(normalized to {normalized!r}); supported values: {allowed}"
        ) from None


def _resolve_quant_mode_name(field: str, value: str | None) -> str | None:
    """Like :func:`_resolve_quant_mode` but return the canonical quant-mode
    *name*, validated
    against the field's enum. ``None`` means "use the model default"."""
    mode = _resolve_quant_mode(field, value)
    return mode.name if mode is not None else None


def _pad_nextn_accept_rates(
    nextn_accept_rates: list[float] | str | None,
) -> list[float]:
    """Validate legacy CLI acceptance rates and normalize to five positions."""
    if isinstance(nextn_accept_rates, str):
        try:
            nextn_accept_rates = [
                float(x) for x in nextn_accept_rates.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "ais_nextn_accept_rates must be comma-separated floats, got "
                f"{nextn_accept_rates!r}"
            ) from exc
    if not nextn_accept_rates:
        return list(_DEFAULT_NEXTN_ACCEPT_RATES)
    rates = list(nextn_accept_rates)
    # Rates are acceptance probabilities; out-of-range or non-finite values
    # would silently skew calc_expectation rather than surface a config error.
    if any(not math.isfinite(r) or not 0.0 <= r <= 1.0 for r in rates):
        raise ValueError(
            f"ais_nextn_accept_rates must be finite floats in [0, 1], got {rates}"
        )
    if len(rates) < _NEXTN_ACCEPT_RATES_LEN:
        rates = rates + [0.0] * (_NEXTN_ACCEPT_RATES_LEN - len(rates))
    elif len(rates) > _NEXTN_ACCEPT_RATES_LEN:
        rates = rates[:_NEXTN_ACCEPT_RATES_LEN]
    return rates


class AisSession:
    """One canonical AISimulate estimator with static latency query adapters.

    The estimator owns identity, data selection and tuning. Dynamo only maps its
    existing static query convention to one forward-pass workload at a time.
    """

    def __init__(self, config, *, worker_type: str | None = None):
        from aisimulate_core.sdk import RustForwardPassPerfModel

        if os.environ.get("DYNAMO_AIS_DISABLE_COMPILED_ENGINE"):
            raise ValueError(
                "AIS perf modeling requires the compiled canonical estimator"
            )
        payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        if worker_type is not None and payload.get("worker_type") != worker_type:
            raise ValueError(
                "config.worker_type conflicts with the requested worker role"
            )
        self._nextn = int(payload.get("nextn", 0))
        speculation = payload.get("speculation") or {}
        if speculation.get("kind") == "ngram":
            self._nextn = int(
                speculation.get("params", {}).get("num_speculative_tokens", 0)
            )
        self._estimator = RustForwardPassPerfModel.best_available(payload)
        diagnostics = self._estimator.diagnostics()
        provenance = diagnostics.get("provenance") or {}
        self.config = provenance.get("config", payload)
        if (
            self._nextn
            and provenance.get("selected_estimation_mode") == "fpm_interpolation"
        ):
            raise ValueError(
                "AIS FPM interpolation does not support speculative verification queries"
            )
        if diagnostics.get("readiness") != "ready":
            raise ValueError(
                "AIS static latency queries require a ready estimator; "
                "cold fpm_regression must receive observations before use"
            )

    def _estimate(self, scheduled: dict[str, int]) -> float:
        result = self._estimator.estimate_forward_pass_time_ms(
            {"scheduled_requests": scheduled}
        )
        if result is None or not math.isfinite(result) or result < 0:
            raise RuntimeError("AIS estimator has no finite latency for this workload")
        return result

    def predict_prefill(
        self, batch_size: int, effective_isl: int, prefix: int
    ) -> float:
        """Predict milliseconds for newly computed tokens and cached prefix."""
        if batch_size <= 0 or effective_isl <= 0 or prefix < 0:
            raise ValueError(
                "batch_size/effective_isl must be positive and prefix nonnegative"
            )
        return self._estimate(
            {
                "num_prefill_requests": batch_size,
                "sum_prefill_tokens": batch_size * effective_isl,
                "sum_prefill_kv_tokens": batch_size * prefix,
            }
        )

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict generation milliseconds, preserving static stride and KV convention."""
        if batch_size <= 0 or isl < 0 or osl < 0:
            raise ValueError(
                "batch_size must be positive and sequence lengths nonnegative"
            )
        batch = batch_size * (self._nextn + 1)
        total = 0.0
        for step in range(0, max(osl - 1, 0), DEFAULT_STATIC_STRIDE):
            latency = self._estimate(
                {
                    "num_decode_requests": batch,
                    "sum_decode_kv_tokens": batch * (isl + step + 1),
                }
            )
            total += latency * min(DEFAULT_STATIC_STRIDE, osl - 1 - step)
        return total


def create_session(config, *, worker_type: str | None = None) -> AisSession:
    """Construct the canonical estimator for a static latency consumer."""
    return AisSession(config, worker_type=worker_type)


def estimate_canonical_num_gpu_blocks(config, **scheduler_options) -> int:
    """Estimate capacity from the exact resolved canonical estimator identity."""
    from aisimulate.aic import materialize_aic_num_gpu_blocks

    payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    payload.setdefault("estimation_mode", "auto")
    payload.setdefault("fallback_policy", "deny")
    lowered = materialize_aic_num_gpu_blocks(
        {
            **{
                name: value
                for name, value in scheduler_options.items()
                if value is not None
            },
            "timing_model": {"type": "external", "provider": "aic", "config": payload},
        }
    )
    return int(lowered["num_gpu_blocks"])
