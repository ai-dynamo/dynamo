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
_KV_CAPACITY_BACKENDS = frozenset(DEFAULT_BACKEND_VERSIONS)
DEFAULT_STATIC_STRIDE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9


def _validate_kv_capacity_backend(backend_name: str) -> None:
    if backend_name not in _KV_CAPACITY_BACKENDS:
        supported = ", ".join(sorted(_KV_CAPACITY_BACKENDS))
        raise ValueError(
            "AIC KV cache capacity estimation does not support "
            f"backend {backend_name!r}; supported backends: {supported}. "
            "Set num_gpu_blocks explicitly for this backend."
        )


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Preserve explicit versions; otherwise use the release database current slot."""
    if backend_version is not None:
        return backend_version
    return DEFAULT_BACKEND_VERSIONS.get(backend_name, DEFAULT_BACKEND_VERSIONS["vllm"])


def _normalize_aic_quant_mode(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None
    if value == "int4":
        return "int4_wo"
    return value


def _resolve_quant_mode(field: str, value: str | None):
    """Resolve a dtype-override string to aiconfigurator's per-field quant-mode
    enum, or ``None`` to use the model default.

    The four quant fields accept *different* value sets (e.g. KV cache only
    supports ``bfloat16``/``int8``/``fp8``), so the string -> enum lookup is per
    field. On an unsupported value, raise a clear ``ValueError`` naming the
    field and its allowed values instead of letting an opaque ``KeyError``
    escape from deep inside aiconfigurator. ``field`` is one of ``gemm``,
    ``moe``, ``fmha``, ``kvcache``, ``comm``.
    """
    normalized = _normalize_aic_quant_mode(value)
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
            f"unsupported AIC {field} quant mode {value!r} "
            f"(normalized to {normalized!r}); supported values: {allowed}"
        ) from None


def _resolve_quant_mode_name(field: str, value: str | None) -> str | None:
    """Like :func:`_resolve_quant_mode` but return the canonical quant-mode
    *name* (the string aiconfigurator's string-keyed APIs expect), validated
    against the field's enum. ``None`` means "use the model default"."""
    mode = _resolve_quant_mode(field, value)
    return mode.name if mode is not None else None


def _pad_nextn_accept_rates(
    nextn_accept_rates: list[float] | str | None,
) -> list[float]:
    """Normalize accept-rates for the released ``aiconfigurator`` wheel.

    The upper AIC wheel still accepts the fixed length-5 conditional-rate
    contract. When rates are omitted entirely we preserve its CLI default;
    shorter lists are zero-padded and longer lists are truncated.
    """
    if isinstance(nextn_accept_rates, str):
        try:
            nextn_accept_rates = [
                float(x) for x in nextn_accept_rates.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "aic_nextn_accept_rates must be comma-separated floats, got "
                f"{nextn_accept_rates!r}"
            ) from exc
    if not nextn_accept_rates:
        return list(_DEFAULT_NEXTN_ACCEPT_RATES)
    rates = list(nextn_accept_rates)
    # Rates are acceptance probabilities; out-of-range or non-finite values
    # would silently skew calc_expectation rather than surface a config error.
    if any(not math.isfinite(r) or not 0.0 <= r <= 1.0 for r in rates):
        raise ValueError(
            f"aic_nextn_accept_rates must be finite floats in [0, 1], got {rates}"
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

    def __init__(
        self,
        backend_name: str | None = None,
        system: str | None = None,
        model_path: str | None = None,
        tp_size: int | None = None,
        backend_version: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        attention_dp_size: int | None = None,
        gemm_dtype: str | None = None,
        moe_dtype: str | None = None,
        fmha_dtype: str | None = None,
        kv_cache_dtype: str | None = None,
        comm_dtype: str | None = None,
        nextn: int | None = None,
        nextn_accept_rates: list[float] | str | None = None,
        *,
        config=None,
        worker_type: str | None = None,
    ):
        from aisimulate_core.sdk.rust_engine_step import (
            ForwardPassPerfModelConfig,
            RustForwardPassPerfModel,
        )

        if os.environ.get("DYNAMO_AIS_DISABLE_COMPILED_ENGINE") or os.environ.get(
            "DYNAMO_AIC_DISABLE_COMPILED_ENGINE"
        ):
            raise ValueError(
                "AIS perf modeling requires the compiled canonical estimator"
            )
        if config is not None:
            legacy_identity = (
                backend_name,
                system,
                model_path,
                tp_size,
                backend_version,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                gemm_dtype,
                moe_dtype,
                fmha_dtype,
                kv_cache_dtype,
                comm_dtype,
                nextn,
            )
            if any(value is not None for value in legacy_identity):
                raise ValueError(
                    "config cannot be combined with legacy model identity arguments"
                )
            payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
            if worker_type is not None and payload.get("worker_type") != worker_type:
                raise ValueError(
                    "config.worker_type conflicts with the requested worker role"
                )
        else:
            if backend_name is None or system is None or model_path is None:
                raise ValueError("AIS model, system, and backend are required")
            payload = ForwardPassPerfModelConfig(
                model=model_path,
                system=system,
                backend=backend_name,
                worker_type=worker_type or "aggregated",
                backend_version=resolve_backend_version(backend_name, backend_version),
                tp=tp_size if tp_size is not None else 1,
                attention_dp=attention_dp_size or 1,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                gemm_quant_mode=_resolve_quant_mode_name("gemm", gemm_dtype),
                moe_quant_mode=_resolve_quant_mode_name("moe", moe_dtype),
                fmha_quant_mode=_resolve_quant_mode_name("fmha", fmha_dtype),
                kvcache_quant_mode=_resolve_quant_mode_name("kvcache", kv_cache_dtype),
                comm_quant_mode=_resolve_quant_mode_name("comm", comm_dtype),
                nextn=nextn or 0,
            ).to_dict()
        self._nextn = int(payload.get("nextn", 0))
        speculation = payload.get("speculation") or {}
        if speculation.get("kind") == "ngram":
            self._nextn = int(
                speculation.get("params", {}).get("num_speculative_tokens", 0)
            )
        if payload.get("nextn", 0):
            _pad_nextn_accept_rates(nextn_accept_rates)
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


def create_session(*args, **kwargs) -> AisSession:
    """Construct the canonical estimator; legacy flat inputs are adapted once."""
    return AisSession(*args, **kwargs)


# Compatibility imports only. New callers use AisSession.
AicSession = AisSession


def estimate_num_gpu_blocks(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    mem_fraction_static: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
) -> int:
    """Estimate rank-local KV cache blocks for mocker/replay AIC configs.

    Delegates the budget math to aiconfigurator-core's unified
    ``sdk.memory.estimate_num_gpu_blocks`` (the single source of truth for the
    AIC memory estimator) instead of recomputing it here. The result is
    per rank (per single GPU): AIC's memory model is already sharded for the
    configured TP/DP shape, so the caller must not multiply it by TP or DP.

    The backend selects which memory-fraction knob applies, mapped onto AIC's
    ``memory_fraction_kind``/``memory_fraction_value``:

    - ``vllm``   -> ``of_total`` with ``gpu_memory_utilization`` (fraction of total HBM)
    - ``sglang`` -> ``of_total`` with ``mem_fraction_static``
    - ``trtllm`` -> ``of_free`` with ``free_gpu_memory_fraction`` (fraction of the
      HBM left after the model is loaded)
    """
    _validate_kv_capacity_backend(backend_name)

    if backend_name == "trtllm":
        memory_fraction_kind = "of_free"
        memory_fraction_value = (
            free_gpu_memory_fraction
            if free_gpu_memory_fraction is not None
            else DEFAULT_FREE_GPU_MEMORY_FRACTION
        )
    elif backend_name == "sglang":
        memory_fraction_kind = "of_total"
        memory_fraction_value = (
            mem_fraction_static
            if mem_fraction_static is not None
            else DEFAULT_MEM_FRACTION_STATIC
        )
    else:  # vllm
        memory_fraction_kind = "of_total"
        memory_fraction_value = gpu_memory_utilization

    # Imported lazily from the compatibility namespace shipped by AISimulate.
    # An AIC-backed call requires AISimulate and fails fast when it is absent.
    # TODO: account for whether specdec is enabled (pass `nextn=...`). Currently
    #   omitted due to a downstream AIC bug where `_get_memory_usage` predicts
    #   negative KV capacity with Eagle.
    try:
        from aisimulate_core.sdk.memory import (
            estimate_num_gpu_blocks as aic_estimate_num_gpu_blocks,
        )
    except ImportError as exc:
        missing = exc.name or ""
        if missing in {"aisimulate_core", "aiconfigurator_core"} or missing.startswith(
            ("aisimulate_core.", "aiconfigurator_core.")
        ):
            raise RuntimeError(
                "aisimulate is required for AIC KV-cache estimation but is "
                "not installed"
            ) from exc
        raise

    # AIC's non-KV memory is independent of batch size (activations track
    # max_num_tokens), so the fixed max_batch_size here does not affect the result.
    return int(
        aic_estimate_num_gpu_blocks(
            model_path,
            system,
            backend_name,
            backend_version=resolve_backend_version(backend_name, backend_version),
            scheduler_block_size=block_size,
            max_num_tokens=max_num_batched_tokens,
            max_batch_size=1,
            memory_fraction_kind=memory_fraction_kind,
            memory_fraction_value=memory_fraction_value,
            tp_size=tp_size,
            attention_dp_size=(
                attention_dp_size if attention_dp_size is not None else 1
            ),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=_resolve_quant_mode_name("gemm", gemm_dtype),
            moe_quant_mode=_resolve_quant_mode_name("moe", moe_dtype),
            fmha_quant_mode=_resolve_quant_mode_name("fmha", fmha_dtype),
            kvcache_quant_mode=_resolve_quant_mode_name("kvcache", kv_cache_dtype),
            comm_quant_mode=_resolve_quant_mode_name("comm", comm_dtype),
        )
    )


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
