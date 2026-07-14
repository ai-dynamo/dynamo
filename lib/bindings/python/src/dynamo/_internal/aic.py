# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import importlib
import logging
import math
import os

logger = logging.getLogger(__name__)

_NEXTN_ACCEPT_RATES_LEN = 5
# AIC CLI default when accept-rates are omitted (``cli/main.py:795``).
_DEFAULT_NEXTN_ACCEPT_RATES = [0.85, 0.3, 0.0, 0.0, 0.0]

# Default backend versions match the AIC v0.9.0 perf DB.
DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.19.0",
    "sglang": "0.5.10",
    "trtllm": "1.3.0rc10",
}
_KV_CAPACITY_BACKENDS = frozenset(DEFAULT_BACKEND_VERSIONS)
DEFAULT_STATIC_STRIDE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9


class AicMemoryEstimatorUnavailableError(RuntimeError):
    """Raised when the optional AIC KV-cache estimator cannot be imported."""


def _validate_kv_capacity_backend(backend_name: str) -> None:
    if backend_name not in _KV_CAPACITY_BACKENDS:
        supported = ", ".join(sorted(_KV_CAPACITY_BACKENDS))
        raise ValueError(
            "AIC KV cache capacity estimation does not support "
            f"backend {backend_name!r}; supported backends: {supported}. "
            "Set num_gpu_blocks explicitly for this backend."
        )


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Return the pinned backend version used for AIC perf lookups."""
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
    from aiconfigurator.sdk import common

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
    """Normalize accept-rates to AIC's fixed length-5 slot.

    AIC caps MTP draft tokens at 5 (``ModelConfig.nextn`` "at most mtp5",
    ``sdk/config.py:28``) and ``calc_expectation`` indexes into the list up
    to ``nextn``. When rates are omitted entirely we fall back to AIC's CLI
    default (``cli/main.py:795``); an explicit shorter list is zero-padded and
    a longer one is truncated, so callers never trip over IndexError downstream.
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
        gemm_dtype: str | None = None,
        moe_dtype: str | None = None,
        fmha_dtype: str | None = None,
        kv_cache_dtype: str | None = None,
        comm_dtype: str | None = None,
        nextn: int | None = None,
        nextn_accept_rates: list[float] | str | None = None,
        systems_path: str | None = None,
    ):
        aic = _load_aiconfigurator()
        version = resolve_backend_version(backend_name, backend_version)

        database_kwargs = (
            {"systems_paths": systems_path} if systems_path is not None else {}
        )
        database = aic["get_database"](
            system=system, backend=backend_name, version=version, **database_kwargs
        )
        if database is None:
            supported = (
                aic["get_supported_databases"](**database_kwargs)
                .get(system, {})
                .get(backend_name, [])
            )
            supported_versions = ", ".join(supported) if supported else "<none>"
            raise RuntimeError(
                "AIC perf database not found for "
                f"system={system!r}, backend={backend_name!r}, version={version!r}. "
                f"Supported versions for this system/backend: {supported_versions}"
            )

        model_config_kwargs: dict = dict(
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=attention_dp_size or 1,
        )
        # Quantization overrides drive the per-op perf-DB lookups (GEMM/MoE/FMHA
        # precision) and the KV-cache element size, so predicted latency tracks
        # the quantized deployment instead of the model's default dtype. Omit
        # unset fields so ModelConfig keeps its own defaults.
        for cfg_key, field, dtype in (
            ("gemm_quant_mode", "gemm", gemm_dtype),
            ("moe_quant_mode", "moe", moe_dtype),
            ("fmha_quant_mode", "fmha", fmha_dtype),
            ("kvcache_quant_mode", "kvcache", kv_cache_dtype),
            ("comm_quant_mode", "comm", comm_dtype),
        ):
            quant_mode = _resolve_quant_mode(field, dtype)
            if quant_mode is not None:
                model_config_kwargs[cfg_key] = quant_mode
        if nextn:
            # Mirror the Rust 1..=5 contract; AIC indexes accept_rates up to
            # nextn, so >5 would IndexError in calc_expectation.
            if not 1 <= nextn <= _NEXTN_ACCEPT_RATES_LEN:
                raise ValueError(
                    f"nextn must be 1..={_NEXTN_ACCEPT_RATES_LEN} when set, got {nextn}"
                )
            model_config_kwargs["nextn"] = nextn
            model_config_kwargs["nextn_accept_rates"] = _pad_nextn_accept_rates(
                nextn_accept_rates
            )
        model_config = aic["config"].ModelConfig(**model_config_kwargs)
        model = aic["get_model"](
            model_path=model_path,
            model_config=model_config,
            backend_name=backend_name,
        )
        backend = aic["get_backend"](backend_name)
        self._session = aic["InferenceSession"](
            model=model, database=database, backend=backend
        )
        self._backend = backend
        self._backend_name = backend_name
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

        # Phase 1.5: compile the model's op list to a Rust Engine ONCE, so each
        # predict call is a single Rust dispatch instead of a per-call Python
        # walk over model.context_ops / generation_ops. Falls back to the
        # Python op-walk if aiconfigurator predates Phase 1.5 or the build fails.
        self._engine = self._build_compiled_engine()

    def _build_compiled_engine(self):
        """Build a cached aiconfigurator_core EngineHandle from the already-built
        model, or return None to fall back to the Python op-walk."""
        if os.environ.get("DYNAMO_AIC_DISABLE_COMPILED_ENGINE"):
            logger.info(
                "AIC compiled-engine path disabled via env; using Python op-walk."
            )
            return None
        try:
            from aiconfigurator.sdk.rust_engine_step import _cached_engine_handle
        except Exception as exc:  # aiconfigurator without the Phase 1.5 engine
            logger.info(
                "AIC compiled-engine path unavailable (%s); using Python op-walk.",
                exc,
            )
            return None
        try:
            handle = _cached_engine_handle(self._model, self._database)
            logger.info("AIC compiled-engine path active (Phase 1.5 Rust engine).")
            return handle
        except Exception as exc:
            logger.warning(
                "AIC compiled-engine build failed (%s); using Python op-walk.", exc
            )
            return None

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
        if self._engine is not None:
            # The engine's predict_prefill_latency takes the FULL isl and
            # subtracts `prefix` internally, whereas the caller already gives us
            # the post-prefix `effective_isl`. Pass effective_isl + prefix so the
            # engine recovers the same effective length (and keeps prefix for the
            # KV-cache-aware context-attention cost).
            return self._engine.predict_prefill_latency(
                batch_size, effective_isl + prefix, prefix
            )
        return self._predict_context_latency(batch_size, effective_isl, prefix)

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict decode (generation) latency in ms."""
        if self._engine is not None:
            return self._engine.predict_decode_latency(batch_size, isl, osl)
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
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
    nextn: int | None = None,
    nextn_accept_rates: list[float] | str | None = None,
    systems_path: str | None = None,
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
        gemm_dtype=gemm_dtype,
        moe_dtype=moe_dtype,
        fmha_dtype=fmha_dtype,
        kv_cache_dtype=kv_cache_dtype,
        comm_dtype=comm_dtype,
        nextn=nextn,
        nextn_accept_rates=nextn_accept_rates,
        systems_path=systems_path,
    )


def estimate_num_gpu_blocks(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    systems_path: str | None = None,
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

    Delegates the budget math to aiconfigurator's unified
    ``sdk.memory.estimate_num_gpu_blocks`` (the single source of truth as of
    aiconfigurator 0.8.0, PR #1159) instead of recomputing it here. The result is
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

    # Imported lazily: aiconfigurator is an optional dependency (the `mocker`
    # extra), so importing it at module scope would break callers that never run
    # AIC estimation. Report a typed error so callers can choose their policy:
    # mocker warns and uses its existing default block count, while direct callers
    # still receive an actionable error. Mirrors `_load_aiconfigurator`.
    # TODO: account for whether specdec is enabled (pass `nextn=...`). Currently
    #   omitted due to a downstream AIC bug where `_get_memory_usage` predicts
    #   negative KV capacity with Eagle.
    try:
        memory = importlib.import_module("aiconfigurator.sdk.memory")
    except ModuleNotFoundError as exc:
        if exc.name == "aiconfigurator.sdk.memory":
            raise AicMemoryEstimatorUnavailableError(
                "aiconfigurator.sdk.memory is required for AIC KV-cache estimation; "
                "install a compatible aiconfigurator version"
            ) from exc
        if exc.name in {"aiconfigurator", "aiconfigurator.sdk"}:
            raise RuntimeError(
                "aiconfigurator is required for AIC KV-cache estimation but is not "
                "installed; install the 'mocker' extra or set num_gpu_blocks "
                "explicitly (e.g. --num-gpu-blocks-override)"
            ) from exc
        raise

    # AIC's non-KV memory is independent of batch size (activations track
    # max_num_tokens), so the fixed max_batch_size here does not affect the result.
    return int(
        memory.estimate_num_gpu_blocks(
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
            systems_path=systems_path,
        )
    )
