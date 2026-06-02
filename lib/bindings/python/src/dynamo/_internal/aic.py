# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import logging
import math

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
_BYTES_PER_GIB = 1 << 30


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
        nextn: int | None = None,
        nextn_accept_rates: list[float] | str | None = None,
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

        model_config_kwargs: dict = dict(
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=attention_dp_size or 1,
        )
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

    def estimate_num_gpu_blocks(
        self,
        *,
        block_size: int,
        max_num_batched_tokens: int,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        mem_fraction_static: float | None = None,
        free_gpu_memory_fraction: float | None = None,
    ) -> int:
        """Estimate rank-local KV cache blocks from AIC's per-GPU memory model."""
        _validate_kv_capacity_backend(self._backend_name)
        if block_size <= 0:
            raise ValueError(
                f"block_size must be positive, got block_size={block_size}"
            )
        if max_num_batched_tokens <= 0:
            raise ValueError(
                "max_num_batched_tokens must be positive, "
                f"got max_num_batched_tokens={max_num_batched_tokens}"
            )
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1], "
                f"got gpu_memory_utilization={gpu_memory_utilization}"
            )

        if mem_fraction_static is None:
            mem_fraction_static = DEFAULT_MEM_FRACTION_STATIC
        if not 0 < mem_fraction_static <= 1:
            raise ValueError(
                "mem_fraction_static must be in (0, 1], "
                f"got mem_fraction_static={mem_fraction_static}"
            )

        if free_gpu_memory_fraction is None:
            free_gpu_memory_fraction = DEFAULT_FREE_GPU_MEMORY_FRACTION
        if not 0 < free_gpu_memory_fraction <= 1:
            raise ValueError(
                "free_gpu_memory_fraction must be in (0, 1], "
                f"got free_gpu_memory_fraction={free_gpu_memory_fraction}"
            )

        # AIC's memory model is already rank-local for the configured TP/DP shape.
        # The returned weight/KV numbers have been sharded, so mocker should not
        # multiply the resulting block count by TP or DP.
        memory = self._backend._get_memory_usage(
            self._model,
            self._database,
            batch_size=1,
            beam_width=1,
            isl=0,
            osl=0,
            num_tokens=max_num_batched_tokens,
        )
        gpu_capacity_bytes = float(self._database.system_spec["gpu"]["mem_capacity"])
        # AIC exposes KV bytes for one sequence of block_size tokens; that is the
        # byte cost of one scheduler block on this rank.
        block_bytes = float(self._model.get_kvcache_bytes_per_sequence(block_size))
        if block_bytes <= 0:
            raise ValueError(
                f"AIC returned non-positive KV block size: block_bytes={block_bytes}"
            )

        # Non-KV memory footprint AIC models for this rank (everything resident
        # besides the KV pool: weights, activations, runtime). The vLLM and
        # TRT-LLM budgets below both derive from it; SGLang uses its own static
        # figure instead.
        non_kv_bytes = (
            float(memory["total"]) - float(memory.get("kvcache", 0.0))
        ) * _BYTES_PER_GIB

        if self._backend_name == "sglang":
            # SGLang's knob reserves a static fraction of HBM for weights,
            # runtime allocations, and the KV pool. AIC reports those static
            # non-KV allocations separately, in GiB.
            static_non_kv_bytes = (
                float(memory["weights"])
                + float(memory["nccl"])
                + float(memory["others"])
            ) * _BYTES_PER_GIB
            kv_budget_bytes = (
                gpu_capacity_bytes * mem_fraction_static - static_non_kv_bytes
            )
            fraction_name = "mem_fraction_static"
            fraction_value = mem_fraction_static
        elif self._backend_name == "trtllm":
            # TRT-LLM allocates `free_gpu_memory_fraction` of the memory that
            # remains *after* the model is loaded — unlike vLLM's
            # `gpu_memory_utilization`, which is a fraction of *total* memory.
            free_bytes = gpu_capacity_bytes - non_kv_bytes
            kv_budget_bytes = free_bytes * free_gpu_memory_fraction
            fraction_name = "free_gpu_memory_fraction"
            fraction_value = free_gpu_memory_fraction
        else:
            # vLLM's knob caps total engine memory. Subtract AIC's modeled
            # non-KV footprint from that cap to get the KV budget.
            kv_budget_bytes = gpu_capacity_bytes * gpu_memory_utilization - non_kv_bytes
            fraction_name = "gpu_memory_utilization"
            fraction_value = gpu_memory_utilization

        if kv_budget_bytes <= 0:
            raise ValueError(
                "AIC estimated no available KV cache memory: "
                f"backend={self._backend_name!r}, {fraction_name}={fraction_value}, "
                f"kv_budget_gib={kv_budget_bytes / _BYTES_PER_GIB:.3f}"
            )

        num_gpu_blocks = math.floor(kv_budget_bytes / block_bytes)
        if num_gpu_blocks <= 0:
            raise ValueError(
                "AIC estimated fewer than one KV cache block: "
                f"backend={self._backend_name!r}, block_size={block_size}, "
                f"block_bytes={block_bytes:.0f}, "
                f"kv_budget_gib={kv_budget_bytes / _BYTES_PER_GIB:.3f}"
            )
        return num_gpu_blocks


def create_session(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    nextn: int | None = None,
    nextn_accept_rates: list[float] | str | None = None,
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
        nextn=nextn,
        nextn_accept_rates=nextn_accept_rates,
    )


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
) -> int:
    """Estimate rank-local KV cache blocks for mocker/replay AIC configs."""
    _validate_kv_capacity_backend(backend_name)
    # TODO: account for whether specdec is enabled, (i.e. pass in `nextn=...`
    #   to `create_session``). Currently omitted to downstream AIC calculation
    #   bug causing `_get_memory_usage` to predict negative KV capacity w/ Eagle.
    session = create_session(
        backend_name=backend_name,
        system=system,
        model_path=model_path,
        tp_size=tp_size,
        backend_version=backend_version,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
    )
    return session.estimate_num_gpu_blocks(
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        mem_fraction_static=mem_fraction_static,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )
