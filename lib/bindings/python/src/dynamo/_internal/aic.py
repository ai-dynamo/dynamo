# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.14.0",
    "sglang": "0.5.6.post2",
}
DEFAULT_STATIC_STRIDE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
_BYTES_PER_GIB = 1 << 30


def _validate_supported_backend(backend_name: str) -> None:
    if backend_name not in DEFAULT_BACKEND_VERSIONS:
        supported = ", ".join(sorted(DEFAULT_BACKEND_VERSIONS))
        raise ValueError(
            f"unsupported AIC backend {backend_name!r}; supported backends: {supported}"
        )


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Return the pinned backend version used for AIC perf lookups."""
    _validate_supported_backend(backend_name)
    if backend_version is not None:
        return backend_version
    return DEFAULT_BACKEND_VERSIONS[backend_name]


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
    ) -> int:
        """Estimate rank-local KV cache blocks from AIC's per-GPU memory model."""
        _validate_supported_backend(self._backend_name)
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
        else:
            # vLLM's knob caps total engine memory. Subtract AIC's modeled
            # non-KV footprint from that cap to get the KV budget.
            non_kv_bytes = (
                float(memory["total"]) - float(memory.get("kvcache", 0.0))
            ) * _BYTES_PER_GIB
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


def estimate_num_gpu_blocks(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    mem_fraction_static: float | None = None,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
) -> int:
    """Estimate rank-local KV cache blocks for mocker/replay AIC configs."""
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
    )
