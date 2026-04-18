# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process AIC interpolation sweep that bootstraps the planner's regression.

This module replaces the profiler's NPZ-based AIC interpolation for rapid
mode. It runs the sweep in the planner pod at startup and produces
``ForwardPassMetrics`` directly — no disk I/O, no ConfigMap round-trip.

The FPM conventions match the thorough-mode path so both bootstrap sources
are interchangeable from the regression models' perspective:

* **Prefill** — one FPM per sweep ISL. ``num_prefill_requests=1``,
  ``sum_prefill_tokens=isl``, ``wall_time = per-rank TTFT``. AIC's
  ``estimate_prefill_perf`` with ``batch_size=1`` and the correct
  ``attention_dp_size`` returns per-rank latency by construction, so no DP
  scaling is needed here.

* **Decode** — one FPM per (ISL, aggregate-num-request) sample.
  ``num_decode_requests = aggregate_num_req``,
  ``sum_decode_kv_tokens = aggregate_num_req * (isl + osl/2)``,
  ``wall_time = per-step ITL``. Matches thorough's aggregate semantics. We
  convert aggregate → per-rank before calling AIC (AIC's
  ``RuntimeConfig.batch_size`` is per-attention-DP-rank, per the
  ``TrtllmWideEPDeepSeekModel`` comment in aiconfigurator/sdk/models.py).

The 7 MoE-DEP bugs that silently corrupted the old profiler path are fixed
here: every AIC call uses :func:`picked_to_aic_model_config_kwargs` so
``moe_tp_size`` / ``moe_ep_size`` / ``attention_dp_size`` reach AIC's
``ModelConfig``; ``get_max_kv_tokens`` is scaled up by ``attention_dp_size``
to aggregate; and the decode concurrency sweep is a plain linear sweep
(the DP-multiples constraint was a thorough-mode routing requirement that
doesn't apply to a static simulator).
"""

import logging
from typing import TYPE_CHECKING

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)

if TYPE_CHECKING:
    from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator

logger = logging.getLogger(__name__)

_PREFILL_CONTEXT_MARGIN = 512  # mirror profile_prefill.py: leave room for chat template


def run_aic_interpolation(
    spec: AICInterpolationSpec,
    component_type: SubComponentType,
) -> list[ForwardPassMetrics]:
    """Run the AIC interpolation sweep and return synthetic FPMs.

    Lazy-imports ``aiconfigurator`` — callers should catch ``ImportError``
    and fall back to the file-based loader if the dependency is missing.
    """
    from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator

    estimator = AIConfiguratorPerfEstimator(
        hf_id=spec.hf_id,
        system=spec.system,
        backend=spec.backend,
    )

    if component_type == SubComponentType.PREFILL:
        return _sweep_prefill(estimator, spec)
    if component_type == SubComponentType.DECODE:
        return _sweep_decode(estimator, spec)
    raise ValueError(
        f"Unsupported component_type for AIC interpolation: {component_type}"
    )


def _sweep_prefill(
    estimator: "AIConfiguratorPerfEstimator",
    spec: AICInterpolationSpec,
) -> list[ForwardPassMetrics]:
    """Sweep prefill ISL, emit one FPM per point (per-rank semantics)."""
    pick = spec.prefill_pick
    kwargs = picked_to_aic_model_config_kwargs(pick)
    max_ctx = spec.sweep_max_context_length - _PREFILL_CONTEXT_MARGIN
    if max_ctx <= 100:
        raise ValueError(
            f"sweep_max_context_length {spec.sweep_max_context_length} is too "
            f"small to profile prefill (need > {100 + _PREFILL_CONTEXT_MARGIN})"
        )

    step = max(1, (max_ctx - 100) // spec.prefill_interpolation_granularity)
    fpms: list[ForwardPassMetrics] = []
    for isl in range(100, max_ctx, step):
        perf = estimator.estimate_prefill_perf(isl, **kwargs)
        ttft_ms = perf.get("context_latency")
        if ttft_ms is None or ttft_ms <= 0:
            logger.warning(
                "AIC returned invalid context_latency=%s for isl=%s; skipping",
                ttft_ms,
                isl,
            )
            continue
        fpms.append(_prefill_fpm(isl, ttft_ms))

    if len(fpms) < 3:
        raise RuntimeError(
            f"AIC prefill sweep produced only {len(fpms)} valid points; need >= 3"
        )
    return fpms


def _sweep_decode(
    estimator: "AIConfiguratorPerfEstimator",
    spec: AICInterpolationSpec,
) -> list[ForwardPassMetrics]:
    """Sweep decode (ISL, aggregate num_request), emit one FPM per point."""
    pick = spec.decode_pick
    kwargs = picked_to_aic_model_config_kwargs(pick)
    attention_dp = max(1, pick.dp)

    # get_max_kv_tokens returns per-rank (AIC's memory accounting is per-GPU).
    # Each DP rank has its own KV cache, so aggregate max = per_rank × dp.
    per_rank_max_kv = estimator.get_max_kv_tokens(spec.isl, spec.osl, **kwargs)
    if per_rank_max_kv <= 0:
        raise RuntimeError(
            "AIC get_max_kv_tokens returned %s; pick does not fit on GPU"
            % per_rank_max_kv
        )
    max_kv_tokens_aggregate = per_rank_max_kv * attention_dp

    osl_sweep = 500  # mirror profile_decode.py: short OSL for stable ITL measurement
    isl_step = max(
        1,
        (spec.sweep_max_context_length - osl_sweep)
        // spec.decode_interpolation_granularity,
    )
    fpms: list[ForwardPassMetrics] = []
    for isl in range(100, spec.sweep_max_context_length - osl_sweep, isl_step):
        ctx_len = isl + osl_sweep / 2.0
        max_concurrency_aggregate = max_kv_tokens_aggregate // (isl + osl_sweep)
        if max_concurrency_aggregate <= 0:
            logger.warning(
                "max_kv_tokens_aggregate=%s too small for isl=%s osl=%s; stopping sweep",
                max_kv_tokens_aggregate,
                isl,
                osl_sweep,
            )
            break

        for num_req_aggregate in _concurrency_sweep(
            max_concurrency_aggregate, spec.decode_interpolation_granularity
        ):
            # AIC RuntimeConfig.batch_size is per-attention-DP-rank.
            batch_size_per_rank = max(1, num_req_aggregate // attention_dp)
            perf = estimator.estimate_perf(
                isl,
                osl_sweep,
                batch_size_per_rank,
                mode="decode",
                **kwargs,
            )
            itl_ms = perf.get("tpot")
            if itl_ms is None or itl_ms <= 0:
                logger.warning(
                    "AIC returned invalid tpot=%s for isl=%s num_req_agg=%s; skipping",
                    itl_ms,
                    isl,
                    num_req_aggregate,
                )
                continue
            fpms.append(_decode_fpm(num_req_aggregate, ctx_len, itl_ms))

    if len(fpms) < 3:
        raise RuntimeError(
            f"AIC decode sweep produced only {len(fpms)} valid points; need >= 3"
        )
    return fpms


def _concurrency_sweep(max_concurrency: int, granularity: int) -> list[int]:
    """Linear sweep of integer concurrency levels from 1 to ``max_concurrency``.

    Unlike the thorough-mode sweep (``get_num_request_range``) we do not need
    multiples of ``attention_dp_size`` here — AIC is a static simulator, not
    a round-robin request router, so DP alignment is not required.
    """
    if max_concurrency <= 0 or granularity <= 0:
        return []
    if max_concurrency <= granularity:
        return list(range(1, max_concurrency + 1))
    step = (max_concurrency - 1) / (granularity - 1)
    points = {max(1, 1 + int(round(i * step))) for i in range(granularity)}
    return sorted(points)


def _prefill_fpm(isl: int, ttft_ms: float) -> ForwardPassMetrics:
    """Per-rank, single-request prefill FPM (matches thorough-mode convention)."""
    return ForwardPassMetrics(
        wall_time=float(ttft_ms) / 1000.0,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=int(isl),
        ),
    )


def _decode_fpm(
    num_request_aggregate: int,
    ctx_len: float,
    itl_ms: float,
) -> ForwardPassMetrics:
    """Aggregate-batch decode FPM (matches thorough-mode convention)."""
    sum_kv = int(round(num_request_aggregate * ctx_len))
    return ForwardPassMetrics(
        wall_time=float(itl_ms) / 1000.0,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=int(num_request_aggregate),
            sum_decode_kv_tokens=sum_kv,
        ),
    )


__all__ = [
    "run_aic_interpolation",
    "PickedParallelConfig",  # re-exported for convenience in callers
]
