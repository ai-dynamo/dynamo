# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic specs for replay_optimize, shaped to mirror DGDR v1beta1.

Field names follow DGDR's lowerCamelCase convention so that the eventual
upstream-back into `deploy/operator/api/v1beta1/...` is a clean merge.
Method names follow Python / Pydantic convention (snake_case), matching how
`dgdr_v1beta1_types.py` itself defines its one internal method
(`_validate_sla_options`).

DGDR shapes we clone / extend:
- `EngineSpec`    — local extension (DGDR has `model`/`backend` flat on the
                    outer; we need engine-args carriers)
- `HardwareSpec`  — subset clone of DGDR.HardwareSpec (gpuSku + totalGpus only)
- `WorkloadSpec`  — DGDR.WorkloadSpec + replay extensions, unified synthetic/
                    trace with a `traceFile` discriminator
- `SLASpec`       — DGDR.SLASpec field names + p95 variants; we explicitly
                    do NOT clone DGDR's "ttft+itl XOR e2eLatency" validator
                    because our optimizer treats them as independent additive
                    penalties (all three may be set)
- `RouterSpec`    — our sweep-oriented router config; analogous location to
                    DGDR.KVRouterSpec but different semantics (runtime flag
                    there vs. dev-time sweep here)
- `ReplayOptimizeSpec` — top-level bundle, analog to
                    DGDR.DynamoGraphDeploymentRequestSpec
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.profiler.utils.dgdr_v1beta1_types import BackendType, GPUSKUType

from .constants import (
    AIC_BACKEND_VERSIONS,
    DEFAULT_MAX_PARALLEL_EVALS,
    DEFAULT_OVERLAP_SCORE_WEIGHTS,
)


class RouterMode(str, Enum):
    """Router mode for the replay search.

    `BOTH` triggers a combined sweep across `KV_ROUTER` and `ROUND_ROBIN`;
    `round_robin` collapses `overlapWeights` to `(0.0,)` (guardrail #5).
    Subclasses `str` for Pydantic coercion and wire-compatibility with the
    existing `router_mode` field on `DenseReplayState` / `DenseAggReplayState`.
    """

    KV_ROUTER = "kv_router"
    ROUND_ROBIN = "round_robin"
    BOTH = "both"


class ReplayObjective(str, Enum):
    """Optimization objective driving state ranking in the search."""

    THROUGHPUT = "throughput"
    MEAN_TTFT = "mean_ttft"
    MEAN_E2E_LATENCY = "mean_e2e_latency"

    def score(self, report: Mapping[str, Any]) -> float:
        if self is ReplayObjective.THROUGHPUT:
            return float(report["output_throughput_tok_s"])
        if self is ReplayObjective.MEAN_TTFT:
            return -float(report["mean_ttft_ms"])
        return -float(report["mean_e2e_latency_ms"])


class EngineSpec(BaseModel):
    """Model + backend + engine-arg carriers.

    DGDR has `model: str` and `backend: BackendType` flat on the outer spec and
    no engine-args equivalent, so this spec is a replay-local extension.

    Carries engine args for both agg and disagg paths; the relevant
    `optimize_dense_*` entry asserts the right fields are populated
    (guardrail #8).

    `backend` has no default — pre-Phase-2 `optimize_dense_*` required it; keep
    the explicit contract so a forgotten backend fails at spec construction
    instead of silently falling through to a vLLM run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str
    backend: BackendType
    baseEngineArgs: MockEngineArgs | None = None
    basePrefillEngineArgs: MockEngineArgs | None = None
    baseDecodeEngineArgs: MockEngineArgs | None = None

    @field_validator("backend", mode="after")
    @classmethod
    def _validate_backend_supported_by_aic(cls, backend: BackendType) -> BackendType:
        # Guardrail #6: DGDR's BackendType allows Auto / Trtllm which AIC
        # doesn't support in replay_optimize; reject at spec-construction time
        # instead of crashing mid-search.
        if backend.value not in AIC_BACKEND_VERSIONS:
            raise ValueError(
                f"backend must be one of {sorted(AIC_BACKEND_VERSIONS)}, "
                f"got {backend.value!r}"
            )
        return backend


class HardwareSpec(BaseModel):
    """GPU budget + SKU. Subset clone of DGDR.HardwareSpec."""

    model_config = ConfigDict(extra="forbid")

    # Guardrail #7: keep the union so exotic AIC systems outside the current
    # GPUSKUType enumeration still work. Pydantic coerces matching strings to
    # the enum; non-matching strings stay as str and reach AIC untouched.
    gpuSku: GPUSKUType | str
    totalGpus: int = Field(gt=0)


_SYNTHETIC_ONLY_FIELDS: tuple[str, ...] = (
    "isl",
    "osl",
    "concurrency",
    "requestRate",
    "requestCount",
)


class WorkloadSpec(BaseModel):
    """Workload, unified across synthetic and trace replay.

    Extends DGDR.WorkloadSpec with:
    - replay-specific synthetic knobs (request_count, shared-prefix / multi-
      turn controls, arrival interval)
    - trace-source fields (`traceFile`, `arrivalSpeedupRatio`)

    `traceFile` acts as a discriminator:
    - when set, the workload is trace-based and the synthetic-only fields
      (`isl`, `osl`, `concurrency`, `requestRate`, `requestCount`) must not
      be populated — the validator rejects mixed mode to avoid silent data loss
    - when unset, the synthetic fields `isl`, `osl`, `concurrency`, and
      `requestCount` are all required
    """

    model_config = ConfigDict(extra="forbid")

    # DGDR base fields
    isl: int | None = None
    osl: int | None = None
    concurrency: float | None = None
    requestRate: float | None = None

    # Replay synthetic extensions
    requestCount: int | None = None
    sharedPrefixRatio: float = 0.0
    numPrefixGroups: int = 0
    turnsPerSession: int = 1
    interTurnDelayMs: float = 0.0
    arrivalIntervalMs: float = 0.0

    # Replay trace-source extensions (mutually exclusive with synthetic fields)
    traceFile: str | None = None
    traceFormat: str = "mooncake"
    arrivalSpeedupRatio: float = 1.0
    traceReplayConcurrency: int | None = None
    traceSharedPrefixRatio: float = 0.0
    traceNumPrefixGroups: int = 0

    @model_validator(mode="after")
    def _validate_source(self) -> "WorkloadSpec":
        if self.traceFile is not None:
            mixed = [
                name
                for name in _SYNTHETIC_ONLY_FIELDS
                if getattr(self, name) is not None
            ]
            if mixed:
                raise ValueError(
                    "trace workload (traceFile set) must not also set synthetic "
                    f"fields: {mixed}"
                )
            if self.traceFormat not in {"mooncake", "applied_compute_agentic"}:
                raise ValueError(
                    "traceFormat must be either 'mooncake' or 'applied_compute_agentic', got "
                    f"{self.traceFormat!r}"
                )
            if (
                self.traceReplayConcurrency is not None
                and self.traceReplayConcurrency < 1
            ):
                raise ValueError("traceReplayConcurrency must be at least 1")
            if not 0.0 <= self.traceSharedPrefixRatio <= 1.0:
                raise ValueError("traceSharedPrefixRatio must be between 0.0 and 1.0")
            if self.traceNumPrefixGroups < 0:
                raise ValueError("traceNumPrefixGroups must be non-negative")
            if self.traceSharedPrefixRatio > 0.0 and self.traceNumPrefixGroups == 0:
                raise ValueError(
                    "traceSharedPrefixRatio > 0 requires traceNumPrefixGroups >= 1"
                )
            if (
                self.traceFormat == "applied_compute_agentic"
                and self.traceReplayConcurrency is None
            ):
                raise ValueError(
                    "traceFormat='applied_compute_agentic' requires traceReplayConcurrency"
                )
            return self

        missing = [
            name
            for name in ("isl", "osl", "concurrency", "requestCount")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "synthetic workload requires "
                + ", ".join(missing)
                + "; or set traceFile for trace replay"
            )
        return self

    @property
    def isTraceBased(self) -> bool:
        return self.traceFile is not None


# SLASpec field names → Rust replay-report keys. The Rust runner emits
# snake_case + `_ms` suffix (`mean_ttft_ms`, `mean_tpot_ms`, ...) while DGDR's
# convention is camelCase + no unit suffix. This table bridges the two; the
# Rust side renaming is out of scope here.
_SLA_REPORT_KEYS: dict[str, str] = {
    "ttft": "mean_ttft_ms",
    "itl": "mean_tpot_ms",
    "e2eLatency": "mean_e2e_latency_ms",
    "p95Ttft": "p95_ttft_ms",
    "p95Itl": "p95_tpot_ms",
    "p95E2eLatency": "p95_e2e_latency_ms",
}


class SLASpec(BaseModel):
    """Latency SLA bounds.

    Guardrail #1 + #2: defaults are `None` (unconstrained); we do NOT clone
    DGDR's `_validate_sla_options` — our optimizer treats ttft / itl /
    e2eLatency as independent additive penalties and the example uses all
    three simultaneously.

    `itl` is the DGDR-native name for what the Rust runner reports as
    `mean_tpot_ms` (inter-token latency == time-per-output-token).
    """

    model_config = ConfigDict(extra="forbid")

    ttft: float | None = None
    itl: float | None = None
    e2eLatency: float | None = None
    p95Ttft: float | None = None
    p95Itl: float | None = None
    p95E2eLatency: float | None = None

    def _active(
        self, report: Mapping[str, Any]
    ) -> Iterator[tuple[str, float | None, float]]:
        """Yield `(field_name, metric_from_report_or_None, bound)` for each set bound."""
        for field_name, report_key in _SLA_REPORT_KEYS.items():
            bound = getattr(self, field_name)
            if bound is None or bound <= 0:
                continue
            value = report.get(report_key)
            yield field_name, None if value is None else float(value), float(bound)

    def violation_penalty(self, report: Mapping[str, Any]) -> float:
        """Sum of positive (metric/bound - 1) across active SLA bounds.

        Missing report key contributes `inf` (fails the feasibility gate)
        rather than silently scoring as zero.
        """
        penalty = 0.0
        for _, metric, bound in self._active(report):
            if metric is None:
                penalty += math.inf
                continue
            penalty += max(metric / bound - 1.0, 0.0)
        return penalty

    def summarize(self, report: Mapping[str, Any]) -> str:
        statuses: list[str] = []
        for field_name, metric, bound in self._active(report):
            if metric is None:
                statuses.append(f"{field_name}=missing<={bound:g} unsatisfied")
                continue
            state = "satisfied" if metric <= bound else "unsatisfied"
            statuses.append(f"{field_name}={metric:.3f}<={bound:g} {state}")
        return "sla=" + ", ".join(statuses) if statuses else "sla=none"

    def aic_task_kwargs(self) -> dict[str, float | None]:
        """Translate to `aiconfigurator.sdk.task.TaskConfig` kwargs.

        AIC's external API still uses `tpot` and `request_latency`; we keep
        those wire names untouched.
        """
        return {
            "ttft": self.ttft,
            "tpot": self.itl,
            "request_latency": self.e2eLatency,
        }


class RouterSpec(BaseModel):
    """Router config for the search.

    Analogous location to DGDR.KVRouterSpec but semantically different: DGDR
    has a single runtime on/off flag, we have a dev-time sweep over overlap
    score weights plus a mode selector.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mode: RouterMode = RouterMode.KV_ROUTER
    # None → fallback to DEFAULT_OVERLAP_SCORE_WEIGHTS (guardrail #3). Empty
    # list rejected (guardrail #4). Round-robin auto-collapse happens in
    # `effectiveOverlapWeights` (guardrail #5).
    overlapWeights: list[float] | None = None
    baseRouterConfig: KvRouterConfig | None = None

    @field_validator("overlapWeights", mode="after")
    @classmethod
    def _reject_empty_weights(cls, weights: list[float] | None) -> list[float] | None:
        if weights is not None and len(weights) == 0:
            raise ValueError("overlapWeights must not be empty")
        return weights

    @property
    def effectiveOverlapWeights(self) -> tuple[float, ...]:
        """Resolve to the concrete weight sweep used by the search."""
        if self.mode is RouterMode.ROUND_ROBIN:
            return (0.0,)
        if self.overlapWeights is None:
            return DEFAULT_OVERLAP_SCORE_WEIGHTS
        return tuple(float(w) for w in self.overlapWeights)


class ReplayOptimizeSpec(BaseModel):
    """Top-level spec; analog to DGDR.DynamoGraphDeploymentRequestSpec."""

    model_config = ConfigDict(extra="forbid")

    engine: EngineSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    sla: SLASpec = Field(default_factory=SLASpec)
    router: RouterSpec = Field(default_factory=RouterSpec)
    objective: ReplayObjective = ReplayObjective.THROUGHPUT
    maxParallelEvals: int = Field(default=DEFAULT_MAX_PARALLEL_EVALS, gt=0)
