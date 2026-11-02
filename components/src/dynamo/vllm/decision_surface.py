# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional native decision surface, initialized once by the worker factory."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from dynamo.common.engine_monitor import (
    ENGINE_HEALTH_CHECK_TIMEOUT,
    EngineHealthMonitorConfig,
)
from dynamo.llm.exceptions import EngineShutdown, HttpError, WorkerOverloaded
from dynamo.runtime import Context

from .decision_dispatch import DecisionDispatcher
from .decision_native_errors import NativeRuntimeError
from .decision_native_registry import (
    NativeFailed,
    NativeProfile,
    NativePublication,
    NativeReady,
)

if TYPE_CHECKING:
    from .decision_heads import HeadExecutionAdapter, PreparedHeadRead
    from .decision_native_lifecycle import NativeAdapter

logger = logging.getLogger(__name__)


def _transient_startup_error(cause):
    return isinstance(cause, (TimeoutError, WorkerOverloaded)) or (
        isinstance(cause, HttpError) and cause.code == 503
    )


@dataclass(frozen=True)
class PoolingProbe:
    canary: PreparedHeadRead
    adapter: HeadExecutionAdapter

    async def run(self, context):
        await self.adapter.verify_liveness(self.canary, context)


class _Stage(Enum):
    CONSTRUCT = auto()
    PREPARE = auto()
    PROVE = auto()
    CANARY = auto()


class _Outcome(Enum):
    READY = auto()
    INELIGIBLE = auto()
    UNAVAILABLE = auto()
    FAILED = auto()


class _StartupQualification:
    """Retain only progress needed to classify failure and keep a proven probe."""

    def __init__(self, kind, engine, profile, abort_monitor, on_engine_dead):
        self.kind, self.engine, self.profile = kind, engine, profile
        self.abort_monitor, self.on_engine_dead = abort_monitor, on_engine_dead
        self.stage = _Stage.CONSTRUCT
        self.adapter = None
        self.probe = None

    async def run(self):
        adapter_type: type[NativeAdapter]
        if self.kind == "head":
            from .decision_heads import HeadExecutionAdapter

            adapter_type = HeadExecutionAdapter
        else:
            from .decision_reads import ReadExecutionAdapter

            adapter_type = ReadExecutionAdapter
        self.adapter = await adapter_type.create(
            self.engine,
            self.profile,
            abort_monitor=self.abort_monitor,
            on_engine_dead=self.on_engine_dead,
        )
        self.stage = _Stage.PREPARE
        payload, prepared = await self.adapter.prepare_canary()
        if self.kind == "read":
            return _Outcome.READY
        self.stage = _Stage.PROVE
        candidate = PoolingProbe(prepared.reads[0], self.adapter)
        await candidate.run(Context())
        self.probe = candidate
        self.stage = _Stage.CANARY
        for attempt in range(2):
            context = Context()
            try:
                await self.adapter.execute(
                    payload,
                    context,
                    request_id=context.id(),
                    cancelled=lambda: context.is_stopped() or context.is_killed(),
                    trace_headers=context.trace_headers(),
                )
                return _Outcome.READY
            except EngineShutdown:
                raise
            except Exception as cause:
                if attempt or not _transient_startup_error(cause):
                    raise

    def classify(self, cause):
        if self.stage is _Stage.CONSTRUCT and isinstance(cause, NativeRuntimeError):
            logger.info("vLLM native %s decisions ineligible: %s", self.kind, cause)
            return _Outcome.INELIGIBLE
        if _transient_startup_error(cause):
            logger.warning(
                "vLLM native %s startup qualification unavailable: %s",
                self.kind,
                type(cause).__name__,
            )
            return _Outcome.UNAVAILABLE
        logger.exception("vLLM native %s decisions disabled", self.kind)
        return _Outcome.FAILED


@dataclass(frozen=True)
class NativeDecisionSurface(DecisionDispatcher):
    adapter: NativeAdapter | None = None
    pooling_probe: PoolingProbe | None = None
    failed_profile: NativeProfile | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.adapter is not None and self.adapter.kind != self.kind:
            raise ValueError("native readiness does not match the surface kind")
        if self.pooling_probe is not None and self.kind != "head":
            raise ValueError("pooling health requires a head surface")
        if self.failed_profile is not None and (
            self.adapter is not None or self.failed_profile.kind != self.kind
        ):
            raise ValueError("native qualification failure conflicts with readiness")

    @property
    def spec(self) -> NativeProfile | None:
        return self.adapter.spec if self.adapter is not None else None

    @property
    def publication(self) -> NativePublication | None:
        if self.adapter is not None:
            return NativeReady(self.adapter.spec)
        if self.failed_profile is not None:
            return NativeFailed(self.failed_profile)
        return None

    @classmethod
    async def create(
        cls,
        engine,
        profile: NativeProfile,
        *,
        abort_monitor,
        on_engine_dead=None,
    ):
        """Only construction-time incompatibility is ineligible; later contracts fail."""
        kind = profile.kind
        qualification = _StartupQualification(
            kind, engine, profile, abort_monitor, on_engine_dead
        )
        timeout = EngineHealthMonitorConfig.from_env().check_timeout
        # One deadline includes cold queries, preparation, liveness and canary attempts.
        timeout = timeout if timeout > 0 else ENGINE_HEALTH_CHECK_TIMEOUT
        try:
            outcome = await asyncio.wait_for(qualification.run(), timeout=timeout)
        except EngineShutdown:
            raise
        except Exception as cause:
            outcome = qualification.classify(cause)
        return cls(
            kind,
            adapter=qualification.adapter if outcome is _Outcome.READY else None,
            pooling_probe=qualification.probe,
            failed_profile=profile if outcome is _Outcome.FAILED else None,
        )

    def health_payload(self, model_name):
        if self.pooling_probe is None:
            return None
        # HealthCheckPayload adds the canonical marker and operator override.
        return {"native_pooling_health": True, "model": model_name}

    async def _run_decision(self, request, context):
        if self.adapter is None:
            return await super()._run_decision(request, context)
        return await self.adapter.execute(
            request,
            context,
            request_id=context.id(),
            cancelled=lambda: context.is_stopped() or context.is_killed(),
            trace_headers=context.trace_headers(),
        )

    async def _probe(self, context):
        if self.pooling_probe is None:
            return await super()._probe(context)
        await self.pooling_probe.run(context)
