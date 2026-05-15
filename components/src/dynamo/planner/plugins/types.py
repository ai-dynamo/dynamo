# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 mirror of the v1 plugin proto messages.

These classes are 1:1 with the proto messages in
``dynamo.planner.plugins.proto.v1.plugin_pb2``; field names, types, and
optional-ness must match exactly.

**Why a Pydantic mirror?**
1. **In-process plugin path** (`InProcessTransport`) directly invokes
   plugin Python objects; Pydantic instances are far cleaner than proto
   builder pattern + wrapped optionals.
2. **Test construction**: ``OverrideResult(targets=[ComponentTarget(...)])``
   reads naturally vs proto repeated-field setters.
3. **JSON serialization** for audit logs / debug: Pydantic ``model_dump()``
   gives JSON directly.

**Lock-step contract** (the round-trip test enforces): any change to
the proto MUST come with a matching change here, verified by
``tests/plugins/proto/test_round_trip.py``.

Plugin implementations are encouraged to use these classes internally;
the ``call(method, request)`` transport boundary still uses proto
generated messages (see ``_proto_bridge.py`` for converters).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# Enums (mirror proto enum integer values exactly)
# ----------------------------------------------------------------------------


class HoldPolicy(IntEnum):
    """Mirrors plugin_pb2.HoldPolicy."""

    ACCEPT_WHEN_IDLE = 0
    HOLD_LAST = 1


class CircuitState(IntEnum):
    """Mirrors plugin_pb2.CircuitState."""

    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class OverrideType(IntEnum):
    """Mirrors plugin_pb2.OverrideType."""

    SET = 0
    AT_LEAST = 1
    AT_MOST = 2


# ----------------------------------------------------------------------------
# Shared base config — strict; reject extras
# ----------------------------------------------------------------------------


class _ProtoMirror(BaseModel):
    """Shared base: strict, immutable hash for testing equality."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


# ----------------------------------------------------------------------------
# PluginRegistry messages
# ----------------------------------------------------------------------------


class RegisterRequest(_ProtoMirror):
    plugin_id: str
    plugin_type: Literal["predict", "propose", "reconcile", "constrain"]
    priority: int = 0
    endpoint: str = ""
    version: str = ""
    execution_interval_seconds: float = 0.0
    hold_policy: HoldPolicy = HoldPolicy.ACCEPT_WHEN_IDLE
    needs: list[str] = Field(default_factory=list)
    protocol_version: str = ""
    auth_token: str = ""
    fpm_encoding: str = ""
    request_timeout_seconds: float = 0.0


class RegisterResponse(_ProtoMirror):
    accepted: bool = False
    reject_reason: str = ""
    negotiated_protocol_version: str = ""


class HeartbeatRequest(_ProtoMirror):
    plugin_id: str = ""


class HeartbeatResponse(_ProtoMirror):
    ok: bool = False


class UnregisterRequest(_ProtoMirror):
    plugin_id: str = ""
    reason: str = ""


class UnregisterResponse(_ProtoMirror):
    ok: bool = False


class ListPluginsRequest(_ProtoMirror):
    stage_filter: str = ""
    include_disabled: bool = False


class PluginInfo(_ProtoMirror):
    plugin_id: str = ""
    plugin_type: str = ""
    priority: int = 0
    version: str = ""
    protocol_version: str = ""
    enabled: bool = False
    is_builtin: bool = False
    transport: Literal["", "in_process", "grpc"] = ""
    circuit_state: CircuitState = CircuitState.CLOSED
    evaluations_total: int = 0
    last_call_at_seconds_ago: float = 0.0
    cache_age_seconds: float = 0.0


class ListPluginsResponse(_ProtoMirror):
    plugins: list[PluginInfo] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Pipeline context + observation types
# ----------------------------------------------------------------------------


class TrafficMetrics(_ProtoMirror):
    duration_s: float = 0.0
    num_req: float = 0.0
    isl: float = 0.0
    osl: float = 0.0


class FpmData(_ProtoMirror):
    """bytes interpretation per RegisterRequest.fpm_encoding (msgspec | proto | json)."""

    prefill_engines: dict[str, bytes] = Field(default_factory=dict)
    decode_engines: dict[str, bytes] = Field(default_factory=dict)


class WorkerState(_ProtoMirror):
    ready_prefill: Optional[int] = None
    ready_decode: Optional[int] = None
    expected_prefill: Optional[int] = None
    expected_decode: Optional[int] = None


class ObservationData(_ProtoMirror):
    traffic: Optional[TrafficMetrics] = None
    fpm: Optional[FpmData] = None
    workers: Optional[WorkerState] = None


class PredictionData(_ProtoMirror):
    """All three prediction fields are ``Optional[float]``.

    ``chain_augment`` partial-merge uses field set/unset to distinguish
    "I assert this value (even 0.0)" vs "no opinion, preserve previous".
    Proto3 ``optional`` modifier on the corresponding proto fields preserves
    this; here in Pydantic, ``Optional[float] = None`` carries the same
    semantics — ``None`` means unset, any concrete float (including 0.0)
    means asserted.
    """

    predicted_num_req: Optional[float] = None
    predicted_isl: Optional[float] = None
    predicted_osl: Optional[float] = None
    source: str = ""


class ComponentTarget(_ProtoMirror):
    """One scaling target per component instance.

    ``replicas=None`` means "no opinion on this component" (v9 semantics).
    ``component_name=None`` means "the default pool of this sub_component_type".
    ``type`` is meaningful inside OverrideResult; ignored in ScalingProposal.
    """

    sub_component_type: str
    component_name: Optional[str] = None
    replicas: Optional[int] = None
    type: OverrideType = OverrideType.SET


class ScalingProposal(_ProtoMirror):
    """Output of RECONCILE/CONSTRAIN. ComponentTarget.type is unused here."""

    targets: list[ComponentTarget] = Field(default_factory=list)
    reason: str = ""
    source: str = ""


class PipelineContext(_ProtoMirror):
    """Full context flowing through PREDICT -> PROPOSE -> RECONCILE -> CONSTRAIN."""

    request_id: str = ""
    decision_id: str = ""
    observations: Optional[ObservationData] = None
    predictions: Optional[PredictionData] = None
    proposal: Optional[ScalingProposal] = None
    constrained: Optional[ScalingProposal] = None


# ----------------------------------------------------------------------------
# OverrideResult / Accept / Reject + stage-specific request/response
# ----------------------------------------------------------------------------


class AcceptResult(_ProtoMirror):
    """Empty marker; proto AcceptResult is empty too."""


class RejectResult(_ProtoMirror):
    reason: str = ""


class OverrideResult(_ProtoMirror):
    targets: list[ComponentTarget] = Field(default_factory=list)
    reason: str = ""


# Stage request/response — `result` is a discriminated union; Pydantic
# expresses it as a literal `Optional` of three kinds, only one set at a time.
# We use a `result_kind` tag + payload field to mirror proto3 oneof semantics
# explicitly. Round-trip test verifies equivalence.


class _StageOneofResponse(_ProtoMirror):
    """Common base for stage responses with proto3 ``oneof result``.

    ``result_kind`` + payload mirror proto3's ``WhichOneof('result')``.
    Exactly one of ``accept`` / ``override`` / ``reject`` should be set
    (matching ``result_kind``); validators enforce it.
    """

    result_kind: Literal["", "accept", "override", "reject"] = ""
    accept: Optional[AcceptResult] = None
    override: Optional[OverrideResult] = None
    reject: Optional[RejectResult] = None
    final: bool = False

    def model_post_init(self, __context: Any) -> None:
        # Auto-derive result_kind from set fields if not explicit
        set_kinds = [k for k in ("accept", "override", "reject") if getattr(self, k) is not None]
        if self.result_kind == "" and len(set_kinds) == 1:
            object.__setattr__(self, "result_kind", set_kinds[0])
        elif len(set_kinds) > 1:
            raise ValueError(
                f"oneof violation: at most one of accept/override/reject may be set; "
                f"got {set_kinds}"
            )
        elif self.result_kind != "" and self.result_kind not in set_kinds:
            raise ValueError(
                f"result_kind={self.result_kind!r} but corresponding payload not set"
            )


class PredictStageRequest(_ProtoMirror):
    context: Optional[PipelineContext] = None


class PredictStageResponse(_ProtoMirror):
    """PREDICT plugin output — chain-augment partial-merge.

    ``predictions=None`` ≈ AcceptResult (no opinion);
    ``final=True`` stops the chain (v11: should only be used by lowest-priority
    plugin to avoid breaking chain before higher-priority plugins run).
    """

    predictions: Optional[PredictionData] = None
    reason: str = ""
    final: bool = False


class ProposeStageRequest(_ProtoMirror):
    context: Optional[PipelineContext] = None


class ProposeStageResponse(_StageOneofResponse):
    """PROPOSE plugin output. ``final=True`` completely overrides other plugins'
    outputs in this stage (multiple final → priority number smallest wins).
    REJECT > final priority (v11 G-2)."""


class ProposeResult(_ProtoMirror):
    """Per-plugin propose output passed to RECONCILE plugins.

    Mirrors proto ProposeResult; ``priority`` is the originating plugin's
    priority (RECONCILE plugins reweight / filter based on this).
    """

    plugin_id: str = ""
    result_kind: Literal["", "accept", "override", "reject"] = ""
    accept: Optional[AcceptResult] = None
    override: Optional[OverrideResult] = None
    reject: Optional[RejectResult] = None
    priority: int = 0


class ReconcileStageRequest(_ProtoMirror):
    context: Optional[PipelineContext] = None
    proposals: list[ProposeResult] = Field(default_factory=list)


class ReconcileStageResponse(_StageOneofResponse):
    """RECONCILE plugin output. Same final semantics as ProposeStageResponse."""


class ConstrainStageRequest(_ProtoMirror):
    context: Optional[PipelineContext] = None


class ConstrainStageResponse(_StageOneofResponse):
    """CONSTRAIN plugin output. SET targets are silently dropped at runtime
    (v11: register-time static rejection is infeasible). ``final`` is silently
    ignored in CONSTRAIN."""


# ----------------------------------------------------------------------------
# PluginLifecycle messages (v10 YAGNI: only Bootstrap + Reset)
# ----------------------------------------------------------------------------


class BootstrapRequest(_ProtoMirror):
    bootstrap_data: bytes = b""
    hints: dict[str, str] = Field(default_factory=dict)


class BootstrapResponse(_ProtoMirror):
    ok: bool = False
    message: str = ""


class ResetRequest(_ProtoMirror):
    reason: str = ""


class ResetResponse(_ProtoMirror):
    ok: bool = False
    message: str = ""


__all__ = [
    # Enums
    "HoldPolicy",
    "CircuitState",
    "OverrideType",
    # PluginRegistry
    "RegisterRequest",
    "RegisterResponse",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "UnregisterRequest",
    "UnregisterResponse",
    "ListPluginsRequest",
    "PluginInfo",
    "ListPluginsResponse",
    # Pipeline context + observation
    "TrafficMetrics",
    "FpmData",
    "WorkerState",
    "ObservationData",
    "PredictionData",
    "ComponentTarget",
    "ScalingProposal",
    "PipelineContext",
    # Stage payloads
    "AcceptResult",
    "RejectResult",
    "OverrideResult",
    "ProposeResult",
    # Stage request/response
    "PredictStageRequest",
    "PredictStageResponse",
    "ProposeStageRequest",
    "ProposeStageResponse",
    "ReconcileStageRequest",
    "ReconcileStageResponse",
    "ConstrainStageRequest",
    "ConstrainStageResponse",
    # PluginLifecycle
    "BootstrapRequest",
    "BootstrapResponse",
    "ResetRequest",
    "ResetResponse",
]
