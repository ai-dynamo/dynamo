# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HoldPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACCEPT_WHEN_IDLE: _ClassVar[HoldPolicy]
    HOLD_LAST: _ClassVar[HoldPolicy]

class CircuitState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLOSED: _ClassVar[CircuitState]
    OPEN: _ClassVar[CircuitState]
    HALF_OPEN: _ClassVar[CircuitState]

class OverrideType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SET: _ClassVar[OverrideType]
    AT_LEAST: _ClassVar[OverrideType]
    AT_MOST: _ClassVar[OverrideType]
ACCEPT_WHEN_IDLE: HoldPolicy
HOLD_LAST: HoldPolicy
CLOSED: CircuitState
OPEN: CircuitState
HALF_OPEN: CircuitState
SET: OverrideType
AT_LEAST: OverrideType
AT_MOST: OverrideType

class RegisterRequest(_message.Message):
    __slots__ = ("plugin_id", "plugin_type", "priority", "endpoint", "version", "execution_interval_seconds", "hold_policy", "needs", "protocol_version", "auth_token", "requires_produced_fields", "observation_window_seconds")
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    HOLD_POLICY_FIELD_NUMBER: _ClassVar[int]
    NEEDS_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_PRODUCED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_WINDOW_SECONDS_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    plugin_type: str
    priority: int
    endpoint: str
    version: str
    execution_interval_seconds: float
    hold_policy: HoldPolicy
    needs: _containers.RepeatedScalarFieldContainer[str]
    protocol_version: str
    auth_token: str
    requires_produced_fields: _containers.RepeatedScalarFieldContainer[str]
    observation_window_seconds: float
    def __init__(self, plugin_id: _Optional[str] = ..., plugin_type: _Optional[str] = ..., priority: _Optional[int] = ..., endpoint: _Optional[str] = ..., version: _Optional[str] = ..., execution_interval_seconds: _Optional[float] = ..., hold_policy: _Optional[_Union[HoldPolicy, str]] = ..., needs: _Optional[_Iterable[str]] = ..., protocol_version: _Optional[str] = ..., auth_token: _Optional[str] = ..., requires_produced_fields: _Optional[_Iterable[str]] = ..., observation_window_seconds: _Optional[float] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("accepted", "reject_reason", "negotiated_protocol_version")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REJECT_REASON_FIELD_NUMBER: _ClassVar[int]
    NEGOTIATED_PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    reject_reason: str
    negotiated_protocol_version: str
    def __init__(self, accepted: bool = ..., reject_reason: _Optional[str] = ..., negotiated_protocol_version: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("plugin_id", "auth_token")
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    auth_token: str
    def __init__(self, plugin_id: _Optional[str] = ..., auth_token: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class UnregisterRequest(_message.Message):
    __slots__ = ("plugin_id", "reason", "auth_token")
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    reason: str
    auth_token: str
    def __init__(self, plugin_id: _Optional[str] = ..., reason: _Optional[str] = ..., auth_token: _Optional[str] = ...) -> None: ...

class UnregisterResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class ListPluginsRequest(_message.Message):
    __slots__ = ("stage_filter", "include_disabled")
    STAGE_FILTER_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DISABLED_FIELD_NUMBER: _ClassVar[int]
    stage_filter: str
    include_disabled: bool
    def __init__(self, stage_filter: _Optional[str] = ..., include_disabled: bool = ...) -> None: ...

class ListPluginsResponse(_message.Message):
    __slots__ = ("plugins",)
    PLUGINS_FIELD_NUMBER: _ClassVar[int]
    plugins: _containers.RepeatedCompositeFieldContainer[PluginInfo]
    def __init__(self, plugins: _Optional[_Iterable[_Union[PluginInfo, _Mapping]]] = ...) -> None: ...

class PluginInfo(_message.Message):
    __slots__ = ("plugin_id", "plugin_type", "priority", "version", "protocol_version", "enabled", "is_builtin", "transport", "circuit_state", "evaluations_total", "last_call_at_seconds_ago", "cache_age_seconds")
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    IS_BUILTIN_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    CIRCUIT_STATE_FIELD_NUMBER: _ClassVar[int]
    EVALUATIONS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    LAST_CALL_AT_SECONDS_AGO_FIELD_NUMBER: _ClassVar[int]
    CACHE_AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    plugin_type: str
    priority: int
    version: str
    protocol_version: str
    enabled: bool
    is_builtin: bool
    transport: str
    circuit_state: CircuitState
    evaluations_total: int
    last_call_at_seconds_ago: float
    cache_age_seconds: float
    def __init__(self, plugin_id: _Optional[str] = ..., plugin_type: _Optional[str] = ..., priority: _Optional[int] = ..., version: _Optional[str] = ..., protocol_version: _Optional[str] = ..., enabled: bool = ..., is_builtin: bool = ..., transport: _Optional[str] = ..., circuit_state: _Optional[_Union[CircuitState, str]] = ..., evaluations_total: _Optional[int] = ..., last_call_at_seconds_ago: _Optional[float] = ..., cache_age_seconds: _Optional[float] = ...) -> None: ...

class PipelineContext(_message.Message):
    __slots__ = ("request_id", "decision_id", "observations", "predictions", "proposal", "constrained")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DECISION_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    PROPOSAL_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    decision_id: str
    observations: ObservationData
    predictions: PredictionData
    proposal: ScalingProposal
    constrained: ScalingProposal
    def __init__(self, request_id: _Optional[str] = ..., decision_id: _Optional[str] = ..., observations: _Optional[_Union[ObservationData, _Mapping]] = ..., predictions: _Optional[_Union[PredictionData, _Mapping]] = ..., proposal: _Optional[_Union[ScalingProposal, _Mapping]] = ..., constrained: _Optional[_Union[ScalingProposal, _Mapping]] = ...) -> None: ...

class ObservationData(_message.Message):
    __slots__ = ("traffic", "fpm", "workers")
    TRAFFIC_FIELD_NUMBER: _ClassVar[int]
    FPM_FIELD_NUMBER: _ClassVar[int]
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    traffic: TrafficMetrics
    fpm: FpmData
    workers: WorkerState
    def __init__(self, traffic: _Optional[_Union[TrafficMetrics, _Mapping]] = ..., fpm: _Optional[_Union[FpmData, _Mapping]] = ..., workers: _Optional[_Union[WorkerState, _Mapping]] = ...) -> None: ...

class TrafficMetrics(_message.Message):
    __slots__ = ("duration_s", "num_req", "isl", "osl", "kv_hit_rate", "accept_length")
    DURATION_S_FIELD_NUMBER: _ClassVar[int]
    NUM_REQ_FIELD_NUMBER: _ClassVar[int]
    ISL_FIELD_NUMBER: _ClassVar[int]
    OSL_FIELD_NUMBER: _ClassVar[int]
    KV_HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    ACCEPT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    duration_s: float
    num_req: float
    isl: float
    osl: float
    kv_hit_rate: float
    accept_length: float
    def __init__(self, duration_s: _Optional[float] = ..., num_req: _Optional[float] = ..., isl: _Optional[float] = ..., osl: _Optional[float] = ..., kv_hit_rate: _Optional[float] = ..., accept_length: _Optional[float] = ...) -> None: ...

class FpmData(_message.Message):
    __slots__ = ("prefill_engines", "decode_engines")
    class PrefillEnginesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    class DecodeEnginesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    PREFILL_ENGINES_FIELD_NUMBER: _ClassVar[int]
    DECODE_ENGINES_FIELD_NUMBER: _ClassVar[int]
    prefill_engines: _containers.ScalarMap[str, bytes]
    decode_engines: _containers.ScalarMap[str, bytes]
    def __init__(self, prefill_engines: _Optional[_Mapping[str, bytes]] = ..., decode_engines: _Optional[_Mapping[str, bytes]] = ...) -> None: ...

class WorkerState(_message.Message):
    __slots__ = ("ready_prefill", "ready_decode", "expected_prefill", "expected_decode", "prefill_scaling_in_progress", "decode_scaling_in_progress")
    READY_PREFILL_FIELD_NUMBER: _ClassVar[int]
    READY_DECODE_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_PREFILL_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_DECODE_FIELD_NUMBER: _ClassVar[int]
    PREFILL_SCALING_IN_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    DECODE_SCALING_IN_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    ready_prefill: int
    ready_decode: int
    expected_prefill: int
    expected_decode: int
    prefill_scaling_in_progress: bool
    decode_scaling_in_progress: bool
    def __init__(self, ready_prefill: _Optional[int] = ..., ready_decode: _Optional[int] = ..., expected_prefill: _Optional[int] = ..., expected_decode: _Optional[int] = ..., prefill_scaling_in_progress: bool = ..., decode_scaling_in_progress: bool = ...) -> None: ...

class PredictionData(_message.Message):
    __slots__ = ("predicted_num_req", "predicted_isl", "predicted_osl", "source", "predicted_kv_hit_rate", "predicted_accept_length")
    PREDICTED_NUM_REQ_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_ISL_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_OSL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_KV_HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_ACCEPT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    predicted_num_req: float
    predicted_isl: float
    predicted_osl: float
    source: str
    predicted_kv_hit_rate: float
    predicted_accept_length: float
    def __init__(self, predicted_num_req: _Optional[float] = ..., predicted_isl: _Optional[float] = ..., predicted_osl: _Optional[float] = ..., source: _Optional[str] = ..., predicted_kv_hit_rate: _Optional[float] = ..., predicted_accept_length: _Optional[float] = ...) -> None: ...

class ScalingProposal(_message.Message):
    __slots__ = ("targets", "reason", "source")
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    targets: _containers.RepeatedCompositeFieldContainer[ComponentTarget]
    reason: str
    source: str
    def __init__(self, targets: _Optional[_Iterable[_Union[ComponentTarget, _Mapping]]] = ..., reason: _Optional[str] = ..., source: _Optional[str] = ...) -> None: ...

class ComponentTarget(_message.Message):
    __slots__ = ("sub_component_type", "replicas", "type")
    SUB_COMPONENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    sub_component_type: str
    replicas: int
    type: OverrideType
    def __init__(self, sub_component_type: _Optional[str] = ..., replicas: _Optional[int] = ..., type: _Optional[_Union[OverrideType, str]] = ...) -> None: ...

class OverrideResult(_message.Message):
    __slots__ = ("targets", "reason")
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    targets: _containers.RepeatedCompositeFieldContainer[ComponentTarget]
    reason: str
    def __init__(self, targets: _Optional[_Iterable[_Union[ComponentTarget, _Mapping]]] = ..., reason: _Optional[str] = ...) -> None: ...

class AcceptResult(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RejectResult(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class PredictStageRequest(_message.Message):
    __slots__ = ("context",)
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: PipelineContext
    def __init__(self, context: _Optional[_Union[PipelineContext, _Mapping]] = ...) -> None: ...

class PredictStageResponse(_message.Message):
    __slots__ = ("predictions", "reason", "final")
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    predictions: PredictionData
    reason: str
    final: bool
    def __init__(self, predictions: _Optional[_Union[PredictionData, _Mapping]] = ..., reason: _Optional[str] = ..., final: bool = ...) -> None: ...

class ProposeStageRequest(_message.Message):
    __slots__ = ("context",)
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: PipelineContext
    def __init__(self, context: _Optional[_Union[PipelineContext, _Mapping]] = ...) -> None: ...

class ProposeStageResponse(_message.Message):
    __slots__ = ("accept", "override", "reject", "final")
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    OVERRIDE_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    accept: AcceptResult
    override: OverrideResult
    reject: RejectResult
    final: bool
    def __init__(self, accept: _Optional[_Union[AcceptResult, _Mapping]] = ..., override: _Optional[_Union[OverrideResult, _Mapping]] = ..., reject: _Optional[_Union[RejectResult, _Mapping]] = ..., final: bool = ...) -> None: ...

class ReconcileStageRequest(_message.Message):
    __slots__ = ("context", "proposals")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PROPOSALS_FIELD_NUMBER: _ClassVar[int]
    context: PipelineContext
    proposals: _containers.RepeatedCompositeFieldContainer[ProposeResult]
    def __init__(self, context: _Optional[_Union[PipelineContext, _Mapping]] = ..., proposals: _Optional[_Iterable[_Union[ProposeResult, _Mapping]]] = ...) -> None: ...

class ProposeResult(_message.Message):
    __slots__ = ("plugin_id", "accept", "override", "reject", "priority")
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    OVERRIDE_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    accept: AcceptResult
    override: OverrideResult
    reject: RejectResult
    priority: int
    def __init__(self, plugin_id: _Optional[str] = ..., accept: _Optional[_Union[AcceptResult, _Mapping]] = ..., override: _Optional[_Union[OverrideResult, _Mapping]] = ..., reject: _Optional[_Union[RejectResult, _Mapping]] = ..., priority: _Optional[int] = ...) -> None: ...

class ReconcileStageResponse(_message.Message):
    __slots__ = ("accept", "override", "reject", "final")
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    OVERRIDE_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    accept: AcceptResult
    override: OverrideResult
    reject: RejectResult
    final: bool
    def __init__(self, accept: _Optional[_Union[AcceptResult, _Mapping]] = ..., override: _Optional[_Union[OverrideResult, _Mapping]] = ..., reject: _Optional[_Union[RejectResult, _Mapping]] = ..., final: bool = ...) -> None: ...

class ConstrainStageRequest(_message.Message):
    __slots__ = ("context",)
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: PipelineContext
    def __init__(self, context: _Optional[_Union[PipelineContext, _Mapping]] = ...) -> None: ...

class ConstrainStageResponse(_message.Message):
    __slots__ = ("accept", "override", "reject", "final")
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    OVERRIDE_FIELD_NUMBER: _ClassVar[int]
    REJECT_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    accept: AcceptResult
    override: OverrideResult
    reject: RejectResult
    final: bool
    def __init__(self, accept: _Optional[_Union[AcceptResult, _Mapping]] = ..., override: _Optional[_Union[OverrideResult, _Mapping]] = ..., reject: _Optional[_Union[RejectResult, _Mapping]] = ..., final: bool = ...) -> None: ...

class BootstrapRequest(_message.Message):
    __slots__ = ("bootstrap_data", "hints")
    class HintsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    BOOTSTRAP_DATA_FIELD_NUMBER: _ClassVar[int]
    HINTS_FIELD_NUMBER: _ClassVar[int]
    bootstrap_data: bytes
    hints: _containers.ScalarMap[str, str]
    def __init__(self, bootstrap_data: _Optional[bytes] = ..., hints: _Optional[_Mapping[str, str]] = ...) -> None: ...

class BootstrapResponse(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool = ..., message: _Optional[str] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool = ..., message: _Optional[str] = ...) -> None: ...
