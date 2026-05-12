# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional converters between proto generated messages and Pydantic mirror.

Used by:
- Round-trip tests in ``tests/plugins/proto/test_round_trip.py``
- ``InProcessTransport`` boundary (proto in / proto out, but plugin
  authors can use Pydantic internally and convert)

Conversion strategy:
- Pydantic ``.model_dump(exclude_none=True)`` → dict → proto via field
  assignment (handles oneof, optional, repeated, map, nested message)
- Proto ``MessageToDict(preserving_proto_field_name=True)`` → dict → Pydantic
  via ``cls.model_validate(...)``

Edge cases handled:
- proto3 ``optional`` fields: Pydantic ``None`` ↔ proto ``HasField`` False
- proto3 ``oneof``: Pydantic ``result_kind`` ↔ proto ``WhichOneof``
- ``map<string, bytes>``: bytes preserved end-to-end
- repeated nested messages: list ↔ repeated field
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from google.protobuf import json_format
from google.protobuf.message import Message
from pydantic import BaseModel

from dynamo.planner.plugins import types as pyd
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb

PydT = TypeVar("PydT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Pydantic → proto
# ---------------------------------------------------------------------------

# Pydantic class → proto class lookup table. Keys are Pydantic class objects
# (NOT strings, to allow IDE refactor).
_PYD_TO_PROTO: dict[Type[BaseModel], Type[Message]] = {
    pyd.RegisterRequest: pb.RegisterRequest,
    pyd.RegisterResponse: pb.RegisterResponse,
    pyd.HeartbeatRequest: pb.HeartbeatRequest,
    pyd.HeartbeatResponse: pb.HeartbeatResponse,
    pyd.UnregisterRequest: pb.UnregisterRequest,
    pyd.UnregisterResponse: pb.UnregisterResponse,
    pyd.ListPluginsRequest: pb.ListPluginsRequest,
    pyd.ListPluginsResponse: pb.ListPluginsResponse,
    pyd.PluginInfo: pb.PluginInfo,
    pyd.TrafficMetrics: pb.TrafficMetrics,
    pyd.FpmData: pb.FpmData,
    pyd.WorkerState: pb.WorkerState,
    pyd.ObservationData: pb.ObservationData,
    pyd.PredictionData: pb.PredictionData,
    pyd.ComponentTarget: pb.ComponentTarget,
    pyd.ScalingProposal: pb.ScalingProposal,
    pyd.PipelineContext: pb.PipelineContext,
    pyd.AcceptResult: pb.AcceptResult,
    pyd.RejectResult: pb.RejectResult,
    pyd.OverrideResult: pb.OverrideResult,
    pyd.ProposeResult: pb.ProposeResult,
    pyd.PredictStageRequest: pb.PredictStageRequest,
    pyd.PredictStageResponse: pb.PredictStageResponse,
    pyd.ProposeStageRequest: pb.ProposeStageRequest,
    pyd.ProposeStageResponse: pb.ProposeStageResponse,
    pyd.ReconcileStageRequest: pb.ReconcileStageRequest,
    pyd.ReconcileStageResponse: pb.ReconcileStageResponse,
    pyd.ConstrainStageRequest: pb.ConstrainStageRequest,
    pyd.ConstrainStageResponse: pb.ConstrainStageResponse,
    pyd.BootstrapRequest: pb.BootstrapRequest,
    pyd.BootstrapResponse: pb.BootstrapResponse,
    pyd.ResetRequest: pb.ResetRequest,
    pyd.ResetResponse: pb.ResetResponse,
}


def proto_class_for(pyd_cls: Type[BaseModel]) -> Type[Message]:
    """Look up the proto class corresponding to a Pydantic mirror class."""
    if pyd_cls not in _PYD_TO_PROTO:
        raise KeyError(f"No proto class registered for Pydantic class {pyd_cls.__name__}")
    return _PYD_TO_PROTO[pyd_cls]


def pydantic_to_proto(pyd_msg: BaseModel, proto_cls: Type[Message] | None = None) -> Message:
    """Convert a Pydantic mirror instance to its proto generated equivalent.

    Uses JSON intermediate (``Pydantic.model_dump_json()`` →
    ``json_format.Parse``) which correctly handles:
    - optional fields (None values are excluded from JSON, so HasField stays False)
    - oneof fields (Pydantic's ``result_kind`` + payload mapping reconstructs to oneof)
    - map fields, repeated nested messages, bytes (base64 in JSON)
    """
    if proto_cls is None:
        proto_cls = proto_class_for(type(pyd_msg))
    data = _pyd_to_dict(pyd_msg)
    pb_msg = proto_cls()
    return json_format.ParseDict(data, pb_msg, ignore_unknown_fields=False)


def _pyd_to_dict(pyd_msg: BaseModel) -> dict[str, Any]:
    """Pydantic → dict suitable for ``json_format.ParseDict``.

    Uses Pydantic ``mode="python"`` (preserves bytes; safer than "json"
    which UTF-8-decodes bytes), then walks the dict to:
    - Convert IntEnum values to int (json_format expects int for proto enum)
    - Base64-encode bytes (json_format wire format for proto bytes)
    - Strip ``result_kind`` oneof tag fields (Pydantic-only convenience)
    """
    data = pyd_msg.model_dump(mode="python", exclude_none=True)
    return _normalize(data)


def _normalize(d: Any) -> Any:
    """Recursively convert IntEnum → int, bytes → base64 string, strip oneof tags."""
    import base64
    from enum import IntEnum

    if isinstance(d, dict):
        out: dict[str, Any] = {}
        kind: str | None = d.get("result_kind") if "result_kind" in d else None
        for k, v in d.items():
            if k == "result_kind":
                continue
            # If a oneof payload key but doesn't match kind, skip
            if kind not in (None, "") and k in ("accept", "override", "reject") and k != kind:
                continue
            out[k] = _normalize(v)
        return out
    if isinstance(d, list):
        return [_normalize(x) for x in d]
    if isinstance(d, IntEnum):
        return int(d)
    if isinstance(d, bytes):
        return base64.b64encode(d).decode("ascii")
    return d


# ---------------------------------------------------------------------------
# proto → Pydantic
# ---------------------------------------------------------------------------

_PROTO_TO_PYD: dict[Type[Message], Type[BaseModel]] = {v: k for k, v in _PYD_TO_PROTO.items()}


def pydantic_class_for(proto_cls: Type[Message]) -> Type[BaseModel]:
    if proto_cls not in _PROTO_TO_PYD:
        raise KeyError(f"No Pydantic class registered for proto class {proto_cls.__name__}")
    return _PROTO_TO_PYD[proto_cls]


def proto_to_pydantic(pb_msg: Message, pyd_cls: Type[PydT] | None = None) -> PydT:
    """Convert a proto message to its Pydantic mirror equivalent.

    Uses ``MessageToDict(preserving_proto_field_name=True, including_default_value_fields=False)``
    which gives field names matching Pydantic mirror exactly, and correctly
    omits unset optional fields (HasField=False) so Pydantic sees None.
    """
    target_cls: Type[BaseModel] = pyd_cls if pyd_cls is not None else pydantic_class_for(type(pb_msg))
    data = json_format.MessageToDict(
        pb_msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=False,
        use_integers_for_enums=True,  # Pydantic IntEnum expects int values
    )

    # Recursively inject result_kind for any nested message with oneof
    data = _inject_oneof_kinds(data, pb_msg)

    # Decode base64-encoded bytes fields based on Pydantic schema annotation
    data = _decode_bytes_by_pyd_schema(data, target_cls)

    return target_cls.model_validate(data)  # type: ignore[return-value]


def _decode_bytes_by_pyd_schema(d: Any, pyd_cls: Type[BaseModel]) -> Any:
    """Walk Pydantic schema; base64-decode any bytes / dict[str,bytes] field.

    Inspects ``model_fields`` annotations to detect bytes-typed fields.
    Recurses into nested Pydantic message types.
    """
    import base64
    import typing

    if not isinstance(d, dict):
        return d

    out: dict[str, Any] = {}
    fields = pyd_cls.model_fields
    for k, v in d.items():
        if k not in fields:
            out[k] = v
            continue
        ann = fields[k].annotation
        # Strip Optional[X] -> X
        origin = typing.get_origin(ann)
        if origin is typing.Union or (origin is not None and str(origin) == "types.UnionType"):
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(args) == 1:
                ann = args[0]
                origin = typing.get_origin(ann)

        # Singular bytes
        if ann is bytes and isinstance(v, str):
            out[k] = base64.b64decode(v)
        # dict[str, bytes]
        elif origin is dict:
            args = typing.get_args(ann)
            if len(args) == 2 and args[1] is bytes and isinstance(v, dict):
                out[k] = {kk: (base64.b64decode(vv) if isinstance(vv, str) else vv) for kk, vv in v.items()}
            else:
                out[k] = v
        # Singular nested Pydantic message
        elif isinstance(ann, type) and issubclass(ann, BaseModel) and isinstance(v, dict):
            out[k] = _decode_bytes_by_pyd_schema(v, ann)
        # list[NestedPydantic]
        elif origin is list:
            args = typing.get_args(ann)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel) and isinstance(v, list):
                out[k] = [_decode_bytes_by_pyd_schema(x, args[0]) if isinstance(x, dict) else x for x in v]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _inject_oneof_kinds(d: Any, pb_msg: Message) -> Any:
    """Recursively inject ``result_kind`` for proto messages with ``oneof result``.

    Walks the proto message structure via ``ListFields`` (returns set fields
    only) which avoids descriptor attribute access patterns that trigger
    C++ binding errors in some protobuf versions.
    """
    if not isinstance(d, dict):
        return d

    # Check if THIS message has a oneof named "result"
    if hasattr(pb_msg, "WhichOneof"):
        try:
            kind = pb_msg.WhichOneof("result")
            d["result_kind"] = kind if kind is not None else ""
        except (ValueError, KeyError):
            # No oneof named "result" on this message
            pass

    # Recurse into nested messages via ListFields (only set fields)
    for field, value in pb_msg.ListFields():
        name = field.name
        if name not in d:
            continue
        # Detect map<K, V> — Python repr is dict, not list
        if isinstance(value, dict) and not isinstance(d[name], list):
            continue
        # Singular nested Message — Message is not iterable
        if isinstance(value, Message):
            d[name] = _inject_oneof_kinds(d[name], value)
            continue
        # Repeated message field — RepeatedCompositeContainer (iterable but
        # no __iter__ attr; use iter() to detect)
        try:
            children = list(iter(value))
        except TypeError:
            continue
        if children and isinstance(children[0], Message) and isinstance(d[name], list):
            for i, child_pb in enumerate(children):
                if i < len(d[name]):
                    d[name][i] = _inject_oneof_kinds(d[name][i], child_pb)
    return d


__all__ = [
    "proto_class_for",
    "pydantic_class_for",
    "pydantic_to_proto",
    "proto_to_pydantic",
]
