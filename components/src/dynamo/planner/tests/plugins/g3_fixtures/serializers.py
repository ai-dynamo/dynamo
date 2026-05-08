# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON serializers for G3 behavior parity fixtures.

Round-trip serialization for ``TickInput`` / ``PlannerEffects`` /
``ScheduledTick`` and supporting types. The goal is **deterministic**
JSON output—same Python object always serializes to the same bytes,
so fixtures locked at a git tag remain stable across re-runs of the
dump tool.

FPM payloads (``ForwardPassMetrics``) are encoded as msgspec base64 to
preserve the native wire format used in production.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Any, Optional

import msgspec
from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)

# msgspec encoder/decoder for ForwardPassMetrics (preserves wire format).
_FPM_ENCODER = msgspec.json.Encoder()
_FPM_DECODER = msgspec.json.Decoder(ForwardPassMetrics)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_fpm(fpm: ForwardPassMetrics) -> str:
    """Encode a ForwardPassMetrics as base64-wrapped msgspec JSON."""
    raw = _FPM_ENCODER.encode(fpm)
    return base64.b64encode(raw).decode("ascii")


def decode_fpm(blob: str) -> ForwardPassMetrics:
    """Decode a base64 msgspec JSON string back to ForwardPassMetrics."""
    raw = base64.b64decode(blob.encode("ascii"))
    return _FPM_DECODER.decode(raw)


def encode_fpm_observations(obs: Optional[FpmObservations]) -> Optional[dict[str, Any]]:
    if obs is None:
        return None

    def _encode_engine_dict(
        engines: Optional[dict[tuple[str, int], ForwardPassMetrics]]
    ) -> Optional[dict[str, str]]:
        if engines is None:
            return None
        # Tuple key "worker_id:dp_rank" -> base64 msgspec FPM
        # Sort keys for deterministic output.
        return {
            f"{worker_id}:{dp_rank}": encode_fpm(fpm)
            for (worker_id, dp_rank), fpm in sorted(engines.items())
        }

    return {
        "prefill": _encode_engine_dict(obs.prefill),
        "decode": _encode_engine_dict(obs.decode),
    }


def decode_fpm_observations(blob: Optional[dict[str, Any]]) -> Optional[FpmObservations]:
    if blob is None:
        return None

    def _decode_engine_dict(
        engine_blob: Optional[dict[str, str]]
    ) -> Optional[dict[tuple[str, int], ForwardPassMetrics]]:
        if engine_blob is None:
            return None
        out: dict[tuple[str, int], ForwardPassMetrics] = {}
        for key, fpm_blob in engine_blob.items():
            worker_id, dp_rank_str = key.split(":")
            out[(worker_id, int(dp_rank_str))] = decode_fpm(fpm_blob)
        return out

    return FpmObservations(
        prefill=_decode_engine_dict(blob.get("prefill")),
        decode=_decode_engine_dict(blob.get("decode")),
    )


def _asdict(obj: Any) -> Any:
    """Recursive dataclass -> dict, preserving Optional and primitive types."""
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj  # primitive


def encode_tick_input(tick_input: TickInput) -> dict[str, Any]:
    return {
        "now_s": tick_input.now_s,
        "traffic": _asdict(tick_input.traffic),
        "worker_counts": _asdict(tick_input.worker_counts),
        "fpm_observations": encode_fpm_observations(tick_input.fpm_observations),
    }


def decode_tick_input(blob: dict[str, Any]) -> TickInput:
    traffic_blob = blob.get("traffic")
    worker_counts_blob = blob.get("worker_counts")
    return TickInput(
        now_s=blob["now_s"],
        traffic=TrafficObservation(**traffic_blob) if traffic_blob else None,
        worker_counts=WorkerCounts(**worker_counts_blob) if worker_counts_blob else None,
        fpm_observations=decode_fpm_observations(blob.get("fpm_observations")),
    )


def encode_scheduled_tick(tick: ScheduledTick) -> dict[str, Any]:
    return _asdict(tick)


def decode_scheduled_tick(blob: dict[str, Any]) -> ScheduledTick:
    return ScheduledTick(**blob)


def encode_planner_effects(effects: PlannerEffects) -> dict[str, Any]:
    return {
        "scale_to": _asdict(effects.scale_to),
        "next_tick": _asdict(effects.next_tick),
        "diagnostics": _asdict(effects.diagnostics),
    }


def decode_planner_effects(blob: dict[str, Any]) -> PlannerEffects:
    scale_to_blob = blob.get("scale_to")
    next_tick_blob = blob.get("next_tick")
    diag_blob = blob.get("diagnostics")
    return PlannerEffects(
        scale_to=ScalingDecision(**scale_to_blob) if scale_to_blob else None,
        next_tick=ScheduledTick(**next_tick_blob) if next_tick_blob else None,
        diagnostics=TickDiagnostics(**diag_blob) if diag_blob else TickDiagnostics(),
    )


def encode_worker_capabilities(caps: WorkerCapabilities) -> dict[str, Any]:
    return _asdict(caps)


def decode_worker_capabilities(blob: dict[str, Any]) -> WorkerCapabilities:
    prefill = blob.get("prefill")
    decode = blob.get("decode")
    return WorkerCapabilities(
        prefill=EngineCapabilities(**prefill) if prefill else None,
        decode=EngineCapabilities(**decode) if decode else None,
    )
