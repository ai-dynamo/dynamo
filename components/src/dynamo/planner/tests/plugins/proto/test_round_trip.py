# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip equivalence: Pydantic mirror ↔ proto generated.

Verifies the lock-step contract between ``plugins/types.py`` and
``plugins/proto/v1/plugin_pb2.py``:
- Pydantic → proto → Pydantic produces the same Pydantic instance
- proto → Pydantic → proto produces the same proto wire bytes

Picked up by existing ``planner-test`` CI job via pytest markers.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins import types as pyd
from dynamo.planner.plugins._proto_bridge import (
    _PYD_TO_PROTO,
    proto_to_pydantic,
    pydantic_to_proto,
)
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Coverage: every Pydantic class has a proto class registered (and vice versa)
# ---------------------------------------------------------------------------


def test_class_coverage_pydantic_side():
    """Every Pydantic mirror message must have a proto counterpart."""
    pyd_classes = {
        cls
        for name in pyd.__all__
        for cls in [getattr(pyd, name)]
        if isinstance(cls, type) and issubclass(cls, pyd._ProtoMirror)
    }
    registered = set(_PYD_TO_PROTO.keys())
    missing = pyd_classes - registered
    assert not missing, f"Pydantic classes missing proto registration: {sorted(c.__name__ for c in missing)}"


def test_class_coverage_proto_side():
    """Every proto message in plugin_pb2 must have a Pydantic counterpart.

    Verifies no proto message was added without updating the Pydantic mirror.
    """
    proto_msgs = set(pb.DESCRIPTOR.message_types_by_name.keys())
    registered = {p.__name__ for p in _PYD_TO_PROTO.values()}
    missing = proto_msgs - registered
    assert not missing, f"Proto messages missing Pydantic mirror: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Round-trip cases — one per representative scenario
# ---------------------------------------------------------------------------


def _round_trip_pyd(pyd_msg: pyd.BaseModel) -> pyd.BaseModel:
    """Pydantic → proto → Pydantic; assert equality."""
    pb_msg = pydantic_to_proto(pyd_msg)
    pyd_back = proto_to_pydantic(pb_msg)
    assert pyd_back == pyd_msg, (
        f"Pydantic round-trip mismatch:\n"
        f"  original: {pyd_msg!r}\n"
        f"  back:     {pyd_back!r}"
    )
    return pyd_back


def _round_trip_wire(pyd_msg: pyd.BaseModel) -> bytes:
    """Pydantic → proto → wire bytes; serialize twice should match."""
    pb1 = pydantic_to_proto(pyd_msg)
    wire = pb1.SerializeToString()
    pb2 = type(pb1).FromString(wire)
    wire2 = pb2.SerializeToString()
    assert wire == wire2, "wire bytes not deterministic across re-serialize"
    return wire


# ---- PluginRegistry messages ----


def test_register_request_full():
    """RegisterRequest with all 12 fields populated."""
    msg = pyd.RegisterRequest(
        plugin_id="test-plugin",
        plugin_type="propose",
        priority=10,
        endpoint="grpc://plugin.example.com:9090",
        version="1.2.3",
        execution_interval_seconds=30.0,
        hold_policy=pyd.HoldPolicy.HOLD_LAST,
        needs=["observations.traffic", "observations.fpm.prefill"],
        protocol_version="1.0",
        auth_token="secret-token-bytes",
        fpm_encoding="msgspec",
        request_timeout_seconds=5.0,
    )
    _round_trip_pyd(msg)
    _round_trip_wire(msg)


def test_register_request_minimal():
    """RegisterRequest with only required fields (rest defaults)."""
    msg = pyd.RegisterRequest(plugin_id="x", plugin_type="predict")
    _round_trip_pyd(msg)


def test_register_response_accepted():
    msg = pyd.RegisterResponse(accepted=True, negotiated_protocol_version="1.0")
    _round_trip_pyd(msg)


def test_register_response_rejected():
    msg = pyd.RegisterResponse(accepted=False, reject_reason="protocol_version_unsupported")
    _round_trip_pyd(msg)


def test_plugin_info_runtime_state():
    msg = pyd.PluginInfo(
        plugin_id="builtin-throughput-propose",
        plugin_type="propose",
        priority=50,
        version="1.0",
        protocol_version="1.0",
        enabled=True,
        is_builtin=True,
        transport="in_process",
        circuit_state=pyd.CircuitState.CLOSED,
        evaluations_total=1234,
        last_call_at_seconds_ago=2.5,
        cache_age_seconds=0.0,
    )
    _round_trip_pyd(msg)


def test_list_plugins_response_multi():
    msg = pyd.ListPluginsResponse(
        plugins=[
            pyd.PluginInfo(plugin_id="a", plugin_type="propose"),
            pyd.PluginInfo(plugin_id="b", plugin_type="constrain", circuit_state=pyd.CircuitState.OPEN),
        ]
    )
    _round_trip_pyd(msg)


# ---- PipelineContext + observation messages ----


def test_pipeline_context_minimal():
    """request_id only; all other fields None."""
    msg = pyd.PipelineContext(request_id="req-123")
    _round_trip_pyd(msg)


def test_pipeline_context_full():
    """All 6 fields populated."""
    msg = pyd.PipelineContext(
        request_id="req-456",
        decision_id="decision-789",
        observations=pyd.ObservationData(
            traffic=pyd.TrafficMetrics(duration_s=60.0, num_req=1500, isl=3000, osl=150),
            fpm=pyd.FpmData(
                prefill_engines={"engine-0": b"\x01\x02\x03binary-fpm-payload"},
                decode_engines={"engine-1": b"\xff\xfe\xfd"},
            ),
            workers=pyd.WorkerState(
                ready_prefill=4, ready_decode=8, expected_prefill=4, expected_decode=10
            ),
        ),
        predictions=pyd.PredictionData(
            predicted_num_req=1800.0,
            predicted_isl=3000.0,
            predicted_osl=160.0,
            source="builtin-load-predictor",
        ),
        proposal=pyd.ScalingProposal(
            targets=[
                pyd.ComponentTarget(sub_component_type="prefill", replicas=6),
                pyd.ComponentTarget(sub_component_type="decode", replicas=12),
            ],
            reason="scaling up due to predicted_num_req increase",
            source="merged",
        ),
        constrained=pyd.ScalingProposal(
            targets=[
                pyd.ComponentTarget(sub_component_type="prefill", replicas=6),
                pyd.ComponentTarget(sub_component_type="decode", replicas=10),  # capped by AT_MOST
            ],
        ),
    )
    _round_trip_pyd(msg)
    _round_trip_wire(msg)


def test_prediction_data_optional_unset_vs_zero():
    """v11 critical: optional float fields distinguish None from 0.0."""
    # All None
    p1 = pyd.PredictionData(source="builtin")
    pb1 = pydantic_to_proto(p1)
    assert not pb1.HasField("predicted_num_req")
    assert not pb1.HasField("predicted_isl")
    assert not pb1.HasField("predicted_osl")

    # Explicit 0.0 (rare but valid)
    p2 = pyd.PredictionData(predicted_num_req=0.0)
    pb2 = pydantic_to_proto(p2)
    assert pb2.HasField("predicted_num_req"), "predicted_num_req=0.0 must round-trip as set"
    assert pb2.predicted_num_req == 0.0
    assert not pb2.HasField("predicted_isl")  # still unset

    # Round-trip back: None stays None, 0.0 stays 0.0
    p1_back = proto_to_pydantic(pb1)
    assert p1_back.predicted_num_req is None
    p2_back = proto_to_pydantic(pb2)
    assert p2_back.predicted_num_req == 0.0
    assert p2_back.predicted_isl is None


def test_component_target_optional_replicas():
    """Unset replicas = 'no opinion' (v9 semantics)."""
    ct1 = pyd.ComponentTarget(sub_component_type="prefill")  # replicas unset
    pb1 = pydantic_to_proto(ct1)
    assert not pb1.HasField("replicas")
    assert not pb1.HasField("component_name")

    ct1_back = proto_to_pydantic(pb1)
    assert ct1_back.replicas is None
    assert ct1_back.component_name is None


def test_component_target_with_pool_name():
    """Hierarchical pool naming (e.g. 'prefill-pool-A')."""
    ct = pyd.ComponentTarget(
        sub_component_type="prefill",
        component_name="pool-A",
        replicas=8,
        type=pyd.OverrideType.SET,
    )
    _round_trip_pyd(ct)


def test_override_result_multi_target_mixed_types():
    """One OverrideResult can carry SET + AT_LEAST + AT_MOST per component."""
    msg = pyd.OverrideResult(
        targets=[
            pyd.ComponentTarget(sub_component_type="prefill", replicas=10, type=pyd.OverrideType.SET),
            pyd.ComponentTarget(sub_component_type="decode", replicas=4, type=pyd.OverrideType.AT_LEAST),
            pyd.ComponentTarget(sub_component_type="decode", replicas=8, type=pyd.OverrideType.AT_MOST),
        ],
        reason="blended throughput + load decision",
    )
    _round_trip_pyd(msg)


def test_fpm_data_bytes_preserved():
    """map<string, bytes> preserved through proto wire (base64 in JSON intermediate)."""
    msg = pyd.FpmData(
        prefill_engines={
            "p0": bytes(range(256)),  # all byte values
            "p1": b"",  # empty
        },
        decode_engines={"d0": b"\x00\xff\x42"},
    )
    msg_back = _round_trip_pyd(msg)
    assert msg_back.prefill_engines["p0"] == bytes(range(256))
    assert msg_back.prefill_engines["p1"] == b""
    assert msg_back.decode_engines["d0"] == b"\x00\xff\x42"


# ---- Stage request/response with oneof ----


def test_propose_stage_response_accept():
    msg = pyd.ProposeStageResponse(accept=pyd.AcceptResult())
    msg_back = _round_trip_pyd(msg)
    assert msg_back.result_kind == "accept"
    assert msg_back.accept is not None
    assert msg_back.override is None
    assert msg_back.reject is None
    assert msg_back.final is False


def test_propose_stage_response_override_with_final():
    msg = pyd.ProposeStageResponse(
        override=pyd.OverrideResult(
            targets=[pyd.ComponentTarget(sub_component_type="prefill", replicas=20)],
            reason="emergency override",
        ),
        final=True,
    )
    msg_back = _round_trip_pyd(msg)
    assert msg_back.result_kind == "override"
    assert msg_back.override is not None
    assert msg_back.override.targets[0].replicas == 20
    assert msg_back.final is True


def test_propose_stage_response_reject():
    msg = pyd.ProposeStageResponse(reject=pyd.RejectResult(reason="budget exceeded"))
    msg_back = _round_trip_pyd(msg)
    assert msg_back.result_kind == "reject"
    assert msg_back.reject is not None
    assert msg_back.reject.reason == "budget exceeded"


def test_propose_stage_response_oneof_violation():
    """Cannot set multiple oneof payloads."""
    with pytest.raises(Exception, match="oneof"):
        pyd.ProposeStageResponse(
            accept=pyd.AcceptResult(),
            reject=pyd.RejectResult(reason="bad"),
        )


def test_predict_stage_response_partial():
    """PredictionData partial set — only num_req."""
    msg = pyd.PredictStageResponse(
        predictions=pyd.PredictionData(predicted_num_req=1500.0, source="user-llm-predictor"),
        final=False,
    )
    msg_back = _round_trip_pyd(msg)
    assert msg_back.predictions is not None
    assert msg_back.predictions.predicted_num_req == 1500.0
    assert msg_back.predictions.predicted_isl is None
    assert msg_back.predictions.predicted_osl is None


def test_reconcile_stage_request_with_proposals():
    msg = pyd.ReconcileStageRequest(
        context=pyd.PipelineContext(request_id="req-x"),
        proposals=[
            pyd.ProposeResult(
                plugin_id="builtin-throughput-propose",
                priority=50,
                result_kind="override",
                override=pyd.OverrideResult(
                    targets=[pyd.ComponentTarget(sub_component_type="prefill", replicas=6, type=pyd.OverrideType.AT_LEAST)],
                ),
            ),
            pyd.ProposeResult(
                plugin_id="builtin-load-propose",
                priority=10,
                result_kind="override",
                override=pyd.OverrideResult(
                    targets=[pyd.ComponentTarget(sub_component_type="prefill", replicas=8)],
                ),
            ),
            pyd.ProposeResult(
                plugin_id="user-quiet-plugin",
                priority=100,
                result_kind="accept",
                accept=pyd.AcceptResult(),
            ),
        ],
    )
    msg_back = _round_trip_pyd(msg)
    assert len(msg_back.proposals) == 3
    assert msg_back.proposals[1].priority == 10


def test_constrain_stage_response_at_least_at_most():
    """CONSTRAIN typically returns AT_LEAST + AT_MOST (no SET)."""
    msg = pyd.ConstrainStageResponse(
        override=pyd.OverrideResult(
            targets=[
                pyd.ComponentTarget(sub_component_type="prefill", replicas=2, type=pyd.OverrideType.AT_LEAST),
                pyd.ComponentTarget(sub_component_type="prefill", replicas=20, type=pyd.OverrideType.AT_MOST),
            ],
            reason="builtin-budget-constrain: min_endpoint=2 max_gpu_budget=20",
        ),
    )
    _round_trip_pyd(msg)


# ---- PluginLifecycle messages ----


def test_bootstrap_request_with_data_and_hints():
    msg = pyd.BootstrapRequest(
        bootstrap_data=b"\x00\x01\x02benchmark FPM serialized\xff",
        hints={"regression_kind": "prefill", "model_size": "70b"},
    )
    msg_back = _round_trip_pyd(msg)
    assert msg_back.bootstrap_data == b"\x00\x01\x02benchmark FPM serialized\xff"
    assert msg_back.hints["regression_kind"] == "prefill"


def test_reset_request_with_reason():
    msg = pyd.ResetRequest(reason="config_reload")
    _round_trip_pyd(msg)


# ---- Wire-bytes deterministic for representative messages ----


@pytest.mark.parametrize(
    "msg",
    [
        pyd.RegisterRequest(plugin_id="x", plugin_type="propose", priority=10),
        pyd.OverrideResult(targets=[pyd.ComponentTarget(sub_component_type="prefill", replicas=8)]),
        pyd.PipelineContext(request_id="r"),
    ],
    ids=["RegisterRequest", "OverrideResult", "PipelineContext"],
)
def test_wire_deterministic(msg):
    """Same Pydantic input → same wire bytes (no field reordering)."""
    pb1 = pydantic_to_proto(msg)
    pb2 = pydantic_to_proto(msg)
    assert pb1.SerializeToString() == pb2.SerializeToString()
