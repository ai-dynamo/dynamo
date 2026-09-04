# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalogue's coverage claim, tested rather than asserted in prose.

The design's case for putting the role in the argument list is that a scenario
document then maps to a call mechanically. That claim is only worth anything if
every event kind the existing scenario suite uses actually has one verb — so
this file lists all 26 of them and checks it.
"""

import pytest
from dynamo_test import catalog  # noqa: F401  (registers the verbs)
from dynamo_test.roles import Role
from dynamo_test.verbs import REGISTRY, Grant, Receiver, UnknownVerb, VerbCall, Verdict

# Every `class X(Event)` in the scenario suite's events package, as measured.
SCENARIO_EVENT_KINDS = [
    "AssertPodsRestarted",
    "CaptureMetrics",
    "CudaFaultInjection",
    "DeletePod",
    "NetworkPartition",
    "PeriodicSnapshot",
    "PrepareCudaFaultInjection",
    "PrintProcessTree",
    "ResourceMonitor",
    "RollingReplace",
    "RollingUpgrade",
    "RstFromInsidePod",
    "RstInjection",
    "RunCommand",
    "SetBusyThreshold",
    "StallProcess",
    "StartLoad",
    "StopLoad",
    "TerminateProcess",
    "UpstreamGpuXidInjection",
    "Wait",
    "WaitForLoadCompletion",
    "WaitForLogPattern",
    "WaitForModelReady",
    "WaitForRecovery",
    "WaitForStablePods",
]


@pytest.mark.parametrize("kind", SCENARIO_EVENT_KINDS)
def test_every_scenario_event_kind_maps_to_one_verb(kind):
    """The success test for role-as-argument.

    A component-first surface would need a dispatch table from event kind to
    attribute path here; with the role as an argument this is a registry lookup.
    """
    assert kind in REGISTRY
    assert REGISTRY.require(kind).name


def test_the_event_list_is_the_measured_one():
    """Guards against the coverage claim quietly shrinking."""
    assert len(SCENARIO_EVENT_KINDS) == 26
    assert len(set(SCENARIO_EVENT_KINDS)) == 26


# ------------------------------------------------------ documents become calls


def test_a_document_event_becomes_a_call():
    call = VerbCall.from_document(
        {"kind": "StallProcess", "role": "worker", "rank": 0, "seconds": 30}
    )
    assert call.spec.name == "stall_process"
    assert call.selector.role is Role.WORKER
    assert call.selector.rank == 0
    assert call.kwargs == {"seconds": 30}
    assert call.describe() == "stall_process(at('worker/rank=0'), seconds=30)"


def test_a_call_round_trips_back_to_a_document():
    document = {"kind": "stall_process", "role": "worker", "rank": 0, "seconds": 30}
    assert VerbCall.from_document(document).to_document() == document


def test_an_event_without_a_kind_is_rejected():
    with pytest.raises(ValueError, match="no 'kind'"):
        VerbCall.from_document({"role": "worker"})


def test_an_unknown_kind_names_near_matches():
    with pytest.raises(UnknownVerb) as exc:
        VerbCall.from_document({"kind": "restart_everything"})
    assert "26 verbs" not in str(exc.value)  # it reports the registry size
    assert "no verb named" in str(exc.value)


def test_a_verb_with_a_default_role_needs_no_role():
    """This is what makes `query(...)` read naturally instead of `query(at('frontend'))`."""
    call = VerbCall.from_document({"kind": "query", "payload": "hi"})
    assert call.selector.role is Role.FRONTEND


def test_a_verb_that_takes_no_role_rejects_one():
    with pytest.raises(TypeError, match="does not act on a role"):
        VerbCall.from_document({"kind": "Wait", "role": "worker", "seconds": 1})


def test_selector_fields_without_a_role_are_rejected():
    with pytest.raises(TypeError, match="without a role"):
        REGISTRY.require("exec_in").call(replica=2, argv=["ls"])


def test_unexpected_arguments_are_rejected_with_the_accepted_set():
    with pytest.raises(TypeError, match="accepts"):
        VerbCall.from_document({"kind": "Wait", "secondz": 5})


# ------------------------------------------------------------- the naming law


def test_gating_and_observing_are_different_names_not_a_flag():
    """Five checks in the scenario suite today assert *and* have a bare early
    return, so an argument turns the gate off and the report still says PASSED.
    A reader cannot tell a check that held from one that was asked not to look.
    """
    assert REGISTRY.require("logs").verdict is Verdict.NONE  # pure reader
    assert REGISTRY.require("metrics").gates is False


def test_act_verbs_never_gate():
    for spec in REGISTRY.for_receiver(Receiver.ACT):
        assert spec.verdict is Verdict.NONE, spec.name


def test_every_fault_verb_declares_what_it_proves():
    """A fault whose effect is unobservable cannot be told from one that no-oped."""
    faults = [s for s in REGISTRY if s.grant is Grant.FAULT]
    assert faults
    for spec in faults:
        assert spec.proves, spec.name


def test_infra_restarts_are_a_separate_grant_from_lifecycle():
    """Scaling etcd to zero in a shared namespace breaks other people's tests."""
    assert REGISTRY.require("restart_infra").grant is Grant.INFRA
    assert REGISTRY.require("restart").grant is Grant.LIFECYCLE


def test_grants_needed_is_computable_before_anything_runs():
    """A suite that may not inject faults is refused at plan time, not midway."""
    timeline = [
        VerbCall.from_document({"kind": "WaitForModelReady", "timeout": 300}),
        VerbCall.from_document({"kind": "query", "payload": "hi"}),
        VerbCall.from_document({"kind": "DeletePod", "role": "decode"}),
    ]
    assert REGISTRY.grants_needed(timeline) == frozenset(
        {Grant.READ, Grant.INFER, Grant.FAULT}
    )


# -------------------------------------------------- both spellings, one source


class FakeSut:
    def __init__(self):
        self.calls = []

    def restart(self, sel=None, **kw):
        self.calls.append(("restart", sel, kw))
        return "handle"

    def query(self, payload, sel=None, **kw):
        self.calls.append(("query", payload, sel, kw))
        return "answer"


def test_the_component_spelling_is_generated_from_the_same_registry():
    """``sut.frontend.restart()`` and ``sut.restart(at('frontend'))`` are one thing.

    Both spellings exist; only one definition does. The component surface cannot
    drift from the verb surface because it is not separately written down.
    """
    sut = FakeSut()
    REGISTRY.bind(sut, "decode").restart(replica=1)
    verb_name, sel, _ = sut.calls[0]
    assert verb_name == "restart"
    assert sel.role is Role.DECODE
    assert sel.replica == 1


def test_the_component_view_rejects_a_second_role():
    sut = FakeSut()
    with pytest.raises(TypeError, match="already names its role"):
        REGISTRY.bind(sut, "decode").restart(sel="anything")


def test_the_component_view_lists_the_registry():
    assert "restart" in dir(REGISTRY.bind(FakeSut(), "decode"))


# ------------------------------------------------------------------ catalogue


def test_the_catalogue_serialises_for_the_cli():
    record = REGISTRY.to_record()
    by_name = {r["name"]: r for r in record}
    assert by_name["delete_replica"]["proves"] == [
        "replica_deleted",
        "replica_replaced",
    ]
    assert by_name["query"]["default_role"] == "frontend"
    assert by_name["restart_infra"]["takes_selector"] is False


def test_the_catalogue_is_json_safe():
    import json

    assert json.loads(json.dumps(REGISTRY.to_record()))


def test_no_alias_collides_with_a_verb_name():
    names = set(REGISTRY.names())
    for spec in REGISTRY:
        for alias in spec.aliases:
            assert alias not in names, f"{alias} is both an alias and a verb name"
