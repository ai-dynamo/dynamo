# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole thing, end to end, with no cluster.

A real subprocess, real HTTP, a real restart, real evidence collection, and a
real judgement over the sealed bundle. Nothing here is mocked — if the phase
machine, the provider, the verb façade or the seal is wrong, these fail.

This is the demonstration that the abstraction is usable, not just coherent.
"""

import socket
import sys
import time
from pathlib import Path

import pytest
from dynamo_test.evidence import Outcome, Recorder, Verdict
from dynamo_test.providers import LocalProvider, LocalRole
from dynamo_test.roles import PortName, Role, RoleBinding, RoleTable, at
from dynamo_test.sut import NotGranted, Phase, PhaseError, Sut
from dynamo_test.verbs import Grant

STUB = Path(__file__).parent / "stub_frontend.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def deployment(tmp_path):
    """A one-role deployment: a frontend, as a local subprocess."""
    port = free_port()
    provider = LocalProvider(
        {
            Role.FRONTEND: LocalRole(
                role=Role.FRONTEND,
                argv=[
                    sys.executable,
                    str(STUB),
                    "--port",
                    str(port),
                    "--model",
                    "Qwen/Qwen3-0.6B",
                    "--max-model-len",
                    "2048",
                ],
                port=port,
            )
        },
        log_dir=tmp_path / "logs",
    )
    roles = RoleTable(
        {
            Role.FRONTEND: RoleBinding(
                role=Role.FRONTEND,
                service="Frontend",
                log_key="frontend",
                ports={PortName.SERVICE: port},
            )
        }
    )
    yield provider, roles, port
    provider.shutdown()


def make_sut(
    provider,
    roles,
    tmp_path,
    grants=(Grant.READ, Grant.INFER, Grant.LIFECYCLE),
):
    return Sut(
        provider,
        roles,
        Recorder(tmp_path / "bundles", "run-e2e"),
        grants=grants,
        collectors=[provider.collect_into],
    )


# --------------------------------------------------------------- the happy path


def test_start_query_collect_judge(deployment, tmp_path):
    """The full loop, in the order the design says it must happen."""
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path)

    with sut:
        sut.start(at("frontend"))
        sut.wait_serving(timeout=30)

        # Verb-first: the role is a default, so this is just query().
        answer = sut.query("hello")
        assert answer.value.is_known
        assert answer.value.require()["choices"][0]["text"] == "echo: hello"

        models = sut.models()
        assert models.value.require()["data"][0]["id"] == "Qwen/Qwen3-0.6B"

    # Now in CHECK: judge the sealed bundle.
    evidence = sut.evidence()
    assert evidence.require_valid_run().verdict is Verdict.PASSED

    def expect_frontend_served_a_request(ev):
        log = ev.text("roles/frontend/current.log")
        if not log.is_known:
            return Outcome(
                "expect_frontend_served_a_request", Verdict.INVALID, log.detail
            )
        served = "request" in log.require()
        return Outcome(
            "expect_frontend_served_a_request",
            Verdict.PASSED if served else Verdict.FAILED,
            f"{log.require().count('request ')} request line(s) in the log",
            evidence=("roles/frontend/current.log",),
        )

    verdict, outcomes = evidence.judge([expect_frontend_served_a_request])
    assert verdict is Verdict.PASSED, [o.detail for o in outcomes]


def test_the_timeline_records_every_verb(deployment, tmp_path):
    """A handle per verb, in order, with what it resolved to."""
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path)
    with sut:
        sut.start(at("frontend"))
        sut.wait_serving(timeout=30)
        sut.query("a")

    rows = sut.evidence().rows("timeline.jsonl").require()
    assert [r["verb"] for r in rows] == ["start", "wait_serving", "query"]
    assert all(r["elapsed_s"] >= 0 for r in rows)
    assert rows[0]["resolved"]["pid"] > 0


# ------------------------------------------------------------ the phase machine


def test_acting_after_teardown_is_refused_and_says_what_to_do(deployment, tmp_path):
    """The bug this machine exists to make unwritable.

    A check that reads the system after teardown is measuring the teardown.
    """
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path)
    with sut:
        sut.start(at("frontend"))
        sut.wait_serving(timeout=30)

    assert sut.phase is Phase.CHECK
    with pytest.raises(PhaseError) as exc:
        sut.query("too late")
    assert "ACT verb" in str(exc.value)
    assert "may not exist any more" in str(exc.value)


def test_evidence_is_not_readable_before_the_seal(deployment, tmp_path):
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path)
    with sut:
        with pytest.raises(PhaseError, match="CHECK-phase call"):
            sut.evidence()


def test_collection_happens_even_when_the_body_fails(deployment, tmp_path):
    """A failed run is exactly when its evidence matters most."""
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path)

    with pytest.raises(ZeroDivisionError):
        with sut:
            sut.start(at("frontend"))
            sut.wait_serving(timeout=30)
            sut.query("before the explosion")
            1 / 0

    evidence = sut.evidence()
    log = evidence.text("roles/frontend/current.log")
    assert log.is_known, "the log must be collected despite the failure"
    assert "request " in log.require()
    assert len(evidence.rows("timeline.jsonl").require()) == 3


# ------------------------------------------------------------------- grants


def test_a_verb_without_its_grant_is_refused_before_anything_happens(
    deployment, tmp_path
):
    """Refused at the call, before the provider is touched.

    Halfway through a fault is the worst place to discover a suite was not
    allowed to inject it.
    """
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path, grants=(Grant.READ, Grant.INFER))
    with sut:
        with pytest.raises(NotGranted):
            sut.start(at("frontend"))
    # Nothing was started, so nothing has a log — the refusal came first.
    assert provider.restart_count(at("frontend")).is_unknown


def test_lifecycle_needs_its_grant(deployment, tmp_path):
    provider, roles, port = deployment
    sut = make_sut(provider, roles, tmp_path, grants=(Grant.READ,))
    with sut:
        with pytest.raises(NotGranted) as exc:
            sut.restart(at("frontend"))
    assert "'lifecycle' grant" in str(exc.value)
    assert "read" in str(exc.value)


# ------------------------------------------------------------------ restart


def test_restart_changes_a_flag_and_the_service_reflects_it(deployment, tmp_path):
    """Settings go through ArgV, so the flag is replaced rather than appended.

    Appending gives `--max-model-len 2048 --max-model-len 4096`; the process
    takes one and the test believes the other.
    """
    provider, roles, port = deployment
    sut = make_sut(
        provider, roles, tmp_path, grants=(Grant.READ, Grant.INFER, Grant.LIFECYCLE)
    )
    with sut:
        first = sut.start(at("frontend"))
        sut.wait_serving(timeout=30)
        assert sut.query("x").value.require()["max_model_len"] == 2048

        sut.restart(at("frontend"), max_model_len=4096)
        sut.wait_serving(timeout=30)
        assert sut.query("y").value.require()["max_model_len"] == 4096

        # The flag was replaced, not appended.
        argv = provider._roles[Role.FRONTEND].argv
        assert argv.count("--max-model-len") == 1
        sut.require_restarted(at("frontend"), since=first, within=10)


def test_restart_count_is_unknown_before_the_first_start(deployment, tmp_path):
    """ "Never started" and "started once" are different facts."""
    provider, roles, port = deployment
    assert provider.restart_count(at("frontend")).is_unknown
    provider.start(at("frontend"))
    assert provider.restart_count(at("frontend")).require() == 0


# ----------------------------------------------------------------- failures


def test_an_unreachable_role_is_unknown_not_absent(deployment, tmp_path):
    """Nothing was started, so nothing is known — not "it returned nothing"."""
    provider, roles, port = deployment
    got = provider.request(at("frontend"), "/v1/models", timeout=2)
    assert got.is_unknown
    assert "unreachable" in got.detail


def test_an_http_error_is_absent_with_its_status(deployment, tmp_path):
    """A 503 is a real answer from a running service, not a failure to reach it.

    Collapsing the two is how "the server refused" gets reported as "the server
    is down", and they call for different responses.
    """
    port = free_port()
    provider = LocalProvider(
        {
            Role.FRONTEND: LocalRole(
                role=Role.FRONTEND,
                argv=[
                    sys.executable,
                    str(STUB),
                    "--port",
                    str(port),
                    "--fail-after",
                    "1",
                ],
                port=port,
            )
        },
        log_dir=tmp_path / "logs2",
    )
    try:
        provider.start(at("frontend"))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if provider.request(at("frontend"), "/v1/models", timeout=1).is_known:
                break
            time.sleep(0.05)
        assert provider.request(
            at("frontend"), "/v1/completions", method="POST", body={"prompt": "1"}
        ).is_known
        second = provider.request(
            at("frontend"), "/v1/completions", method="POST", body={"prompt": "2"}
        )
        assert second.is_absent
        assert "503" in second.detail
    finally:
        provider.shutdown()


def test_stop_is_scoped_to_the_role(deployment, tmp_path):
    """A stop that tears down everything makes fault tests vacuous."""
    provider, roles, port = deployment
    provider.start(at("frontend"))
    assert provider.replicas(at("frontend")).require() == 1
    provider.stop(at("frontend"))
    assert provider.replicas(at("frontend")).require() == 0
    # The log survives the process, so evidence about a dead role is still readable.
    assert provider.logs(at("frontend")).is_known


def test_an_unknown_role_names_the_ones_that_exist(deployment, tmp_path):
    provider, roles, port = deployment
    with pytest.raises(KeyError, match="frontend"):
        provider.start(at("decode"))


# ------------------------------------------------- both façade spellings work


def test_the_component_spelling_reaches_the_same_verb(deployment, tmp_path):
    """`sut.component("frontend").query(...)` is `sut.query(..., at("frontend"))`."""
    provider, roles, port = deployment
    sut = make_sut(
        provider, roles, tmp_path, grants=(Grant.READ, Grant.INFER, Grant.LIFECYCLE)
    )
    with sut:
        sut.start(at("frontend"))
        sut.wait_serving(timeout=30)
        handle = sut.component("frontend").query("via the component view")
    assert (
        handle.value.require()["choices"][0]["text"] == "echo: via the component view"
    )
    assert handle.selector.role is Role.FRONTEND
