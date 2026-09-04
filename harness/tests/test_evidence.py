# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.evidence`.

The properties worth having are the ones that stop a run reporting success it
did not earn: an artifact nobody collected must not read as an empty one, a run
that judged nothing must not pass, and a check that crashes must not be scored
as a product failure.
"""

import json

import pytest
from dynamo_test.evidence import (
    BundleError,
    Evidence,
    NotDeclared,
    Outcome,
    Recorder,
    Sealed,
    Verdict,
    combine,
)


@pytest.fixture
def recorder(tmp_path):
    return Recorder(tmp_path, "run-1")


@pytest.fixture
def sealed(tmp_path):
    r = Recorder(tmp_path, "run-1")
    r.declare("requests/main.jsonl", "query", min_rows=1)
    r.declare("roles/frontend/0/current.log", "log-scrape")
    r.write_rows(
        "requests/main.jsonl", [{"id": "a", "ok": True}, {"id": "b", "ok": False}]
    )
    r.write_text("roles/frontend/0/current.log", "started\nserving\n")
    r.note("plan", {"name": "demo"})
    r.seal()
    return r.evidence()


# --------------------------------------------------------------- verdicts


def test_invalid_outranks_failed():
    """A run that did not measure has not judged the product.

    Reporting a failure from missing evidence attributes a harness problem to
    whoever owns the code under test.
    """
    assert Verdict.INVALID.rank < Verdict.FAILED.rank < Verdict.PASSED.rank
    assert (
        combine(
            [
                Outcome("a", Verdict.FAILED),
                Outcome("b", Verdict.INVALID),
                Outcome("c", Verdict.PASSED),
            ]
        )
        is Verdict.INVALID
    )


def test_verdicts_have_distinct_exit_codes():
    """So a CI lane can retry an invalid run and escalate a failed one."""
    assert Verdict.PASSED.exit_code == 0
    assert Verdict.FAILED.exit_code == 1
    assert Verdict.INVALID.exit_code == 3
    assert Verdict.OBSERVED.exit_code == 0


def test_judging_nothing_is_invalid_not_passed():
    """The largest false green available: a run that checked nothing."""
    assert combine([]) is Verdict.INVALID


def test_observed_never_lowers_the_verdict():
    assert combine([Outcome("a", Verdict.PASSED), Outcome("b", Verdict.OBSERVED)]) is (
        Verdict.PASSED
    )


# ------------------------------------------------------------- collecting


def test_writing_an_undeclared_artifact_is_refused(recorder):
    """Without a promise there is nothing for absence to contradict."""
    with pytest.raises(NotDeclared, match="never declared"):
        recorder.write_text("stray.log", "hello")


def test_nothing_can_be_written_after_the_seal(recorder):
    recorder.declare("a.txt", "p")
    recorder.write_text("a.txt", "x")
    recorder.seal()
    with pytest.raises(Sealed):
        recorder.write_text("a.txt", "y")
    with pytest.raises(Sealed):
        recorder.declare("b.txt", "p")


def test_the_bundle_cannot_be_read_before_it_is_sealed(recorder):
    recorder.declare("a.txt", "p")
    with pytest.raises(BundleError, match="not sealed"):
        recorder.evidence()


def test_an_artifact_name_may_not_escape_the_bundle(recorder):
    recorder.declare("../escape.txt", "p")
    with pytest.raises(BundleError, match="relative path"):
        recorder.write_text("../escape.txt", "x")


# ------------------------------------------------------------------ seal


def test_a_kept_promise_is_valid(sealed):
    assert sealed.seal.is_valid
    assert sealed.require_valid_run().verdict is Verdict.PASSED


def test_a_declared_artifact_that_never_arrives_invalidates_the_run(tmp_path):
    r = Recorder(tmp_path, "run-2")
    r.declare("requests/main.jsonl", "query", min_rows=1)
    seal = r.seal()
    assert not seal.is_valid
    assert "requests/main.jsonl" in seal.missing
    assert "never written" in seal.why() or "declared but never written" in seal.why()


def test_an_empty_file_does_not_satisfy_a_row_promise(tmp_path):
    """ "The file exists" is a weak promise; an empty requests log proves nothing."""
    r = Recorder(tmp_path, "run-3")
    r.declare("requests/main.jsonl", "load", min_rows=1)
    r.write_rows("requests/main.jsonl", [])
    seal = r.seal()
    assert not seal.is_valid
    assert seal.short == (("requests/main.jsonl", 1, 0),)
    assert "promised 1, got 0" in seal.why()


def test_an_optional_artifact_may_be_absent(tmp_path):
    r = Recorder(tmp_path, "run-4")
    r.declare("maybe.log", "p", required=False)
    assert r.seal().is_valid


# ------------------------------------------------------------- judgement


def test_an_uncollected_artifact_is_unknown_not_empty(sealed):
    """The property the whole module exists for.

    A check that globs, finds nothing, and concludes "no errors" has answered a
    question about the product using the failure of the measurement.
    """
    fact = sealed.text("roles/worker/0/current.log")
    assert fact.is_unknown
    assert "nothing declared this artifact" in fact.detail
    assert "roles/frontend/0/current.log" in fact.detail  # says what *was* collected


def test_reads_go_through_the_producer_index(sealed):
    assert sealed.produced_by("requests/main.jsonl").require() == "query"
    assert sealed.artifacts() == (
        "requests/main.jsonl",
        "roles/frontend/0/current.log",
    )


def test_rows_are_parsed_and_counted(sealed):
    rows = sealed.rows("requests/main.jsonl")
    assert len(rows.require()) == 2
    assert rows.require()[0]["id"] == "a"


def test_a_malformed_row_makes_the_whole_artifact_unknown(tmp_path):
    """Skipping a bad line silently changes the denominator of every rate."""
    r = Recorder(tmp_path, "run-5")
    r.declare("requests/main.jsonl", "load")
    r.write_text("requests/main.jsonl", '{"ok": true}\nnot json\n')
    r.seal()
    fact = r.evidence().rows("requests/main.jsonl")
    assert fact.is_unknown
    assert "line 2" in fact.detail


def test_an_invalid_run_short_circuits_the_checks(tmp_path):
    """No point asking whether the product behaved if nothing was measured."""
    r = Recorder(tmp_path, "run-6")
    r.declare("requests/main.jsonl", "load", min_rows=1)
    r.seal()
    ran = []

    def expect_something(ev):
        ran.append(True)
        return Outcome("expect_something", Verdict.PASSED)

    verdict, outcomes = r.evidence().judge([expect_something])
    assert verdict is Verdict.INVALID
    assert ran == [], "checks must not run against an invalid bundle"
    assert len(outcomes) == 1


def test_a_check_that_raises_is_invalid_not_failed(sealed):
    """A broken check is a harness problem, not a product regression."""

    def expect_boom(ev):
        raise RuntimeError("kaboom")

    verdict, outcomes = sealed.judge([expect_boom])
    assert verdict is Verdict.INVALID
    boom = next(o for o in outcomes if o.check == "expect_boom")
    assert "RuntimeError" in boom.detail
    assert "kaboom" in boom.detail


def test_a_check_returning_the_wrong_type_is_invalid(sealed):
    def expect_true(ev):
        return True

    verdict, outcomes = sealed.judge([expect_true])
    assert verdict is Verdict.INVALID
    assert "expected an Outcome" in next(
        o.detail for o in outcomes if o.check == "expect_true"
    )


def test_a_failing_check_fails_the_run(sealed):
    def expect_all_requests_succeeded(ev):
        rows = ev.rows("requests/main.jsonl").require()
        bad = [r for r in rows if not r["ok"]]
        return Outcome(
            "expect_all_requests_succeeded",
            Verdict.PASSED if not bad else Verdict.FAILED,
            f"{len(bad)} of {len(rows)} failed",
            evidence=("requests/main.jsonl",),
        )

    verdict, outcomes = sealed.judge([expect_all_requests_succeeded])
    assert verdict is Verdict.FAILED
    assert "1 of 2 failed" in outcomes[-1].detail


# --------------------------------------------------------------- replay


def test_a_sealed_bundle_is_judgeable_offline(tmp_path):
    """The point of sealing: the same verdict, hours later, from the directory.

    Nothing here touches a cluster, so a CI failure can be re-scored on a laptop
    without reproducing the deployment.
    """
    r = Recorder(tmp_path, "run-7")
    r.declare("requests/main.jsonl", "query", min_rows=1)
    r.write_rows("requests/main.jsonl", [{"ok": True}])
    r.note("plan", {"name": "demo"})
    r.seal()

    reopened = Evidence.open(tmp_path / "run-7")
    assert reopened.run_id == "run-7"
    assert reopened.run["plan"] == {"name": "demo"}
    assert reopened.rows("requests/main.jsonl").require() == ({"ok": True},)
    assert reopened.require_valid_run().verdict is Verdict.PASSED


def test_opening_a_directory_that_is_not_a_bundle_says_so(tmp_path):
    with pytest.raises(BundleError, match="not a sealed evidence bundle"):
        Evidence.open(tmp_path)


def test_findings_are_written_so_a_run_can_be_rescored(sealed, tmp_path):
    def observe_request_count(ev):
        return Outcome("observe_request_count", Verdict.OBSERVED, "2 requests")

    verdict, outcomes = sealed.judge([observe_request_count])
    path = sealed.write_findings(outcomes)
    written = json.loads(path.read_text())
    assert written["verdict"] == verdict.value
    assert any(o["check"] == "observe_request_count" for o in written["outcomes"])
    assert all("verdict" in o for o in written["outcomes"])
