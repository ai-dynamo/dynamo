# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic rates and disabled-path behavior, without serving dependencies."""

import json
from unittest.mock import patch

import pytest

from dynamo.duru_router.diagnostics import PerformanceLogger, emit

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_disabled_diagnostics_do_not_inspect_snapshot(monkeypatch):
    monkeypatch.delenv("DURU_DIAGNOSTICS", raising=False)
    PerformanceLogger().observe(None, None)


def test_periodic_rates_use_deltas_and_only_active_roots(monkeypatch, caplog):
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1")
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    reporter = PerformanceLogger()
    snapshot = dict(
        time=110,
        observation_time=109,
        run_id="run",
        output_tokens=20,
        accepted_steps=10,
        requests={"request": {}},
        roots={
            "live": dict(U=2, active_agents=1),
            "unknown": dict(U=None, active_agents=1),
            "finished": dict(U=100, active_agents=0),
        },
    )
    with patch("dynamo.duru_router.diagnostics.time.monotonic", side_effect=[0, 4, 5]):
        reporter.observe(snapshot, 0)
        snapshot.update(output_tokens=70, accepted_steps=20)
        reporter.observe(snapshot, 2)
        assert not caplog.records
        reporter.observe(snapshot, 3)
    row = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert row["output_tokens_per_s"] == 10
    assert row["accepted_steps_per_s"] == 2
    assert row["active_roots"] == 2
    assert row["measured_roots"] == 1
    assert row["u_min"] == row["u_max"] == row["u_mean"] == 2
    assert row["accounting_lag_s"] == 1
    assert row["pending_events"] == 3


def test_invalid_score_diagnostics_are_json_null(caplog):
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    emit("admission", U=float("nan"), effective_score=float("inf"))
    row = json.loads(caplog.records[0].getMessage().split(" ", 1)[1])
    assert row == dict(U=None, effective_score=None)
