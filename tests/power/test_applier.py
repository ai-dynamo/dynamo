# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the applier's state machine with fake actuators, readers and clocks.

Everything under test lives in ``dynamo.power.applier_core``, which imports
neither ``pynvml`` nor ``epd``; the NVML-backed pieces in
``dynamo.power.applier`` are not imported here.
"""

import csv
import importlib
import json
import sys

import pytest

from dynamo.power.applier_core import (
    STATUS_CLAMPED,
    STATUS_NOT_LANDED,
    STATUS_OK,
    STATUS_UNVERIFIED,
    TRACE_COLUMNS,
    Applier,
    CommandResult,
    DcgmiActuator,
    GpuLimits,
    SudoSmiActuator,
    TraceWriter,
    build_parser,
    classify,
    parse_dcgmi_group_listing,
    setpoint_missed,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

NODE = "hecate0001"
BUDGET_W = 1500
DEFAULT_W = 1400.0
STALE_S = 90.0


class FakeLimits:
    """Programmable NVML stand-in: what the hardware reports back."""

    def __init__(self, gpus, default_w=DEFAULT_W, constraints=(300.0, 2300.0)):
        self.setpoint = {gpu: default_w for gpu in gpus}
        self.enforced = {gpu: default_w for gpu in gpus}
        self.default_w = default_w
        self.constraints = constraints
        self.unreadable = set()
        self.enforced_ceiling = {}
        self.enforced_override = {}

    def set_hardware(self, gpu, watts):
        self.setpoint[gpu] = float(watts)
        if gpu in self.enforced_override:
            self.enforced[gpu] = float(self.enforced_override[gpu])
        else:
            self.enforced[gpu] = min(
                float(watts), self.enforced_ceiling.get(gpu, float(watts))
            )

    def read(self, gpu):
        if gpu in self.unreadable:
            return GpuLimits(None, None)
        return GpuLimits(self.setpoint[gpu], self.enforced[gpu])

    def default_limit_w(self, gpu):
        return self.default_w

    def constraints_w(self, gpu):
        return self.constraints


class FakeActuator:
    """Records every apply; lands the caps on the fake hardware when ``land``."""

    name = "fake"

    def __init__(self, limits, land=True):
        self.limits = limits
        self.land = land
        self.calls = []
        self.closed = False

    def apply(self, limits_w):
        self.calls.append(dict(limits_w))
        if self.land:
            for gpu, watts in limits_w.items():
                self.limits.set_hardware(gpu, watts)

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeRunner:
    """Captures command lines and answers with canned results."""

    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def __call__(self, argv):
        self.calls.append(list(argv))
        if self.results:
            return self.results.pop(0)
        return CommandResult(0, "", "")


def make_harness(tmp_path, **applier_overrides):
    """One applier for GPUs (2, 0, 1) of NODE with fake collaborators."""
    gpus = (2, 0, 1)
    limits = FakeLimits(gpus)
    actuator = FakeActuator(limits)
    clock = FakeClock(1_000_000.0)
    monotonic = FakeClock(500.0)
    decision_file = tmp_path / "epd" / "actuation" / "power_decision.json"
    trace_path = tmp_path / "applier_trace.csv"
    kwargs = dict(
        decision_file=str(decision_file),
        node=NODE,
        gpu_indices=gpus,
        budget_w_per_gpu=BUDGET_W,
        actuator=actuator,
        limits=limits,
        trace=TraceWriter(str(trace_path)),
        stale_after_s=STALE_S,
        restore_default_on_exit=True,
        clock=clock,
        monotonic=monotonic,
    )
    kwargs.update(applier_overrides)
    applier = Applier(**kwargs)

    def write_decision(sequence, watts, age_s=0.0, node=NODE):
        decision_file.parent.mkdir(parents=True, exist_ok=True)
        decision_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "sequence": sequence,
                    "timestamp": clock.now - age_s,
                    "per_node_gpu_power_limit_w": {node: watts, "other-node": 999},
                }
            )
        )

    def tick(seconds):
        clock.advance(seconds)
        monotonic.advance(seconds)

    def trace_rows():
        if not trace_path.exists():
            return []
        with open(trace_path, newline="") as handle:
            return list(csv.DictReader(handle))

    return {
        "applier": applier,
        "actuator": actuator,
        "limits": limits,
        "write": write_decision,
        "tick": tick,
        "rows": trace_rows,
        "decision_file": decision_file,
    }


@pytest.fixture
def harness(tmp_path):
    return make_harness(tmp_path)


# ----------------------------------------------------------- apply & verify


def test_first_decision_is_applied_per_gpu_in_order_and_traced(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    assert harness["actuator"].calls == [{0: 1200, 1: 1200, 2: 1200}]
    assert list(harness["actuator"].calls[0]) == [0, 1, 2]
    rows = harness["rows"]()
    assert list(rows[0]) == list(TRACE_COLUMNS)
    assert [
        (int(r["gpu"]), r["requested"], r["setpoint"], r["enforced"], r["status"])
        for r in rows
    ] == [
        (0, "1200", "1200", "1200", STATUS_OK),
        (1, "1200", "1200", "1200", STATUS_OK),
        (2, "1200", "1200", "1200", STATUS_OK),
    ]
    assert all(
        state.status == STATUS_OK and not state.retry
        for state in harness["applier"].states.values()
    )


def test_unchanged_decision_is_not_reapplied_but_new_watts_are(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["tick"](30)
    harness["write"](2, 1200)
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 1
    harness["tick"](30)
    harness["write"](3, 1250)
    harness["applier"].poll_once()
    assert harness["actuator"].calls[-1] == {0: 1250, 1: 1250, 2: 1250}


def test_missed_setpoint_is_retried_until_it_lands(harness):
    harness["actuator"].land = False
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    assert harness["applier"].states[0].status == STATUS_NOT_LANDED
    assert harness["applier"].states[0].retry
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 2
    harness["actuator"].land = True
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 3
    assert all(
        state.status == STATUS_OK and not state.retry
        for state in harness["applier"].states.values()
    )
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 3


def test_enforced_above_a_landed_setpoint_is_not_landed_but_not_retried(harness):
    harness["limits"].enforced_override[1] = 1300.0  # EPD's "diverged" case
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    state = harness["applier"].states[1]
    assert (state.status, state.retry) == (STATUS_NOT_LANDED, False)
    assert harness["applier"].states[0].status == STATUS_OK
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 1
    row = [r for r in harness["rows"]() if r["gpu"] == "1"][0]
    assert (row["setpoint"], row["enforced"], row["status"]) == (
        "1200",
        "1300",
        STATUS_NOT_LANDED,
    )


def test_unverified_gpu_is_applied_once_and_not_retried(harness):
    harness["limits"].unreadable.add(1)
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    assert harness["applier"].states[1].status == STATUS_UNVERIFIED
    assert not harness["applier"].states[1].retry
    assert harness["applier"].states[0].status == STATUS_OK
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 1
    row = [r for r in harness["rows"]() if r["gpu"] == "1"][0]
    assert (row["setpoint"], row["enforced"], row["status"]) == (
        "",
        "",
        STATUS_UNVERIFIED,
    )


def test_hardware_clamp_below_setpoint_is_reported_as_clamped(harness):
    harness["limits"].enforced_ceiling[2] = 1000.0
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    assert harness["applier"].states[2].status == STATUS_CLAMPED
    assert harness["applier"].states[0].status == STATUS_OK


def test_request_outside_settable_range_is_clamped_before_writing(harness):
    harness["limits"].constraints = (600.0, 2300.0)
    harness["write"](1, 400)
    harness["applier"].poll_once()
    assert harness["actuator"].calls == [{0: 600, 1: 600, 2: 600}]
    assert all(
        state.status == STATUS_CLAMPED for state in harness["applier"].states.values()
    )
    row = harness["rows"]()[0]
    assert (row["requested"], row["setpoint"], row["status"]) == (
        "400",
        "600",
        STATUS_CLAMPED,
    )


def test_decision_without_this_node_applies_nothing(harness):
    harness["write"](1, 1200, node="someone-else")
    harness["applier"].poll_once()
    assert harness["actuator"].calls == []
    assert harness["rows"]() == []


# ------------------------------------------------------------------ floor


def test_floor_raises_a_low_cap_before_writing_and_reports_clamped(tmp_path):
    h = make_harness(tmp_path, min_w_per_gpu=900)
    h["write"](1, 700)
    h["applier"].poll_once()
    assert h["actuator"].calls == [{0: 900, 1: 900, 2: 900}]
    assert all(
        state.status == STATUS_CLAMPED and not state.retry
        for state in h["applier"].states.values()
    )
    row = h["rows"]()[0]
    assert (row["requested"], row["setpoint"], row["enforced"], row["status"]) == (
        "700",
        "900",
        "900",
        STATUS_CLAMPED,
    )
    h["tick"](30)
    h["write"](2, 1200)
    h["applier"].poll_once()
    assert h["actuator"].calls[-1] == {0: 1200, 1: 1200, 2: 1200}
    assert all(state.status == STATUS_OK for state in h["applier"].states.values())


def test_floor_is_not_applied_to_the_hardware_default_restored_on_exit(tmp_path):
    h = make_harness(tmp_path, min_w_per_gpu=900)
    h["limits"].default_w = 800.0
    h["write"](1, 1200)
    h["applier"].poll_once()
    h["applier"].shutdown()
    assert h["actuator"].calls[-1] == {0: 800, 1: 800, 2: 800}


def test_floor_keeps_the_fail_safe_budget_untouched(tmp_path):
    h = make_harness(tmp_path, min_w_per_gpu=BUDGET_W)
    h["write"](1, 1200, age_s=STALE_S + 1)
    h["applier"].poll_once()
    assert h["actuator"].calls == [{0: BUDGET_W, 1: BUDGET_W, 2: BUDGET_W}]
    assert all(state.status == STATUS_OK for state in h["applier"].states.values())


# ------------------------------------------------------------- fail-safe


def test_stale_decision_falls_back_to_the_uniform_budget_once(harness):
    harness["write"](1, 1200, age_s=STALE_S + 1)
    harness["applier"].poll_once()
    assert harness["applier"].fail_safe_active
    assert harness["actuator"].calls == [{0: BUDGET_W, 1: BUDGET_W, 2: BUDGET_W}]
    harness["applier"].poll_once()
    assert len(harness["actuator"].calls) == 1
    harness["tick"](30)
    harness["write"](2, 1300)
    harness["applier"].poll_once()
    assert not harness["applier"].fail_safe_active
    assert harness["actuator"].calls[-1] == {0: 1300, 1: 1300, 2: 1300}


def test_decision_going_stale_after_tracking_falls_back(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["tick"](STALE_S + 5)
    harness["applier"].poll_once()
    assert harness["applier"].fail_safe_active
    assert harness["actuator"].calls[-1] == {0: BUDGET_W, 1: BUDGET_W, 2: BUDGET_W}


def test_advancing_sequence_overrides_a_skewed_timestamp(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["tick"](30)
    harness["write"](2, 1250, age_s=600)  # controller clock far behind, but alive
    harness["applier"].poll_once()
    assert not harness["applier"].fail_safe_active
    assert harness["actuator"].calls[-1] == {0: 1250, 1: 1250, 2: 1250}
    harness["tick"](STALE_S + 5)  # no new sequence for longer than the window
    harness["applier"].poll_once()
    assert harness["applier"].fail_safe_active


def test_missing_file_after_it_existed_falls_back_but_never_existing_does_nothing(
    harness,
):
    harness["applier"].poll_once()
    assert harness["actuator"].calls == []
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["decision_file"].unlink()
    harness["applier"].poll_once()
    assert harness["applier"].fail_safe_active
    assert harness["actuator"].calls[-1] == {0: BUDGET_W, 1: BUDGET_W, 2: BUDGET_W}


def test_corrupt_file_after_it_existed_falls_back(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["decision_file"].write_text("{not json")
    harness["applier"].poll_once()
    assert harness["applier"].fail_safe_active


# ------------------------------------------------------------------ exit


def test_shutdown_restores_hardware_defaults_when_requested(harness):
    harness["write"](1, 1200)
    harness["applier"].poll_once()
    harness["applier"].shutdown()
    assert harness["actuator"].calls[-1] == {0: 1400, 1: 1400, 2: 1400}
    assert harness["actuator"].closed
    assert harness["rows"]()[-1]["requested"] == "1400"


def test_shutdown_without_restore_only_closes(tmp_path):
    limits = FakeLimits((0,))
    actuator = FakeActuator(limits)
    applier = Applier(
        decision_file=str(tmp_path / "power_decision.json"),
        node=NODE,
        gpu_indices=(0,),
        budget_w_per_gpu=BUDGET_W,
        actuator=actuator,
        limits=limits,
        trace=TraceWriter(None),
    )
    applier.shutdown()
    assert actuator.calls == []
    assert actuator.closed


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget_w_per_gpu": 0},
        {"stale_after_s": 0},
        {"min_w_per_gpu": 0},
        {"min_w_per_gpu": BUDGET_W + 1},
    ],
)
def test_applier_rejects_bad_budget_window_and_floor(tmp_path, overrides):
    limits = FakeLimits((0,))
    values = {"budget_w_per_gpu": BUDGET_W, "stale_after_s": STALE_S, **overrides}
    with pytest.raises(ValueError):
        Applier(
            decision_file=str(tmp_path / "d.json"),
            node=NODE,
            gpu_indices=(0,),
            actuator=FakeActuator(limits),
            limits=limits,
            trace=TraceWriter(None),
            **values,
        )


# ------------------------------------------------------------- actuators


def test_sudo_smi_argument_order_per_gpu_ascending_non_interactive():
    runner = FakeRunner()
    SudoSmiActuator(runner).apply({3: 1200, 1: 1100})
    assert runner.calls == [
        ["sudo", "-n", "nvidia-smi", "-pl", "1100", "-i", "1"],
        ["sudo", "-n", "nvidia-smi", "-pl", "1200", "-i", "3"],
    ]


def test_sudo_smi_failure_is_logged_not_raised(caplog):
    runner = FakeRunner([CommandResult(4, "", "Insufficient permissions")])
    SudoSmiActuator(runner).apply({0: 1200})
    assert "Insufficient permissions" in caplog.text


def test_dcgmi_creates_one_group_for_a_uniform_cap_and_reuses_it():
    created = CommandResult(
        0, 'Successfully created group "g" with a group ID of 7\n', ""
    )
    runner = FakeRunner([created])
    actuator = DcgmiActuator(runner)
    actuator.apply({0: 1200, 1: 1200, 2: 1200, 3: 1200})
    assert runner.calls[0][:3] == ["dcgmi", "group", "-c"]
    assert runner.calls[0][3].startswith("dynamo-power-")
    assert runner.calls[0][4:] == ["-a", "0,1,2,3"]
    assert runner.calls[1] == ["dcgmi", "config", "-g", "7", "--set", "-P", "1200"]
    actuator.apply({0: 1300, 1: 1300, 2: 1300, 3: 1300})
    assert runner.calls[2] == ["dcgmi", "config", "-g", "7", "--set", "-P", "1300"]
    actuator.close()
    assert runner.calls[3] == ["dcgmi", "group", "-d", "7"]


def test_dcgmi_uses_one_group_per_distinct_cap():
    runner = FakeRunner(
        [
            CommandResult(0, 'Successfully created group "a" with a group ID of 3', ""),
            CommandResult(0, "", ""),
            CommandResult(0, 'Successfully created group "b" with a group ID of 4', ""),
            CommandResult(0, "", ""),
        ]
    )
    DcgmiActuator(runner).apply({0: 1100, 1: 1100, 2: 1200})
    assert runner.calls[0][4:] == ["-a", "0,1"]
    assert runner.calls[1] == ["dcgmi", "config", "-g", "3", "--set", "-P", "1100"]
    assert runner.calls[2][4:] == ["-a", "2"]
    assert runner.calls[3] == ["dcgmi", "config", "-g", "4", "--set", "-P", "1200"]


def test_dcgmi_group_creation_failure_skips_the_set(caplog):
    runner = FakeRunner(
        [CommandResult(1, "", "Error: unable to connect to host engine")]
    )
    DcgmiActuator(runner).apply({0: 1200})
    assert len(runner.calls) == 1
    assert "unable to connect" in caplog.text


DCGMI_LISTING = """\
+-------------------+----------------------------------------------------------+
| GROUPS                                                                       |
| 3 groups found.                                                              |
+===================+==========================================================+
| Groups            |                                                          |
| -> 0              |                                                          |
|    -> Group ID    | 0                                                        |
|    -> Group Name  | dynamo-power-4242-0-1-2-3                                |
|    -> Entities    | GPU 0, GPU 1, GPU 2, GPU 3                               |
| -> 1              |                                                          |
|    -> Group ID    | 1                                                        |
|    -> Group Name  | dynamo-power-777-0-1                                     |
|    -> Entities    | GPU 0, GPU 1                                             |
| -> 2              |                                                          |
|    -> Group ID    | 2                                                        |
|    -> Group Name  | someone-elses-group                                      |
|    -> Entities    | GPU 2                                                    |
+-------------------+----------------------------------------------------------+
"""


def test_parse_dcgmi_group_listing_pairs_ids_with_names():
    assert parse_dcgmi_group_listing(DCGMI_LISTING) == [
        ("0", "dynamo-power-4242-0-1-2-3"),
        ("1", "dynamo-power-777-0-1"),
        ("2", "someone-elses-group"),
    ]
    assert parse_dcgmi_group_listing("Error: unable to connect to host engine") == []


def test_dcgmi_cleanup_deletes_only_groups_of_dead_appliers():
    runner = FakeRunner([CommandResult(0, DCGMI_LISTING, "")])
    deleted = DcgmiActuator(runner).cleanup_stale_groups(alive=lambda pid: pid == 777)
    assert deleted == ["0"]
    assert runner.calls == [["dcgmi", "group", "-l"], ["dcgmi", "group", "-d", "0"]]


def test_dcgmi_cleanup_tolerates_a_failed_listing(caplog):
    runner = FakeRunner(
        [CommandResult(1, "", "Error: unable to connect to host engine")]
    )
    assert DcgmiActuator(runner).cleanup_stale_groups() == []
    assert runner.calls == [["dcgmi", "group", "-l"]]
    assert "listing DCGM groups failed" in caplog.text


# ------------------------------------------------------------ classify/CLI


@pytest.mark.parametrize(
    ("requested", "target", "limits", "expected"),
    [
        (1200, 1200, GpuLimits(1200.0, 1200.0), STATUS_OK),
        (1200, 1200, GpuLimits(1203.0, 1196.0), STATUS_OK),
        (400, 600, GpuLimits(600.0, 600.0), STATUS_CLAMPED),
        (700, 900, GpuLimits(900.0, 900.0), STATUS_CLAMPED),
        (1200, 1200, GpuLimits(1200.0, 1000.0), STATUS_CLAMPED),
        (1200, 1200, GpuLimits(1400.0, 1400.0), STATUS_NOT_LANDED),
        (1200, 1200, GpuLimits(1200.0, 1300.0), STATUS_NOT_LANDED),
        (1200, 1200, GpuLimits(None, 1200.0), STATUS_UNVERIFIED),
        (1200, 1200, GpuLimits(1200.0, None), STATUS_UNVERIFIED),
    ],
)
def test_classify(requested, target, limits, expected):
    assert classify(requested, target, limits) == expected


def test_setpoint_missed_only_when_the_programmed_setpoint_differs():
    assert setpoint_missed(1200, GpuLimits(1400.0, 1400.0))
    assert not setpoint_missed(1200, GpuLimits(1200.0, 1300.0))
    assert not setpoint_missed(1200, GpuLimits(1203.0, 1196.0))
    assert not setpoint_missed(1200, GpuLimits(None, None))


def test_trace_writer_without_path_is_a_no_op(tmp_path):
    TraceWriter(None).write(1.0, 0, 1200, 1200.0, 1200.0, STATUS_OK)
    nested = tmp_path / "a" / "b" / "trace.csv"
    TraceWriter(str(nested)).write(1.5, 3, 1200, None, 1199.6, STATUS_UNVERIFIED)
    with open(nested, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        list(TRACE_COLUMNS),
        ["1.500", "3", "1200", "", "1200", STATUS_UNVERIFIED],
    ]


def test_cli_parser_contract():
    required = [
        "--decision-file",
        "/logs/epd/actuation/power_decision.json",
        "--node",
        NODE,
        "--gpu-indices",
        "0,1,2,3",
        "--mode",
        "dcgmi",
        "--budget-w-per-gpu",
        "1500",
    ]
    args = build_parser().parse_args(required)
    assert args.stale_after_s == 90
    assert args.poll_s == 5
    assert args.min_w_per_gpu is None
    assert args.restore_default_on_exit is False
    assert args.trace_csv is None
    assert (
        build_parser().parse_args([*required, "--min-w-per-gpu", "900"]).min_w_per_gpu
        == 900
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--decision-file", "x", "--node", "n", "--gpu-indices", "0"]
        )


def test_applier_core_imports_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` fail
    monkeypatch.delitem(sys.modules, "dynamo.power.applier_core", raising=False)
    module = importlib.import_module("dynamo.power.applier_core")
    assert hasattr(module, "Applier")
