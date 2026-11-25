# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for graph-aware calibration; no accuracy-validation campaign."""

import json
from functools import partial
from pathlib import Path

import pytest

from dynamo.duru_router.calibration import measure as measurement
from dynamo.duru_router.calibration.measure import load, measure
from dynamo.duru_router.calibration.plan import METRIC, atomic_json, design
from dynamo.duru_router.calibration.predict import predict, predict_prefill_table
from dynamo.duru_router.scoring import LiveU

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_graph_neighbors_and_eager_transition():
    plan = design(
        32768,
        64,
        max_prefill=1024,
        capture_sizes=[64, 128, 256],
        prefill_graphs=True,
    )
    for boundary in (64, 128, 256):
        assert {boundary - 1, boundary, boundary + 1} <= set(plan["P"])
    assert plan["P_discontinuities"] == [64, 128, 256]
    assert all((p + d + 1 <= plan["cap"] for p in plan["P"] for d in plan["D"]))
    assert all((d % 64 == 0 for d in plan["D"]))
    assert all((case["split"] == "train" for case in plan["cases"]))
    assert "validation" not in plan


def test_decode_only_graphs_do_not_create_prefill_boundaries():
    graph = design(
        32768, 64, max_prefill=1024, capture_sizes=[73, 149], prefill_graphs=True
    )
    decode_only = design(32768, 64, max_prefill=1024, capture_sizes=[73, 149])
    assert decode_only["P_discontinuities"] == []
    assert graph["K"] == decode_only["K"]
    with pytest.raises(ValueError):
        design(32768, 64, prefill_graphs=True)


def test_live_predictor_keeps_graph_jump():
    table = dict(
        P=[1, 128, 512],
        D=[0, 16],
        P_discontinuities=[256],
        wall_ms=[[1, 1], [2, 2], [100, 100]],
    )
    for p, expected in ((256, 2), (257, 100)):
        assert predict_prefill_table(table, p, 0) == expected
        assert predict(table, dict(phase="prefill", p=p, d=0)) == (expected, False)


def test_live_outer_edge_extrapolation_preserved():
    table = dict(P=[1, 2], D=[0, 1], wall_ms=[[1, 2], [2, 3]])
    value, outside = predict(table, dict(phase="prefill", p=3, d=2))
    assert value == 5
    assert outside
    with pytest.raises(ValueError):
        predict_prefill_table(table, 3, 2)


def test_lookahead_uses_calibrated_chunk_budget():
    book = LiveU(1, {"max_prefill": 4})
    coordinates = []

    def cost(phase, p, d, k):
        coordinates.append((phase, p, d, k))
        return 1.0

    book.cost = cost
    assert book.next_five(10) == 5.0
    assert [row[1] for row in coordinates] == [4, 4, 2, 1, 1]


@pytest.mark.parametrize(
    "k, expected, outside",
    [
        (2, 2 / 44.5, True),
        (43.5, 43.5 / 44.5, True),
        (44.5, 1, False),
        (45, 1.5, False),
        (46.5, 3, True),
    ],
)
def test_decode_prediction_origin_fallback(k, expected, outside):
    table = dict(grid=[dict(K=44.5, wall_ms=1), dict(K=45.5, wall_ms=2)])
    value, actual_outside = predict(table, dict(phase="decode", k=k))
    assert value == pytest.approx(expected)
    assert actual_outside is outside


@pytest.mark.parametrize(
    "costs, k, expected",
    [([2, 1], 48, 1), ([1e308, 1.1e308], 1, 1e308 / 45)],
)
def test_decode_fallback_provides_live_cost(costs, k, expected):
    table = dict(grid=[dict(K=k, wall_ms=ms) for k, ms in zip((45, 46), costs)])
    value, outside = predict(table, dict(phase="decode", k=k))
    assert value == pytest.approx(expected)
    assert outside
    book = LiveU(1, dict(max_prefill=4, decode=table))
    assert book.cost("decode", 1, k - 1, k) == pytest.approx(expected / 1000)
    assert book.extrapolated_cost_evaluations == 1


@pytest.mark.parametrize(
    "axis, coordinate, expected, outside",
    [
        ("p", 2, 1, True),
        ("d", 2, 1, True),
        ("p", 44.5, 1.5, False),
        ("d", 44.5, 1.5, False),
        ("p", 46, 3, True),
        ("d", 46, 3, True),
    ],
)
def test_prefill_prediction_boundary_fallback(axis, coordinate, expected, outside):
    if axis == "p":
        table = dict(P=[44, 45], D=[0, 1], wall_ms=[[1, 1], [2, 2]])
    else:
        table = dict(P=[1, 2], D=[44, 45], wall_ms=[[1, 2], [1, 2]])
    step = dict(phase="prefill", p=1, d=0)
    step[axis] = coordinate
    value, actual_outside = predict(table, step)
    assert value == pytest.approx(expected)
    assert actual_outside is outside


@pytest.mark.parametrize("d, expected", [(0, 1), (500, 2), (1000, 3), (2000, 3)])
def test_prefill_boundary_fallback_retains_cached_context(d, expected):
    table = dict(P=[8, 9], D=[0, 1000], wall_ms=[[1, 3], [2, 4]])
    value, outside = predict(table, dict(phase="prefill", p=2, d=d))
    assert value == pytest.approx(expected)
    assert outside
    book = LiveU(1, dict(max_prefill=9, prefill=table))
    assert book.cost("prefill", 2, d, d + 2) == pytest.approx(expected / 1000)


@pytest.mark.parametrize(
    "p, d, expected",
    [(1, 0, 1), (32, 0, 3), (32, 20, 4.5)],
)
def test_prefill_boundary_fallback_keeps_execution_regimes(p, d, expected):
    table = dict(
        P=[4, 8, 16, 24],
        D=[0, 10],
        P_discontinuities=[8],
        wall_ms=[[1, 2], [3, 4], [10, 11], [3, 4.5]],
    )
    value, outside = predict(table, dict(phase="prefill", p=p, d=d))
    assert value == pytest.approx(expected)
    assert outside


def test_prefill_boundary_fallback_handles_nonfinite_extrapolation():
    table = dict(P=[45, 46], D=[0, 1], wall_ms=[[1e308, 1e308], [1.1e308, 1.1e308]])
    value, outside = predict(table, dict(phase="prefill", p=1, d=0))
    assert value == pytest.approx(1e308)
    assert outside


@pytest.mark.parametrize("p, expected", [(8, 1), (8.5, 1.5)])
def test_prefill_boundary_fallback_above_cached_context_grid(p, expected):
    table = dict(P=[8, 9], D=[1000, 1100], wall_ms=[[2, 1], [3, 2]])
    value, outside = predict(table, dict(phase="prefill", p=p, d=1300))
    assert value == pytest.approx(expected)
    assert outside


@pytest.fixture
def calibration_run(tmp_path):
    root = tmp_path
    ledger = root / "accepted.jsonl"
    ledger.touch()
    plan = dict(
        timing_metric=METRIC,
        cap=1024,
        max_prefill=4,
        P=[1, 4],
        D=[0, 16],
        K=[32, 128],
        P_discontinuities=[],
        repeats=1,
        warmups=1,
        decode_tokens=16,
        decode_warmup_tokens=8,
        cases=[
            *[dict(phase="prefill", P=p, D=d) for d in (0, 16) for p in (1, 4)],
            *[dict(phase="decode", K=k) for k in (32, 128)],
        ],
    )
    contract = dict(timing_metric=METRIC, model="cpu-fixture")
    primed = set()
    clock = 0.0

    def submit(prompt, count, rid, salt):
        nonlocal clock
        cached = int(salt.rsplit("D", 1)[1]) if "-D" in salt and salt in primed else 0
        primed.add(salt)
        steps = []
        for i in range(count):
            p, d = (
                (len(prompt) - cached, cached) if i == 0 else (1, len(prompt) + i - 1)
            )
            begin = clock
            clock += 0.001 * (2 + p + 0.001 * d) if i == 0 else 0.001
            steps.append(
                dict(
                    client_request_id=rid,
                    accepted=True,
                    p=p,
                    d=d,
                    k=p + d,
                    generated_tokens=1,
                    scheduled_at=begin,
                    end=clock,
                )
            )
        with ledger.open("a") as output:
            output.write(json.dumps(dict(event="accepted_output", steps=steps)) + "\n")
        return dict(output_tokens=count, cached_tokens=cached)

    run = partial(
        measure,
        root / "cal",
        contract,
        submit=submit,
        drain=lambda: None,
        ledger=ledger,
        reset=primed.clear,
    )
    return run, plan, contract


def test_measure_resume_and_load_without_accuracy_gate(tmp_path, calibration_run):
    root = tmp_path
    run, plan, contract = calibration_run
    result = run(plan)
    assert "quality_passed" not in result
    assert "validation" not in result
    assert result["prefill"]["wall_ms"][0][0] == pytest.approx(3, rel=0, abs=5e-8)
    loaded = load(root / "cal/calibration.json", contract)
    assert loaded == result
    legacy_path = root / "legacy.json"
    legacy_path.write_text(json.dumps(dict(result, measurement_only=True)))
    legacy = load(legacy_path, contract)
    assert legacy["prefill"] == loaded["prefill"]
    assert legacy["decode"] == loaded["decode"]
    assert run(plan, resume=True) == result
    with pytest.raises(ValueError):
        load(root / "cal/calibration.json", dict(contract, model="other"))
    with pytest.raises(ValueError):
        run(dict(plan, repeats=2), resume=True)
    legacy_plan = dict(
        plan,
        measurement_only=True,
        point_selection="graph_neighbors_and_geometric_context_v1",
    )
    legacy_run = partial(measure, root / "legacy-cal", contract, **run.keywords)
    legacy_result = legacy_run(legacy_plan)
    assert legacy_run(legacy_plan, resume=True) == legacy_result
    result["prefill"]["wall_ms"][0][0] = 0
    (root / "bad.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load(root / "bad.json", contract)


@pytest.mark.parametrize("filename", ["plan.json", "contract.json", "checkpoint.json"])
def test_measure_initialization_failure_can_resume(
    tmp_path, calibration_run, monkeypatch, filename
):
    run, plan, _ = calibration_run

    def fail_initial_write(path, value):
        if path.name == filename:
            raise OSError("initial write interrupted")
        atomic_json(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(measurement, "atomic_json", fail_initial_write)
        with pytest.raises(OSError, match="initial write interrupted"):
            run(plan)
    assert not (tmp_path / "cal").exists()
    assert not list(tmp_path.glob(".cal-*"))
    result = run(plan, resume=True)
    assert result["prefill"]["wall_ms"][0][0] > 0


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_measure_rejects_invalid_timing_before_saving(
    tmp_path, calibration_run, duration
):
    run, plan, _ = calibration_run
    submit = run.keywords["submit"]
    ledger = run.keywords["ledger"]

    def invalid_timing(prompt, count, rid, salt):
        response = submit(prompt, count, rid, salt)
        if len(prompt) == 4:
            lines = ledger.read_text().splitlines()
            record = json.loads(lines[-1])
            step = record["steps"][0]
            step["end"] = step["scheduled_at"] + duration
            lines[-1] = json.dumps(record)
            ledger.write_text("\n".join(lines) + "\n")
        return response

    with pytest.raises(ValueError, match="timing must be finite and positive"):
        run(plan, submit=invalid_timing)
    checkpoint = json.loads((tmp_path / "cal/checkpoint.json").read_text())
    assert [row["P"] for row in checkpoint["points"]] == [1]
    samples = [
        json.loads(line)
        for line in (tmp_path / "cal/samples.jsonl").read_text().splitlines()
    ]
    assert all(row["case"]["P"] == 1 for row in samples)
    result = run(plan, resume=True)
    assert result["prefill"]["wall_ms"][1][0] > 0


def test_measure_checkpoint_failure_is_not_retried(
    tmp_path, calibration_run, monkeypatch
):
    run, plan, _ = calibration_run
    attempts = 0

    def fail_completed_checkpoint(path, value):
        nonlocal attempts
        if path.name == "checkpoint.json" and value["points"]:
            attempts += 1
            raise OSError("checkpoint disk failure")
        atomic_json(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(measurement, "atomic_json", fail_completed_checkpoint)
        with pytest.raises(OSError, match="checkpoint disk failure"):
            run(plan)
    assert attempts == 1
    checkpoint = json.loads((tmp_path / "cal/checkpoint.json").read_text())
    assert checkpoint["points"] == []
    assert run(plan, resume=True)["decode"]["grid"]


def test_measure_probe_error_does_not_trigger_checkpoint_write(
    tmp_path, calibration_run, monkeypatch
):
    run, plan, _ = calibration_run
    initial_written = False

    def save_initial_only(path, value):
        nonlocal initial_written
        if initial_written:
            raise OSError("secondary checkpoint failure")
        atomic_json(path, value)
        if path.name == "checkpoint.json":
            initial_written = True

    def fail_probe(*args):
        raise RuntimeError("original probe failure")

    monkeypatch.setattr(measurement, "atomic_json", save_initial_only)
    with pytest.raises(RuntimeError, match="original probe failure"):
        run(plan, submit=fail_probe)
    assert json.loads((tmp_path / "cal/checkpoint.json").read_text())["points"] == []


def test_atomic_json_interleaved_writes_use_distinct_temporaries(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.json"
    unrelated = tmp_path / "checkpoint.json.tmp"
    unrelated.write_text("unrelated")
    replace = Path.replace
    temporaries = []

    def interleaved_replace(source, target):
        temporaries.append(source)
        if len(temporaries) == 1:
            atomic_json(destination, {"writer": "second"})
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", interleaved_replace)
    atomic_json(destination, {"writer": "first"})
    assert len(set(temporaries)) == 2
    assert json.loads(destination.read_text()) == {"writer": "first"}
    assert unrelated.read_text() == "unrelated"
    assert set(tmp_path.iterdir()) == {destination, unrelated}


def test_atomic_json_replace_failure_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.json"
    atomic_json(destination, {"complete": 1})

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_json(destination, {"complete": 2})
    assert json.loads(destination.read_text()) == {"complete": 1}
    assert list(tmp_path.iterdir()) == [destination]
