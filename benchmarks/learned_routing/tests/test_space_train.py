# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest
from learned_routing.cells import Cell
from learned_routing.paths import Layout
from learned_routing.policy import default_reference
from learned_routing.space import Space, SpaceError
from learned_routing.train import (
    TrainConfig,
    Trainer,
    TrainError,
    fake_quadratic_evaluator,
    objective_value,
    pair_score,
)

SPACES = Path(__file__).resolve().parents[1] / "spaces"


def space(params, **extra):
    raw = {
        "base": {"type": "dynamo-default-cost-fn", "parameters": {}},
        "params": params,
    }
    raw.update(extra)
    return Space(raw)


# -- space transforms ---------------------------------------------------------------------
def test_linear_and_log_transforms_round_trip_and_clamp():
    s = space(
        [
            {
                "path": "parameters.overlap_score_credit",
                "bounds": [0.0, 4.0],
                "init": 1.0,
            },
            {
                "path": "parameters.prefill_load_scale",
                "bounds": [0.1, 10.0],
                "scale": "log",
                "init": 1.0,
            },
        ]
    )
    assert s.x0() == pytest.approx([0.25, 0.5])
    values = s.decode([0.25, 0.5])
    assert values["parameters.overlap_score_credit"] == pytest.approx(1.0)
    assert values["parameters.prefill_load_scale"] == pytest.approx(1.0)
    assert s.decode([1.0, 1.0])["parameters.prefill_load_scale"] == pytest.approx(10.0)
    clamped = s.decode([-0.3, 1.7])
    assert clamped["parameters.overlap_score_credit"] == 0.0
    assert clamped["parameters.prefill_load_scale"] == pytest.approx(10.0)
    assert s.cma_bounds() == [[0.0, 0.0], [1.0, 1.0]]


def test_integers_round_and_unbounded_coordinates_use_units():
    s = space(
        [
            {
                "path": "router_config.router_queue_threshold",
                "kind": "int",
                "bounds": [1, 9],
                "init": 5,
            },
            {
                "path": "parameters.overlap_score_credit",
                "bounds": None,
                "unit": 2.0,
                "init": 3.0,
            },
        ]
    )
    assert s.x0() == pytest.approx([0.5, 1.5])
    values = s.decode([0.56, 1.5])
    assert values["router_config.router_queue_threshold"] == 5 and isinstance(
        values["router_config.router_queue_threshold"], int
    )
    assert values["parameters.overlap_score_credit"] == 3.0
    assert s.cma_bounds() == [[0.0, None], [1.0, None]]
    built = s.spec(s.x0())
    assert built.router_config == {"router_queue_threshold": 5}


def test_vector_pins_lowrank_and_matrix_shapes():
    s = Space(
        {
            "base": {"type": "learned-choice", "parameters": {"feature_set": "v1"}},
            "fixed": {"parameters.temperature": 0.0},
            "params": [
                {
                    "path": "parameters.theta",
                    "kind": "vector",
                    "dim": 4,
                    "bounds": [[-5, 5], [-5, 5], [-5, -0.01], [-5, 5]],
                    "init": [-1, 0, -1, 0],
                    "fixed": {0: -1.0},
                },
                {
                    "path": "parameters.context",
                    "kind": "lowrank",
                    "dim": 4,
                    "rank": 2,
                    "bounds": [-1, 1],
                },
                {
                    "path": "parameters.extra",
                    "kind": "matrix",
                    "shape": [2, 3],
                    "bounds": [0, 1],
                    "init": 0.5,
                },
            ],
        }
    )
    assert s.dimension == 3 + 16 + 6
    spec = s.spec(s.x0())
    assert spec.parameters["theta"][0] == -1.0 and len(spec.parameters["theta"]) == 4
    assert spec.parameters["theta"][2] == pytest.approx(-1.0)
    context = spec.parameters["context"]
    assert (
        len(context["p"]) == 2 and len(context["q"]) == 2 and len(context["p"][0]) == 4
    )
    assert spec.parameters["extra"] == [[0.5] * 3] * 2
    assert spec.parameters["temperature"] == 0.0
    # sign constraint: element 2 can never become positive
    assert s.spec([1.0] * s.dimension).parameters["theta"][2] == pytest.approx(-0.01)


@pytest.mark.parametrize(
    "param, match",
    [
        ({"path": "parameters.x", "bounds": [0, 1], "init": 2.0}, "outside"),
        ({"path": "parameters.x", "bounds": [0, 1], "scale": "log"}, "log scale"),
        ({"path": "parameters.x", "bounds": [2, 1]}, "low <= high"),
        ({"path": "parameters.x", "kind": "tensor"}, "kind"),
        (
            {"path": "parameters.x", "kind": "vector", "dim": 2, "init": [1, 2, 3]},
            "init has",
        ),
    ],
)
def test_invalid_params_are_rejected(param, match):
    with pytest.raises(SpaceError, match=match):
        space([param])


def learned_theta_space(clamp=None, init=(-1.0, 0.0, 0.0, 0.0)):
    param = {
        "path": "parameters.theta",
        "kind": "vector",
        "dim": 4,
        "bounds": [[-1, -1], [-8, 8], [-4, 4], [-2, 2]],
        "init": list(init),
        "fixed": {0: -1.0},
    }
    if clamp is not None:
        param["clamp"] = clamp
    return Space(
        {
            "base": {"type": "learned-choice", "parameters": {"feature_set": "v1"}},
            "params": [param],
            "cma": {"sigma0": 0.05},
        }
    )


def test_clamp_keeps_the_internal_search_and_only_clips_the_decoded_policy():
    plain = learned_theta_space()
    clamped = learned_theta_space(clamp=[None, None, [None, 0.0], [None, 0.0]])
    # Same internal start and box: a start on the constraint stays in the interior of the box.
    assert clamped.x0() == plain.x0() == pytest.approx([0.5, 0.5, 0.5])
    assert clamped.cma_bounds() == plain.cma_bounds()
    assert clamped.spec(clamped.x0()).sha == plain.spec(plain.x0()).sha
    assert clamped.sha != plain.sha
    z = [0.75, 0.9, 0.2]
    assert plain.spec(z).parameters["theta"] == pytest.approx([-1.0, 4.0, 3.2, -1.2])
    assert clamped.spec(z).parameters["theta"] == pytest.approx([-1.0, 4.0, 0.0, -1.2])
    # One pair applies to every free element; the pinned anchor is never clipped.
    every = learned_theta_space(clamp=[None, 0.0])
    assert every.spec([1.0, 1.0, 1.0]).parameters["theta"] == [-1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "clamp, init, match",
    [
        ([None, 0.0], (-1.0, 0.5, 0.0, 0.0), "outside clamp"),
        ([1.0, 0.0], (-1.0, 0.0, 0.0, 0.0), "low <= high"),
        ([[None, 0.0], [None, 0.0]], (-1.0, 0.0, 0.0, 0.0), "per-element"),
        ([None, None, 0.0, None], (-1.0, 0.0, 0.0, 0.0), r"\[low, high\]"),
    ],
)
def test_invalid_clamps_are_rejected(clamp, init, match):
    with pytest.raises(SpaceError, match=match):
        learned_theta_space(clamp=clamp, init=init)


def test_clamped_and_plain_runs_ask_the_same_first_generation(tmp_path):
    # Same seed, start and sigma: CMA-ES proposes identical internal points in generation 0,
    # and the clamped run's policies are the plain run's with theta[2], theta[3] clipped at 0.
    histories = {}
    for name, clamp in (
        ("plain", None),
        ("clamped", [None, None, [None, 0.0], [None, 0.0]]),
    ):
        run = tmp_path / name
        Trainer(
            learned_theta_space(clamp=clamp),
            cells(1, tmp_path),
            [],
            run,
            fake_quadratic_evaluator({"parameters.overlap_score_credit": 1.0}),
            default_reference(),
            config(budget_evals=6, popsize=6, val_every=0),
        ).run()
        histories[name] = gen_lines(run)[0]
    plain, clamped = histories["plain"], histories["clamped"]
    assert [c["z"] for c in plain["candidates"]] == [
        c["z"] for c in clamped["candidates"]
    ]
    space_plain = learned_theta_space()
    for cand in clamped["candidates"]:
        theta = space_plain.spec(cand["z"]).parameters["theta"]
        clipped = theta[:2] + [min(v, 0.0) for v in theta[2:]]
        assert learned_theta_space(clamp=[None, None, [None, 0.0], [None, 0.0]]).spec(
            cand["z"]
        ).parameters["theta"] == pytest.approx(clipped)


def test_example_spaces_load():
    for path in sorted(SPACES.glob("*.yaml")):
        loaded = Space.load(path)
        assert loaded.dimension >= 1, path
        loaded.spec(loaded.x0())
        assert len(loaded.sha) == 64, path


def test_space_sha_reads_pinned_keys_as_element_indices():
    def pinned(keys):
        return Space(
            {
                "base": {"name": "lc", "type": "learned-choice", "parameters": {}},
                "params": [
                    {
                        "path": "parameters.theta",
                        "kind": "vector",
                        "dim": 3,
                        "bounds": [-2.0, 2.0],
                        "init": [-1.0, 0.0, 0.0],
                        "fixed": keys,
                    }
                ],
            }
        )

    assert pinned({0: -1.0}).sha == pinned({"0": -1.0}).sha
    assert pinned({0: -1.0}).sha != pinned({1: 0.0}).sha


# -- objective ----------------------------------------------------------------------------
def test_pair_scores_and_crn_assertion():
    assert pair_score(2.0, 1.0, "ratio", 0.0) == 2.0
    assert pair_score(100.0, 1.0, "clipped_log_ratio", 0.0) == pytest.approx(
        math.log(3)
    )
    rec = {("a", 0): {"m": 2.0}, ("a", 1): {"m": 1.0}, ("b", 0): {"m": 3.0}}
    ref = {("a", 0): {"m": 1.0}, ("a", 1): {"m": 1.0}, ("b", 0): {"m": 1.0}}
    value, per_cell, error = objective_value(
        rec, ref, objective="ratio", metric="m", eps=0.0
    )
    assert (
        per_cell == {"a": 1.5, "b": 3.0}
        and value == pytest.approx(2.25)
        and error is None
    )
    with pytest.raises(TrainError, match="CRN"):
        objective_value(
            rec, {("a", 0): {"m": 1.0}}, objective="ratio", metric="m", eps=0.0
        )
    bad = dict(rec)
    bad[("b", 0)] = {"m": 1.0, "error": "boom"}
    assert objective_value(bad, ref, objective="ratio", metric="m", eps=0.0)[0] is None


# -- trainer ------------------------------------------------------------------------------
def cells(n, tmp_path):
    layout = Layout(tmp_path)
    return [
        Cell(
            raw={
                "cell_id": f"c{i}",
                "num_workers": 4,
                "load": {"mode": "open_speedup", "value": 1.0},
            },
            layout=layout,
        )
        for i in range(n)
    ]


TARGET = {"parameters.overlap_score_credit": 2.5, "parameters.prefill_load_scale": 3.0}


def two_param_space():
    return space(
        [
            {
                "path": "parameters.overlap_score_credit",
                "bounds": [0.0, 4.0],
                "init": 1.0,
            },
            {
                "path": "parameters.prefill_load_scale",
                "bounds": [0.1, 10.0],
                "scale": "log",
                "init": 1.0,
            },
        ],
        cma={"sigma0": 0.25},
    )


def config(**overrides):
    base = dict(
        budget_evals=48,
        popsize=6,
        seed=7,
        replicates=2,
        replicate_pool=4,
        val_every=2,
        val_replicates=2,
    )
    base.update(overrides)
    return TrainConfig(**base)


def gen_lines(run_dir):
    lines = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text().splitlines()
    ]
    drop = {"gen_wall_s"}
    return [{k: v for k, v in line.items() if k not in drop} for line in lines]


def test_trainer_converges_on_fake_objective_and_writes_outputs(tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(
        two_param_space(),
        cells(2, tmp_path),
        cells(1, tmp_path),
        run,
        fake_quadratic_evaluator(TARGET),
        default_reference(),
        config(budget_evals=120),
    )
    status = trainer.run()
    assert status["phase"] == "done" and status["fevals"] == 120
    best = json.loads((run / "best.json").read_text())
    values = best["best_ever_train"]["values"]
    assert values["parameters.overlap_score_credit"] == pytest.approx(2.5, abs=0.3)
    assert values["parameters.prefill_load_scale"] == pytest.approx(3.0, abs=0.5)
    assert best["selected_by_val"]["val_objective"] > 1.5
    assert 'type: "dynamo-default-cost-fn"' in (run / "best_policy.yaml").read_text()


class Interrupting:
    """Evaluator that behaves like a hit deadline (returns None) after ``calls`` calls."""

    def __init__(self, inner, calls):
        self.inner, self.calls = inner, calls

    def __call__(self, tasks, deadline):
        if self.calls <= 0:
            return [None] * len(tasks)
        self.calls -= 1
        return self.inner(tasks, deadline)


@pytest.mark.parametrize("cut_after", [1, 2, 3, 4, 5, 7])
def test_resume_after_interruption_matches_an_uninterrupted_run(tmp_path, cut_after):
    straight = tmp_path / "straight"
    Trainer(
        two_param_space(),
        cells(2, tmp_path),
        cells(1, tmp_path),
        straight,
        fake_quadratic_evaluator(TARGET, noise=0.05),
        default_reference(),
        config(),
    ).run()

    resumed = tmp_path / "resumed"
    first = Trainer(
        two_param_space(),
        cells(2, tmp_path),
        cells(1, tmp_path),
        resumed,
        Interrupting(fake_quadratic_evaluator(TARGET, noise=0.05), cut_after),
        default_reference(),
        config(),
    )
    paused = first.run()
    assert paused["phase"] == "paused"
    second = Trainer(
        two_param_space(),
        cells(2, tmp_path),
        cells(1, tmp_path),
        resumed,
        fake_quadratic_evaluator(TARGET, noise=0.05),
        default_reference(),
        config(),
    )
    done = second.run()
    assert done["phase"] == "done" and done["fevals"] == 48
    assert gen_lines(resumed) == gen_lines(straight)
    a = json.loads((straight / "best.json").read_text())
    b = json.loads((resumed / "best.json").read_text())
    assert (
        a["best_ever_train"] == b["best_ever_train"]
        and a["selected_by_val"] == b["selected_by_val"]
    )


def test_every_generation_uses_one_shared_crn_set(tmp_path):
    seen = []
    inner = fake_quadratic_evaluator(TARGET)

    def spy(tasks, deadline):
        by_spec = defaultdict(set)
        for task in tasks:
            by_spec[task.spec.sha].add((task.cell.cell_id, task.k))
        seen.append(by_spec)
        return inner(tasks, deadline)

    Trainer(
        two_param_space(),
        cells(3, tmp_path),
        [],
        tmp_path / "run",
        spy,
        default_reference(),
        config(budget_evals=24, cells_per_gen=2),
    ).run()
    assert len(seen) == 4
    ks_by_gen = []
    for by_spec in seen:
        sets = list(by_spec.values())
        assert all(
            s == sets[0] for s in sets
        )  # reference and every candidate: same (cell, k)
        assert default_reference().sha in by_spec
        assert len({cell for cell, _ in sets[0]}) == 2
        ks_by_gen.append(sorted({k for _, k in sets[0]}))
    assert ks_by_gen == [
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3],
    ]  # rotation through the replicate pool


def test_resume_refuses_a_changed_setup(tmp_path):
    run = tmp_path / "run"
    Trainer(
        two_param_space(),
        cells(2, tmp_path),
        [],
        run,
        fake_quadratic_evaluator(TARGET),
        default_reference(),
        config(budget_evals=12),
    ).run()
    with pytest.raises(TrainError, match="seed"):
        Trainer(
            two_param_space(),
            cells(2, tmp_path),
            [],
            run,
            fake_quadratic_evaluator(TARGET),
            default_reference(),
            config(budget_evals=12, seed=8),
        ).run()


def test_one_dimensional_space_trains(tmp_path):
    s = space(
        [{"path": "parameters.overlap_score_credit", "bounds": [0.0, 4.0], "init": 1.0}]
    )
    status = Trainer(
        s,
        cells(1, tmp_path),
        [],
        tmp_path / "run",
        fake_quadratic_evaluator({"parameters.overlap_score_credit": 3.0}),
        default_reference(),
        config(budget_evals=60),
    ).run()
    best = json.loads((tmp_path / "run" / "best.json").read_text())["best_ever_train"]
    assert status["phase"] == "done"
    assert best["values"]["parameters.overlap_score_credit"] == pytest.approx(
        3.0, abs=0.3
    )
    assert len(best["z"]) == 1


def test_zero_wall_budget_still_progresses_one_generation_per_call(tmp_path):
    # Regression: startup cost (checkpoint load) used to exhaust a short --max-wall-seconds
    # before any evaluation, so repeated calls never progressed.
    straight = tmp_path / "straight"
    Trainer(
        two_param_space(),
        cells(2, tmp_path),
        [],
        straight,
        fake_quadratic_evaluator(TARGET),
        default_reference(),
        config(val_every=0),
    ).run()
    chunked = tmp_path / "chunked"
    calls = 0
    while True:
        calls += 1
        status = Trainer(
            two_param_space(),
            cells(2, tmp_path),
            [],
            chunked,
            fake_quadratic_evaluator(TARGET),
            default_reference(),
            config(val_every=0, max_wall_seconds=0.0),
        ).run()
        if status["phase"] == "done":
            break
        assert status["generation"] == calls  # exactly one generation per call
        assert calls < 50
    assert gen_lines(chunked) == gen_lines(straight)


def test_reeval_diagnostic_uses_fresh_replicates_and_counts_its_tasks(tmp_path):
    seen_ks = []
    inner = fake_quadratic_evaluator(TARGET, noise=0.2)

    def spy(tasks, deadline):
        seen_ks.append(sorted({t.k for t in tasks}))
        return inner(tasks, deadline)

    run = tmp_path / "run"
    status = Trainer(
        two_param_space(),
        cells(1, tmp_path),
        [],
        run,
        spy,
        default_reference(),
        config(budget_evals=6, val_every=0, reeval_frac=0.5),
    ).run()
    gen = [
        json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()
    ][0]
    assert seen_ks == [[0, 1], [2, 3]]  # main evaluation, then fresh replicates
    assert gen["rank_change"]["candidates"] == 3
    # (reference + 6 candidates) x 2 replicates, plus (reference + 3) x 2 for the re-evaluation
    assert status["tasks_requested"] == 14 + 8
