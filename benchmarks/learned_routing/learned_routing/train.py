# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CMA-ES over a :class:`~learned_routing.space.Space`, scored by normalized replay goodput.

**Objective (contract, A2.3).** For candidate c and evaluation set E = {(cell, k)}:

- per (cell, k): ``r = (m_c + eps) / (m_ref + eps)``, where ``m`` is ``--metric`` (default the A2
  windowed goodput ``goodput_rps_window``) and ``ref`` is the reference policy (default
  ``default@defaults``) on the same cell and replicate k (same workload, same policy seed);
- ``ratio`` objective: mean over cells of the mean over k of ``r`` (the contract's mean normalized
  goodput); ``clipped_log_ratio`` (LR-01): the same with ``clip(ln r, -ln 3, ln 3)``.

CMA-ES minimizes ``-objective``. Reference results come from the cache when present.

**Common random numbers (LR-03, A1).** Each generation draws one evaluation set: replicate
indices ``k = (g K + j) mod P`` for ``j < K`` (``--replicates K``, ``--replicate-pool P``) and,
with ``--cells-per-gen``, a seeded subset of train cells. Every candidate and the reference are
evaluated on exactly that set; the driver asserts it before scoring.

**Resumability.** ``DIR/checkpoint.pkl`` is written twice per generation: after ``ask()`` (the
pending candidates, so an interrupted generation re-evaluates the same points, mostly from the
cache) and after ``tell()``. It stores the pycma object and numpy's global RNG state, which pycma
samples from, so a resumed run continues exactly as an uninterrupted one.

**Outputs.** ``history.jsonl`` (one ``gen`` line per generation, ``val`` lines every
``--val-every`` generations for the best-so-far sample and the distribution mean),
``best.json`` (best-ever train sample and the validation-selected candidate, LR-10),
``best_policy.yaml`` (the selected candidate as a harness policy spec) and ``status.json``.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from learned_routing.canon import atomic_write_bytes, atomic_write_text, canonical_yaml
from learned_routing.cells import Cell
from learned_routing.evaluate import Task
from learned_routing.policy import PolicySpec
from learned_routing.space import Space

OBJECTIVES = ("ratio", "clipped_log_ratio")
LOG_CLIP = math.log(3.0)
CHECKPOINT_VERSION = 1


class TrainError(RuntimeError):
    pass


EvaluateFn = Callable[[Sequence[Task], float | None], list]


@dataclass
class TrainConfig:
    budget_evals: int
    popsize: int | None = None
    seed: int = 1
    replicates: int = 2
    replicate_pool: int = 8
    cells_per_gen: int | None = None
    val_every: int = 5
    val_replicates: int = 3
    objective: str = "ratio"
    metric: str = "goodput_rps_window"
    eps: float = 1e-3
    max_wall_seconds: float | None = None
    reeval_frac: float = 0.0

    def validate(self) -> None:
        if self.objective not in OBJECTIVES:
            raise TrainError(f"objective must be one of {OBJECTIVES}")
        if self.replicates < 1 or self.replicate_pool < self.replicates:
            raise TrainError("need 1 <= replicates <= replicate_pool")
        if self.budget_evals < 1:
            raise TrainError("budget_evals must be >= 1")


def pair_score(m: float, m_ref: float, objective: str, eps: float) -> float:
    ratio = (m + eps) / (m_ref + eps)
    if objective == "ratio":
        return ratio
    return min(max(math.log(ratio), -LOG_CLIP), LOG_CLIP)


def objective_value(
    records: dict[tuple[str, int], dict],
    reference: dict[tuple[str, int], dict],
    *,
    objective: str,
    metric: str,
    eps: float,
) -> tuple[float | None, dict, str | None]:
    """(objective, per-cell scores, error). Requires identical (cell, k) coverage (CRN)."""
    if set(records) != set(reference):
        diff = sorted(set(records) ^ set(reference))[:5]
        raise TrainError(
            f"CRN violation: candidate and reference cover different (cell, k), e.g. {diff}"
        )
    per_cell: dict[str, list[float]] = {}
    for key, record in records.items():
        ref = reference[key]
        if record.get("error"):
            return None, {}, f"candidate error on {key}: {record['error']}"
        if ref.get("error"):
            return None, {}, f"reference error on {key}: {ref['error']}"
        m, m_ref = record.get(metric), ref.get(metric)
        if m is None or m_ref is None:
            return None, {}, f"metric {metric} missing on {key}"
        per_cell.setdefault(key[0], []).append(pair_score(m, m_ref, objective, eps))
    cells = {cell: sum(v) / len(v) for cell, v in per_cell.items()}
    return sum(cells.values()) / len(cells), cells, None


class Trainer:
    def __init__(
        self,
        space: Space,
        train_cells: Sequence[Cell],
        val_cells: Sequence[Cell],
        run_dir: Path,
        evaluate: EvaluateFn,
        reference: PolicySpec,
        config: TrainConfig,
    ):
        config.validate()
        if not train_cells:
            raise TrainError("no train cells")
        self.space = space
        self.train_cells = list(train_cells)
        self.val_cells = list(val_cells)
        self.run_dir = Path(run_dir)
        self.evaluate_fn = evaluate
        self.reference = reference
        self.config = config
        if space.dimension < 1:
            raise TrainError("the space has no free coordinates")
        self.pad = 1 if space.dimension == 1 else 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.run_dir / "checkpoint.pkl"
        self.history_path = self.run_dir / "history.jsonl"
        self.identity = {
            "space_sha": space.sha,
            "train_cells": sorted(c.cell_id for c in self.train_cells),
            "val_cells": sorted(c.cell_id for c in self.val_cells),
            "reference_sha": reference.sha,
            "objective": config.objective,
            "metric": config.metric,
            "eps": config.eps,
            "replicates": config.replicates,
            "replicate_pool": config.replicate_pool,
            "cells_per_gen": config.cells_per_gen,
            "seed": config.seed,
            "popsize": config.popsize,
            "val_replicates": config.val_replicates,
        }

    def _unpad(self, z) -> list[float]:
        z = [float(v) for v in z]
        return z[: len(z) - self.pad] if self.pad else z

    # -- state -----------------------------------------------------------------------------
    def _new_state(self) -> dict:
        import cma
        import numpy as np

        options = {
            "seed": self.config.seed,
            "bounds": self.space.cma_bounds(),
            "verbose": -9,
            "tolflatfitness": 10,
            "tolfun": 0,
            "tolfunhist": 0,
            "tolx": 1e-6,
        }
        if self.config.popsize:
            options["popsize"] = self.config.popsize
        stds = self.space.cma_stds()
        if stds is not None:
            options["CMA_stds"] = stds
        options.update(self.space.cma_options)
        x0 = self.space.x0() + [0.5] * self.pad
        if self.pad:
            # pycma does not support 1-D problems; a dummy bounded coordinate is ignored on decode.
            lower, upper = options["bounds"]
            options["bounds"] = [lower + [0.0], upper + [1.0]]
            if "CMA_stds" in options:
                options["CMA_stds"] = list(options["CMA_stds"]) + [1.0]
        es = cma.CMAEvolutionStrategy(x0, self.space.sigma0, options)
        return {
            "version": CHECKPOINT_VERSION,
            "identity": self.identity,
            "es": es,
            "generation": 0,
            "fevals": 0,
            "replays_fresh": 0,
            "tasks_requested": 0,
            "wall_s": 0.0,
            "best": None,
            "selected": None,
            "pending": None,
            "val_due": False,
            "np_random_state": np.random.get_state(),
        }

    def _load_state(self) -> dict:
        if not self.checkpoint_path.exists():
            state = self._new_state()
            atomic_write_text(
                self.run_dir / "identity.json",
                json.dumps(self.identity, indent=1, sort_keys=True),
            )
            return state
        state = pickle.loads(self.checkpoint_path.read_bytes())
        if state.get("version") != CHECKPOINT_VERSION:
            raise TrainError(
                f"checkpoint version {state.get('version')} != {CHECKPOINT_VERSION}"
            )
        if state["identity"] != self.identity:
            changed = sorted(
                k for k in self.identity if state["identity"].get(k) != self.identity[k]
            )
            raise TrainError(
                f"resume mismatch: {changed} differ from {self.checkpoint_path}"
            )
        import numpy as np

        np.random.set_state(state["np_random_state"])
        return state

    def _save_state(self, state: dict) -> None:
        import numpy as np

        state["np_random_state"] = np.random.get_state()
        atomic_write_bytes(self.checkpoint_path, pickle.dumps(state))

    def _history(self, line: dict) -> None:
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    # -- evaluation ------------------------------------------------------------------------
    def eval_set(self, generation: int) -> tuple[list[Cell], list[int]]:
        cfg = self.config
        ks = [
            (generation * cfg.replicates + j) % cfg.replicate_pool
            for j in range(cfg.replicates)
        ]
        cells = self.train_cells
        if cfg.cells_per_gen and cfg.cells_per_gen < len(cells):
            rng = random.Random(f"{cfg.seed}|cells|{generation}")
            cells = sorted(
                rng.sample(cells, cfg.cells_per_gen), key=lambda c: c.cell_id
            )
        return cells, ks

    def score(
        self,
        specs: Sequence[PolicySpec],
        cells: Sequence[Cell],
        ks: Sequence[int],
        deadline: float | None,
    ) -> tuple[list[tuple[float | None, dict, str | None]], int] | None:
        """Objectives for ``specs`` on cells x ks; None if the deadline cut the evaluation."""
        everyone = [self.reference, *specs]
        tasks = [Task(spec, cell, k) for cell in cells for k in ks for spec in everyone]
        records = self.evaluate_fn(tasks, deadline)
        if any(record is None for record in records):
            return None
        by_spec: dict[int, dict] = {i: {} for i in range(len(everyone))}
        width = len(everyone)
        for index, record in enumerate(records):
            task = tasks[index]
            got = (
                record.get("cell_id"),
                record.get("repeat"),
                record.get("policy_sha"),
            )
            if got != (task.cell.cell_id, task.k, task.spec.sha):
                raise TrainError(
                    f"CRN violation: task {index} expected "
                    f"{(task.cell.cell_id, task.k, task.spec.sha[:12])}, got {got}"
                )
            by_spec[index % width][(task.cell.cell_id, task.k)] = record
        reference = by_spec[0]
        ref_errors = sorted(
            key for key, record in reference.items() if record.get("error")
        )
        if ref_errors:
            raise TrainError(
                f"reference {self.reference.name} errored on {ref_errors[:3]}: "
                f"{reference[ref_errors[0]]['error']}; rerun to retry (errors are not cached)"
            )
        out = [
            objective_value(
                by_spec[i],
                reference,
                objective=self.config.objective,
                metric=self.config.metric,
                eps=self.config.eps,
            )
            for i in range(1, width)
        ]
        fresh = sum(1 for r in records if r is not None and not r.get("cached"))
        self._tasks_last = len(tasks)
        return out, fresh

    # -- main loop -------------------------------------------------------------------------
    def candidate_record(
        self, z: Sequence[float], objective: float | None, generation: int, **extra
    ) -> dict:
        spec = self.space.spec(list(z))
        return {
            "generation": generation,
            "z": [float(v) for v in z],
            "values": self.space.decode(list(z)),
            "objective": objective,
            "policy_sha": spec.sha,
            "spec": spec.to_dict(),
            **extra,
        }

    def _write_best(self, state: dict) -> None:
        payload = {
            "identity": self.identity,
            "generation": state["generation"],
            "fevals": state["fevals"],
            "best_ever_train": state["best"],
            "selected_by_val": state["selected"],
            "selection_rule": (
                "argmax validation objective over validated candidates (best-so-far train sample "
                "and CMA distribution mean every --val-every generations); falls back to the "
                "best-ever train sample until a validation exists (LR-10)"
            ),
        }
        atomic_write_text(
            self.run_dir / "best.json", json.dumps(payload, indent=1, sort_keys=True)
        )
        chosen = state["selected"] or state["best"]
        if chosen is not None:
            spec = dict(chosen["spec"])
            spec[
                "name"
            ] = f"{self.space.base.get('name', spec.get('type'))}@{self.run_dir.name}"
            atomic_write_text(self.run_dir / "best_policy.yaml", canonical_yaml(spec))

    def _status(self, state: dict, phase: str, **extra) -> dict:
        status = {
            "phase": phase,
            "generation": state["generation"],
            "fevals": state["fevals"],
            "budget_evals": self.config.budget_evals,
            "tasks_requested": state.get("tasks_requested", 0),
            "replays_fresh": state["replays_fresh"],
            "wall_s": round(state["wall_s"], 2),
            "best_train_objective": (state["best"] or {}).get("objective"),
            "selected_val_objective": (state["selected"] or {}).get("val_objective"),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **extra,
        }
        atomic_write_text(
            self.run_dir / "status.json", json.dumps(status, indent=1, sort_keys=True)
        )
        return status

    def _validate(self, state: dict, deadline: float | None) -> bool:
        if not self.val_cells:
            return True
        es = state["es"]
        candidates = []
        if state["best"] is not None:
            candidates.append(("best_so_far", state["best"]["z"]))
        candidates.append(("mean", self._unpad(es.result.xfavorite)))
        specs = [self.space.spec(z) for _, z in candidates]
        ks = list(range(self.config.val_replicates))
        result = self.score(specs, self.val_cells, ks, deadline)
        if result is None:
            return False
        scores, fresh = result
        state["replays_fresh"] += fresh
        state["tasks_requested"] = state.get("tasks_requested", 0) + self._tasks_last
        for (which, z), (value, per_cell, error) in zip(candidates, scores):
            line = self.candidate_record(
                z, None, state["generation"], kind="val", which=which
            )
            line.update(
                val_objective=value, val_per_cell=per_cell, error=error, val_ks=ks
            )
            self._history(line)
            if value is None:
                continue
            if state["selected"] is None or value > state["selected"]["val_objective"]:
                state["selected"] = {
                    k: line[k]
                    for k in (
                        "generation",
                        "z",
                        "values",
                        "policy_sha",
                        "spec",
                        "which",
                    )
                } | {"val_objective": value, "val_per_cell": per_cell}
        return True

    def run(self) -> dict:
        cfg = self.config
        state = self._load_state()
        # The clock starts after the checkpoint load, and every call attempts at least one
        # evaluation, so a short --max-wall-seconds can never livelock on startup cost.
        started = time.monotonic()
        deadline = (
            None if cfg.max_wall_seconds is None else started + cfg.max_wall_seconds
        )
        es = state["es"]
        base_wall = state["wall_s"]
        progressed = False

        def elapsed() -> float:
            return base_wall + (time.monotonic() - started)

        while True:
            if state.get("val_due"):
                # Owed by the last completed generation (also after a resume mid-validation).
                if not self._validate(state, deadline):
                    state["wall_s"] = elapsed()
                    self._save_state(state)
                    self._write_best(state)
                    return self._status(
                        state, "paused", reason="max_wall_seconds (validation)"
                    )
                state["val_due"] = False
                state["wall_s"] = elapsed()
                self._save_state(state)
                self._write_best(state)
            lam = es.popsize
            if state["pending"] is None and state["fevals"] + lam > cfg.budget_evals:
                state["wall_s"] = elapsed()
                self._save_state(state)
                self._write_best(state)
                return self._status(state, "done", reason="budget_evals")
            if state["pending"] is None and es.stop():
                state["wall_s"] = elapsed()
                self._save_state(state)
                self._write_best(state)
                return self._status(state, "done", reason=f"cma_stop:{dict(es.stop())}")
            if progressed and deadline is not None and time.monotonic() >= deadline:
                state["wall_s"] = elapsed()
                self._save_state(state)
                return self._status(state, "paused", reason="max_wall_seconds")
            progressed = True
            if state["pending"] is None:
                state["pending"] = [list(map(float, z)) for z in es.ask()]
                self._save_state(state)
            pending = state["pending"]
            generation = state["generation"]
            cells, ks = self.eval_set(generation)
            gen_started = time.monotonic()
            specs = [self.space.spec(self._unpad(z)) for z in pending]
            result = self.score(specs, cells, ks, deadline)
            if result is None:
                state["wall_s"] = elapsed()
                self._save_state(state)
                return self._status(
                    state, "paused", reason="max_wall_seconds (mid-generation)"
                )
            scores, fresh = result
            values = [s[0] for s in scores]
            finite = [v for v in values if v is not None]
            worst = min(finite) - 1.0 if finite else -1e3
            fitness = [-(v if v is not None else worst) for v in values]
            es.tell(pending, fitness)
            state["fevals"] += len(pending)
            state["replays_fresh"] += fresh
            state["tasks_requested"] = (
                state.get("tasks_requested", 0) + self._tasks_last
            )
            for z, (value, per_cell, error) in zip(pending, scores):
                if value is None:
                    continue
                if state["best"] is None or value > state["best"]["objective"]:
                    state["best"] = self.candidate_record(
                        self._unpad(z), value, generation, per_cell=per_cell
                    )
            rank_change = None
            if cfg.reeval_frac > 0 and finite:
                rank_change = self._reeval(pending, values, generation, deadline)
                if rank_change is not None:
                    state["tasks_requested"] += rank_change["tasks"]
            self._history(
                {
                    "kind": "gen",
                    "generation": generation,
                    "fevals": state["fevals"],
                    "tasks_requested": state["tasks_requested"],
                    "cells": [c.cell_id for c in cells],
                    "ks": ks,
                    "objectives": values,
                    "candidates": [
                        {"z": self._unpad(z), "policy_sha": spec.sha, "objective": v}
                        for z, spec, v in zip(pending, specs, values)
                    ],
                    "errors": [s[2] for s in scores if s[2]],
                    "best_gen": max(finite) if finite else None,
                    "mean_gen": sum(finite) / len(finite) if finite else None,
                    "best_so_far": state["best"]["objective"]
                    if state["best"]
                    else None,
                    "sigma": float(es.sigma),
                    "axis_ratio": float(es.D.max() / es.D.min())
                    if hasattr(es, "D")
                    else None,
                    "mean_values": self.space.decode(self._unpad(es.result.xfavorite)),
                    "gen_wall_s": time.monotonic() - gen_started,
                    "rank_change": rank_change,
                }
            )
            state["generation"] = generation + 1
            state["pending"] = None
            state["val_due"] = (
                bool(cfg.val_every) and state["generation"] % cfg.val_every == 0
            )
            state["wall_s"] = elapsed()
            self._save_state(state)
            self._write_best(state)
            self._status(state, "running")

    def _reeval(self, pending, values, generation: int, deadline) -> dict | None:
        """UH-CMA-ES-style diagnostic (LR-03): re-score a fraction on fresh replicates."""
        cfg = self.config
        count = max(2, math.ceil(cfg.reeval_frac * len(pending)))
        order = sorted(
            range(len(pending)),
            key=lambda i: -(values[i] if values[i] is not None else -1e9),
        )
        chosen = order[:count]
        cells, _ = self.eval_set(generation)
        fresh_ks = [
            (generation * cfg.replicates + cfg.replicates + j) % cfg.replicate_pool
            for j in range(cfg.replicates)
        ]
        result = self.score(
            [self.space.spec(self._unpad(pending[i])) for i in chosen],
            cells,
            fresh_ks,
            deadline,
        )
        if result is None:
            return None
        new_values = [s[0] for s in result[0]]
        first = sorted(range(count), key=lambda i: -(values[chosen[i]] or -1e9))
        second = sorted(
            range(count),
            key=lambda i: -(new_values[i] if new_values[i] is not None else -1e9),
        )
        moved = sum(a != b for a, b in zip(first, second))
        return {
            "candidates": count,
            "ks": fresh_ks,
            "rank_positions_changed": moved,
            "values": new_values,
            "tasks": self._tasks_last,
        }


# -- fake evaluator (tests and the lr-train smoke without replays) --------------------------
def fake_quadratic_evaluator(
    target: dict[str, float], noise: float = 0.0
) -> EvaluateFn:
    """Records whose ``goodput_rps_window`` peaks at ``target`` (dotted path -> value).

    The reference (no parameters at the target paths) scores 1.0; a candidate scores
    ``2 exp(-sum (x - target)^2)``, so the normalized objective is maximal (2.0) at the target.
    """

    def value_at(spec: PolicySpec, path: str):
        node = {"parameters": spec.parameters, "router_config": spec.router_config}
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def evaluate(tasks, deadline):
        out = []
        for task in tasks:
            xs = [value_at(task.spec, p) for p in target]
            if all(x is None for x in xs):
                m = 1.0
            else:
                dist = sum(
                    (float(x or 0.0) - t) ** 2 for x, t in zip(xs, target.values())
                )
                m = 2.0 * math.exp(-dist)
            if noise:
                rng = random.Random(f"{task.cell.cell_id}|{task.k}|{task.spec.sha}")
                m *= 1.0 + noise * rng.uniform(-1, 1)
            out.append(
                {
                    "policy_sha": task.spec.sha,
                    "cell_id": task.cell.cell_id,
                    "repeat": task.k,
                    "goodput_rps_window": m,
                    "error": None,
                    "cached": False,
                }
            )
        return out

    return evaluate
