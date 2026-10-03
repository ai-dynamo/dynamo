# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate (policy, cell, replicate) tasks through the result cache and the slot pool.

Each task resolves to a cache key (:mod:`learned_routing.cache`). A cached, error-free record is
returned as is; otherwise the task becomes a worker job. Duplicate keys within one call run once.
Every record that comes back is emitted to the optional ``results.jsonl`` with fresh cell labels
(split, family, ...), ``cached`` and ``run_id``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from learned_routing import HARNESS_VERSION
from learned_routing.cache import ResultCache, bindings_build_id, cache_key
from learned_routing.canon import canonical_json
from learned_routing.cells import Cell, cell_summary, resolve_replicate, source_sha256
from learned_routing.e0 import METHOD as E0_METHOD
from learned_routing.goodput import validate_measure
from learned_routing.paths import NUM_SLOTS, Layout
from learned_routing.policy import PolicySpec
from learned_routing.pool import EvalPool, Job, JobOutcome
from learned_routing.slots import SlotPool

AGENTIC_FORMATS = frozenset({"weka", "agentic_mooncake", "agentic-mooncake"})


@dataclass(frozen=True)
class Task:
    spec: PolicySpec
    cell: Cell
    k: int


@dataclass
class Planned:
    task: Task
    key: str
    identity: dict
    cached: dict | None
    job: Job | None


class Evaluator:
    def __init__(
        self,
        layout: Layout,
        *,
        concurrency: int = NUM_SLOTS,
        num_slots: int = NUM_SLOTS,
        keep_per_request: bool = True,
        timeout_s: float | None = None,
        timeout_mult: float = 20.0,
        min_timeout_s: float = 120.0,
        log_dir: Path | None = None,
        results_path: Path | None = None,
        results_fields: tuple[str, ...] | None = None,
        refresh: bool = False,
        verbose: bool = True,
        run_id: str | None = None,
        slots_dir: Path | None = None,
    ):
        self.layout = layout
        self.cache = ResultCache(layout.cache_dir)
        self.build = bindings_build_id(layout.cache_dir)
        self.keep_per_request = keep_per_request
        self.timeout_s = timeout_s
        self.timeout_mult = timeout_mult
        self.min_timeout_s = min_timeout_s
        self.results_path = Path(results_path) if results_path else None
        self.results_fields = results_fields
        self.refresh = refresh
        self.verbose = verbose
        self.run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        self.job_dir = layout.runs_dir / "tmp" / "jobs"
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = (
            Path(log_dir) if log_dir else layout.runs_dir / "logs" / "lr-workers"
        )
        self.pool = EvalPool(
            SlotPool(Path(slots_dir) if slots_dir else layout.slots_dir, num_slots),
            concurrency=concurrency,
            log_dir=self.log_dir,
            job_dir=self.job_dir,
            build_id=self.build["build_id"],
            build_check_dir=layout.cache_dir,
        )
        self._cell_sha: dict[str, str] = {}
        self._engines: dict[str, dict] = {}
        self.counts = {
            "done": 0,
            "cached": 0,
            "error": 0,
            "timeout": 0,
            "crashed": 0,
            "skipped": 0,
        }

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def cell_sha(self, cell: Cell) -> str:
        memo = canonical_json(cell.raw)
        if memo not in self._cell_sha:
            self._cell_sha[memo] = cell.content_sha()
        return self._cell_sha[memo]

    def _engine(self, cell: Cell) -> dict:
        path = str(cell.engine_path())
        if path not in self._engines:
            self._engines[path] = json.loads(Path(path).read_text())
        return self._engines[path]

    def plan(self, task: Task) -> Planned:
        spec, cell, k = task.spec, task.cell, task.k
        # A bad measurement rule fails here, before any replay (build audit r1 F1).
        validate_measure(cell.measure, open_loop=cell.is_open_loop)
        cell_sha = self.cell_sha(cell)
        rep = resolve_replicate(cell, k)
        key = cache_key(
            policy_sha=spec.sha,
            cell_id=cell.cell_id,
            cell_sha=cell_sha,
            repeat=k,
            protocol=rep.protocol,
            harness_version=HARNESS_VERSION,
            build_id=self.build["build_id"],
        )
        spec_path = spec.write(self.layout.policies_dir)
        policy_seed = spec.effective_seed(rep.policy_seed)
        yaml_path = spec.write_replay_yaml(self.layout.policies_dir, policy_seed)
        source = None if cell.is_synthetic else cell.trace_source()
        identity = {
            "cache_key": key,
            "policy_sha": spec.sha,
            "policy_name": spec.name,
            "policy_type": spec.policy_type,
            "router_mode": spec.router_mode,
            "policy_spec_path": str(spec_path),
            "policy_yaml": None if yaml_path is None else str(yaml_path),
            "cell_id": cell.cell_id,
            "cell_sha": cell_sha,
            "repeat": k,
            "replicate_protocol": rep.protocol,
            "replicate_seed": rep.replicate_seed,
            "policy_seed": policy_seed,
            "trace_sha256": rep.trace_sha256,
            "trace_content_sha256": (
                None if source is None else source_sha256(source, cell.trace_format)
            ),
            "replicate_degenerate": rep.degenerate,
            "harness_version": HARNESS_VERSION,
            "build_id": self.build["build_id"],
            "sla": cell.sla,
            "measure": cell.measure,
            "load_value": cell.load.get("value"),
        }
        cached = None if self.refresh else self.cache.get(key)
        if cached is not None:
            return Planned(task, key, identity, cached, None)
        engine = self._engine(cell)
        replay = {
            "kind": "synthetic" if cell.is_synthetic else "trace",
            "num_workers": cell.num_workers,
            "router_mode": spec.router_mode,
            "policy_yaml": identity["policy_yaml"],
            "router_config": dict(spec.router_config),
            "load_kwargs": cell.load_kwargs(source),
            "sla_ttft_ms": cell.sla["ttft_ms"],
            "sla_itl_ms": cell.sla["itl_ms"],
        }
        if cell.is_synthetic:
            replay.update(
                synthetic=cell.synthetic_spec(), arrival_seed=rep.arrival_seed
            )
        else:
            replay.update(
                trace_path=str(rep.trace_path),
                trace_format=cell.trace_format,
                trace_block_size=cell.raw.get("trace_block_size"),
                execution_model=engine.get("model")
                if cell.trace_format in AGENTIC_FORMATS
                else None,
                replay_options=cell.replay_options(),
            )
        payload = {
            "job_id": uuid.uuid4().hex,
            "per_request_path": (
                str(self.cache.per_request_path(key)) if self.keep_per_request else None
            ),
            "engine_json": str(cell.engine_path()),
            "engine_overrides": dict(cell.raw.get("engine_overrides") or {}),
            "replay": replay,
            "scoring": {
                "sla": cell.sla,
                "measure": cell.measure,
                "open_loop": cell.is_open_loop,
                "occupancy_cap": None if cell.is_open_loop else int(cell.load["value"]),
                "num_workers": cell.num_workers,
                "e0_cache_dir": str(self.layout.e0_dir),
                "e0_method": E0_METHOD,
            },
            "record": identity,
        }
        timeout = self.timeout_s or max(
            self.min_timeout_s, self.timeout_mult * cell.expected_cost_s(source)
        )
        identity["timeout_s"] = timeout
        return Planned(task, key, identity, None, Job(payload, timeout, label=key[:12]))

    def _emit(self, record: dict, cell: Cell, cached: bool) -> dict:
        out = dict(record)
        out.update(cell_summary(cell))
        out.update(cached=cached, run_id=self.run_id)
        if self.results_path is not None:
            line = out
            if self.results_fields:
                line = {key: out.get(key) for key in self.results_fields}
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("a") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        return out

    def evaluate(
        self, tasks: Sequence[Task], *, deadline: float | None = None
    ) -> list[dict | None]:
        """Records aligned with ``tasks``; ``None`` where the deadline skipped a task."""
        planned: list[Planned] = []
        errors: dict[int, dict] = {}
        for index, task in enumerate(tasks):
            try:
                planned.append(self.plan(task))
            except Exception as exc:  # a bad cell or spec becomes an error record
                record = {
                    "policy_sha": task.spec.sha,
                    "policy_name": task.spec.name,
                    "policy_type": task.spec.policy_type,
                    "cell_id": task.cell.cell_id,
                    "repeat": task.k,
                    "harness_version": HARNESS_VERSION,
                    "error": f"plan_error: {type(exc).__name__}: {exc}",
                }
                errors[index] = self._emit(record, task.cell, cached=False)
                self.counts["error"] += 1
                planned.append(None)  # type: ignore[arg-type]
        results: dict[str, dict | None] = {}
        jobs: list[Job] = []
        job_cells: dict[str, Cell] = {}
        for item in planned:
            if item is None or item.key in results or item.key in job_cells:
                continue
            if item.cached is not None:
                # The name is a label outside the cache key: report the caller's, not the one
                # the entry was first evaluated under.
                record = {**item.cached, "policy_name": item.task.spec.name}
                results[item.key] = self._emit(record, item.task.cell, cached=True)
                self.counts["cached"] += 1
            else:
                jobs.append(item.job)
                job_cells[item.key] = item.task.cell
        total = len(jobs)
        finished = 0
        started = time.monotonic()

        def on_outcome(outcome: JobOutcome) -> None:
            nonlocal finished
            key = outcome.job.payload["record"]["cache_key"]
            if outcome.status == "skipped":
                results[key] = None
                self.counts["skipped"] += 1
                return
            record = outcome.record
            if outcome.status == "done":
                self.cache.put(key, record)
            self.counts[outcome.status] += 1
            results[key] = self._emit(record, job_cells[key], cached=False)
            finished += 1
            if self.verbose and (finished % 20 == 0 or finished == total):
                elapsed = time.monotonic() - started
                print(
                    f"[lr-eval] {finished}/{total} replays in {elapsed:.0f} s "
                    f"({json.dumps(self.counts)})",
                    file=sys.stderr,
                    flush=True,
                )

        self.pool.run(jobs, deadline=deadline, on_outcome=on_outcome)
        aligned: list[dict | None] = []
        for index, item in enumerate(planned):
            aligned.append(errors[index] if item is None else results.get(item.key))
        return aligned
