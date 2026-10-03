# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cell specs (CONTRACT "Cell spec"): loading, content identity, replay arguments.

A cell's **content SHA** covers everything that changes what a replay of it does or how it is
scored, and nothing that only locates or labels it:

- dropped: split labels (``split``, ``holdout_axis``, ``holdout_axes``, ``selection_exposed``),
  ``notes``, ``segment``, and locations (``trace_files``, ``engine_ref``, ``derived_meta``), which
  differ between the campaign root and a remote bundle;
- dropped: workload-generator metadata that never reaches replay or scoring
  (``expected_cost_s``, which only sets the timeout; the trace-time hints ``measure_trace``,
  ``trace_rows``, ``cache_pressure_ref``; the label ``transform_tag``);
- added: the SHA-256 of each trace source's content and of the engine's ``mock_engine_args`` and
  model.

So moving a cell between splits or into a bundle keeps its cache entries, while changing its
trace, load, SLA, measurement window or engine invalidates them.

Harness extensions to the contract cell (all optional):

- ``synthetic``: ``run_synthetic_trace_replay`` arguments (``input_tokens``, ``output_tokens``,
  ``request_count``, ``turns_per_session``, ``inter_turn_delay_ms``, ``shared_prefix_ratio``,
  ``num_prefix_groups``) with ``trace_format: "synthetic"`` and no ``trace_files``;
- ``sla``: ``{"itl_ms", "e2e_slowdown", "ttft_ms"}`` (Amendment A2: TTFT is normally null);
- ``measure``: the measurement-window rule (see :mod:`learned_routing.goodput`);
- ``engine_overrides``: ``MockEngineArgs`` overrides such as ``speedup_ratio`` (LR-14);
- ``replay_options``: ``weka_nested_timestamp_basis``, ``trace_shared_prefix_ratio``,
  ``trace_num_prefix_groups``;
- ``expected_cost_s``: replay cost estimate used for the timeout;
- ``arrival_spread_ms``: the trace's arrival logging quantum; replicates re-draw each session's
  first arrival inside its slot (protocol ``crn-spread-v1``, :mod:`learned_routing.replicates`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_routing import replicates
from learned_routing.canon import sha256_json
from learned_routing.paths import Layout

LOAD_MODES = ("open_speedup", "open_rate", "closed_concurrency", "agentic_lanes")
OPEN_MODES = frozenset({"open_speedup", "open_rate"})
NON_CONTENT_KEYS = frozenset(
    {
        "split",
        "holdout_axis",
        "holdout_axes",
        "selection_exposed",
        "notes",
        "segment",
        "trace_files",
        "engine_ref",
        "trace_sha256",
        "derived_meta",
        "expected_cost_s",
        "measure_trace",
        "trace_rows",
        "cache_pressure_ref",
        "transform_tag",
    }
)
REPLAY_OPTIONS = frozenset(
    {
        "weka_nested_timestamp_basis",
        "trace_shared_prefix_ratio",
        "trace_num_prefix_groups",
    }
)
SYNTHETIC_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "request_count",
        "turns_per_session",
        "inter_turn_delay_ms",
        "shared_prefix_ratio",
        "num_prefix_groups",
    }
)


class CellError(ValueError):
    pass


_file_sha_memo: dict[tuple, str] = {}


def source_sha256(path: Path, trace_format: str) -> str:
    """Content SHA of a trace source, memoized per (path, size, mtime)."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, trace_format)
    if key not in _file_sha_memo:
        _file_sha_memo[key] = replicates.source_sha256(path, trace_format)
    return _file_sha_memo[key]


def engine_content(engine: dict) -> dict:
    return {
        "mock_engine_args": engine["mock_engine_args"],
        "model": engine.get("model"),
    }


@dataclass
class Cell:
    raw: dict
    layout: Layout

    @property
    def cell_id(self) -> str:
        return self.raw["cell_id"]

    @property
    def split(self) -> str | None:
        return self.raw.get("split")

    @property
    def family(self) -> str | None:
        return self.raw.get("family")

    @property
    def num_workers(self) -> int:
        return int(self.raw["num_workers"])

    @property
    def trace_format(self) -> str:
        return self.raw.get("trace_format", "mooncake")

    @property
    def is_synthetic(self) -> bool:
        return self.trace_format == "synthetic"

    @property
    def load(self) -> dict:
        return self.raw["load"]

    @property
    def load_mode(self) -> str:
        return self.load["mode"]

    @property
    def is_open_loop(self) -> bool:
        return self.load_mode in OPEN_MODES

    @property
    def sla(self) -> dict:
        sla = dict(self.raw.get("sla") or {})
        return {
            "ttft_ms": sla.get("ttft_ms"),
            "itl_ms": sla.get("itl_ms"),
            "e2e_slowdown": sla.get("e2e_slowdown"),
        }

    @property
    def measure(self) -> dict:
        return dict(self.raw.get("measure") or {})

    def engine_path(self) -> Path:
        ref = self.raw.get("engine_ref")
        return self.layout.resolve_path(ref) if ref else self.layout.engine_json

    def engine(self) -> dict:
        return json.loads(self.engine_path().read_text())

    @property
    def arrival_spread_ms(self) -> float:
        value = float(self.raw.get("arrival_spread_ms") or 0.0)
        if value < 0 or (value and self.trace_format != "mooncake"):
            raise CellError(
                f"{self.cell_id}: arrival_spread_ms {value!r} needs a mooncake trace and >= 0"
            )
        return value

    def trace_source(self) -> Path:
        files = self.raw.get("trace_files") or []
        if len(files) != 1:
            raise CellError(
                f"{self.cell_id}: exactly one trace source (file, or a Weka play directory) "
                f"is supported, got {len(files)}"
            )
        path = self.layout.resolve_path(files[0])
        if not path.exists():
            raise CellError(f"{self.cell_id}: trace source {path} does not exist")
        return path

    def trace_content_sha(self) -> str:
        sha = source_sha256(self.trace_source(), self.trace_format)
        declared = self.raw.get("trace_sha256")
        if declared:
            declared = declared if isinstance(declared, list) else [declared]
            if declared != [sha]:
                raise CellError(
                    f"{self.cell_id}: trace content {sha[:16]} does not match the declared "
                    f"trace_sha256 {[d[:16] for d in declared]}"
                )
        return sha

    def synthetic_spec(self) -> dict:
        spec = dict(self.raw.get("synthetic") or {})
        unknown = sorted(set(spec) - SYNTHETIC_KEYS)
        if unknown:
            raise CellError(f"{self.cell_id}: unknown synthetic keys {unknown}")
        for key in ("input_tokens", "output_tokens", "request_count"):
            if key not in spec:
                raise CellError(f"{self.cell_id}: synthetic.{key} is required")
        return spec

    def content(self) -> dict:
        body = {k: v for k, v in self.raw.items() if k not in NON_CONTENT_KEYS}
        body["engine_content_sha256"] = sha256_json(engine_content(self.engine()))
        if not self.is_synthetic:
            body["trace_content_sha256"] = self.trace_content_sha()
        return body

    def content_sha(self) -> str:
        return sha256_json(self.content())

    def load_kwargs(self, trace_path: Path | None) -> dict:
        mode, value = self.load_mode, self.load.get("value")
        if mode not in LOAD_MODES:
            raise CellError(f"{self.cell_id}: load.mode {mode!r} not in {LOAD_MODES}")
        if value is None:
            raise CellError(
                f"{self.cell_id}: load.value is unset (calibration fills it)"
            )
        if mode == "open_speedup":
            return {"arrival_speedup_ratio": float(value)}
        if mode == "closed_concurrency":
            return {"replay_concurrency": int(value)}
        if mode == "agentic_lanes":
            if self.trace_format not in ("weka", "agentic_mooncake"):
                raise CellError(
                    f"{self.cell_id}: agentic_lanes needs trace_format weka or "
                    f"agentic_mooncake, got {self.trace_format!r}"
                )
            return {"agentic_lanes": int(value)}
        # open_rate
        if self.is_synthetic:
            return {"request_rate": float(value)}
        if self.trace_format != "mooncake" or trace_path is None:
            raise CellError(
                f"{self.cell_id}: open_rate on a trace needs a mooncake file"
            )
        rate = mooncake_native_rate(trace_path)
        return {"arrival_speedup_ratio": float(value) / rate}

    def replay_options(self) -> dict:
        options = dict(self.raw.get("replay_options") or {})
        unknown = sorted(set(options) - REPLAY_OPTIONS)
        if unknown:
            raise CellError(f"{self.cell_id}: unknown replay_options {unknown}")
        return options

    def expected_cost_s(self, trace_path: Path | None) -> float:
        if self.raw.get("expected_cost_s"):
            return float(self.raw["expected_cost_s"])
        if self.is_synthetic:
            spec = self.synthetic_spec()
            rows = int(spec["request_count"]) * int(spec.get("turns_per_session", 1))
            return 2.0 + 1e-3 * rows
        if trace_path is None:
            return 10.0
        if trace_path.is_file() and self.trace_format == "mooncake":
            with trace_path.open("rb") as handle:
                rows = sum(1 for _ in handle)
            return 2.0 + 1e-3 * rows
        size = (
            trace_path.stat().st_size
            if trace_path.is_file()
            else sum(p.stat().st_size for p in trace_path.rglob("*") if p.is_file())
        )
        return 5.0 + size / 5e6


_rate_memo: dict[tuple, float] = {}


def mooncake_native_rate(path: Path) -> float:
    """Requests per second of a Mooncake trace at speedup 1 (rows over timestamp span)."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _rate_memo:
        stamps = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    stamps.append(
                        float(row.get("timestamp", row.get("created_time", 0.0)) or 0.0)
                    )
        span_s = (max(stamps) - min(stamps)) / 1000.0
        if span_s <= 0:
            raise CellError(f"{path}: zero timestamp span; open_rate is undefined")
        _rate_memo[key] = len(stamps) / span_s
    return _rate_memo[key]


def load_cells(path: str | os.PathLike, layout: Layout) -> list[Cell]:
    cells = []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for key in ("cell_id", "num_workers", "load"):
            if key not in raw:
                raise CellError(f"{path}:{number}: cell is missing {key!r}")
        if raw["cell_id"] in seen:
            raise CellError(f"{path}:{number}: duplicate cell_id {raw['cell_id']!r}")
        seen.add(raw["cell_id"])
        cells.append(Cell(raw=raw, layout=layout))
    return cells


@dataclass(frozen=True)
class ResolvedReplicate:
    """What replicate ``k`` of a cell replays: a trace file or a synthetic arrival seed."""

    k: int
    protocol: str
    trace_path: Path | None
    trace_sha256: str | None
    replicate_seed: int
    policy_seed: int
    degenerate: bool
    arrival_seed: int | None = None


_replicate_memo: dict[tuple, ResolvedReplicate] = {}


def resolve_replicate(cell: Cell, k: int) -> ResolvedReplicate:
    """Materialize (idempotently) and memoize replicate ``k`` of ``cell`` (protocol crn-order-v1)."""
    if cell.is_synthetic:
        spec_sha = sha256_json(cell.synthetic_spec())
        seed = replicates.synthetic_arrival_seed(spec_sha, k)
        return ResolvedReplicate(
            k=k,
            protocol=replicates.PROTOCOL,
            trace_path=None,
            trace_sha256=None,
            replicate_seed=seed,
            policy_seed=replicates.policy_seed(k),
            degenerate=False,
            arrival_seed=seed,
        )
    source = cell.trace_source()
    stat = source.stat()
    spread = cell.arrival_spread_ms
    key = (
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        cell.trace_format,
        k,
        spread,
    )
    if key not in _replicate_memo:
        rep = replicates.materialize_replicate(
            source, cell.trace_format, k, cell.layout.replicates_dir, spread_ms=spread
        )
        _replicate_memo[key] = ResolvedReplicate(
            k=k,
            protocol=rep.protocol,
            trace_path=rep.path,
            trace_sha256=rep.sha256,
            replicate_seed=rep.replicate_seed,
            policy_seed=rep.policy_seed,
            degenerate=rep.stats.degenerate,
        )
    return _replicate_memo[key]


def cell_summary(cell: Cell) -> dict[str, Any]:
    """Grouping fields carried into every result record (for lr-report)."""
    return {
        "split": cell.split,
        "family": cell.family,
        "num_workers": cell.num_workers,
        "holdout_axis": cell.raw.get("holdout_axis"),
        "load_mode": cell.load_mode,
        "load_level": cell.load.get("level"),
        "segment": cell.raw.get("segment"),
    }
