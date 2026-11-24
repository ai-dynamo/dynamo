# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``Workload`` spec and its materialization to a Mooncake-JSONL trace.

A synthetic workload composes one value per axis (arrival × shape × prefix); a
real workload (e.g. the Mooncake anchor) wraps an existing trace file. Both
materialize to a path the DynoSim substrate can replay, so the leaderboard
treats real and synthetic workloads identically.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from autoscaling_arena.workloads.axes import RateFn, ShapeFn, poisson_arrivals

# A prefix policy is stateful across a trace, so we store a factory that mints a
# fresh policy per materialization (keeps generation reproducible + isolated).
PrefixFactory = Callable[[], "object"]  # () -> obj with .assign(rng, isl, block_size)


def validate_mooncake_trace(
    path: Path, *, block_size: int, presorted: bool = False
) -> int:
    """Stream-validate a Mooncake JSONL trace and return its request count."""

    if block_size <= 0:
        raise ValueError("trace block size must be positive")
    count = 0
    previous_timestamp: int | float | None = None
    try:
        source = path.open()
    except OSError as exc:
        raise ValueError(f"cannot read trace: {exc}") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")

            timestamp = record.get("timestamp")
            input_length = record.get("input_length")
            output_length = record.get("output_length")
            hash_ids = record.get("hash_ids")
            try:
                finite_timestamp = math.isfinite(float(timestamp))
            except (TypeError, ValueError, OverflowError):
                finite_timestamp = False
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not finite_timestamp
                or timestamp < 0
            ):
                raise ValueError(
                    f"line {line_number}: timestamp must be a finite nonnegative number"
                )
            if (
                isinstance(input_length, bool)
                or not isinstance(input_length, int)
                or input_length <= 0
            ):
                raise ValueError(
                    f"line {line_number}: input_length must be a positive integer"
                )
            if (
                isinstance(output_length, bool)
                or not isinstance(output_length, int)
                or output_length <= 0
            ):
                raise ValueError(
                    f"line {line_number}: output_length must be a positive integer"
                )
            if not isinstance(hash_ids, list) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in hash_ids
            ):
                raise ValueError(
                    f"line {line_number}: hash_ids must be a list of integers"
                )
            expected_hashes = (input_length + block_size - 1) // block_size
            if len(hash_ids) != expected_hashes:
                raise ValueError(
                    f"line {line_number}: input_length {input_length} with block size "
                    f"{block_size} requires {expected_hashes} hashes, got {len(hash_ids)}"
                )
            if (
                presorted
                and previous_timestamp is not None
                and timestamp < previous_timestamp
            ):
                raise ValueError(f"line {line_number}: timestamps are not sorted")
            previous_timestamp = timestamp
            count += 1
    if count == 0:
        raise ValueError("trace contains no request records")
    return count


@dataclass
class Workload:
    """One named workload.

    Either ``static_trace`` (a real recorded trace) or the three synthetic axes
    must be set. ``block_size`` is the token span per ``hash_ids`` entry and must
    match the replay's ``--trace-block-size`` for the prefix structure to land.
    """

    name: str
    description: str
    # Synthetic axes (mutually exclusive with static_trace):
    duration_s: Optional[float] = None
    arrival: Optional[RateFn] = None
    shape: Optional[ShapeFn] = None
    prefix_factory: Optional[PrefixFactory] = None
    block_size: int = 512
    # Real-trace anchor:
    static_trace: Optional[Path] = None
    # Set when a large static Mooncake trace is already sorted by ``timestamp``.
    # Lets ``materialize`` stream + early-stop at a cap instead of
    # loading the whole file to sort it — essential for very large traces.
    presorted: bool = False
    # Optional cap on the number of requests (keeps the earliest-arriving N).
    # Shrinks a run while PRESERVING the arrival rate/intensity — just stops the
    # trace early. ``None`` = full workload. Overridable per-call in materialize().
    max_requests: Optional[int] = None

    @property
    def is_synthetic(self) -> bool:
        return self.static_trace is None

    def generate_records(
        self, seed: int, *, max_requests: Optional[int] = None
    ) -> List[dict]:
        """Deterministically build the Mooncake records for a synthetic workload.

        ``max_requests`` (falling back to the workload's own ``max_requests``)
        keeps only the earliest-arriving N requests.
        """
        if not self.is_synthetic:
            raise ValueError(
                f"workload '{self.name}' wraps a static trace; nothing to generate"
            )
        missing = [
            name
            for name in ("arrival", "shape", "prefix_factory", "duration_s")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"synthetic workload '{self.name}' is missing required axes: "
                + ", ".join(missing)
            )
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(
                f"workload '{self.name}' duration_s must be positive and finite"
            )
        rng = random.Random(seed)
        arrivals = poisson_arrivals(self.arrival, self.duration_s, rng)
        prefix = self.prefix_factory()
        records: List[dict] = []
        for t_ms in arrivals:
            isl, osl = self.shape(rng)
            hash_ids = prefix.assign(rng, isl, self.block_size)
            records.append(
                {
                    "timestamp": int(t_ms),
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": hash_ids,
                }
            )
        records.sort(key=lambda r: r["timestamp"])
        cap = max_requests if max_requests is not None else self.max_requests
        if cap is not None:
            records = records[:cap]
        return records

    def materialize(
        self,
        out_dir: Path,
        seed: int = 0,
        *,
        max_requests: Optional[int] = None,
        arrival_speedup: float = 1.0,
        force_copy_static: bool = False,
    ) -> Path:
        """Return a replayable trace path: the static file, or a freshly written one.

        ``max_requests`` caps the trace to the earliest-arriving N requests
        (preserves intensity, shortens the run). ``arrival_speedup`` divides every
        timestamp by the ratio, compressing the schedule so a *real-time* consumer
        (AIPerf) replays faster — note this also raises the effective request rate.
        Synthetic traces are stable for a given ``seed``. A presorted static
        trace with no transforms is returned as-is unless ``force_copy_static``
        requests private staging; unsorted inputs are normalized into a copy.
        """
        cap = max_requests if max_requests is not None else self.max_requests

        if not self.is_synthetic:
            assert self.static_trace is not None
            if not self.static_trace.exists():
                raise FileNotFoundError(
                    f"static trace for '{self.name}' missing: {self.static_trace}"
                )
            if cap is None and arrival_speedup == 1.0 and self.presorted:
                if not force_copy_static:
                    return self.static_trace
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{self.name}.jsonl"
                try:
                    os.link(self.static_trace, path)
                except OSError:
                    shutil.copyfile(self.static_trace, path)
                return path
            if self.presorted:
                # Already timestamp-sorted: stream + early-stop at the cap so we
                # never load a multi-hundred-MB trace just to slice it.
                records = []
                with self.static_trace.open() as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        records.append(json.loads(line))
                        if cap is not None and len(records) >= cap:
                            break
            else:
                records = [
                    json.loads(line)
                    for line in self.static_trace.read_text().splitlines()
                    if line.strip()
                ]
                records.sort(key=lambda r: r.get("timestamp", 0))
                if cap is not None:
                    records = records[:cap]
        else:
            records = self.generate_records(seed, max_requests=cap)

        if arrival_speedup != 1.0:
            for r in records:
                r["timestamp"] = int(r.get("timestamp", 0) / arrival_speedup)

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.jsonl"
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path
