# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded multi-turn chat sessions written as a Mooncake session trace.

The replay entry point can drive the native synthetic-session source, but that source cannot express
a conversation: through the Python API every request has the same ISL and OSL
(``LengthSpec { stddev: 0 }``), every inter-turn delay is the same constant, and a later turn does
not extend the earlier turns' context (each turn draws fresh unique blocks after the group prefix;
aisimulate-core ``replay/loadgen/trace.rs`` ``Trace::synthetic``). Session affinity would then have
nothing to exploit. This generator writes a Mooncake JSONL in which:

- sessions start as a Poisson process at ``session_rate`` per second over ``duration_s``, and the
  first-turn timestamps are floored to ``arrival_grid_ms`` so sessions that start in the same tick
  tie (as Mooncake's 3 s logging grid does), which gives the ``crn-order-v1`` replicate protocol
  arrival ties to permute;
- each session draws a system prompt from ``num_prefix_groups`` groups with Zipf popularity;
- turn ``k`` sends the whole conversation so far: system prompt, every user message and every
  earlier output, plus the new user message. Its ``hash_ids`` extend turn ``k - 1``'s full blocks;
  a partial last block gets its own id, because filling it changes its content;
- later turns carry ``delay`` (milliseconds after the previous turn completes), so release is causal;
- user-message, output and system-prompt lengths are lognormal, think times exponential, and the
  number of turns is geometric, truncated so that ``ISL + OSL`` stays under ``context_cap``.

All randomness comes from ``random.Random(seed)``, so a spec yields identical bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import canonical_json, sha256_text, write_atomic

GENERATOR_VERSION = "lr-synthetic-sessions-v1"


@dataclass(frozen=True)
class SessionSpec:
    seed: int = 0
    duration_s: float = 900.0
    session_rate: float = 1.0
    arrival_grid_ms: int = 1000
    block_size: int = 64
    num_prefix_groups: int = 16
    group_zipf_s: float = 1.1
    system_tokens_median: float = 2000.0
    system_tokens_sigma: float = 0.6
    system_tokens_range: tuple[int, int] = (256, 12000)
    first_user_median: float = 600.0
    first_user_sigma: float = 1.0
    user_median: float = 250.0
    user_sigma: float = 1.0
    user_range: tuple[int, int] = (8, 16000)
    output_median: float = 250.0
    output_sigma: float = 0.8
    output_range: tuple[int, int] = (1, 2000)
    mean_turns: float = 5.0
    max_turns: int = 24
    think_mean_ms: float = 8000.0
    context_cap: int = 49152

    def validate(self) -> None:
        if self.duration_s <= 0 or self.session_rate <= 0:
            raise ValueError("duration_s and session_rate must be > 0")
        if self.mean_turns < 1 or self.max_turns < 1:
            raise ValueError("mean_turns and max_turns must be >= 1")
        if (
            self.num_prefix_groups < 1
            or self.block_size < 1
            or self.arrival_grid_ms < 1
        ):
            raise ValueError(
                "num_prefix_groups, block_size and arrival_grid_ms must be >= 1"
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("system_tokens_range", "user_range", "output_range"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SessionSpec:
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown session spec keys {unknown}")
        values = dict(data)
        for key in ("system_tokens_range", "user_range", "output_range"):
            if key in values:
                values[key] = tuple(int(v) for v in values[key])
        spec = cls(**values)
        spec.validate()
        return spec

    def key(self) -> str:
        return sha256_text(
            canonical_json({"version": GENERATOR_VERSION, "spec": self.to_dict()})
        )


def _lognormal(
    rng: random.Random, median: float, sigma: float, bounds: tuple[int, int]
) -> int:
    value = int(round(math.exp(math.log(median) + sigma * rng.gauss(0.0, 1.0))))
    return min(max(value, bounds[0]), bounds[1])


def _zipf_cdf(n: int, s: float) -> list[float]:
    weights = [1.0 / (i + 1) ** s for i in range(n)]
    total = sum(weights)
    cdf, acc = [], 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    return cdf


def generate(spec: SessionSpec) -> list[dict]:
    """Rows of a Mooncake session trace; sessions are contiguous and ordered by first arrival."""
    spec.validate()
    rng = random.Random(spec.seed)
    bs = spec.block_size
    group_cdf = _zipf_cdf(spec.num_prefix_groups, spec.group_zipf_s)
    group_tokens = [
        _lognormal(
            rng,
            spec.system_tokens_median,
            spec.system_tokens_sigma,
            spec.system_tokens_range,
        )
        for _ in range(spec.num_prefix_groups)
    ]
    continue_p = 1.0 - 1.0 / spec.mean_turns

    ids: dict = {}

    def block_id(identity) -> int:
        return ids.setdefault(identity, len(ids))

    rows: list[dict] = []
    clock_s = 0.0
    session_index = 0
    while True:
        clock_s += rng.expovariate(spec.session_rate)
        if clock_s >= spec.duration_s:
            break
        stamp = int(clock_s * 1000.0) // spec.arrival_grid_ms * spec.arrival_grid_ms
        u = rng.random()
        group = next(
            i for i, c in enumerate(group_cdf) if u <= c or i == len(group_cdf) - 1
        )
        system = group_tokens[group]
        session_id = f"syn{spec.seed}-s{session_index:06d}"
        context = system  # tokens of conversation so far, before the new user message
        turn = 0
        while True:
            user = _lognormal(
                rng,
                spec.first_user_median if turn == 0 else spec.user_median,
                spec.first_user_sigma if turn == 0 else spec.user_sigma,
                spec.user_range,
            )
            output = _lognormal(
                rng, spec.output_median, spec.output_sigma, spec.output_range
            )
            think_ms = (
                rng.expovariate(1.0 / spec.think_mean_ms)
                if spec.think_mean_ms > 0
                else 0.0
            )
            more = rng.random() < continue_p
            isl = context + user
            if isl + output > spec.context_cap:
                break
            full = isl // bs
            hash_ids = [
                block_id(("g", group, b))
                if (b + 1) * bs <= system
                else block_id(("s", session_id, b))
                for b in range(full)
            ]
            if isl % bs:
                hash_ids.append(block_id(("t", session_id, turn)))
            row = {
                "session_id": session_id,
                "input_length": isl,
                "output_length": output,
                "hash_ids": hash_ids,
            }
            if turn == 0:
                row["timestamp"] = stamp
            else:
                row["delay"] = round(think_ms, 3)
            rows.append(row)
            context = isl + output
            turn += 1
            if not more or turn >= spec.max_turns:
                break
        session_index += 1
    return rows


def write(spec: SessionSpec, path: Path) -> str:
    rows = generate(spec)
    content = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    write_atomic(path, content.encode())
    return sha256_text(content)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write a seeded multi-turn Mooncake session trace."
    )
    parser.add_argument(
        "--spec", default="{}", help="JSON object of SessionSpec fields"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = SessionSpec.from_dict(json.loads(args.spec))
    sha = write(spec, args.out)
    print(json.dumps({"path": str(args.out), "sha256": sha, "spec_key": spec.key()}))


if __name__ == "__main__":
    main()
