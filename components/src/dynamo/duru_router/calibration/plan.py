# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose a sparse grid with dense probes at resolved prefill graph boundaries."""

import json
import math
from pathlib import Path

METRIC = "engine_service_time_v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def geometric_points(maximum, count):
    if maximum < 1 or count < 2:
        raise ValueError("Need a positive range and at least two sparse points")
    return sorted(
        {
            1,
            maximum,
            *(
                round(math.exp(math.log(maximum) * i / (count - 1)))
                for i in range(count)
            ),
        }
    )


def design(
    cap,
    alignment,
    *,
    max_prefill=8192,
    capture_sizes=(),
    prefill_graphs=False,
    prefix_points=8,
    decode_points=12,
    repeats=10,
    warmups=3,
    decode_tokens=80,
):
    """Capture sizes must be resolved sizes usable by single-request prefill.

    vLLM rounds scheduled tokens up to the next captured size. Thus c and c+1
    lie on opposite sides of a dispatch boundary. Decode context K is not a
    graph batch size: ordinary solo decode schedules one token at any K.
    """
    for name, value in (
        ("cap", cap),
        ("alignment", alignment),
        ("max_prefill", max_prefill),
        ("repeats", repeats),
        ("warmups", warmups),
        ("decode_tokens", decode_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if prefix_points < 2 or decode_points < 2 or decode_tokens < 12:
        raise ValueError(
            "Need two points per context axis and at least 12 decode tokens"
        )
    if max_prefill < 2 or cap <= max_prefill + alignment or cap <= decode_tokens + 1:
        raise ValueError("Context must leave room for prefix and output tokens")
    captures = sorted(set(capture_sizes))
    if any(type(c) is not int or c < 1 for c in captures):
        raise ValueError("Capture sizes must be positive integers")
    if prefill_graphs and not captures:
        raise ValueError(
            "Provide resolved capture sizes when prefill graphs are enabled"
        )
    boundaries = [c for c in captures if c < max_prefill] if prefill_graphs else []
    p = set(geometric_points(max_prefill, 12))
    for c in boundaries:
        p.update(n for n in (c - 1, c, c + 1) if 1 <= n <= max_prefill)
    maximum_d = (cap - max_prefill - 1) // alignment * alignment
    d = sorted(
        {
            0,
            alignment,
            maximum_d,
            *(
                n // alignment * alignment
                for n in geometric_points(maximum_d, prefix_points)
            ),
        }
    )
    k = geometric_points(cap - decode_tokens, decode_points)
    p = sorted(p)
    cases = [dict(phase="prefill", P=a, D=b, split="train") for b in d for a in p]
    cases += [dict(phase="decode", K=a, split="train") for a in k]
    return dict(
        schema_version=1,
        timing_metric=METRIC,
        P=p,
        D=d,
        K=k,
        P_discontinuities=boundaries,
        cases=cases,
        cap=cap,
        alignment=alignment,
        max_prefill=max_prefill,
        repeats=repeats,
        warmups=warmups,
        decode_tokens=decode_tokens,
        decode_warmup_tokens=8,
        capture_sizes=captures,
        prefill_graphs=prefill_graphs,
    )
