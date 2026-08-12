# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable human and JSON output for the public AISimulate CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import yaml

from .sweeper.config import Candidate, SmartSearchConfig
from .sweeper.reporting import serialize_sweep_results

_METRIC_LABELS = {
    "output_throughput_tok_s": "output throughput (tok/s)",
    "mean_ttft_ms": "mean TTFT (ms)",
    "mean_tpot_ms": "mean TPOT (ms)",
    "mean_e2e_latency_ms": "mean E2E latency (ms)",
    "mean_output_token_throughput_per_user": "per-user throughput (tok/s/user)",
    "goodput_output_throughput_tok_s": "goodput (tok/s)",
    "gpu_hours": "GPU hours",
    "duration_ms": "duration (ms)",
}


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return candidate.model_dump(mode="json")


def _print_candidate(candidate: Candidate, *, rank: int | None = None) -> None:
    heading = f"candidate {rank}" if rank is not None else "resolved prediction"
    print(f"{heading}:")
    print("  configuration:")
    config_yaml = yaml.safe_dump(
        candidate.config,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()
    for line in config_yaml.splitlines():
        print(f"    {line}")
    print(f"  used GPUs: {candidate.used_gpus}")
    print(f"  score: {candidate.score:.6g}")
    if candidate.objectives:
        print("  objectives:")
        for name, value in candidate.objectives.items():
            print(f"    {name}: {value:.6g}")
    print("  performance:")
    if not candidate.metrics:
        print("    <no metrics>")
    for name, value in candidate.metrics.items():
        print(f"    {_METRIC_LABELS.get(name, name)}: {value:.6g}")


def print_prediction(*, stack: str, candidate: Candidate, output: str) -> None:
    """Print one resolved prediction."""

    if output == "json":
        print(
            json.dumps(
                {
                    "command": "predict",
                    "stack": stack,
                    "candidate": _candidate_payload(candidate),
                },
                sort_keys=True,
            )
        )
        return
    print(f"AISimulate prediction (stack={stack})")
    _print_candidate(candidate)


def print_recommendations(
    *,
    stack: str,
    config: SmartSearchConfig,
    candidates: Sequence[Candidate],
    output: str,
    top_n: int,
) -> None:
    """Print ranked recommendations or a Pareto frontier."""

    if output == "json":
        result = serialize_sweep_results(config, candidates)
        print(
            json.dumps(
                {
                    "command": "recommend",
                    "stack": stack,
                    **result,
                },
                sort_keys=True,
            )
        )
        return
    result_label = "Pareto frontier" if config.goal.is_pareto else "ranked candidates"
    shown = candidates if config.goal.is_pareto else candidates[:top_n]
    print(f"AISimulate recommendation (stack={stack})")
    print(f"{result_label}: {len(candidates)} feasible")
    for rank, candidate in enumerate(shown, start=1):
        _print_candidate(candidate, rank=rank)
