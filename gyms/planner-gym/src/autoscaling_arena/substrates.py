# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named hardware/model substrate presets for the Arena.

A substrate is the *simulated deployment* an autoscaler runs on: model + system +
backend + perf-DB version + parallelism (TP/MoE-TP/MoE-EP/attention-DP) + GPUs per
worker. It is deliberately separate from the workload (traffic) and the autoscaler
(decision-maker) — the same substrate is shared by every autoscaler in a match so
only the decision-maker varies.

Each preset is a combination AIS actually has perf data for (verified against the
aiconfigurator perf DB). ``engine_args()`` renders the JSON the Dynamo replay
runtime consumes; ``gpus_per_worker`` feeds the substrate's per-engine GPU
count (and thus GPU-hour accounting).

The preset backend is emitted both as DynoSim's ``engine_type`` and as the AIS
performance-data backend. First-class Match Config engines may decouple those
values explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Substrate:
    key: str
    model: str
    system: str
    backend: str
    backend_version: Optional[str] = None
    tp_size: int = 1
    moe_tp_size: Optional[int] = None
    moe_ep_size: Optional[int] = None
    attention_dp_size: Optional[int] = None
    gpus_per_worker: int = (
        1  # GPUs per prefill/decode engine (tp_size * attention_dp_size)
    )
    default_gpu_budget: int = 32
    # Extra raw engine-arg overrides merged verbatim (e.g. num_gpu_blocks to bypass
    # the KV-capacity estimate when weights don't fit the estimator's memory model).
    extra_engine_args: dict = field(default_factory=dict)

    def engine_args(self) -> str:
        perf_config = {
            "model": self.model,
            "system": self.system,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "tp": self.tp_size,
            "moe_tp_size": self.moe_tp_size,
            "moe_ep_size": self.moe_ep_size,
            "attention_dp": self.attention_dp_size,
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        }
        args = {
            **self.extra_engine_args,
            "engine_type": self.backend,
            "tensor_parallel_size": self.tp_size,
            "dp_size": self.attention_dp_size or 1,
            "ais_perf_config": {
                key: value for key, value in perf_config.items() if value is not None
            },
        }
        return json.dumps(args)


SUBSTRATES = {
    # gpt-oss-120b: fits one GPU per worker (MoE tp1/ep1). Perf data on h200_sxm/vllm/0.19.0.
    "gpt_oss": Substrate(
        key="gpt_oss",
        model="openai/gpt-oss-120b",
        system="h200_sxm",
        backend="vllm",
        backend_version="0.19.0",
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        gpus_per_worker=1,
        default_gpu_budget=32,
    ),
}


def get_substrate(key: str) -> Substrate:
    if key not in SUBSTRATES:
        raise KeyError(f"unknown substrate '{key}'; known: {sorted(SUBSTRATES)}")
    return SUBSTRATES[key]
