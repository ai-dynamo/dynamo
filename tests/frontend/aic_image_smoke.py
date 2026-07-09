# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke the AIC payload and native engine in the shipped frontend image."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import aiconfigurator
import aiconfigurator_core
from aiconfigurator.sdk.engine import compile_engine
from aiconfigurator.sdk.memory import estimate_num_gpu_blocks
from aiconfigurator.sdk.task_v2 import Task

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.mocker import AicEngineConfig, EnginePerfLimits, RustEnginePerfModel


def main() -> None:
    assert metadata.version("aiconfigurator") == "0.10.0"
    assert aiconfigurator_core and compile_engine and estimate_num_gpu_blocks and Task

    package_root = Path(aiconfigurator.__file__).resolve().parent
    assert (package_root / "model_configs/Qwen--Qwen3-32B_config.json").is_file()
    assert (package_root / "systems/h200_sxm.yaml").is_file()
    parquet_files = list(
        (package_root / "systems/data/h200_sxm/vllm/0.14.0").glob("*.parquet")
    )
    assert parquet_files
    for path in parquet_files:
        with path.open("rb") as handle:
            assert handle.read(4) == b"PAR1"

    model = RustEnginePerfModel.from_native(
        aic_config=AicEngineConfig(
            model_name="Qwen/Qwen3-32B",
            backend="vllm",
            system_name="h200_sxm",
            backend_version="0.14.0",
            tp_size=1,
            attention_dp_size=1,
        ),
        worker_type="prefill",
        limits=EnginePerfLimits(
            max_num_batched_tokens=4096,
            max_num_seqs=128,
            max_kv_tokens=1_000_000,
        ),
    )
    estimate = model.estimate_forward_pass_time(
        [
            ForwardPassMetrics(
                scheduled_requests=ScheduledRequestMetrics(
                    num_prefill_requests=1,
                    sum_prefill_tokens=1024,
                )
            )
        ]
    )
    assert estimate is not None and estimate > 0.0
    diagnostics = json.loads(model.diagnostics())
    assert diagnostics["source"] == "aic"
    assert diagnostics["readiness"] == "ready"
    assert diagnostics["last_warning"] is None


if __name__ == "__main__":
    main()
