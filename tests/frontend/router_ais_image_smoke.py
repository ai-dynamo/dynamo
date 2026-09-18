# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke the experimental router-side AIS path in the frontend image."""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Keep this smoke offline: the selected model config and perf database are
    # shipped inside aiconfigurator-core.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Parse the same frontend flags users enable for the live KV-router path.
    sys.argv = [
        "dynamo.frontend",
        "--router-mode",
        "kv",
        "--router-prefill-load-model",
        "ais",
        "--dyn-chat-processor",
        "dynamo",
        "--ais-backend",
        "vllm",
        "--ais-system",
        "h200_sxm",
        "--ais-model-path",
        "Qwen/Qwen3-32B",
        "--ais-backend-version",
        "current",
    ]

    import json

    from dynamo.frontend.main import parse_args
    from dynamo.llm import AisPerfConfig, KvRouterConfig
    from dynamo.mocker import MockEngineArgs
    from dynamo.replay import run_synthetic_trace_replay

    config, _, _ = parse_args()
    assert config.router_mode == "kv"
    assert config.router_prefill_load_model == "ais"
    perf = AisPerfConfig(**config.ais_perf_kwargs())
    engine_args = MockEngineArgs.from_json(
        json.dumps(
            {
                "ais_perf_config": perf.to_dict(),
                "num_gpu_blocks": 1024,
            }
        )
    )
    # Exercise Dynamo's actual embedded Rust callback and Replay engine together.
    # This catches a wheel/crate EngineSpec mismatch that an SDK-only query misses.
    report = run_synthetic_trace_replay(
        input_tokens=1024,
        output_tokens=2,
        request_count=2,
        num_workers=2,
        extra_engine_args=engine_args,
        ais_perf_config=perf,
        router_mode="kv_router",
        router_config=KvRouterConfig(router_prefill_load_model="ais"),
        replay_concurrency=1,
    )
    assert report.summary


if __name__ == "__main__":
    main()
