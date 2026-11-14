# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke the experimental router-side AIS path in the frontend image."""

from __future__ import annotations

import json
import os
import sys
import warnings

PERF_MODEL = {
    "model": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "current",
    "worker_type": "aggregated",
}


def main() -> None:
    # Keep this smoke offline: the selected model config and perf database are
    # shipped in the AISimulate wheel's aisimulate_core package.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # An old AIS input only warns; here it fails, so the smoke stays on the
    # input that reads the value with the AISimulate loader in the image.
    # The pattern is Dynamo's sentence: a library's notice must not fail it.
    warnings.filterwarnings(
        "error",
        message=r".* deprecated and will be removed in v\d+\.\d+; use ",
        category=FutureWarning,
    )

    # Parse the same frontend flags users enable for the live KV-router path.
    sys.argv = [
        "dynamo.frontend",
        "--router-mode",
        "kv",
        "--router-prefill-load-model",
        "ais",
        "--dyn-chat-processor",
        "dynamo",
        "--ais-perf-model",
        json.dumps(PERF_MODEL),
    ]

    from dynamo.frontend.main import parse_args
    from dynamo.llm import AisPerfConfig, KvRouterConfig
    from dynamo.mocker.config import normalize_mocker_config
    from dynamo.replay import run_synthetic_trace_replay

    config, _, _ = parse_args()
    assert config.router_mode == "kv"
    assert config.router_prefill_load_model == "ais"
    perf = AisPerfConfig(**config.ais_perf_kwargs())
    engine_args = normalize_mocker_config(
        {
            "engine": {
                "num_gpu_blocks": 1024,
                "timing_model": {
                    "type": "external",
                    "provider": "ais",
                    "config": perf.to_dict(),
                },
            }
        }
    )
    # Exercise Dynamo's actual embedded Rust callback and Replay engine together.
    # This catches a wheel/crate EngineSpec mismatch that an SDK-only query misses.
    report = run_synthetic_trace_replay(
        input_tokens=1024,
        output_tokens=2,
        request_count=2,
        num_workers=2,
        extra_engine_args=engine_args,
        router_ais_perf_model=PERF_MODEL,
        router_mode="kv_router",
        router_config=KvRouterConfig(router_prefill_load_model="ais"),
        replay_concurrency=1,
    )
    assert report.summary


if __name__ == "__main__":
    main()
