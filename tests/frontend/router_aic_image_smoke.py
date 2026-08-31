# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke the experimental router-side AIC path in the frontend image."""

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
        "aic",
        "--dyn-chat-processor",
        "dynamo",
        "--aic-backend",
        "vllm",
        "--aic-system",
        "h200_sxm",
        "--aic-model-path",
        "Qwen/Qwen3-32B",
        "--aic-backend-version",
        "0.19.0",
    ]

    from aiconfigurator_core.sdk.engine import EngineHandle

    from dynamo.frontend.main import parse_args
    from dynamo.llm import AicPerfConfig

    config, _, _ = parse_args()
    assert config.router_mode == "kv"
    assert config.router_prefill_load_model == "aic"
    assert config.aic_backend is not None
    assert config.aic_system is not None
    assert config.aic_model_path is not None

    # This is the object passed by dynamo.frontend into the native KV router.
    assert AicPerfConfig(**config.aic_perf_kwargs())

    # Exercise the compiled predictor and packaged perf data that the router's
    # native AIC prefill-load estimator consumes at startup.
    engine = EngineHandle.compile(
        model_path=config.aic_model_path,
        system=config.aic_system,
        backend=config.aic_backend,
        backend_version=config.aic_backend_version,
        tp_size=config.aic_tp_size,
    )
    assert engine.predict_prefill_latency(1, 1024, 0) > 0.0


if __name__ == "__main__":
    main()
