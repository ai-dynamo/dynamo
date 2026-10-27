# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime requirements for native sidecar serving tests."""

import importlib.util
import os
from pathlib import Path

import pytest


def trtllm_serves_openengine() -> bool:
    """Probe the installed package without importing TensorRT-LLM."""
    spec = importlib.util.find_spec("tensorrt_llm")
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    return any(
        (Path(root) / "grpc/openengine/servicer.py").is_file()
        for root in locations or ()
    )


def require_trtllm_openengine() -> None:
    """Fail required TRT-LLM CI, or skip optional local runs, if unavailable."""
    if trtllm_serves_openengine():
        return
    reason = "the installed TensorRT-LLM runtime has no OpenEngine servicer"
    if os.environ.get("DYNAMO_TEST_SIDECAR_BACKEND") == "trtllm":
        pytest.fail(reason)
    pytest.skip(reason)
