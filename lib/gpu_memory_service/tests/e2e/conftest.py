# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        default="facebook/opt-125m",
        help="HuggingFace model id or local path to use for the E2E test "
        "(default: facebook/opt-125m).",
    )
    parser.addoption(
        "--vllm-port",
        type=int,
        default=18_700,
        help="TCP port for the vLLM OpenAI-compatible server (default: 18700).",
    )
    parser.addoption(
        "--vllm-startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for vLLM to become healthy (default: 300).",
    )
    parser.addoption(
        "--restore-workers",
        type=int,
        default=4,
        help="Number of parallel worker threads for restoring GMS state from "
        "disk (default: 4).  Passed as --workers to gms-storage-client load.",
    )
