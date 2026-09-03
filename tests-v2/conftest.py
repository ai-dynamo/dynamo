# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builds the Dynamo a test receives, from configuration given at run time.

    # deploy a released container and test it
    pytest tests-v2 --dynamo-image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0

    # or attach to something already running (query-only)
    pytest tests-v2 --dynamo-url http://localhost:8000
"""

from __future__ import annotations

import pytest
from dynamo_harness import DEFAULT_IMAGE, Docker, Dynamo


def pytest_addoption(parser):
    group = parser.getgroup("dynamo")
    group.addoption(
        "--dynamo-url",
        default=None,
        help="Attach to an already-running frontend (query only).",
    )
    group.addoption(
        "--dynamo-image",
        default=None,
        help=f"Deploy this container image. Default: {DEFAULT_IMAGE}",
    )
    group.addoption(
        "--dynamo-backend", default="vllm", choices=["vllm", "sglang", "trtllm"]
    )
    group.addoption("--dynamo-model", default="Qwen/Qwen3-0.6B")
    group.addoption("--dynamo-port", type=int, default=8000)
    group.addoption(
        "--dynamo-gpus", default="all", help="--gpus value; 'none' to disable."
    )
    group.addoption("--dynamo-hf-cache", default=None, help="Host HF cache to mount.")
    group.addoption("--dynamo-ready-timeout", type=float, default=900.0)
    group.addoption(
        "--dynamo-tool-parser",
        default=None,
        help="Worker --dyn-tool-call-parser (e.g. hermes). Enables TOOL_CALLING.",
    )
    group.addoption(
        "--dynamo-reasoning-parser",
        default=None,
        help="Worker --dyn-reasoning-parser (e.g. qwen3). Enables REASONING_PARSER.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_deployment: requires lifecycle control")
    config.addinivalue_line("markers", "e2e: end-to-end test against a real deployment")


@pytest.fixture(scope="session")
def dynamo(request):
    """A Dynamo, deployed or attached, ready to serve."""
    opt = request.config.getoption
    url = opt("--dynamo-url")

    if url:
        instance = Dynamo.attach(url, model=None)
    else:
        gpus = opt("--dynamo-gpus")
        worker_args = []
        if opt("--dynamo-tool-parser"):
            worker_args += ["--dyn-tool-call-parser", opt("--dynamo-tool-parser")]
        if opt("--dynamo-reasoning-parser"):
            worker_args += ["--dyn-reasoning-parser", opt("--dynamo-reasoning-parser")]
        instance = Dynamo.deploy(
            Docker(
                worker_args=worker_args,
                image=opt("--dynamo-image") or DEFAULT_IMAGE,
                model=opt("--dynamo-model"),
                backend=opt("--dynamo-backend"),
                port=opt("--dynamo-port"),
                gpus=None if gpus in ("none", "") else gpus,
                hf_cache=opt("--dynamo-hf-cache"),
            )
        )

    try:
        instance.wait_until_serving(timeout=opt("--dynamo-ready-timeout"))
    except Exception:
        if instance.deployment is not None:
            print("\n---- deployment logs ----")
            try:
                print(instance.deployment.logs(tail=80))
            except Exception:
                pass
        instance.close()
        raise

    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def _skip_without_deployment(request):
    """`needs_deployment` tests skip when attached rather than failing late."""
    if request.node.get_closest_marker("needs_deployment"):
        instance = request.getfixturevalue("dynamo")
        if (
            instance.deployment is None
            or type(instance.deployment).__name__ == "Attached"
        ):
            pytest.skip(
                "needs lifecycle control; run with --dynamo-image, not --dynamo-url"
            )
