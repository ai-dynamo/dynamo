# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Usage: run.py TESTS_ONLY_DIRECTORY OUTPUT_DIRECTORY [pytest options]."""

import importlib
import json
import os
import sys
from pathlib import Path


def unavailable_shutdown_helper(*args, **kwargs):
    """Reject accidental use of the vLLM-only helper on older runtimes."""
    raise RuntimeError("This vLLM runtime lacks the requested shutdown helper")


def main():
    """Bind Dynamo before importing selected tests, then return upstream pytest's status."""
    root, output = (Path(arg).resolve() for arg in sys.argv[1:3])
    if (root / "vllm").exists():
        raise ValueError("Use a tests-only tree so the installed vLLM is not shadowed")
    nodes = json.loads(Path(__file__).with_name("selection.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    # shortcut: vLLM 0.28 lacks this imported helper; remove after its support ends.
    engine_utils = importlib.import_module("vllm.v1.engine.utils")
    if not hasattr(engine_utils, "get_engine_process_shutdown_timeout"):
        engine_utils.get_engine_process_shutdown_timeout = unavailable_shutdown_helper
    upstream_utils = importlib.import_module("tests.utils")
    upstream_utils.RemoteOpenAIServer = importlib.import_module(
        "dynamo_server"
    ).RemoteDynamoServer
    pytest = importlib.import_module("pytest")
    return pytest.main(
        [
            "-v",
            "-s",
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "no:cacheprovider",
            "-o",
            "asyncio_mode=auto",
            "--junitxml",
            str(output / "junit.xml"),
            "--basetemp",
            str(output / "tmp"),
            *nodes,
            *sys.argv[3:],
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
