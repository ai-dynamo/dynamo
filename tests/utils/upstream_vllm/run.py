# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Usage: run.py TESTS_ONLY_DIRECTORY OUTPUT_DIRECTORY [pytest options]."""

import importlib
import json
import os
import sys
from pathlib import Path

# Reuse Dynamo's port helpers without importing its conflicting tests package.
sys.path.append(str(Path(__file__).resolve().parents[1]))


class Report:
    """Record discovered node IDs and identify tests that use a Dynamo server fixture."""

    def __init__(self, output, server_type):
        """Write collection under output; server_type identifies adapted fixture values."""
        self.output = output
        self.server_type = server_type

    def pytest_collection_finish(self, session):
        """Keep full IDs, including parametrization, for the outer Dynamo report."""
        nodes = [item.nodeid for item in session.items]
        (self.output / "collection.json").write_text(json.dumps(nodes, indent=2))
        for item in session.items:
            item.user_properties.append(("upstream_nodeid", item.nodeid))

    def pytest_runtest_call(self, item):
        """Label actual server-fixture use; vLLM-only passes are not Dynamo coverage."""
        target = (
            "dynamo"
            if any(
                isinstance(value, self.server_type) for value in item.funcargs.values()
            )
            else "vllm-only"
        )
        item.user_properties.append(("upstream_target", target))


def unavailable_shutdown_helper(*args, **kwargs):
    """Reject accidental use of the vLLM-only helper on older runtimes."""
    raise RuntimeError("This vLLM runtime lacks the requested shutdown helper")


def main():
    """Bind Dynamo, discover the chat directory and return upstream pytest's status."""
    root, output = (Path(arg).resolve() for arg in sys.argv[1:3])
    if (root / "vllm").exists():
        raise ValueError("Use a tests-only tree so the installed vLLM is not shadowed")
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
            "pytest_mock",
            "-p",
            "no:cacheprovider",
            "-o",
            "asyncio_mode=auto",
            "--junitxml",
            str(output / "junit.xml"),
            "--basetemp",
            str(output / "tmp"),
            "tests/entrypoints/openai/chat_completion",
            *sys.argv[3:],
        ],
        plugins=[Report(output, upstream_utils.RemoteOpenAIServer)],
    )


if __name__ == "__main__":
    sys.exit(main())
