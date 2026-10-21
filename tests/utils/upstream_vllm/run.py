# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the frozen upstream nodes in an isolated Python process."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


class Selection:
    """Keep upstream collection exact and retain fixture dependencies."""

    def __init__(self, nodes, output):
        """Bind one pytest invocation to its ordered allowlist and evidence directory."""
        self.nodes = nodes
        self.output = output

    def pytest_collection_finish(self, session):
        """Fail unexpected collection before any fixture starts an engine."""
        actual = [item.nodeid for item in session.items]
        if actual != self.nodes:
            raise RuntimeError(f"Unexpected upstream collection: {actual}")
        (self.output / "collection.json").write_text(
            json.dumps(
                {item.nodeid: item.fixturenames for item in session.items}, indent=2
            )
        )


def unavailable_shutdown_helper(*args, **kwargs):
    """Fail if newer test utilities invoke a helper absent from the installed runtime."""
    raise RuntimeError("This vLLM runtime lacks the requested shutdown helper")


def main():
    """Verify sources, bind the server class before test imports and return pytest status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "dynamo"], required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    root, output = args.upstream.resolve(), args.output.resolve()
    if (root / "vllm").exists():
        parser.error(
            "Supply a tests-only tree so the installed runtime is not shadowed"
        )
    for name, expected in selection["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            parser.error(f"Changed upstream input: {name}")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["UPSTREAM_CHAT_OUTPUT"] = str(output)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # This process belongs only to upstream pytest; do not inherit Dynamo's -m/-p flags.
    os.environ.pop("PYTEST_ADDOPTS", None)
    # Imports must follow path setup: both repositories have a package named tests.
    engine_utils = importlib.import_module("vllm.v1.engine.utils")
    legacy_shutdown = not hasattr(engine_utils, "get_engine_process_shutdown_timeout")
    if legacy_shutdown:
        engine_utils.get_engine_process_shutdown_timeout = unavailable_shutdown_helper
    upstream_utils = importlib.import_module("tests.utils")
    servers = importlib.import_module("dynamo_server")
    base = (
        servers.ReferenceServer
        if args.backend == "reference"
        else servers.RemoteDynamoServer
    )

    class PinnedServer(base):
        """Preserve the fixture body while applying the recorded resource/model profile."""

        def __init__(self, model, vllm_serve_args, **kwargs):
            """Pin model revision and bound KV allocation equally on both endpoints."""
            profile = next(p for p in selection["models"].values() if p["id"] == model)
            super().__init__(
                model,
                [
                    *vllm_serve_args,
                    "--revision",
                    profile["revision"],
                    "--tokenizer-revision",
                    profile["revision"],
                    "--served-model-name",
                    model,
                    "--max-num-seqs",
                    "8",
                    "--kv-cache-memory-bytes",
                    "1073741824",
                    "--no-enable-prefix-caching",
                ],
                **kwargs,
            )

    if legacy_shutdown:
        # shortcut: released 0.28 lacks this helper; use its generic shutdown grace.
        servers.ReferenceServer._get_process_termination_timeout = (
            upstream_utils.RemoteVLLMServer._get_process_termination_timeout
        )
    upstream_utils.RemoteOpenAIServer = PinnedServer
    nodes = [case["nodeid"] for case in selection["cases"]]
    (output / "environment.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "vllm_test_source": selection["vllm_test_source"],
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "ai-dynamo", "openai", "pytest")
                },
                "legacy_shutdown_guard": legacy_shutdown,
                "models": selection["models"],
            },
            indent=2,
        )
    )
    pytest = importlib.import_module("pytest")
    argv = [
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
    ]
    if args.collect_only:
        argv.append("--collect-only")
    return pytest.main(argv + nodes, plugins=[Selection(nodes, output)])


if __name__ == "__main__":
    sys.exit(main())
