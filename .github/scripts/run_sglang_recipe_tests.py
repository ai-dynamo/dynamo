# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run recipe contracts against a pinned upstream adapter in the SGLang image."""

import argparse
import hashlib
import importlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest


class RequireRecipeCases:
    def __init__(self, adapter: Path, revision: str):
        self.adapter = adapter
        self.revision = revision

    def pytest_configure(self, config):
        # Initialize pytest's plugin rewriting before SGLang imports anyio.
        parser = importlib.import_module("sglang.srt.parser")
        parser.__path__ = [str(self.adapter.parent), *parser.__path__]
        bridge = importlib.import_module("dynamo.frontend.sglang_deepseek")
        if (
            bridge.recipe_renderer is None
            or Path(bridge.recipe_renderer.__file__) != self.adapter
        ):
            raise RuntimeError("The pinned recipe adapter was not loaded")
        recipe = importlib.import_module("deepseek_recipe")
        tokenizers = importlib.import_module("tokenizers")
        print(
            f"Recipe adapter {self.revision}; "
            f"deepseek-recipe {recipe.__version__}; tokenizers {tokenizers.__version__}",
            flush=True,
        )

    def pytest_sessionfinish(self, session):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if session.testscollected == 0 or reporter.stats.get("skipped"):
            reporter.write_sep("!", "Recipe CI requires collected cases without skips")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    url = (
        "https://raw.githubusercontent.com/sgl-project/sglang/"
        f"{args.revision}/python/sglang/srt/parser/deepseek_recipe_renderer.py"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != args.sha256:
        raise RuntimeError("Upstream recipe adapter SHA256 mismatch")

    repo = Path(__file__).resolve().parents[2]
    tests = repo / "components/src/dynamo/frontend/tests/test_sglang_deepseek_recipe.py"
    sys.path.insert(0, str(repo / "components/src"))
    with tempfile.TemporaryDirectory(prefix="sglang-recipe-") as adapter_dir:
        adapter = Path(adapter_dir) / "deepseek_recipe_renderer.py"
        adapter.write_bytes(source)
        return pytest.main(
            ["-q", "--durations=5", f"--confcutdir={tests.parent}", str(tests)],
            plugins=[RequireRecipeCases(adapter, args.revision)],
        )


if __name__ == "__main__":
    raise SystemExit(main())
