# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts/collect_dynamo_debug_bundle.py"
spec = importlib.util.spec_from_file_location("collect_dynamo_debug_bundle", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT}")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class DeploymentPodSelectorTest(unittest.TestCase):
    def test_selector_combinations(self):
        cases = [
            (None, None, None),
            (None, "app=worker", "app=worker"),
            ("qwen", None, "nvidia.com/dynamo-graph-deployment-name=qwen"),
            (
                "qwen",
                "app=worker",
                "nvidia.com/dynamo-graph-deployment-name=qwen,app=worker",
            ),
        ]

        for deployment_name, selector, expected in cases:
            with self.subTest(deployment_name=deployment_name, selector=selector):
                self.assertEqual(
                    bundle.deployment_pod_selector(deployment_name, selector), expected
                )

    def test_main_scopes_both_pod_queries(self):
        calls = []

        def fake_run(cmd, timeout):
            del timeout
            calls.append(cmd)
            stdout = json.dumps({"items": []}) if cmd[-2:] == ["-o", "json"] else ""
            return {"cmd": cmd, "returncode": 0, "stdout": stdout, "stderr": ""}

        with tempfile.TemporaryDirectory() as outdir:
            argv = [
                str(SCRIPT),
                "--namespace",
                "demo",
                "--deployment-name",
                "qwen",
                "--selector",
                "app=worker",
                "--outdir",
                outdir,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bundle, "run", side_effect=fake_run),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(bundle.main(), 0)

        selector = "nvidia.com/dynamo-graph-deployment-name=qwen,app=worker"
        self.assertIn(
            ["kubectl", "get", "pods", "-n", "demo", "-o", "wide", "-l", selector],
            calls,
        )
        self.assertIn(
            ["kubectl", "get", "pods", "-n", "demo", "-l", selector, "-o", "json"],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
