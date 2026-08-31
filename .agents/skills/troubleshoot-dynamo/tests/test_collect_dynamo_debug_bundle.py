# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_dynamo_debug_bundle.py"
SPEC = importlib.util.spec_from_file_location("collect_dynamo_debug_bundle", SCRIPT)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class DebugBundleTest(unittest.TestCase):
    def test_redact_distinguishes_credentials_from_token_telemetry(self) -> None:
        source = (
            '{"api_key":"json-secret","access_token":"access-secret",'
            '"author":"alice","tokenizer":"Qwen","prompt_tokens":128,'
            '"token_count":4,"monkey":"banana"}\n'
            "CUSTOM_AUTH=plain-secret\n"
        )
        output = bundle.redact(source)

        self.assertNotIn("json-secret", output)
        self.assertNotIn("access-secret", output)
        self.assertNotIn("plain-secret", output)
        self.assertIn('"author":"alice"', output)
        self.assertIn('"tokenizer":"Qwen"', output)
        self.assertIn('"prompt_tokens":128', output)
        self.assertIn('"token_count":4', output)
        self.assertIn('"monkey":"banana"', output)
        self.assertEqual(output.count("<redacted>"), 3)

    def test_pod_discovery_files_do_not_persist_pod_specs(self) -> None:
        result = {
            "cmd": ["kubectl", "get", "pod", "worker-0", "-o", "json"],
            "returncode": 0,
            "stdout": '{"env": [{"name": "CUSTOM_AUTH", "value": "sensitive"}]}',
            "stderr": "",
        }
        with tempfile.TemporaryDirectory() as tempdir:
            bundle.write_pod_discovery_result(Path(tempdir), "pod_json", result)
            output = (Path(tempdir) / "pod_json.txt").read_text()

        self.assertNotIn("sensitive", output)
        self.assertIn("pod JSON used only for local discovery", output)

    def test_deployment_and_explicit_selectors_are_combined(self) -> None:
        self.assertEqual(
            bundle.deployment_pod_selector("qwen", "app=worker"),
            "nvidia.com/dynamo-graph-deployment-name=qwen,app=worker",
        )

    def test_named_deployment_scopes_pods_and_failed_reads_return_nonzero(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], timeout: int) -> dict[str, object]:
            del timeout
            calls.append(cmd)
            failed = cmd[1:3] == ["get", "events"]
            return {
                "cmd": cmd,
                "returncode": 1 if failed else 0,
                "stdout": '{"items": []}' if cmd[-2:] == ["-o", "json"] else "",
                "stderr": "forbidden" if failed else "",
            }

        with tempfile.TemporaryDirectory() as tempdir:
            argv = [
                str(SCRIPT),
                "--namespace",
                "demo",
                "--deployment-name",
                "qwen",
                "--outdir",
                tempdir,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bundle, "run", side_effect=fake_run),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                result = bundle.main()

            summary = json.loads((Path(tempdir) / "summary.json").read_text())

        self.assertEqual(result, 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["failed_commands"], ["events"])
        self.assertEqual(
            summary["pod_selector"],
            "nvidia.com/dynamo-graph-deployment-name=qwen",
        )
        pod_call = next(cmd for cmd in calls if cmd[1:3] == ["get", "pods"])
        self.assertEqual(
            pod_call[-2:],
            ["-l", "nvidia.com/dynamo-graph-deployment-name=qwen"],
        )

    def test_complete_top_level_collection_returns_zero(self) -> None:
        def fake_run(cmd: list[str], timeout: int) -> dict[str, object]:
            del timeout
            return {
                "cmd": cmd,
                "returncode": 0,
                "stdout": '{"items": []}' if cmd[-2:] == ["-o", "json"] else "",
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as tempdir:
            argv = [str(SCRIPT), "--namespace", "demo", "--outdir", tempdir]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bundle, "run", side_effect=fake_run),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                result = bundle.main()
            summary = json.loads((Path(tempdir) / "summary.json").read_text())

        self.assertEqual(result, 0)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["failed_commands"], [])

    def test_current_log_rbac_failure_makes_bundle_incomplete(self) -> None:
        def fake_run(cmd: list[str], timeout: int) -> dict[str, object]:
            del timeout
            if cmd[1:3] == ["get", "pods"] and cmd[-2:] == ["-o", "json"]:
                stdout = '{"items": [{"metadata": {"name": "worker-0"}}]}'
                return {"cmd": cmd, "returncode": 0, "stdout": stdout, "stderr": ""}
            if cmd[1:4] == ["get", "pod", "worker-0"]:
                stdout = '{"spec": {"containers": [{"name": "main"}]}}'
                return {"cmd": cmd, "returncode": 0, "stdout": stdout, "stderr": ""}
            if cmd[1] == "logs":
                return {
                    "cmd": cmd,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "forbidden",
                }
            return {"cmd": cmd, "returncode": 0, "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as tempdir:
            argv = [str(SCRIPT), "--namespace", "demo", "--outdir", tempdir]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bundle, "run", side_effect=fake_run),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                result = bundle.main()
            summary = json.loads((Path(tempdir) / "summary.json").read_text())

        self.assertEqual(result, 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["failed_commands"], ["logs_container_worker-0_main"])
        self.assertEqual(
            summary["optional_failed_commands"],
            ["logs_previous_container_worker-0_main"],
        )

    def test_documented_output_dir_alias_is_accepted(self) -> None:
        def fake_run(cmd: list[str], timeout: int) -> dict[str, object]:
            del timeout
            return {
                "cmd": cmd,
                "returncode": 0,
                "stdout": '{"items": []}' if cmd[-2:] == ["-o", "json"] else "",
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as tempdir:
            argv = [str(SCRIPT), "--namespace", "demo", "--output-dir", tempdir]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bundle, "run", side_effect=fake_run),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                result = bundle.main()

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
