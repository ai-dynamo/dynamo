#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Sequence


DRIVER = Path(__file__).with_name("remote-command-driver.py")


def load_driver_module():
    spec = importlib.util.spec_from_file_location("glm52_remote_command_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load remote command driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteCommandDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "lock"
        self.root.mkdir()
        self.state = self.root / "command"
        self.env = os.environ.copy()
        self.env["GLM52_REMOTE_COMMAND_ROOT"] = str(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, *arguments: str) -> dict[str, object]:
        completed = subprocess.run(
            [sys.executable, str(DRIVER), *arguments],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        return json.loads(completed.stdout)

    def start(
        self,
        invocation_id: str,
        command: Sequence[str],
    ) -> dict[str, object]:
        return self.invoke(
            "start",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            invocation_id,
            "--",
            *command,
        )

    def status(self, invocation_id: str) -> dict[str, object]:
        return self.invoke(
            "status",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            invocation_id,
        )

    def wait_finished(
        self,
        invocation_id: str,
        timeout: float = 10,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.status(invocation_id)
            if value["state"] == "finished":
                return value
            self.assertIn(value["state"], {"starting", "running"})
            time.sleep(0.01)
        self.fail("detached command did not finish")

    def test_concurrent_idempotent_start_launches_once(self) -> None:
        marker = Path(self.temporary.name) / "launches"
        command = [
            sys.executable,
            "-c",
            "import sys, time; "
            "open(sys.argv[1], 'a', encoding='utf-8').write('launch\\n'); "
            "time.sleep(0.3)",
            str(marker),
        ]
        arguments = [
            sys.executable,
            str(DRIVER),
            "start",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            "same-attempt",
            "--",
            *command,
        ]
        first = subprocess.Popen(
            arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        second = subprocess.Popen(
            arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn(json.loads(stdout)["state"], {"running", "finished"})
        result = self.wait_finished("same-attempt")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(marker.read_text().splitlines(), ["launch"])

    def test_concurrent_lock_acquisition_converges(self) -> None:
        artifact_root = Path(self.temporary.name) / "artifacts"
        artifact_root.mkdir()
        lock = artifact_root / ".campaign-run.lock"
        state = lock / "command"
        environment = self.env.copy()
        environment["GLM52_REMOTE_COMMAND_ROOT"] = str(lock)
        arguments = [
            sys.executable,
            str(DRIVER),
            "acquire",
            "--state-dir",
            str(state),
            "--invocation-id",
            "same-lock-attempt",
            "--variant",
            "dynamo-vllm",
            "--campaign-phase",
            "ab",
            "--attestation",
            "result/runtime-continuity.json",
            "--argv-sha256",
            "a" * 64,
            "--deployment-sha256",
            "b" * 64,
            "--acquired-at",
            "2026-07-05T00:00:00Z",
        ]
        processes = [
            subprocess.Popen(
                arguments,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            for _ in range(2)
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout)["state"], "acquired")
        owner = json.loads((lock / "owner.json").read_text())
        self.assertEqual(owner["invocation_id"], "same-lock-attempt")

    def test_same_invocation_rejects_different_argv(self) -> None:
        self.start(
            "mismatch",
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
        )
        mismatch = self.start(
            "mismatch",
            [sys.executable, "-c", "raise SystemExit(0)"],
        )
        self.assertEqual(mismatch["state"], "error")
        self.assertIn("does not match", str(mismatch["error"]))
        self.wait_finished("mismatch")

    def test_exact_argv_and_private_files(self) -> None:
        received = Path(self.temporary.name) / "argv.json"
        expected = ["", "two words", "line\nbreak", "*", "$()", "'\"", "--flag"]
        self.start(
            "argv",
            [
                sys.executable,
                "-c",
                "import json, sys; "
                "open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps(sys.argv[2:]))",
                str(received),
                *expected,
            ],
        )
        result = self.wait_finished("argv")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(json.loads(received.read_text()), expected)
        self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o700)
        for path in self.state.iterdir():
            expected_mode = 0o700 if path.is_dir() else 0o600
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected_mode, path)
        self.assertFalse(list(self.state.glob("*.tmp")))
        self.assertFalse(list(self.state.glob(".*.tmp")))

    def test_persists_distinct_exit_codes(self) -> None:
        for exit_code in (0, 7, 255):
            with self.subTest(exit_code=exit_code):
                if self.state.exists():
                    shutil.rmtree(self.state)
                self.start(
                    f"exit-{exit_code}",
                    [sys.executable, "-c", f"raise SystemExit({exit_code})"],
                )
                result = self.wait_finished(f"exit-{exit_code}")
                self.assertEqual(result["exit_code"], exit_code)

    def test_missing_command_is_127(self) -> None:
        self.start("missing", ["/definitely/missing/glm52-command"])
        result = self.wait_finished("missing")
        self.assertEqual(result["exit_code"], 127)

    def test_started_metadata_failure_kills_and_reaps_child(self) -> None:
        module = load_driver_module()
        pid_path = Path(self.temporary.name) / "child.pid"
        self.state.mkdir(mode=0o700)
        (self.state / "command.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "invocation_id": "metadata-failure",
                    "argv": [
                        sys.executable,
                        "-c",
                        "import os, sys, time; "
                        "open(sys.argv[1], 'w', encoding='utf-8').write(str(os.getpid())); "
                        "time.sleep(60)",
                        str(pid_path),
                    ],
                }
            )
        )
        original_atomic_json = module.atomic_json

        def fail_started(path, payload):
            if path.name == "started.json":
                time.sleep(0.1)
                raise FileNotFoundError("injected started metadata failure")
            original_atomic_json(path, payload)

        module.atomic_json = fail_started
        module.run_command(self.state, "metadata-failure")
        status = json.loads((self.state / "status.json").read_text())
        self.assertEqual(status["exit_code"], 125)
        self.assertIn("injected started metadata failure", status["driver_error"])
        child_pid = int(pid_path.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_terminate_publishes_terminal_status(self) -> None:
        self.start(
            "terminate",
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        result = self.invoke(
            "terminate",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            "terminate",
            "--timeout",
            "5",
        )
        self.assertEqual(result["state"], "finished")
        self.assertEqual(result["exit_code"], 143)

    def test_terminate_during_start_prevents_late_launch(self) -> None:
        marker = Path(self.temporary.name) / "must-not-run"
        self.state.mkdir(mode=0o700)
        (self.state / "command.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "invocation_id": "cancel-before-run",
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; from pathlib import Path; "
                        "Path(sys.argv[1]).touch()",
                        str(marker),
                    ],
                }
            )
        )
        cancelled = self.invoke(
            "terminate",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            "cancel-before-run",
            "--timeout",
            "0",
        )
        self.assertEqual(cancelled["exit_code"], 143)
        subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "_run",
                "--state-dir",
                str(self.state),
                "--invocation-id",
                "cancel-before-run",
            ],
            env=self.env,
            check=True,
        )
        self.assertFalse(marker.exists())
        self.assertEqual(self.status("cancel-before-run")["exit_code"], 143)

    def test_term_trap_cannot_turn_cancellation_into_success(self) -> None:
        ready = Path(self.temporary.name) / "ready"
        self.start(
            "term-trap",
            [
                sys.executable,
                "-c",
                "import signal, sys, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                "Path(sys.argv[1]).touch(); time.sleep(60)",
                str(ready),
            ],
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        self.assertTrue(ready.exists())
        result = self.invoke(
            "terminate",
            "--state-dir",
            str(self.state),
            "--invocation-id",
            "term-trap",
            "--timeout",
            "5",
        )
        self.assertEqual(result["state"], "finished")
        self.assertEqual(result["exit_code"], 143)
        self.assertTrue(result["cancelled"])


if __name__ == "__main__":
    unittest.main()
