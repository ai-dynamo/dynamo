# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import http.client
import importlib
import io
import json
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

drive_omnigent = importlib.import_module("drive_omnigent")


class OmnigentConfigTest(unittest.TestCase):
    def test_provider_config_uses_responses_and_external_key_reference(self):
        config = drive_omnigent.provider_config(
            "http://127.0.0.1:8000",
            "test-model",
        )

        provider = config["providers"]["dynamo"]["openai"]
        self.assertEqual(provider["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(provider["api_key_ref"], "env:DYNAMO_API_KEY")
        self.assertEqual(provider["wire_api"], "responses")
        self.assertEqual(provider["models"]["default"], "test-model")

    def test_build_invocation_is_headless_pinned_and_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "omnigent"
            cwd = root / "workspace"
            runtime = root / "runtime"
            launch = cwd / "launch"
            repo.mkdir()
            cwd.mkdir()
            launch.mkdir()

            with mock.patch.object(
                drive_omnigent,
                "_safe_sandbox_backend",
                return_value="linux_bwrap",
            ):
                invocation = drive_omnigent.build_invocation(
                    omnigent_repo=repo,
                    cwd=cwd,
                    runtime_root=runtime,
                    launch_cwd=launch,
                    base_url="https://dynamo.example.com/v1",
                    model="test-model",
                    prompt="inspect the workspace",
                    codex_path=root / "codex",
                    capability="verify",
                    source_environment={
                        "PATH": "/usr/bin",
                        "DYNAMO_API_KEY": "test-dynamo-key",
                        "AWS_SECRET_ACCESS_KEY": "must-not-leak",
                        "GITHUB_TOKEN": "must-not-leak",
                        "OPENAI_API_KEY": "must-not-leak",
                    },
                )
                run_index = invocation.command.index(
                    "run", invocation.command.index("omnigent") + 1
                )
                agent_path = Path(invocation.command[run_index + 1])
                agent_config = json.loads(
                    (agent_path / "config.yaml").read_text(encoding="utf-8")
                )

        self.assertEqual(
            invocation.command[:5],
            ("uvx", "--from", "uv==0.11.8", "uv", "run"),
        )
        self.assertIn("--isolated", invocation.command)
        self.assertIn("--frozen", invocation.command)
        self.assertIn("--no-dev", invocation.command)
        self.assertEqual(
            invocation.command[invocation.command.index("--server") + 1], "local"
        )
        self.assertEqual(
            invocation.command[invocation.command.index("--harness") + 1], "codex"
        )
        self.assertEqual(
            invocation.command[invocation.command.index("--model") + 1], "test-model"
        )
        self.assertNotIn("--no-session", invocation.command)
        self.assertIn("--no-log", invocation.command)
        self.assertEqual(
            invocation.environment["HARNESS_CODEX_CWD"], str(cwd.resolve())
        )
        self.assertEqual(
            invocation.environment["OMNIGENT_CONFIG_HOME"], str(runtime / "config")
        )
        self.assertEqual(
            invocation.environment["OMNIGENT_DATA_DIR"], str(runtime / "data")
        )
        self.assertEqual(invocation.environment["DYNAMO_API_KEY"], "test-dynamo-key")
        self.assertEqual(
            invocation.environment["OMNIGENT_DYNAMO_API_KEY"], "test-dynamo-key"
        )
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", invocation.environment)
        self.assertNotIn("GITHUB_TOKEN", invocation.environment)
        self.assertNotIn("OPENAI_API_KEY", invocation.environment)
        self.assertEqual(
            invocation.environment["OMNIGENT_CODEX_PATH"],
            str((root / "codex").resolve()),
        )
        self.assertNotIn("HARNESS_CODEX_OS_ENV", invocation.environment)
        self.assertEqual(
            agent_config["executor"],
            {
                "type": "omnigent",
                "model": "test-model",
                "config": {"harness": "codex"},
            },
        )
        self.assertEqual(agent_config["skills"], "none")
        os_environment = agent_config["os_env"]
        self.assertEqual(os_environment["type"], "caller_process")
        self.assertEqual(os_environment["cwd"], str(cwd.resolve()))
        self.assertEqual(os_environment["sandbox"]["type"], "linux_bwrap")
        self.assertEqual(
            os_environment["sandbox"]["write_paths"],
            [str(cwd.resolve()), str(runtime / "codex-source"), str(runtime / "tmp")],
        )
        self.assertEqual(os_environment["sandbox"]["env_passthrough"], [])
        prompt = invocation.command[invocation.command.index("-p") + 1]
        self.assertIn("Verification-only task", prompt)
        self.assertIn("do not create, edit, rename, or delete files", prompt)

    def test_normalize_base_url_rejects_non_http_url(self):
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            drive_omnigent.normalize_base_url("localhost:8000")

    def test_act_capability_is_explicit_but_never_requests_full_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "omnigent"
            cwd = root / "workspace"
            launch = cwd / "launch"
            repo.mkdir()
            cwd.mkdir()
            launch.mkdir()
            with mock.patch.object(
                drive_omnigent,
                "_safe_sandbox_backend",
                return_value="darwin_seatbelt",
            ):
                invocation = drive_omnigent.build_invocation(
                    omnigent_repo=repo,
                    cwd=cwd,
                    runtime_root=root / "runtime",
                    launch_cwd=launch,
                    base_url="http://127.0.0.1:8000",
                    model="test-model",
                    prompt="update one file",
                    codex_path=root / "codex",
                    capability="act",
                    source_environment={"PATH": "/usr/bin"},
                )
                run_index = invocation.command.index(
                    "run", invocation.command.index("omnigent") + 1
                )
                agent_path = Path(invocation.command[run_index + 1])
                os_environment = json.loads(
                    (agent_path / "config.yaml").read_text(encoding="utf-8")
                )["os_env"]

        prompt = invocation.command[invocation.command.index("-p") + 1]
        self.assertIn("Workspace edits are explicitly authorized", prompt)
        self.assertNotEqual(os_environment["sandbox"]["type"], "none")
        self.assertTrue(os_environment["sandbox"]["write_paths"])
        self.assertEqual(invocation.environment["DYNAMO_API_KEY"], "dummy")

    def test_linux_sandbox_fails_closed_without_bubblewrap(self):
        with (
            mock.patch.object(drive_omnigent.sys, "platform", "linux"),
            mock.patch.object(drive_omnigent.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "requires bubblewrap"):
                drive_omnigent._safe_sandbox_backend({"PATH": "/usr/bin"})


class CodexPinTest(unittest.TestCase):
    def test_resolve_codex_cli_accepts_exact_pinned_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex = Path(temp_dir) / "codex"
            codex.write_text("#!/bin/sh\n", encoding="utf-8")
            codex.chmod(0o755)
            completed = subprocess.CompletedProcess(
                (str(codex), "--version"),
                0,
                stdout="codex-cli 0.147.0\n",
                stderr="",
            )
            with mock.patch.object(
                drive_omnigent.subprocess, "run", return_value=completed
            ):
                resolved = drive_omnigent.resolve_codex_cli(codex, {"PATH": "/usr/bin"})

        self.assertEqual(resolved, codex.resolve())

    def test_resolve_codex_cli_rejects_version_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex = Path(temp_dir) / "codex"
            codex.write_text("#!/bin/sh\n", encoding="utf-8")
            codex.chmod(0o755)
            completed = subprocess.CompletedProcess(
                (str(codex), "--version"),
                0,
                stdout="codex-cli 0.148.0\n",
                stderr="",
            )
            with mock.patch.object(
                drive_omnigent.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(
                    ValueError, "expected exact version 0.147.0"
                ):
                    drive_omnigent.resolve_codex_cli(codex, {"PATH": "/usr/bin"})


class ExecutionCleanupTest(unittest.TestCase):
    def _invocation(self, root: Path) -> drive_omnigent.Invocation:
        (root / "runtime").mkdir()
        launch = root / "workspace" / "launch"
        launch.mkdir()
        return drive_omnigent.Invocation(
            command=("omnigent-test", "run"),
            environment={"PATH": "/usr/bin", "DYNAMO_API_KEY": "test-key"},
            omnigent_repo=root / "omnigent",
            runtime_root=root / "runtime",
            launch_cwd=launch.resolve(),
            sandbox_backend="darwin_seatbelt",
        )

    def test_execute_invocation_stops_after_success_and_removes_empty_codex_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "workspace"
            cwd.mkdir()
            invocation = self._invocation(root)

            def fake_run(command, **kwargs):
                del kwargs
                if command != invocation.command:
                    return subprocess.CompletedProcess(
                        command, 0, stdout="stopped", stderr=""
                    )
                (cwd / ".codex-tmp").mkdir()
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with mock.patch.object(
                drive_omnigent.subprocess, "run", side_effect=fake_run
            ) as run:
                execution = drive_omnigent.execute_invocation(
                    invocation, cwd=cwd, timeout=30, capture_output=True
                )

            self.assertEqual(run.call_count, 2)
            self.assertIn("-c", run.call_args_list[1].args[0])
            self.assertNotIn("stop", run.call_args_list[1].args[0])
            self.assertEqual(run.call_args_list[0].kwargs["cwd"], invocation.launch_cwd)
            self.assertEqual(run.call_args_list[1].kwargs["cwd"], invocation.launch_cwd)
            self.assertTrue(execution.result.ok)
            self.assertTrue(execution.cleanup.ok)
            self.assertTrue(execution.codex_temp_clean)
            self.assertFalse((cwd / ".codex-tmp").exists())

    def test_execute_invocation_still_stops_after_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "workspace"
            cwd.mkdir()
            invocation = self._invocation(root)
            results = [
                subprocess.TimeoutExpired(invocation.command, 30, stderr="hung"),
                subprocess.CompletedProcess(
                    ("omnigent", "stop"), 0, stdout="stopped", stderr=""
                ),
            ]
            with mock.patch.object(
                drive_omnigent.subprocess, "run", side_effect=results
            ) as run:
                execution = drive_omnigent.execute_invocation(
                    invocation, cwd=cwd, timeout=30, capture_output=True
                )

        self.assertEqual(run.call_count, 2)
        self.assertTrue(execution.result.timed_out)
        self.assertEqual(execution.result.stderr, "hung")
        self.assertTrue(execution.cleanup.ok)

    def test_codex_temp_cleanup_waits_for_late_empty_directory(self):
        path = mock.Mock(spec=Path)
        path.rmdir.side_effect = [OSError("still draining"), None]

        with mock.patch.object(drive_omnigent.time, "sleep") as sleep:
            clean = drive_omnigent._remove_empty_codex_temp(path)

        self.assertTrue(clean)
        self.assertEqual(path.rmdir.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_execute_invocation_rejects_preexisting_codex_temp_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "workspace"
            cwd.mkdir()
            (cwd / ".codex-tmp").mkdir()
            invocation = self._invocation(root)

            with (
                mock.patch.object(drive_omnigent.subprocess, "run") as run,
                self.assertRaisesRegex(
                    ValueError, "pre-existing Codex temporary state"
                ),
            ):
                drive_omnigent.execute_invocation(
                    invocation, cwd=cwd, timeout=30, capture_output=True
                )

            run.assert_not_called()

    def test_cleanup_failure_has_structured_stderr_and_dirty_runtime_status(self):
        execution = drive_omnigent.Execution(
            result=drive_omnigent.ProcessStatus(0, "", ""),
            cleanup=drive_omnigent.ProcessStatus(
                1, "partial stop", "runner still active"
            ),
            codex_temp_clean=False,
        )

        diagnostic = drive_omnigent.execution_diagnostic(
            execution, runtime_removed=False
        )

        self.assertTrue(diagnostic["cleanup"]["attempted"])
        self.assertEqual(diagnostic["cleanup"]["exit_code"], 1)
        self.assertEqual(diagnostic["cleanup"]["stderr"], "runner still active")
        self.assertFalse(diagnostic["codex_temp_clean"])
        self.assertFalse(diagnostic["disposable_runtime_removed"])

    def test_cleanup_diagnostic_redacts_dynamo_credential(self):
        execution = drive_omnigent.Execution(
            result=drive_omnigent.ProcessStatus(1, "", "gateway rejected secret-token"),
            cleanup=drive_omnigent.ProcessStatus(
                1, "secret-token", "could not stop secret-token"
            ),
            codex_temp_clean=False,
        )

        diagnostic = drive_omnigent.execution_diagnostic(
            execution,
            runtime_removed=False,
            sensitive_values=("secret-token",),
        )

        serialized = json.dumps(diagnostic)
        self.assertNotIn("secret-token", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_run_emits_structured_cleanup_failure_to_stderr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "workspace"
            cwd.mkdir()
            invocation = self._invocation(root)
            execution = drive_omnigent.Execution(
                result=drive_omnigent.ProcessStatus(0, "", ""),
                cleanup=drive_omnigent.ProcessStatus(
                    1, "partial stop", "runner still active"
                ),
                codex_temp_clean=True,
            )
            args = types.SimpleNamespace(
                omnigent_repo=root / "omnigent",
                codex_bin=None,
                cwd=cwd,
                base_url="http://127.0.0.1:8000",
                model="test-model",
                prompt="inspect",
                capability="verify",
                timeout=30,
            )
            stderr = io.StringIO()
            with (
                mock.patch.object(drive_omnigent, "validate_checkout"),
                mock.patch.object(
                    drive_omnigent,
                    "resolve_codex_cli",
                    return_value=root / "codex",
                ),
                mock.patch.object(
                    drive_omnigent, "build_invocation", return_value=invocation
                ),
                mock.patch.object(
                    drive_omnigent, "execute_invocation", return_value=execution
                ),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = drive_omnigent._run_dynamo(args)

        payload = json.loads(stderr.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(
            payload["omnigent_execution"]["cleanup"]["stderr"],
            "runner still active",
        )
        self.assertTrue(payload["omnigent_execution"]["disposable_runtime_removed"])


class CaptureAssessmentTest(unittest.TestCase):
    def test_capture_requires_the_assistant_reply_to_be_consumed(self):
        execution = drive_omnigent.Execution(
            result=drive_omnigent.ProcessStatus(0, "", ""),
            cleanup=drive_omnigent.ProcessStatus(0, "", ""),
            codex_temp_clean=True,
        )
        assessment = {
            "protocol_compatible": True,
            "safe_sandbox_observed": True,
            "assistant_reply_seen": False,
        }

        self.assertFalse(
            drive_omnigent._capture_is_successful(
                assessment, execution, runtime_removed=True
            )
        )
        assessment["assistant_reply_seen"] = True
        self.assertTrue(
            drive_omnigent._capture_is_successful(
                assessment, execution, runtime_removed=True
            )
        )

    def test_capture_models_endpoint_accepts_codex_version_query(self):
        state = drive_omnigent.CaptureState()
        server = drive_omnigent.CaptureServer(state)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            connection.request("GET", "/v1/models?client_version=0.147.0")
            response = connection.getresponse()
            response.read()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

        self.assertEqual(response.status, 200)

    def test_capture_proves_protocol_but_reports_missing_session_final(self):
        request = drive_omnigent.CapturedRequest(
            method="POST",
            path="/v1/responses",
            headers={
                "authorization": "Bearer test-key",
                "thread-id": "thread-123",
                "x-codex-turn-metadata": '{"sandbox":"seatbelt"}',
            },
            body={"model": "test-model", "stream": True},
        )

        assessment = drive_omnigent.assess_capture(
            [request],
            expected_model="test-model",
            expected_api_key="test-key",
        )

        self.assertTrue(assessment["protocol_compatible"])
        self.assertEqual(assessment["thread_ids"], ["thread-123"])
        self.assertTrue(assessment["session_affinity_ok"])
        self.assertTrue(assessment["safe_sandbox_observed"])
        self.assertEqual(assessment["observed_sandbox_modes"], ["seatbelt"])
        self.assertFalse(assessment["persistent_thread_reuse_verified"])
        self.assertFalse(assessment["session_final_seen"])
        self.assertFalse(assessment["lifecycle_qualified"])

    def test_capture_accepts_background_request_with_its_own_thread(self):
        requests = [
            drive_omnigent.CapturedRequest(
                method="POST",
                path="/v1/responses",
                headers={
                    "authorization": "Bearer test-key",
                    "thread-id": thread_id,
                    "x-codex-turn-metadata": '{"sandbox":"read-only"}',
                },
                body={"model": "test-model", "stream": True, "input": prompt},
            )
            for thread_id, prompt in (
                ("thread-main", "main"),
                ("thread-title", "title"),
            )
        ]

        assessment = drive_omnigent.assess_capture(
            requests,
            expected_model="test-model",
            expected_api_key="test-key",
        )

        self.assertTrue(assessment["protocol_compatible"])
        self.assertEqual(assessment["unique_thread_count"], 2)
        self.assertTrue(assessment["session_affinity_ok"])
        self.assertFalse(assessment["persistent_thread_reuse_verified"])

    def test_capture_rejects_unsandboxed_codex_metadata(self):
        request = drive_omnigent.CapturedRequest(
            method="POST",
            path="/v1/responses",
            headers={
                "authorization": "Bearer test-key",
                "thread-id": "thread-123",
                "x-codex-turn-metadata": '{"sandbox":"none"}',
            },
            body={"model": "test-model", "stream": True},
        )

        assessment = drive_omnigent.assess_capture(
            [request],
            expected_model="test-model",
            expected_api_key="test-key",
        )

        self.assertTrue(assessment["protocol_compatible"])
        self.assertFalse(assessment["safe_sandbox_observed"])
        self.assertEqual(assessment["observed_sandbox_modes"], ["none"])

    def test_capture_requires_codex_thread_id(self):
        request = drive_omnigent.CapturedRequest(
            method="POST",
            path="/v1/responses",
            headers={"authorization": "Bearer test-key"},
            body={"model": "test-model", "stream": True},
        )

        assessment = drive_omnigent.assess_capture(
            [request],
            expected_model="test-model",
            expected_api_key="test-key",
        )

        self.assertFalse(assessment["protocol_compatible"])
        self.assertFalse(assessment["session_affinity_ok"])


if __name__ == "__main__":
    unittest.main()
