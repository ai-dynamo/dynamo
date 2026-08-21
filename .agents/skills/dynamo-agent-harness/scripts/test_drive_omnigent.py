# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http.client
import importlib
import tempfile
import threading
import unittest
from pathlib import Path

drive_omnigent = importlib.import_module("drive_omnigent")


class OmnigentConfigTest(unittest.TestCase):
    def test_provider_config_uses_responses_and_external_key_reference(self):
        config = drive_omnigent.provider_config(
            "http://127.0.0.1:8000",
            "test-model",
            "DYNAMO_API_KEY",
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
            repo.mkdir()
            cwd.mkdir()

            invocation = drive_omnigent.build_invocation(
                omnigent_repo=repo,
                cwd=cwd,
                runtime_root=runtime,
                base_url="https://dynamo.example.com/v1",
                model="test-model",
                prompt="inspect the workspace",
                api_key_env="DYNAMO_API_KEY",
            )

        self.assertEqual(
            invocation.command[:5],
            ("uvx", "--from", "uv==0.11.8", "uv", "run"),
        )
        self.assertIn("--isolated", invocation.command)
        self.assertIn("--frozen", invocation.command)
        self.assertIn("--no-dev", invocation.command)
        self.assertIn("--harness", invocation.command)
        self.assertIn("codex", invocation.command)
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
        self.assertEqual(invocation.environment["DYNAMO_API_KEY"], "dummy")
        self.assertEqual(invocation.environment["OMNIGENT_DYNAMO_API_KEY"], "dummy")

    def test_normalize_base_url_rejects_non_http_url(self):
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            drive_omnigent.normalize_base_url("localhost:8000")


class CaptureAssessmentTest(unittest.TestCase):
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
        self.assertFalse(assessment["persistent_thread_reuse_verified"])
        self.assertFalse(assessment["session_final_seen"])
        self.assertFalse(assessment["lifecycle_qualified"])

    def test_capture_accepts_background_request_with_its_own_thread(self):
        requests = [
            drive_omnigent.CapturedRequest(
                method="POST",
                path="/v1/responses",
                headers={"authorization": "Bearer test-key", "thread-id": thread_id},
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
