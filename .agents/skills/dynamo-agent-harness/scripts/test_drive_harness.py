# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import json
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

if importlib.util.find_spec("acp") is None:
    acp = ModuleType("acp")
    acp.PROTOCOL_VERSION = 1
    acp.spawn_agent_process = None
    acp.text_block = lambda text: text
    sys.modules["acp"] = acp

drive_harness = importlib.import_module("drive_harness")


class EmptyClient:
    def start_turn(self):
        pass

    def response(self):
        return ""


class EmptyConnection:
    async def prompt(self, **kwargs):
        return None


class PromptTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_response_emits_error_without_raising(self):
        with patch.object(drive_harness, "emit") as emit:
            await drive_harness.prompt(
                EmptyConnection(), EmptyClient(), "session-1", "hello"
            )

        emit.assert_called_once_with(
            {
                "type": "error",
                "session_id": "session-1",
                "ok": False,
                "error": "agent returned no text response",
            }
        )


class SessionFinalTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = drive_harness.HarnessConfig(
            command=("npx",),
            environment={},
            gateway_url="http://dynamo.example/v1",
            openai_url="http://dynamo.example/v1",
            api_key="test-key",
            mode="read-only",
            model="test-model",
            session_model="test-model",
        )

    async def test_sends_terminal_signal_for_exact_codex_thread(self):
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b""

        with (
            patch.object(
                drive_harness.urlrequest, "urlopen", return_value=response
            ) as urlopen,
            patch.object(drive_harness, "emit") as emit,
        ):
            await drive_harness.send_session_final(
                self.config,
                "codex-thread-1",
                3.0,
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://dynamo.example/v1/chat/completions")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("X-dynamo-session-id"), "codex-thread-1")
        self.assertEqual(request.get_header("X-dynamo-session-final"), "true")
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        emit.assert_called_once_with(
            {"type": "session_final", "session_id": "codex-thread-1", "ok": True}
        )

    async def test_terminal_failure_is_visible_and_fails_closed(self):
        with (
            patch.object(
                drive_harness.urlrequest,
                "urlopen",
                side_effect=drive_harness.urlerror.URLError("offline"),
            ),
            patch.object(drive_harness, "emit") as emit,
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            await drive_harness.send_session_final(
                self.config,
                "codex-thread-1",
                3.0,
            )

        emit.assert_called_once_with(
            {
                "type": "session_final",
                "session_id": "codex-thread-1",
                "ok": False,
                "error": "session-final request failed: offline",
            }
        )


if __name__ == "__main__":
    unittest.main()
