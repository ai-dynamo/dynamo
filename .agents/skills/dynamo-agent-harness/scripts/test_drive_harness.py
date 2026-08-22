# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

if importlib.util.find_spec("acp") is None:
    acp = ModuleType("acp")
    acp.PROTOCOL_VERSION = 1
    acp.spawn_agent_process = None
    acp.text_block = lambda text: text
    sys.modules["acp"] = acp

drive_harness = importlib.import_module("drive_harness")


def codex_args(cwd: Path, *, session_final: bool) -> Namespace:
    return Namespace(
        harness="codex",
        base_url="http://dynamo.example",
        model="test-model",
        cwd=cwd,
        add_dir=[],
        capability="verify",
        api_key_env="DYNAMO_API_KEY",
        session_final=session_final,
        session_final_timeout=3.0,
    )


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


class EnvironmentTest(unittest.TestCase):
    def test_codex_child_receives_only_allowlisted_runtime_and_dynamo_values(self):
        parent_environment = {
            "PATH": "/test/bin",
            "HOME": "/test/home",
            "TMPDIR": "/test/tmp",
            "XDG_RUNTIME_DIR": "/test/runtime",
            "LANG": "en_US.UTF-8",
            "DYNAMO_API_KEY": "dynamo-key",
            "CODEX_CONFIG": '{"web_search":"live"}',
            "GITHUB_TOKEN": "must-not-leak",
            "KUBECONFIG": "/sensitive/kubeconfig",
        }
        args = codex_args(Path("."), session_final=False)

        with patch.dict(os.environ, parent_environment, clear=True):
            config = drive_harness.build_config(args)

        self.assertEqual(
            config.environment,
            {
                "PATH": "/test/bin",
                "HOME": "/test/home",
                "TMPDIR": "/test/tmp",
                "XDG_RUNTIME_DIR": "/test/runtime",
                "LANG": "en_US.UTF-8",
                "DYNAMO_API_KEY": "dynamo-key",
                "CODEX_CONFIG": json.dumps(
                    {
                        "model": "test-model",
                        "model_reasoning_effort": "medium",
                    }
                ),
                "NO_BROWSER": "1",
            },
        )

    def test_child_environment_requires_home(self):
        with (
            patch.dict(os.environ, {"PATH": "/test/bin"}, clear=True),
            self.assertRaisesRegex(ValueError, "HOME must be set"),
        ):
            drive_harness.build_child_environment("DYNAMO_API_KEY")

    def test_api_key_cannot_replace_managed_or_runtime_value(self):
        with patch.dict(os.environ, {"HOME": "/test/home"}, clear=True):
            for name in ("PATH", "CODEX_CONFIG"):
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "conflicts"
                ):
                    drive_harness.build_child_environment(name)


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


class FakeConnection:
    def __init__(
        self,
        timeline: list[str],
        *,
        prompt_error: BaseException | None = None,
        session_id: str = "codex-thread-1",
    ):
        self.timeline = timeline
        self.prompt_error = prompt_error
        self.session_id = session_id

    async def initialize(self, **kwargs):
        del kwargs
        return SimpleNamespace(agent_info=None)

    async def authenticate(self, **kwargs):
        del kwargs

    async def new_session(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            session_id=self.session_id,
            modes=SimpleNamespace(available_modes=[SimpleNamespace(id="read-only")]),
            config_options=[],
        )

    async def set_session_mode(self, **kwargs):
        del kwargs

    async def prompt(self, **kwargs):
        del kwargs
        self.timeline.append("prompt")
        if self.prompt_error is not None:
            raise self.prompt_error
        return SimpleNamespace(stop_reason="end_turn", usage=None)


class FakeSpawnContext:
    def __init__(self, connection: FakeConnection, timeline: list[str]):
        self.connection = connection
        self.timeline = timeline

    async def __aenter__(self):
        self.timeline.append("process_open")
        return self.connection, SimpleNamespace(stderr=None)

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.timeline.append("process_closed")


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def run_with_connection(
        self,
        connection: FakeConnection,
        stdin: str,
        *,
        session_final: bool = True,
        final_error: Exception | None = None,
    ) -> tuple[list[str], AsyncMock]:
        timeline = connection.timeline

        async def finalize(*args, **kwargs):
            del args, kwargs
            timeline.append("session_final")
            if final_error is not None:
                raise final_error

        finalizer = AsyncMock(side_effect=finalize)
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                drive_harness,
                "spawn_agent_process",
                return_value=FakeSpawnContext(connection, timeline),
            ),
            patch.object(drive_harness.shutil, "which", return_value="/test/npx"),
            patch.object(drive_harness, "send_session_final", finalizer),
            patch.object(drive_harness, "emit"),
            patch.object(sys, "stdin", io.StringIO(stdin)),
        ):
            await drive_harness.run(
                codex_args(Path(temporary_directory), session_final=session_final)
            )
        return timeline, finalizer

    async def test_normal_close_finalizes_once_after_process_close(self):
        timeline: list[str] = []
        timeline, finalizer = await self.run_with_connection(
            FakeConnection(timeline),
            '{"close":true}\n',
        )

        self.assertEqual(timeline, ["process_open", "process_closed", "session_final"])
        finalizer.assert_awaited_once()
        self.assertEqual(finalizer.await_args.args[1], "codex-thread-1")

    async def test_failed_turn_preserves_error_and_finalizes_after_close(self):
        timeline: list[str] = []
        with self.assertRaisesRegex(RuntimeError, "turn failed"):
            await self.run_with_connection(
                FakeConnection(timeline, prompt_error=RuntimeError("turn failed")),
                '{"prompt":"fail"}\n',
            )

        self.assertEqual(
            timeline,
            ["process_open", "prompt", "process_closed", "session_final"],
        )

    async def test_keyboard_interrupt_finalizes_after_close(self):
        timeline: list[str] = []
        with self.assertRaises(KeyboardInterrupt):
            await self.run_with_connection(
                FakeConnection(timeline, prompt_error=KeyboardInterrupt()),
                '{"prompt":"interrupt"}\n',
            )

        self.assertEqual(
            timeline,
            ["process_open", "prompt", "process_closed", "session_final"],
        )

    async def test_stock_mode_never_sends_terminal_signal(self):
        timeline: list[str] = []
        timeline, finalizer = await self.run_with_connection(
            FakeConnection(timeline),
            '{"close":true}\n',
            session_final=False,
        )

        self.assertEqual(timeline, ["process_open", "process_closed"])
        finalizer.assert_not_awaited()

    async def test_empty_session_id_fails_without_finalization(self):
        timeline: list[str] = []
        with self.assertRaisesRegex(RuntimeError, "empty session ID"):
            await self.run_with_connection(
                FakeConnection(timeline, session_id=""),
                '{"close":true}\n',
            )

        self.assertEqual(timeline, ["process_open", "process_closed"])

    async def test_primary_error_survives_finalization_failure(self):
        timeline: list[str] = []
        with self.assertRaisesRegex(RuntimeError, "turn failed") as raised:
            await self.run_with_connection(
                FakeConnection(timeline, prompt_error=RuntimeError("turn failed")),
                '{"prompt":"fail"}\n',
                final_error=RuntimeError("finalization failed"),
            )

        self.assertIn(
            "ThunderAgent session finalization also failed: finalization failed",
            raised.exception.__notes__,
        )
        self.assertEqual(
            timeline,
            ["process_open", "prompt", "process_closed", "session_final"],
        )


if __name__ == "__main__":
    unittest.main()
