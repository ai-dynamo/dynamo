# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

CI = Path(__file__).resolve().parents[1]
BASE_SPEC = importlib.util.spec_from_file_location(
    "epd_image_runner", CI / "run_image_test.py"
)
image_test = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(image_test)
SPEC = importlib.util.spec_from_file_location("epd_runner", CI / "run_epd_test.py")
runner = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"run_image_test": image_test}):
    SPEC.loader.exec_module(runner)


class EpdHarnessTests(unittest.TestCase):
    def test_missing_or_wrong_role_and_backend_fail_without_skip(self):
        for role in ("encoder", "prefill", "decode"):
            capability = dict(
                schema_version=1,
                encoder_only=role == "encoder",
                accepts_cache=role != "encoder",
                transfer_backends=[runner.BACKEND],
            )
            runner.require_encoder_support(
                SimpleNamespace(encoder=SimpleNamespace(**capability)), role
            )
            invalid = [
                None,
                {**capability, "schema_version": 0},
                {**capability, "encoder_only": role != "encoder"},
                {**capability, "transfer_backends": []},
            ]
            if role != "encoder":
                invalid.append({**capability, "accepts_cache": False})
            for fields in invalid:
                info = SimpleNamespace(
                    encoder=SimpleNamespace(**fields) if fields else None
                )
                with (
                    self.subTest(role=role, fields=fields),
                    self.assertRaisesRegex(RuntimeError, "Published nightly lacks"),
                ):
                    runner.require_encoder_support(info, role)

    def test_graceful_parent_stop_does_not_signal_the_worker_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("")
            process = Mock(pid=123)
            process.poll.return_value = None
            process.wait.return_value = 0
            with (
                patch.object(runner, "_child_pidfds", return_value=[]),
                patch.object(image_test, "stop_engine") as forced,
            ):
                runner._stop_engine(process, log)
            process.send_signal.assert_called_once_with(signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=60)
            forced.assert_not_called()

    def test_forced_or_nonzero_shutdown_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("")
            for result in (7, subprocess.TimeoutExpired("engine", 60)):
                process = Mock(pid=123)
                process.poll.return_value = None
                process.wait.side_effect = (
                    result if isinstance(result, Exception) else None
                )
                process.wait.return_value = result
                with (
                    self.subTest(result=result),
                    patch.object(runner, "_child_pidfds", return_value=[]),
                    patch.object(image_test, "stop_engine") as forced,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "forced shutdown|exited with 7"
                    ):
                        runner._stop_engine(process, log)
                    forced.assert_called_once_with(process)

    def test_worker_timeout_is_killed_even_after_parent_has_exited(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("")
            process = Mock(pid=123)
            process.poll.return_value = 0
            process.wait.return_value = 0
            with (
                patch.object(runner, "_child_pidfds", return_value=[17]),
                patch.object(
                    runner,
                    "_wait_pidfds",
                    side_effect=[subprocess.TimeoutExpired("workers", 30), None],
                ),
                patch.object(image_test, "stop_engine"),
                patch.object(runner.signal, "pidfd_send_signal") as kill,
                patch.object(runner.os, "close") as close,
                self.assertRaisesRegex(RuntimeError, "forced shutdown"),
            ):
                runner._stop_engine(process, log)
            process.send_signal.assert_not_called()
            kill.assert_called_once_with(17, signal.SIGKILL)
            close.assert_called_once_with(17)

    def test_cleanup_failure_still_stops_siblings_and_retains_original_error(self):
        stopped = []

        def stop(pid):
            stopped.append(pid)
            if pid == 13:
                raise RuntimeError("decode cleanup failed")

        with self.assertRaisesRegex(RuntimeError, "decode cleanup failed") as error:
            with contextlib.ExitStack() as cleanup:
                for pid in (11, 12, 13):
                    cleanup.callback(stop, pid)
                raise ValueError("Rust adapter failed")
        self.assertEqual(stopped, [13, 12, 11])
        self.assertIsInstance(error.exception.__context__, ValueError)
        self.assertEqual(str(error.exception.__context__), "Rust adapter failed")

    def test_native_ipc_warning_is_failure_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text(runner.FATAL_LOG_MESSAGES[0])
            process = Mock(pid=123)
            process.poll.return_value = 0
            process.wait.return_value = 0
            with (
                patch.object(runner, "_child_pidfds", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "Native engine failure"),
            ):
                runner._stop_engine(process, log)

    def test_capability_failure_reaps_every_started_role_in_consumer_first_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            processes = [Mock(pid=number) for number in (11, 12, 13)]
            stopped = []
            with (
                patch.object(
                    runner.subprocess, "Popen", side_effect=processes
                ) as start,
                patch.object(image_test, "wait_for_engine"),
                patch.object(
                    runner,
                    "_stop_engine",
                    side_effect=lambda process, log: stopped.append(process.pid),
                ),
                patch.object(
                    runner,
                    "_check_capabilities",
                    side_effect=[None, None, RuntimeError("missing decode capability")],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "missing decode capability"):
                    with contextlib.ExitStack() as cleanup:
                        for role in ("encoder", "prefill", "decode"):
                            runner._start_engine(
                                Path("model"),
                                "vision",
                                "a" * 40,
                                role,
                                True,
                                results,
                                cleanup,
                            )
            self.assertEqual(stopped, [13, 12, 11])
            for role in ("prefill", "decode"):
                options = json.loads((results / f"{role}.json").read_text())
                self.assertEqual(options["kv_cache_config"]["max_tokens"], 2048)
                self.assertFalse(options["kv_cache_config"]["enable_block_reuse"])
                self.assertEqual(
                    options["multimodal_config"]["encoder_cache_max_bytes"], 0
                )
                self.assertTrue(options["disable_mm_encoder"])
                self.assertEqual(
                    options["cache_transceiver_config"], {"backend": "NIXL"}
                )
            for call in start.call_args_list:
                self.assertEqual(
                    call.kwargs["env"]["TRTLLM_OPENENGINE_ENCODER_TRANSFER"], "nixl"
                )

    def test_second_image_and_fresh_usage_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "response.json"
            tokenizer = Mock()
            tokenizer.decode.side_effect = lambda tokens, **kwargs: (
                "Red, blue" if tokens == [1] else "Blue, red"
            )
            cases = [
                {
                    "token_ids": [index],
                    "prompt_tokens": 82,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                }
                for index in (1, 2)
            ]
            path.write_text(json.dumps({"cases": cases}))
            runner._validate_cases(path, tokenizer)
            for invalid in (
                [cases[0]],
                [{**cases[0], "cached_tokens": 1}, cases[1]],
                [cases[0], cases[0]],
            ):
                path.write_text(json.dumps({"cases": invalid}))
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaises((RuntimeError, AssertionError)),
                ):
                    runner._validate_cases(path, tokenizer)


if __name__ == "__main__":
    unittest.main()
