# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import psutil

from tests.utils import managed_process

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
                accepts_handoff=role != "encoder",
                supported_transfer_backends=(
                    [] if role == "encoder" else [runner.BACKEND]
                ),
                supports_abort_cleanup=role == "encoder",
                requires_handoff=role != "encoder",
            )
            if role == "encoder":
                capability["transfer_backend"] = runner.BACKEND

            def info_for(fields):
                if fields is None:
                    return SimpleNamespace(HasField=lambda name: False)
                return SimpleNamespace(
                    HasField=lambda name: name == "encoder",
                    encoder=SimpleNamespace(
                        **fields,
                        HasField=lambda name: name in fields,
                    ),
                )

            runner.require_encoder_support(info_for(capability), role)
            invalid = [
                None,
                {**capability, "schema_version": 0},
                {**capability, "schema_version": 2},
                {**capability, "encoder_only": role != "encoder"},
                (
                    {**capability, "transfer_backend": "unknown"}
                    if role == "encoder"
                    else {**capability, "supported_transfer_backends": []}
                ),
                {k: v for k, v in capability.items() if k != "encoder_only"},
            ]
            if role != "encoder":
                invalid.append({**capability, "accepts_handoff": False})
                invalid.append({**capability, "requires_handoff": False})
            else:
                invalid.append({**capability, "supports_abort_cleanup": False})
                invalid.append({**capability, "requires_handoff": True})
                invalid.append(
                    {**capability, "supported_transfer_backends": [runner.BACKEND]}
                )
                runner.require_encoder_support(
                    info_for(
                        {k: v for k, v in capability.items() if k != "requires_handoff"}
                    ),
                    role,
                )
            for fields in invalid:
                info = info_for(fields)
                with (
                    self.subTest(role=role, fields=fields),
                    self.assertRaisesRegex(RuntimeError, "Published nightly lacks"),
                ):
                    runner.require_encoder_support(info, role)

    def start_test_engine(self, directory, *, detached=False, orphan=False, message=""):
        ready = directory / "ready"
        script = textwrap.dedent(
            """
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            directory, detached, orphan, message = sys.argv[1:]
            directory = Path(directory)
            child = subprocess.Popen([
                sys.executable, "-c",
                "import signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[2] == 'True' "
                "else lambda *_: sys.exit(0)); Path(sys.argv[1]).touch(); time.sleep(60)",
                str(directory / "child-ready"), orphan,
            ], start_new_session=detached == "True")

            def stop(*_):
                child.terminate()
                child.wait(timeout=3)
                sys.exit(0)

            signal.signal(signal.SIGTERM, stop)
            while not (directory / "child-ready").exists():
                time.sleep(0.01)
            print(message, flush=True)
            (directory / "ready").write_text(str(child.pid))
            while not (directory / "exit").exists():
                time.sleep(0.01)
            """
        )
        process = runner.ManagedProcess(
            command=[
                sys.executable,
                "-c",
                script,
                str(directory),
                str(detached),
                str(orphan),
                message,
            ],
            log_dir=str(directory),
            display_name="test-engine",
            health_check_funcs=[ready.exists],
            timeout=5,
            graceful_shutdown=True,
            terminate_all_matching_process_names=False,
        )
        self.addCleanup(process.__exit__, None, None, None)
        process.__enter__()
        child = psutil.Process(int(ready.read_text()))
        child.create_time()
        self.addCleanup(self.kill_test_child, child)
        return process, child

    @staticmethod
    def kill_test_child(child):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    def assert_child_stopped(self, child):
        self.assertTrue(
            not child.is_running() or child.status() == psutil.STATUS_ZOMBIE,
            f"Engine worker {child.pid} survived teardown",
        )

    def test_graceful_parent_stop_does_not_signal_the_worker_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            process, child = self.start_test_engine(Path(tmp))
            with patch.object(
                managed_process.os, "killpg", wraps=os.killpg
            ) as signal_group:
                runner._stop_engine(process)
            signal_group.assert_not_called()
            self.assert_child_stopped(child)
            self.assertIsNone(process.proc)

    def test_nonzero_shutdown_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = runner.ManagedProcess(
                command=[sys.executable, "-c", "raise SystemExit(7)"],
                log_dir=tmp,
                display_name="failed-engine",
                graceful_shutdown=True,
                terminate_all_matching_process_names=False,
            )
            process.__enter__()
            process.proc.wait(timeout=3)
            with self.assertRaisesRegex(RuntimeError, "exited with 7"):
                runner._stop_engine(process)
            self.assertIsNone(process.proc)

    def test_worker_timeout_is_killed_even_after_parent_has_exited(self):
        for detached in (False, True):
            with self.subTest(detached=detached), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                process, child = self.start_test_engine(
                    directory, detached=detached, orphan=True
                )
                with runner.ManagedProcess(
                    command=[sys.executable, "-c", "import time; time.sleep(60)"],
                    log_dir=tmp,
                    display_name="unrelated-engine",
                    terminate_all_matching_process_names=False,
                ) as sentinel:
                    (directory / "exit").touch()
                    process.proc.wait(timeout=3)
                    self.assertTrue(child.is_running())
                    process.stop_gracefully = functools.partial(
                        process.stop_gracefully, timeout=0.1, worker_timeout=0.1
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "required forced shutdown.*test-engine.log.txt"
                    ):
                        runner._stop_engine(process)
                    self.assertIsNone(sentinel.proc.poll())
                self.assert_child_stopped(child)
                self.assertIsNone(process.proc)

    def test_session_tracks_worker_created_after_startup_when_parent_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            script = textwrap.dedent(
                """
                import subprocess, sys, time
                from pathlib import Path
                directory = Path(sys.argv[1])
                (directory / "parent-ready").touch()
                while not (directory / "spawn").exists():
                    time.sleep(0.01)
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                (directory / "child-pid").write_text(str(child.pid))
                """
            )
            process = runner.ManagedProcess(
                command=[sys.executable, "-c", script, tmp],
                log_dir=tmp,
                display_name="late-worker-engine",
                health_check_funcs=[(directory / "parent-ready").exists],
                timeout=5,
                graceful_shutdown=True,
                terminate_all_matching_process_names=False,
            )
            self.addCleanup(process.__exit__, None, None, None)
            process.__enter__()
            (directory / "spawn").touch()
            process.proc.wait(timeout=3)
            child = psutil.Process(int((directory / "child-pid").read_text()))
            child.create_time()
            self.addCleanup(self.kill_test_child, child)
            process.stop_gracefully = functools.partial(
                process.stop_gracefully, timeout=0.1, worker_timeout=0.1
            )
            with self.assertRaisesRegex(RuntimeError, "required forced shutdown"):
                runner._stop_engine(process)
            self.assert_child_stopped(child)

    def test_cleanup_failure_still_stops_siblings_and_retains_original_error(self):
        stopped = []
        rust_failure = ValueError("Rust adapter failed")

        class TestProcess:
            def __init__(self, *, display_name, log_dir, **kwargs):
                self.role = display_name.removesuffix("-engine")
                self.log_path = Path(log_dir) / f"{display_name}.log.txt"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                stopped.append(self.role)
                if self.role == "decode":
                    raise RuntimeError("decode cleanup failed")

            def read_logs(self):
                return ""

        def rust_test(binary, env, selected, results, **kwargs):
            if selected == image_test.EPD_TESTS["image-epd-disaggregated"]:
                raise rust_failure
            Path(env["TRTLLM_E2E_RESULT"]).write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "token_ids": [index],
                                "prompt_tokens": 82,
                                "completion_tokens": 2,
                                "cached_tokens": 0,
                            }
                            for index in (1, 2)
                        ]
                    }
                )
            )

        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = image_test.FIXTURES["image"][
            "token_ids"
        ]
        tokenizer.decode.side_effect = lambda tokens, **kwargs: (
            "Red, blue" if tokens == [1] else "Blue, red"
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        )
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            binary = results / "rust-tests"
            binary.touch()
            selected = image_test.EPD_TESTS["image-epd-disaggregated"]
            args = SimpleNamespace(
                scenario="image-epd-disaggregated", binary=binary, model_path=results
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": SimpleNamespace(snapshot_download=Mock()),
                        "transformers": transformers,
                    },
                ),
                patch.object(runner, "ManagedProcess", TestProcess),
                patch.object(runner, "_check_capabilities"),
                patch.object(
                    image_test,
                    "load_runtime_config",
                    return_value={"model": "vision", "model_revision": "a" * 40},
                ),
                patch.object(
                    runner.subprocess,
                    "check_output",
                    side_effect=[
                        f"{selected}: test\n",
                        "tests::e2e::e2e_real_openengine_image: test\n",
                    ],
                ),
                patch.object(image_test, "routing_fixture", return_value=("a", "b")),
                patch.object(image_test, "run_rust_test", side_effect=rust_test),
                self.assertRaisesRegex(RuntimeError, "decode cleanup failed") as error,
            ):
                runner.run(args, results)
        self.assertEqual(stopped, ["baseline", "decode", "prefill", "encoder"])
        chain = []
        context = error.exception
        while context is not None:
            chain.append(context)
            context = context.__context__
        self.assertIn(rust_failure, chain)

    def test_native_ipc_warning_is_failure_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            process, child = self.start_test_engine(
                Path(tmp), message=runner.FATAL_LOG_MESSAGES[0]
            )
            with self.assertRaisesRegex(
                RuntimeError, "Native engine failure.*test-engine.log.txt"
            ):
                runner._stop_engine(process)
            self.assert_child_stopped(child)

    def test_capability_failure_reaps_every_started_role_in_consumer_first_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            processes = [MagicMock(pid=number) for number in (11, 12, 13)]
            stopped = []
            with (
                patch.object(runner, "ManagedProcess", side_effect=processes) as start,
                patch.object(
                    runner,
                    "_stop_engine",
                    side_effect=lambda process: stopped.append(process.pid),
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
                self.assertTrue(call.kwargs["graceful_shutdown"])
                self.assertFalse(call.kwargs["terminate_all_matching_process_names"])
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
