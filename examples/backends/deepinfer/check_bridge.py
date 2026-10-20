# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused owner/callback/failure checks against the real Rust extension."""

import argparse
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import _deepinfer_openengine
from engine import RuntimeHandle
from mock_engine import MockDeepInferEngine


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for engine/server transition")
        time.sleep(0.001)


class BridgeChecks(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            model_dir=MODEL_DIR,
            engine="mock",
            engine_options={},
            context_length=8192,
            max_requests=2,
            tick_ms=1,
        )

    def test_owner_and_closed_consumer(self):
        owners, aborted, loads = set(), [], []

        class AuditedEngine(MockDeepInferEngine):
            def __init__(self, tokenizer):
                owners.add(threading.get_ident())
                super().__init__(tokenizer)

            def add_request(self, **kwargs):
                owners.add(threading.get_ident())
                return super().add_request(**kwargs)

            def step(self):
                owners.add(threading.get_ident())
                return super().step()

            def abort_request(self, request_id):
                owners.add(threading.get_ident())
                aborted.append(request_id)
                super().abort_request(request_id)

            def close(self):
                owners.add(threading.get_ident())
                super().close()

        runtime = RuntimeHandle(self.args(), AuditedEngine)
        try:
            runtime.set_status_callback(lambda load, error: loads.append(load))
            runtime.submit(
                "closed",
                [0, 128804],
                [],
                {"max_tokens": 100, "ignore_eos": True},
                lambda **_: False,
            )
            wait_for(lambda: aborted == ["closed"])
        finally:
            runtime.close()
        self.assertEqual(len(owners), 1)
        self.assertNotIn(threading.get_ident(), owners)
        self.assertFalse(runtime._thread.is_alive())
        self.assertEqual(runtime.load, 0)
        self.assertEqual(loads, [0, 1, 0])

    def test_initialization_failure_reaps_owner(self):
        def fail(_):
            raise ValueError("initialization failure")

        threads = set(threading.enumerate())
        with self.assertRaisesRegex(ValueError, "initialization failure"):
            RuntimeHandle(self.args(), fail)
        self.assertEqual(set(threading.enumerate()), threads)

    def test_engine_failure_stops_rust_server(self):
        class FailingEngine(MockDeepInferEngine):
            def step(self):
                raise RuntimeError("step failure")

        runtime = RuntimeHandle(self.args(), FailingEngine)
        server = _deepinfer_openengine.start_server(runtime, "mock", "127.0.0.1:0")
        try:
            runtime.submit(
                "failure",
                [0],
                [],
                {"max_tokens": 10, "ignore_eos": False},
                lambda **_: True,
            )
            wait_for(lambda: not server.is_alive())
            self.assertIn("step failure", runtime.error)
            with self.assertRaisesRegex(RuntimeError, "DeepInfer engine failed"):
                server.shutdown()
        finally:
            server.shutdown()
            runtime.close()
        self.assertFalse(runtime._thread.is_alive())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    MODEL_DIR = parser.parse_args().model_dir
    unittest.main(argv=[__file__])
