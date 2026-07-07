#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prefill_lock.py")
SPEC = importlib.util.spec_from_file_location("prefill_lock", MODULE_PATH)
assert SPEC and SPEC.loader
prefill_lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill_lock)


class PrefillLockTest(unittest.TestCase):
    def test_lock_has_exclusive_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_dir = Path(temporary) / ".campaign-run.lock"
            state_dir = Path(temporary) / "state"
            prefill_lock.acquire(lock_dir, "first", "verified", state_dir)
            self.assertEqual(prefill_lock.read_owner(lock_dir)["invocation_id"], "first")
            with self.assertRaisesRegex(RuntimeError, "already held"):
                prefill_lock.acquire(lock_dir, "second", "verified", state_dir)
            with self.assertRaisesRegex(RuntimeError, "another invocation"):
                prefill_lock.release(lock_dir, "second")
            status = Path(temporary) / "status.json"
            status.write_text(json.dumps({"state": "running", "completed": 328}))
            prefill_lock.record_exit(lock_dir, "first", status, 137)
            terminal = json.loads(status.read_text())
            self.assertEqual(terminal["state"], "failed")
            self.assertEqual(terminal["process_exit_code"], 137)
            status.write_text(json.dumps({"state": "complete", "completed": 500}))
            prefill_lock.record_exit(lock_dir, "first", status, 0)
            self.assertEqual(json.loads(status.read_text())["state"], "complete")
            status.write_text(json.dumps({"state": "complete", "completed": 500}))
            prefill_lock.record_exit(lock_dir, "first", status, 1)
            failed_complete = json.loads(status.read_text())
            self.assertEqual(failed_complete["state"], "failed")
            self.assertEqual(failed_complete["process_exit_code"], 1)
            with self.assertRaisesRegex(RuntimeError, "another lock owner"):
                prefill_lock.record_exit(lock_dir, "second", status, 1)
            status.unlink()
            with self.assertRaises(FileNotFoundError):
                prefill_lock.record_exit(lock_dir, "first", status, 1)
            prefill_lock.release(lock_dir, "first")
            self.assertFalse(lock_dir.exists())


if __name__ == "__main__":
    unittest.main()
