#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
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
            prefill_lock.release(lock_dir, "first")
            self.assertFalse(lock_dir.exists())


if __name__ == "__main__":
    unittest.main()
