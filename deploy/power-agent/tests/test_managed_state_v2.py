# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import managed_state


def _transactional_record(dgd_uid: str) -> dict:
    return {
        "controlMode": managed_state.TRANSACTIONAL_CONTROL_MODE,
        "dgdUID": dgd_uid,
        "component": "prefill",
        "podUID": "pod-1",
        "allocationID": "pod-1/main/GPU-a",
        "targetWatts": 350,
    }


class TestManagedStateV2(unittest.TestCase):
    def test_transactional_round_trip_preserves_ownership(self):
        state = {
            "version": managed_state.STATE_VERSION,
            "managed": {"GPU-a": _transactional_record("dgd-1")},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "managed.json"
            managed_state.save_managed_state(state, path)
            loaded = managed_state.load_managed_state(path)

        self.assertEqual(loaded, state)

    def test_wrong_dgd_enrollment_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "managed.json"
            managed_state.enroll_managed_gpu(
                "GPU-a", _transactional_record("dgd-old"), path
            )
            with self.assertRaises(managed_state.ManagedStateError):
                managed_state.enroll_managed_gpu(
                    "GPU-a", _transactional_record("dgd-new"), path
                )
            loaded = managed_state.load_managed_state(path)

        self.assertEqual(loaded["managed"]["GPU-a"]["dgdUID"], "dgd-old")

    def test_empty_file_is_valid_first_boot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "managed.json"
            path.write_text("", encoding="utf-8")
            loaded = managed_state.load_managed_state(path)

        self.assertEqual(loaded, managed_state.empty_managed_state())

    def test_versionless_foreign_schema_is_corrupt_not_empty_v1(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "managed.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(managed_state.ManagedStateError):
                managed_state.load_managed_state(path)


if __name__ == "__main__":
    unittest.main()
