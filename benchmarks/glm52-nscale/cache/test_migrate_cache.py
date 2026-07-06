# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("migrate_cache.py")


class CacheMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        (self.source / "docker/registry/v2/repositories/example").mkdir(
            parents=True
        )
        self.destination.mkdir()
        (self.source / "docker/registry/v2/blob").write_bytes(b"payload")
        (self.source / "docker/registry/v2/repositories/example/link").symlink_to(
            "../../blob"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_migration(
        self, *, check: bool = True, initialize_empty: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT)]
        if initialize_empty:
            command.append("--initialize-empty")
        else:
            command.extend(["--source", str(self.source)])
        command.extend(["--destination", str(self.destination)])
        return subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_copy_is_hashed_and_idempotent(self) -> None:
        first = json.loads(self.run_migration().stdout)
        marker_path = self.destination / ".glm52-migration-v1.json"
        marker_bytes = marker_path.read_bytes()
        self.assertEqual(
            first["marker_sha256"], hashlib.sha256(marker_bytes).hexdigest()
        )
        self.assertEqual(first["marker"], json.loads(marker_bytes))
        self.assertEqual(
            (self.destination / "dockerhub-registry/docker/registry/v2/blob").read_bytes(),
            b"payload",
        )
        self.assertTrue(
            (
                self.destination
                / "dockerhub-registry/docker/registry/v2/repositories/example/link"
            ).is_symlink()
        )

        second = json.loads(self.run_migration().stdout)
        self.assertEqual(second, first)

    def test_explicit_empty_initialization(self) -> None:
        result = json.loads(self.run_migration(initialize_empty=True).stdout)
        self.assertEqual(result["marker"]["source"], "empty-initialization")
        self.assertEqual(result["marker"]["file_count"], 0)
        self.assertEqual(result["marker"]["bytes"], 0)
        self.assertTrue(
            (
                self.destination
                / "dockerhub-registry/docker/registry/v2/repositories"
            ).is_dir()
        )

    def test_tampered_marker_is_rejected(self) -> None:
        self.run_migration()
        marker_path = self.destination / ".glm52-migration-v1.json"
        marker = json.loads(marker_path.read_text())
        marker["bytes"] += 1
        marker_path.write_text(json.dumps(marker))
        result = self.run_migration(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("marker mismatch", result.stderr)

    def test_incomplete_final_is_rejected(self) -> None:
        (self.destination / "dockerhub-registry").mkdir()
        result = self.run_migration(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete migration", result.stderr)

    def test_unexpected_destination_entry_is_rejected(self) -> None:
        (self.destination / "unexpected").write_text("data")
        result = self.run_migration(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("destination is not empty", result.stderr)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable")
    def test_special_source_entry_is_rejected(self) -> None:
        os.mkfifo(self.source / "unsupported.fifo")
        result = self.run_migration(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported cache entry", result.stderr)


if __name__ == "__main__":
    unittest.main()
