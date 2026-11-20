# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise published version identity and freshness through the real schema export."""

import contextlib
import io
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from generate_aisimulate_schema import generate, generated_files


class SchemaExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        repository = Path(__file__).resolve().parents[3]
        for relative in (
            "pyproject.toml",
            "container/deps/requirements.aisimulate.txt",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repository / relative).read_bytes())

    def test_installed_version_must_match_the_pinned_release(self):
        for relative in (
            "pyproject.toml",
            "container/deps/requirements.aisimulate.txt",
        ):
            path = self.root / relative
            path.write_text(
                path.read_text().replace(f"aisim=={version('aisim')}", "aisim==0.0.0")
            )

        with self.assertRaisesRegex(ValueError, "install the pinned aisim==0.0.0"):
            generated_files(self.root)

    def test_freshness_check_rejects_changed_active_schema(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(generate(self.root, check=False))
        self.assertTrue(generate(self.root, check=True))

        active = self.root / "deploy/operator/internal/aisimulate/schemas/latest.json"
        active.write_text(active.read_text().replace('"$schema"', '"stale-schema"', 1))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(generate(self.root, check=True))
