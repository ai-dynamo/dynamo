# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise source identity and freshness through the real pinned schema export."""

import contextlib
import io
import os
import tempfile
import tomllib
import unittest
from pathlib import Path

from generate_aisimulate_schema import generate, generated_files


class SchemaExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = Path(os.environ["AISIMULATE_SOURCE_DIR"])
        repository = Path(__file__).resolve().parents[3]
        for relative in (
            "Cargo.toml",
            "lib/bindings/python/Cargo.toml",
            "pyproject.toml",
            "container/deps/requirements.aisimulate.txt",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repository / relative).read_bytes())

    def test_source_change_with_unchanged_python_version_requires_new_export(self):
        revision = tomllib.loads((self.root / "Cargo.toml").read_text())["patch"][
            "crates-io"
        ]["aisim-core"]["rev"]
        for relative in ("Cargo.toml", "lib/bindings/python/Cargo.toml"):
            path = self.root / relative
            path.write_text(path.read_text().replace(revision, "0" * 40))

        with self.assertRaisesRegex(ValueError, "clean checkout of " + "0" * 40):
            generated_files(self.root, self.source)

    def test_freshness_check_rejects_changed_active_schema(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(generate(self.root, self.source, check=False))
        self.assertTrue(generate(self.root, self.source, check=True))

        active = self.root / "deploy/operator/internal/aisimulate/schemas/latest.json"
        active.write_text(active.read_text().replace('"$schema"', '"stale-schema"', 1))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(generate(self.root, self.source, check=True))
