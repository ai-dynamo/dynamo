#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from source_provenance import (
    SourceProvenanceError,
    build_source_provenance,
    verify_source_provenance,
)


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "eval/subdir").mkdir(parents=True)
        (self.root / "campaign.env").write_text("MAX_MODEL_LEN=409600\n")
        (self.root / "eval/run.sh").write_text("#!/bin/sh\n")
        (self.root / "eval/subdir/helper.py").write_text("VALUE = 1\n")
        self.commit = "c" * 40
        self.document = build_source_provenance(
            self.root,
            source_commit=self.commit,
            source_branch="rmccormick/glm52",
            bundle_sha256="b" * 64,
        )
        self.provenance = self.root / "source-provenance.json"
        self.provenance.write_text(json.dumps(self.document, indent=2) + "\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_tree_returns_path_free_compact_identity(self) -> None:
        compact = verify_source_provenance(self.provenance, self.root, self.commit)
        self.assertEqual(compact["source_commit"], self.commit)
        self.assertEqual(compact["source_file_count"], 3)
        self.assertEqual(compact["eval_file_count"], 2)
        serialized = json.dumps(compact)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("source_files", compact)

    def test_changed_missing_and_unexpected_sources_are_rejected(self) -> None:
        (self.root / "eval/run.sh").write_text("changed\n")
        with self.assertRaisesRegex(SourceProvenanceError, "changed=.*eval/run.sh"):
            verify_source_provenance(self.provenance, self.root, self.commit)

        (self.root / "eval/run.sh").write_text("#!/bin/sh\n")
        (self.root / "eval/subdir/helper.py").unlink()
        with self.assertRaisesRegex(SourceProvenanceError, "missing=.*helper.py"):
            verify_source_provenance(self.provenance, self.root, self.commit)

        (self.root / "eval/subdir/helper.py").write_text("VALUE = 1\n")
        (self.root / "eval/shadow.py").write_text("raise SystemExit\n")
        with self.assertRaisesRegex(SourceProvenanceError, "unexpected=.*shadow.py"):
            verify_source_provenance(self.provenance, self.root, self.commit)

    def test_commit_and_manifest_aggregates_are_bound(self) -> None:
        with self.assertRaisesRegex(SourceProvenanceError, "serving deployment"):
            verify_source_provenance(self.provenance, self.root, "d" * 40)

        document = dict(self.document)
        document["eval_tree_sha256"] = "0" * 64
        self.provenance.write_text(json.dumps(document))
        with self.assertRaisesRegex(SourceProvenanceError, "eval_tree_sha256"):
            verify_source_provenance(self.provenance, self.root, self.commit)

    def test_python_bytecode_is_not_campaign_source(self) -> None:
        cache = self.root / "eval/__pycache__"
        cache.mkdir()
        (cache / "helper.cpython-312.pyc").write_bytes(b"generated")
        verify_source_provenance(self.provenance, self.root, self.commit)

    def test_permission_mode_drift_is_rejected(self) -> None:
        script = self.root / "eval/run.sh"
        script.chmod(0o755)
        with self.assertRaisesRegex(SourceProvenanceError, "changed=.*eval/run.sh"):
            verify_source_provenance(self.provenance, self.root, self.commit)

    def test_timestamp_and_branch_types_are_validated(self) -> None:
        for field, value, message in (
            ("generated_at", "not-a-time", "generated_at"),
            ("source_branch", 7, "source_branch"),
        ):
            with self.subTest(field=field):
                document = dict(self.document)
                document[field] = value
                self.provenance.write_text(json.dumps(document))
                with self.assertRaisesRegex(SourceProvenanceError, message):
                    verify_source_provenance(self.provenance, self.root, self.commit)


if __name__ == "__main__":
    unittest.main()
