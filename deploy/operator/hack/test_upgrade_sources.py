# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release capture must execute the selected production tree, not post-cut main."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_upgrade_fixtures as fixtures


class UpgradeCaptureSourceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@example.com")
        chart = self.root / fixtures.CHART
        chart.parent.mkdir(parents=True)
        chart.write_text("appVersion: 1.6.0\n")
        self.worker = self.root / "deploy/operator/internal/dynamo/worker.go"
        self.worker.parent.mkdir(parents=True)
        self.worker.write_text("package dynamo\n")
        self.git("add", ".")
        self.git(
            "-c", "commit.gpgsign=false", "commit", "-qm", "released production tree"
        )
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.git("tag", "v1.6.0")
        replacement = patch.object(fixtures, "ROOT", self.root)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.published = {
            "tag_name": "v1.6.0",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-10-20",
        }
        self.official_revision = self.revision
        self.real_output = subprocess.check_output

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True)

    def upstream_output(self, command, **kwargs):
        if command[0] == "gh":
            return json.dumps(
                self.published
                if "/releases/" in command[-1]
                else {"sha": self.official_revision}
            )
        return self.real_output(command, **kwargs)

    def test_release_capture_rejects_dirty_and_post_cut_production(self):
        with patch.object(
            fixtures.subprocess, "check_output", side_effect=self.upstream_output
        ):
            source = fixtures.capture_source("1.6.0", release_tag="v1.6.0")
            self.assertEqual(source["sourceRevision"], self.revision)
            harness = self.worker.with_name("capture_test.go")
            harness.write_text("package dynamo\n")
            self.assertEqual(
                fixtures.capture_source("1.6.0", release_tag="v1.6.0"), source
            )
            self.worker.write_text("package dynamo // post-cut lowering change\n")
            with self.assertRaisesRegex(ValueError, "production tree differs"):
                fixtures.capture_source("1.6.0", release_tag="v1.6.0")
            self.git("add", ".")
            self.git("-c", "commit.gpgsign=false", "commit", "-qm", "post-cut main")
            with self.assertRaisesRegex(ValueError, "production tree differs"):
                fixtures.capture_source("1.6.0", release_tag="v1.6.0")

    def test_candidate_requires_immutable_matching_clean_source(self):
        source = fixtures.capture_source("1.6.0", candidate_sha=self.revision)
        self.assertEqual(source["sourceKind"], "release-candidate")
        with self.assertRaisesRegex(ValueError, "full immutable"):
            fixtures.capture_source("1.6.0", candidate_sha="release/1.6.0")
        with self.assertRaisesRegex(ValueError, "appVersion"):
            fixtures.capture_source("1.7.0", candidate_sha=self.revision)
        self.worker.with_name("untracked.go").write_text("package dynamo\n")
        with self.assertRaisesRegex(ValueError, "production tree differs"):
            fixtures.capture_source("1.6.0", candidate_sha=self.revision)

    def test_release_requires_published_upstream_tag(self):
        with patch.object(
            fixtures.subprocess, "check_output", side_effect=self.upstream_output
        ):
            for field, invalid in (
                ("published_at", None),
                ("draft", True),
                ("prerelease", True),
            ):
                with self.subTest(field=field):
                    original = self.published[field]
                    self.published[field] = invalid
                    with self.assertRaisesRegex(
                        ValueError, "published stable upstream"
                    ):
                        fixtures.capture_source("1.6.0", release_tag="v1.6.0")
                    self.published[field] = original
            self.official_revision = "0" * 40
            with self.assertRaisesRegex(ValueError, "published stable upstream"):
                fixtures.capture_source("1.6.0", release_tag="v1.6.0")
        for release_tag, candidate_sha in ((None, None), ("v1.6.0", self.revision)):
            with self.subTest(release_tag=release_tag, candidate_sha=candidate_sha):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    fixtures.capture_source("1.6.0", release_tag, candidate_sha)
