# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location(
    "resolve_nightly", Path(__file__).resolve().parents[1] / "resolve_nightly.py"
)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


class NightlySelectionTests(unittest.TestCase):
    def test_latest_means_timestamp_not_version_or_feature_branch(self):
        self.assertEqual(
            resolver.latest_tag(
                [
                    "1.9.0rc1.dev202610260007",
                    "1.4.0rc2.dev202610270008",
                    "feat-qwen3.5-opt-1.4.0a0.dev202610281318",
                    "1.3.0a0.dev202610280406",
                    "1.4.0b1.dev202610290406",
                    "1.4.0rc3",
                    "latest",
                    "nightly",
                ]
            ),
            "1.4.0rc2.dev202610270008",
        )

    def test_no_regular_nightly_fails(self):
        with self.assertRaisesRegex(ValueError, "No regular"):
            resolver.latest_tag(["nightly", "feat-test-1.4.0a0.dev202610281318"])

    def test_final_version_nightly_is_accepted(self):
        self.assertEqual(
            resolver.latest_tag(["1.4.0rc3.dev202610280007", "1.4.0.dev202610290007"]),
            "1.4.0.dev202610290007",
        )

    def test_selection_records_exact_digest_and_source_from_image(self):
        registry = Mock()
        digest = "sha256:" + "a" * 64
        registry.get.side_effect = [
            (json.dumps({"tags": ["1.4.0rc2.dev202610270008"]}).encode(), {}),
            (
                json.dumps(
                    {"config": {"Env": ["TRT_LLM_GIT_COMMIT=" + "b" * 40]}}
                ).encode(),
                {},
            ),
        ]
        registry.manifest.return_value = (
            {"config": {"digest": "sha256:config"}},
            digest,
        )
        resolved = resolver.resolve({"repository": resolver.REPOSITORY}, registry)
        self.assertEqual(
            resolved["image"],
            f"{resolver.REPOSITORY}:1.4.0rc2.dev202610270008@{digest}",
        )
        self.assertEqual(resolved["upstream_commit"], "b" * 40)
        registry.get.assert_called_with("blobs/sha256:config")

    def test_multiarch_pin_retains_index_digest_and_reads_amd64_metadata(self):
        registry = Mock()
        registry.get.side_effect = [
            (json.dumps({"tags": ["1.4.0rc2.dev202610270008"]}).encode(), {}),
            (
                json.dumps(
                    {"config": {"Env": ["TRT_LLM_GIT_COMMIT=" + "c" * 40]}}
                ).encode(),
                {},
            ),
        ]
        registry.manifest.side_effect = [
            (
                {
                    "manifests": [
                        {
                            "platform": {"os": "linux", "architecture": "arm64"},
                            "digest": "sha256:arm",
                        },
                        {
                            "platform": {"os": "linux", "architecture": "amd64"},
                            "digest": "sha256:amd",
                        },
                    ]
                },
                "sha256:index",
            ),
            ({"config": {"digest": "sha256:config"}}, "sha256:amd"),
        ]
        resolved = resolver.resolve({"repository": resolver.REPOSITORY}, registry)
        self.assertTrue(resolved["image"].endswith("@sha256:index"))
        self.assertEqual(resolved["upstream_commit"], "c" * 40)
        registry.manifest.assert_called_with("sha256:amd")

    def test_registry_failure_does_not_select_an_older_image(self):
        registry = Mock()
        registry.get.side_effect = OSError("registry unavailable")
        with self.assertRaisesRegex(OSError, "registry unavailable"):
            resolver.resolve({"repository": resolver.REPOSITORY}, registry)

    def test_manifest_digest_mismatch_fails(self):
        registry = resolver.Registry()
        registry.get = Mock(
            return_value=(b"{}", {"Docker-Content-Digest": "sha256:wrong"})
        )
        with self.assertRaisesRegex(ValueError, "digest"):
            registry.manifest("1.4.0rc2.dev202610270008")

    def test_manifest_by_digest_must_match_the_request(self):
        registry = resolver.Registry()
        registry.get = Mock(return_value=(b"{}", {}))
        with self.assertRaisesRegex(ValueError, "different manifest"):
            registry.manifest("sha256:" + "0" * 64)

    def test_missing_or_invalid_source_commit_fails(self):
        for commit in (None, "", "abc", "g" * 40):
            with self.subTest(commit=commit):
                registry = Mock()
                env = [] if commit is None else ["TRT_LLM_GIT_COMMIT=" + commit]
                registry.get.side_effect = [
                    (json.dumps({"tags": ["1.4.0rc2.dev202610270008"]}).encode(), {}),
                    (json.dumps({"config": {"Env": env}}).encode(), {}),
                ]
                registry.manifest.return_value = (
                    {"config": {"digest": "sha256:config"}},
                    "sha256:" + "a" * 64,
                )
                with self.assertRaisesRegex(ValueError, "source commit"):
                    resolver.resolve({"repository": resolver.REPOSITORY}, registry)

    def test_index_without_exactly_one_amd64_image_fails(self):
        amd64 = {"platform": {"os": "linux", "architecture": "amd64"}}
        for manifests in (
            [],
            [{"platform": {"os": "linux", "architecture": "arm64"}}],
            [dict(amd64, digest="sha256:a"), dict(amd64, digest="sha256:b")],
        ):
            with self.subTest(manifests=manifests):
                registry = Mock()
                registry.get.return_value = (
                    json.dumps({"tags": ["1.4.0rc2.dev202610270008"]}).encode(),
                    {},
                )
                registry.manifest.return_value = (
                    {"manifests": manifests},
                    "sha256:index",
                )
                with self.assertRaisesRegex(ValueError, "exactly one linux/amd64"):
                    resolver.resolve({"repository": resolver.REPOSITORY}, registry)


if __name__ == "__main__":
    unittest.main()
