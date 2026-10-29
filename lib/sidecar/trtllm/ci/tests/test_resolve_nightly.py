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


if __name__ == "__main__":
    unittest.main()
