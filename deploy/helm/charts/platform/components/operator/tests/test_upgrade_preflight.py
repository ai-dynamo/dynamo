# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "files" / "upgrade-preflight" / "check.py"
SPEC = importlib.util.spec_from_file_location("upgrade_preflight", SCRIPT_PATH)
upgrade_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upgrade_preflight)


class ValidateUpgradeTest(unittest.TestCase):
    def test_allows_patch_upgrade(self):
        upgrade_preflight.validate_upgrade("1.5.0", "1.5.3")

    def test_allows_next_minor_upgrade(self):
        upgrade_preflight.validate_upgrade("v1.4.7", "1.5.0-rc.1")

    def test_rejects_skipped_minor(self):
        with self.assertRaisesRegex(
            upgrade_preflight.PreflightError, "every intermediate minor"
        ):
            upgrade_preflight.validate_upgrade("1.2.3", "1.5.0")

    def test_rejects_earlier_minor(self):
        with self.assertRaises(upgrade_preflight.PreflightError):
            upgrade_preflight.validate_upgrade("1.5.0", "1.4.9")

    def test_rejects_major_upgrade(self):
        with self.assertRaises(upgrade_preflight.PreflightError):
            upgrade_preflight.validate_upgrade("1.9.0", "2.0.0")

    def test_rejects_non_semantic_version(self):
        with self.assertRaisesRegex(
            upgrade_preflight.PreflightError, "not a supported semantic"
        ):
            upgrade_preflight.validate_upgrade("latest", "1.5.0")


class InstalledVersionTest(unittest.TestCase):
    def test_reads_operator_deployment_label(self):
        payload = {
            "items": [{"metadata": {"labels": {"app.kubernetes.io/version": "1.4.2"}}}]
        }
        self.assertEqual(upgrade_preflight.installed_version(payload), "1.4.2")

    def test_rejects_ambiguous_operator_deployments(self):
        with self.assertRaisesRegex(upgrade_preflight.PreflightError, "exactly one"):
            upgrade_preflight.installed_version({"items": []})

    def test_rejects_missing_version_label(self):
        with self.assertRaisesRegex(upgrade_preflight.PreflightError, "version label"):
            upgrade_preflight.installed_version(
                {"items": [{"metadata": {"labels": {}}}]}
            )


if __name__ == "__main__":
    unittest.main()
