# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "files/upgrade-preflight/checks/upgrade-path/check.py"
SPEC = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class UpgradePathTest(unittest.TestCase):
    def test_allows_same_minor_and_adjacent_minor_transitions(self):
        for installed, target in (
            ("1.5.0", "1.5.3"),
            ("v1.4.7", "1.5.0-rc.1"),
            ("1.4.7-rc.1+build.5", "1.5.0+build.7"),
            ("1.5.0", "1.4.7"),
        ):
            with self.subTest(installed=installed, target=target):
                preflight.validate_upgrade_path(installed, target)

    def test_rejects_empty_semantic_version_identifiers(self):
        for value in ("1.4.7-rc..1", "1.4.7+build..5"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "not a supported semantic version"
            ):
                preflight.version(value)

    def test_rejects_skipped_minor_upgrade_and_downgrade(self):
        for installed, target, expected in (
            ("1.2.3", "1.5.0", "unsupportedUpgradeOverride=1.5"),
            ("1.5.0", "1.2.3", "unsupportedDowngradeOverride=1.2"),
        ):
            with (
                self.subTest(installed=installed, target=target),
                self.assertRaisesRegex(ValueError, expected),
            ):
                preflight.validate_upgrade_path(installed, target)

    def test_matching_target_minor_override_allows_unsupported_transition(self):
        preflight.validate_upgrade_path("1.2.3", "1.5.0", "1.5")
        preflight.validate_upgrade_path("1.5.3", "1.2.0", "", "1.2")

    def test_installed_minor_override_does_not_allow_unsupported_transition(self):
        for installed, target, upgrade_override, downgrade_override, expected in (
            ("1.2.0", "1.5.0", "1.2", "", "unsupportedUpgradeOverride=1.5"),
            ("1.5.0", "1.2.0", "", "1.5", "unsupportedDowngradeOverride=1.2"),
        ):
            with (
                self.subTest(installed=installed, target=target),
                self.assertRaisesRegex(ValueError, expected),
            ):
                preflight.validate_upgrade_path(
                    installed, target, upgrade_override, downgrade_override
                )

    def test_override_does_not_apply_in_the_other_direction(self):
        with self.assertRaisesRegex(ValueError, "unsupportedDowngradeOverride=1.2"):
            preflight.validate_upgrade_path("1.5.0", "1.2.0", "1.2", "")

    def test_override_requires_minor_version_form(self):
        with self.assertRaisesRegex(ValueError, "must be a minor version in N.M form"):
            preflight.validate_upgrade_path("1.2.0", "1.5.0", "1.5.0")

    def test_matching_override_allows_major_transition(self):
        preflight.validate_upgrade_path("1.9.0", "2.0.0", "2.0")
        preflight.validate_upgrade_path("2.0.0", "1.9.0", "", "1.9")

    def test_rejects_non_semantic_version(self):
        with self.assertRaisesRegex(ValueError, "not a supported semantic version"):
            preflight.validate_upgrade_path("latest", "1.5.0")


class CheckTest(unittest.TestCase):
    def test_allows_initial_install_when_deployment_does_not_exist(self):
        get = Mock(return_value=None)
        preflight.check_upgrade_path(
            get, "dynamo-system", "dynamo-controller-manager", "1.6.0"
        )
        self.assertEqual(
            get.call_args.args,
            (
                "/apis/apps/v1/namespaces/dynamo-system/deployments/"
                "dynamo-controller-manager",
            ),
        )

    def test_reads_installed_version_from_operator_deployment(self):
        get = Mock(
            return_value={
                "metadata": {"labels": {"app.kubernetes.io/version": "1.5.2"}}
            }
        )
        preflight.check_upgrade_path(
            get, "dynamo-system", "dynamo-controller-manager", "1.6.0"
        )

    def test_rejects_missing_installed_version_label(self):
        get = Mock(return_value={"metadata": {"labels": {}}})
        with self.assertRaisesRegex(
            ValueError, "has no app.kubernetes.io/version label"
        ):
            preflight.check_upgrade_path(
                get, "dynamo-system", "dynamo-controller-manager", "1.6.0"
            )


if __name__ == "__main__":
    unittest.main()
