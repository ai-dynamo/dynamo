# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release gate must reject skipped capture even when both versions parse."""
import unittest

from generate_upgrade_fixtures import app_version, validate_versions


class UpgradeVersionTest(unittest.TestCase):
    def test_operator_app_version_is_authoritative(self):
        self.assertEqual(app_version('version: 99.0.0\nappVersion: "1.6.0"\n'), "1.6.0")

    def test_adjacent_development_and_release_boundary(self):
        for current, reconciled, previous in [
            ("1.6.0", "1.5.0", "1.6.0"),
            ("1.6.0", "1.6.0", "1.6.0"),
            ("1.7.0", "1.6.0", "1.6.0"),
        ]:
            with self.subTest(current=current, reconciled=reconciled):
                validate_versions(current, reconciled, previous)

    def test_premature_skipped_and_major_bumps_fail(self):
        for current, reconciled, previous in [
            ("1.7.0", "1.5.0", "1.6.0"),
            ("1.8.0", "1.7.0", "1.6.0"),
            ("2.0.0", "1.9.0", "1.9.0"),
        ]:
            with self.subTest(current=current, reconciled=reconciled):
                with self.assertRaises(ValueError):
                    validate_versions(current, reconciled, previous)


if __name__ == "__main__":
    unittest.main()
