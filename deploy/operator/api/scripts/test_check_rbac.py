# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from check_rbac import Permission, normalize_rules, parse_rules


class CheckRBACTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path("test.yaml")

    def test_normalization_ignores_rule_grouping_and_order(self) -> None:
        combined = [
            {
                "apiGroups": [""],
                "resources": ["services", "configmaps"],
                "verbs": ["watch", "get"],
            }
        ]
        split = [
            {"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get", "watch"]},
            {"apiGroups": [""], "resources": ["services"], "verbs": ["get", "watch"]},
        ]

        self.assertEqual(
            normalize_rules(combined, self.source),
            normalize_rules(split, self.source),
        )

    def test_resource_names_are_part_of_normalized_permission(self) -> None:
        permissions = normalize_rules(
            [
                {
                    "apiGroups": [""],
                    "resources": ["secrets"],
                    "resourceNames": ["webhook-cert"],
                    "verbs": ["get"],
                }
            ],
            self.source,
        )

        self.assertEqual(
            permissions,
            {Permission("resource", "", "secrets", "get", "webhook-cert")},
        )

    def test_helm_expression_in_rules_is_rejected(self) -> None:
        document = [
            "metadata:",
            "  name: manager-role",
            "rules:",
            "{{- if .Values.feature.enabled }}",
            "- apiGroups:",
            '  - ""',
            "  resources:",
            "  - pods",
            "  verbs:",
            "  - get",
            "{{- end }}",
        ]

        with self.assertRaisesRegex(ValueError, "must be unconditional"):
            parse_rules(document, self.source, "manager-role")

    def test_parser_handles_generated_rule_format(self) -> None:
        document = [
            "metadata:",
            "  name: manager-role",
            "rules:",
            "- apiGroups:",
            '  - ""',
            "  resources:",
            "  - pods",
            "  verbs:",
            "  - get",
        ]

        self.assertEqual(
            parse_rules(document, self.source, "manager-role"),
            [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
        )


if __name__ == "__main__":
    unittest.main()
