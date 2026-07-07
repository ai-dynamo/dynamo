# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that marker-generated and platform Helm chart RBAC stay in sync."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# These cluster-scoped permissions stay in dedicated roles when the manager is
# namespace-restricted, so include them in the chart's effective rule set.
AUXILIARY_ROLE_NAMES = (
    "clustertopology-reader",
    "queue-reader",
    "deviceclass-reader",
    "podsnapshotcontents",
)
POLICY_RULE_FIELDS = {
    "apiGroups",
    "nonResourceURLs",
    "resourceNames",
    "resources",
    "verbs",
}
RULE_START = re.compile(r"^- ([A-Za-z][A-Za-z0-9]*):\s*$")
RULE_FIELD = re.compile(r"^  ([A-Za-z][A-Za-z0-9]*):\s*$")
RULE_VALUE = re.compile(r"^  - (.+?)\s*$")


@dataclass(frozen=True, order=True)
class Permission:
    """One normalized RBAC permission."""

    kind: str
    api_group: str
    resource: str
    verb: str
    resource_name: str = ""

    def describe(self) -> str:
        if self.kind == "non-resource":
            return f"{self.verb} {self.resource} (non-resource URL)"

        group = self.api_group or "core"
        suffix = f" (resourceName={self.resource_name})" if self.resource_name else ""
        return f"{self.verb} {group}/{self.resource}{suffix}"


def split_documents(text: str) -> list[list[str]]:
    documents: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "---":
            documents.append([])
        else:
            documents[-1].append(line)
    return documents


def document_name(document: list[str]) -> str:
    for line in document:
        match = re.match(r"^\s*name:\s*(.+?)\s*$", line)
        if match:
            return match.group(1)
    return "<unnamed>"


def parse_scalar(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def parse_rules(
    document: list[str], source: Path, name: str
) -> list[dict[str, list[str]]]:
    try:
        rules_index = next(
            index for index, line in enumerate(document) if line == "rules:"
        )
    except StopIteration:
        return []

    rules: list[dict[str, list[str]]] = []
    current_rule: dict[str, list[str]] | None = None
    current_field: str | None = None

    rule_lines = document[rules_index + 1 :]
    for line_number, line in enumerate(rule_lines, start=rules_index + 2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("{{"):
            raise ValueError(
                f"{source}:{line_number}: Helm condition or expression inside "
                f"{name} rules; "
                "marker-derived manager permissions must be unconditional"
            )

        if match := RULE_START.match(line):
            if current_rule is not None:
                rules.append(current_rule)
            current_field = match.group(1)
            current_rule = {current_field: []}
            continue

        if match := RULE_FIELD.match(line):
            if current_rule is None:
                raise ValueError(
                    f"{source}:{line_number}: policy rule field outside a rule"
                )
            current_field = match.group(1)
            current_rule.setdefault(current_field, [])
            continue

        if match := RULE_VALUE.match(line):
            if current_rule is None or current_field is None:
                raise ValueError(
                    f"{source}:{line_number}: policy rule value outside a field"
                )
            current_rule[current_field].append(parse_scalar(match.group(1)))
            continue

        raise ValueError(
            f"{source}:{line_number}: unsupported policy rule syntax: {line!r}"
        )

    if current_rule is not None:
        rules.append(current_rule)
    return rules


def normalize_rules(rules: list[dict[str, list[str]]], source: Path) -> set[Permission]:
    permissions: set[Permission] = set()
    for rule in rules:
        unknown_fields = set(rule) - POLICY_RULE_FIELDS
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"{source}: unsupported policy rule fields: {fields}")

        verbs = rule.get("verbs", [])
        if not verbs:
            raise ValueError(f"{source}: policy rule has no verbs")

        if non_resource_urls := rule.get("nonResourceURLs"):
            resource_fields = ("apiGroups", "resources", "resourceNames")
            if any(rule.get(field) for field in resource_fields):
                raise ValueError(
                    f"{source}: policy rule mixes resource and non-resource permissions"
                )
            for url in non_resource_urls:
                for verb in verbs:
                    permissions.add(Permission("non-resource", "", url, verb))
            continue

        api_groups = rule.get("apiGroups", [])
        resources = rule.get("resources", [])
        if not api_groups or not resources:
            raise ValueError(
                f"{source}: resource policy rule needs apiGroups and resources"
            )

        resource_names = rule.get("resourceNames") or [""]
        for api_group in api_groups:
            for resource in resources:
                for verb in verbs:
                    for resource_name in resource_names:
                        permissions.add(
                            Permission(
                                "resource", api_group, resource, verb, resource_name
                            )
                        )
    return permissions


def load_generated_permissions(path: Path) -> set[Permission]:
    documents = split_documents(path.read_text())
    rules_documents = []
    for document in documents:
        rules = parse_rules(document, path, document_name(document))
        if rules:
            rules_documents.append(rules)
    if len(rules_documents) != 1:
        raise ValueError(f"{path}: expected exactly one rules-bearing RBAC document")
    return normalize_rules(rules_documents[0], path)


def load_chart_permissions(path: Path) -> set[Permission]:
    manager_rules: list[dict[str, list[str]]] | None = None
    auxiliary_rules: list[dict[str, list[str]]] = []

    for document in split_documents(path.read_text()):
        name = document_name(document)
        rules = parse_rules(document, path, name)
        if not rules:
            continue
        if "manager-role" in name and "manager-rolebinding" not in name:
            if manager_rules is not None:
                raise ValueError(f"{path}: found more than one manager role")
            manager_rules = rules
        elif any(auxiliary_name in name for auxiliary_name in AUXILIARY_ROLE_NAMES):
            auxiliary_rules.extend(rules)
        else:
            raise ValueError(
                f"{path}: unclassified rules-bearing RBAC document {name!r}"
            )

    if manager_rules is None:
        raise ValueError(f"{path}: manager role not found")
    return normalize_rules(manager_rules + auxiliary_rules, path)


def print_permissions(title: str, permissions: set[Permission]) -> None:
    if not permissions:
        return
    print(title, file=sys.stderr)
    for permission in sorted(permissions):
        print(f"  - {permission.describe()}", file=sys.stderr)


def check_rbac(generated_path: Path, chart_path: Path) -> bool:
    generated = load_generated_permissions(generated_path)
    chart = load_chart_permissions(chart_path)
    missing_from_chart = generated - chart
    extra_in_chart = chart - generated

    if missing_from_chart or extra_in_chart:
        print(
            "ERROR: platform Helm chart manager RBAC differs from "
            "marker-generated RBAC.",
            file=sys.stderr,
        )
        print_permissions("Missing from the chart:", missing_from_chart)
        print_permissions("Extra in the chart:", extra_in_chart)
        print(
            "Update both +kubebuilder:rbac markers and the platform chart "
            "manager role.",
            file=sys.stderr,
        )
        return False

    print(f"RBAC rules are in sync ({len(generated)} normalized permissions).")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_role", type=Path)
    parser.add_argument("chart_rbac", type=Path)
    args = parser.parse_args()

    try:
        return 0 if check_rbac(args.generated_role, args.chart_rbac) else 1
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
