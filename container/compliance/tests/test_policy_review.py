# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""New grants need explicit scope; existing debt is visible without inventing approvals."""

import pytest
from compliance.policy.review import classify, review_overrides, review_policy

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("Apache-2.0 OR GPL-2.0-only", "permissive_alternative"),
        ("(Apache-2.0 OR GPL-2.0-only) AND GPL-3.0-only", "reciprocal"),
        ("Apache-2.0 AND MIT", "permissive"),
        ("MIT OR LicenseRef-Unknown", "permissive_alternative"),
        ("MIT AND LicenseRef-Unknown", "review_required"),
        ("GPL-3.0-only OR LicenseRef-Unknown", "review_required"),
        ("Apache-2.0 WITH LLVM-exception", "permissive"),
        ("GPL-3.0-only WITH GCC-exception-3.1", "review_required"),
        ("LGPL-3.0-only", "reciprocal"),
        ("CC-BY-SA-4.0", "reciprocal"),
        ("LicenseRef-NVIDIA-Proprietary", "review_required"),
    ],
)
def test_expression_classification(expression, expected):
    assert classify(expression) == expected


def policy(path, allow='"MIT"', exception=""):
    path.write_text(
        f'[licenses]\nallow = [{allow}]\ndeny = ["GPL-3.0-only"]\nunknown = "deny"\ncopyleft = "deny"\n'
        + exception
    )
    return path


def test_existing_global_grants_are_debt_not_approved_or_automatically_removed(
    tmp_path,
):
    base = policy(tmp_path / "base.toml", '"MIT", "LGPL-3.0-only"')
    current = policy(tmp_path / "current.toml", '"MIT", "LGPL-3.0-only"')
    report = review_policy(current, base)
    assert report["errors"] == []
    assert report["legacy_review_debt"]["status"] == "legacy_review_debt_not_approved"
    assert (
        report["legacy_review_debt"]["global_nonpermissive_or_unclassified"][0][
            "license"
        ]
        == "LGPL-3.0-only"
    )


def test_new_global_reciprocal_and_unknown_grants_fail(tmp_path):
    base = policy(tmp_path / "base.toml")
    current = policy(
        tmp_path / "current.toml", '"MIT", "LGPL-3.0-only", "LicenseRef-Custom"'
    )
    assert len(review_policy(current, base)["errors"]) == 2


def test_new_exception_requires_exact_version_and_use_evidence(tmp_path):
    base = policy(tmp_path / "base.toml")
    row = '\n[[exceptions]]\ntype = "python"\nname = "pkg"\nallow = ["GPL-3.0-only"]\nreason = "Reviewed separate tool"\n'
    current = policy(tmp_path / "current.toml", exception=row)
    assert "exact version" in review_policy(current, base)["errors"][0]
    current.write_text(
        current.read_text()
        + 'version = "1.2.3"\nusage = "Separate-process diagnostic CLI"\nevidence = "https://example.com/review/123"\n'
    )
    assert review_policy(current, base)["errors"] == []


def test_legacy_metadata_correction_can_gain_evidence_without_new_grant(tmp_path):
    base = tmp_path / "base.yaml"
    current = tmp_path / "current.yaml"
    base.write_text(
        "overrides:\n- {ecosystem: python, name: pkg, license: MIT, source: original}\n"
    )
    current.write_text(
        base.read_text().replace("original", "original-and-additional-evidence")
    )
    assert review_overrides(current, base)["errors"] == []
    current.write_text(
        current.read_text().replace("license: MIT", "license: Apache-2.0")
    )
    assert "exact version" in review_overrides(current, base)["errors"][0]


def test_factual_exception_rationale_correction_remains_legacy_debt(tmp_path):
    row = '\n[[exceptions]]\ntype = "python"\nname = "pycountry"\nallow = ["LGPL-2.1-only"]\nreason = "Data only"\n'
    base = policy(tmp_path / "base.toml", exception=row)
    current = policy(
        tmp_path / "current.toml",
        exception=row.replace(
            "Data only", "Python library plus datasets; pending use review"
        ),
    )
    report = review_policy(current, base)
    assert report["errors"] == []
    debt = report["legacy_review_debt"]
    assert debt["status"] == "legacy_review_debt_not_approved"
    assert debt["unscoped_or_undocumented_exceptions"][0]["name"] == "pycountry"
    current.write_text(
        current.read_text().replace(
            '["LGPL-2.1-only"]', '["LGPL-2.1-only", "GPL-3.0-only"]'
        )
    )
    assert "exact version" in review_policy(current, base)["errors"][0]
