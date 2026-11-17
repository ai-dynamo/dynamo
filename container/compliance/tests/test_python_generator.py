# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from compliance import overrides
from compliance.generators import python as python_gen
from compliance.generators.common import render_notices


@pytest.fixture
def audited_overrides(tmp_path, monkeypatch):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "overrides:\n"
        "  - ecosystem: python\n"
        "    name: audited-package\n"
        "    version: '1.0'\n"
        "    license: MIT AND LicenseRef-NVIDIA-Proprietary\n"
        "  - ecosystem: python\n"
        "    name: legacy-package\n"
        "    license: MIT\n"
    )
    monkeypatch.setattr(overrides, "_DEFAULT_OVERRIDES_PATH", path)
    overrides.reset_caches()
    yield
    overrides.reset_caches()


def test_audited_version_corrects_declared_expression_only_for_that_version(
    audited_overrides,
):
    resolve = python_gen._resolve_spdx
    assert resolve("audited-package", "1.0", "LicenseRef-Proprietary", [], []) == (
        "MIT AND LicenseRef-NVIDIA-Proprietary"
    )
    assert (
        resolve("audited-package", "1.1", "LicenseRef-Proprietary", [], [])
        == "LicenseRef-Proprietary"
    )


def test_unversioned_override_remains_a_metadata_fallback(audited_overrides):
    resolve = python_gen._resolve_spdx
    assert resolve("legacy-package", "1.0", "BSD-3-Clause", [], []) == "BSD-3-Clause"
    assert resolve("legacy-package", "1.0", None, [], []) == "MIT"


def test_audited_expression_preserves_all_original_notices(tmp_path, audited_overrides):
    dist_info = tmp_path / "site-packages" / "audited_package-1.0.dist-info"
    licenses = dist_info / "licenses"
    licenses.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Name: audited-package\nVersion: 1.0\n"
        "License-Expression: LicenseRef-Proprietary\n"
    )
    (licenses / "LICENSE").write_text("Original MIT packaging terms.\n")
    (licenses / "THIRD_PARTY_NOTICES.md").write_text(
        "NVIDIA-derived material remains subject to applicable NVIDIA terms.\n"
    )
    components = python_gen.collect_components([dist_info.parent])
    assert len(components) == 1
    assert components[0].spdx == "MIT AND LicenseRef-NVIDIA-Proprietary"
    notices = render_notices("python", components)
    assert "Original MIT packaging terms." in notices
    assert (
        "NVIDIA-derived material remains subject to applicable NVIDIA terms." in notices
    )
