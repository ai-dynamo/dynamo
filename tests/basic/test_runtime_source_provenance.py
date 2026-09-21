# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catch mixed source/wheel/install revisions before image-based CI runs."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/verify_runtime_sources.py"
)
spec = importlib.util.spec_from_file_location("verify_runtime_sources", SCRIPT)
assert spec is not None and spec.loader is not None
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


@pytest.mark.parametrize(
    "wheel_content,installed_content,expected_mismatches",
    [(b"new", b"new", 0), (b"new", b"old", 1), (b"old", b"old", 1), (b"new", None, 1)],
)
def test_detects_stale_wheel_or_install(
    tmp_path, wheel_content, installed_content, expected_mismatches
):
    name = "dynamo/common/example.py"
    source = tmp_path / "repo/components/src" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new")
    source.with_name("_version.py").write_bytes(b"source version")
    installed = tmp_path / "installed" / name
    installed.parent.mkdir(parents=True)
    if installed_content is not None:
        installed.write_bytes(installed_content)
    wheel = tmp_path / "example.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(name, wheel_content)
        archive.writestr("dynamo/common/_version.py", b"generated")
    result = provenance.compare_sources(
        tmp_path / "repo", tmp_path / "installed", wheel
    )
    assert result["checked"] == 1
    assert len(result["mismatches"]) == expected_mismatches
    if expected_mismatches:
        assert result["mismatches"][0]["path"] == name


def test_empty_comparison_is_not_a_pass(tmp_path):
    wheel = tmp_path / "empty.whl"
    with ZipFile(wheel, "w"):
        pass
    with pytest.raises(RuntimeError, match="No Dynamo component sources"):
        provenance.compare_sources(tmp_path, tmp_path, wheel)
