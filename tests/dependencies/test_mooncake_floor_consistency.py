# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Mooncake transfer engine specifier identical in every image.

The vLLM and SGLang images floor mooncake-transfer-engine-cuda13 through their
requirements files. The TensorRT-LLM image installs it into system site from its
template, because requirements.trtllm.txt reaches only the venv, and the template's
guards repeat the floor and the cap as version literals. Nothing else ties the
three together, so a bump in one place would leave the others behind unnoticed.

The layout follows test_pynvvideocodec_spec.py. The template half runs only where
the templates are on disk, and the `mooncake-floor-consistency` pre-commit hook
runs this module as a script against a checkout, where every file must exist.
"""

import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SPEC = ">=0.3.14,<0.4"
DISTRIBUTION = "mooncake-transfer-engine-cuda13"

REQUIREMENTS = (
    "container/deps/requirements.vllm.txt",
    "container/deps/requirements.sglang.txt",
)
TEMPLATE = "container/templates/trtllm_runtime.Dockerfile"

# The template installs the distribution by a quoted specifier.
_INLINE_SPEC = re.compile(r'"mooncake-transfer-engine-cuda13\s*([<>=!~][^"]*)"')
# The guards compare against quoted version literals, for example
# ("0.3.14", "0.4") or Version("0.3.14").
_VERSION_LITERAL = re.compile(r'"(\d+(?:\.\d+)+)"')
# An instruction guards this distribution if it names it in either spelling.
_NAMES = (DISTRIBUTION, DISTRIBUTION.replace("-", "_"))


def _expected_bounds() -> set[str]:
    """The floor and the cap of EXPECTED_SPEC, as the guards write them."""
    bounds = {
        s.version for s in SpecifierSet(EXPECTED_SPEC) if s.operator in (">=", "<")
    }
    assert len(bounds) == 2, f"{EXPECTED_SPEC} needs one >= and one < clause"
    return bounds


def _instructions(text: str) -> list[str]:
    """Split a Dockerfile into instructions, joining backslash continuations.

    Comment lines are dropped first, so prose that names the distribution or a
    version does not count as a guard.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        current.append(line)
        if not line.rstrip().endswith("\\"):
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _requirement(path: str) -> Requirement:
    """Return the one Mooncake line of a requirements file."""
    requirements = ROOT / path
    if not requirements.is_file():
        # The framework runtime-test images ship no container/ directory.
        pytest.skip(f"{path} is not present in this image")
    found: list[Requirement] = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            requirement = Requirement(line)
        except Exception:  # noqa: BLE001 - not a requirement line
            continue
        if canonicalize_name(requirement.name) == DISTRIBUTION:
            found.append(requirement)
    assert len(found) == 1, f"expected one Mooncake line in {path}, got {found}"
    return found[0]


def _template_text() -> str:
    template = ROOT / TEMPLATE
    if not template.is_file():
        pytest.skip(f"{TEMPLATE} is not staged in this image (.dockerignore)")
    return template.read_text(encoding="utf-8")


def _guard_versions(text: str) -> list[set[str]]:
    """The version literals of each instruction that names the distribution."""
    guards = []
    for block in _instructions(text):
        if not any(name in block for name in _NAMES):
            continue
        versions = set(_VERSION_LITERAL.findall(block))
        if versions:
            guards.append(versions)
    return guards


@pytest.mark.parametrize("path", REQUIREMENTS)
def test_requirements_declare_the_expected_spec(path: str) -> None:
    assert _requirement(path).specifier == SpecifierSet(EXPECTED_SPEC)


def test_the_spec_is_bounded_above() -> None:
    specifier = _requirement(REQUIREMENTS[0]).specifier
    assert any(
        s.operator in ("<", "<=", "==", "~=") for s in specifier
    ), f"Mooncake is specified as {specifier}, which has no upper bound"


def test_template_installs_the_same_spec() -> None:
    specs = _INLINE_SPEC.findall(_template_text())
    assert specs, f"{TEMPLATE} does not install {DISTRIBUTION}"
    for spec in specs:
        assert SpecifierSet(spec) == SpecifierSet(EXPECTED_SPEC), f"installs {spec}"


def test_template_guards_on_the_same_bounds() -> None:
    """Every guard compares against exactly the floor and the cap of the spec."""
    guards = _guard_versions(_template_text())
    assert guards, f"no {DISTRIBUTION} guard found in {TEMPLATE}"
    expected = _expected_bounds()
    for versions in guards:
        assert versions == expected, f"a guard compares against {sorted(versions)}"


def test_requirements_missing_from_the_image_skip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_requirements_declare_the_expected_spec(REQUIREMENTS[0])
    with pytest.raises(pytest.skip.Exception):
        test_template_guards_on_the_same_bounds()


def main() -> int:
    """Run every check outside pytest, for the `mooncake-floor-consistency` hook.

    A missing file is an error here, not the skip it is inside an image, because
    a checkout always has every file.
    """
    failures: list[str] = []

    def check(label: str, fn, *args) -> None:
        try:
            fn(*args)
        except BaseException as exc:  # noqa: BLE001 - report, never abort the sweep
            failures.append(f"{label}: {exc}")

    for path in (*REQUIREMENTS, TEMPLATE):
        if not (ROOT / path).is_file():
            failures.append(f"{path}: file not found")
    if not failures:
        check("spec is bounded above", test_the_spec_is_bounded_above)
        for path in REQUIREMENTS:
            check(
                f"requirements spec {path}",
                test_requirements_declare_the_expected_spec,
                path,
            )
        check("template install spec", test_template_installs_the_same_spec)
        check("template guard bounds", test_template_guards_on_the_same_bounds)

    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        print(f"\n{len(failures)} Mooncake spec check(s) failed.")
        return 1
    print(
        f"Mooncake spec {EXPECTED_SPEC} is consistent across "
        f"{len(REQUIREMENTS)} requirements files and {TEMPLATE}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
