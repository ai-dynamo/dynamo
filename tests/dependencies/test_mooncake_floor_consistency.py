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
The checks raise `CheckFailed` explicitly instead of using `assert`, so the hook
also fails under `python -O`.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
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


class CheckFailed(Exception):
    """A file disagrees with EXPECTED_SPEC."""


def _expected_bounds() -> set[str]:
    """The floor and the cap of EXPECTED_SPEC, as the guards write them."""
    bounds = {
        s.version for s in SpecifierSet(EXPECTED_SPEC) if s.operator in (">=", "<")
    }
    if len(bounds) != 2:
        raise CheckFailed(f"{EXPECTED_SPEC} needs one >= and one < clause")
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
        except InvalidRequirement:
            continue  # not a requirement line
        if canonicalize_name(requirement.name) == DISTRIBUTION:
            found.append(requirement)
    if len(found) != 1:
        raise CheckFailed(f"expected one Mooncake line in {path}, got {found}")
    return found[0]


def _template_text() -> str:
    template = ROOT / TEMPLATE
    if not template.is_file():
        pytest.skip(f"{TEMPLATE} is not staged in this image (.dockerignore)")
    return template.read_text(encoding="utf-8")


def _guard_versions(text: str) -> list[set[str]]:
    """The version literals of each command that names the distribution.

    One RUN chains the checks of several packages with `&&`, so the literals are
    read per command, not per instruction.
    """
    guards = []
    for block in _instructions(text):
        for command in block.split("&&"):
            if not any(name in command for name in _NAMES):
                continue
            versions = set(_VERSION_LITERAL.findall(command))
            if versions:
                guards.append(versions)
    return guards


def check_requirements_spec(path: str) -> None:
    specifier = _requirement(path).specifier
    if specifier != SpecifierSet(EXPECTED_SPEC):
        raise CheckFailed(f"declares {specifier}, expected {EXPECTED_SPEC}")


def check_spec_is_bounded_above() -> None:
    specifier = _requirement(REQUIREMENTS[0]).specifier
    if not any(s.operator in ("<", "<=", "==", "~=") for s in specifier):
        raise CheckFailed(
            f"Mooncake is specified as {specifier}, which has no upper bound"
        )


def check_template_install_spec() -> None:
    specs = _INLINE_SPEC.findall(_template_text())
    if not specs:
        raise CheckFailed(f"{TEMPLATE} does not install {DISTRIBUTION}")
    for spec in specs:
        try:
            matches = SpecifierSet(spec) == SpecifierSet(EXPECTED_SPEC)
        except InvalidSpecifier as exc:
            raise CheckFailed(f"cannot parse {spec!r}: {exc}") from exc
        if not matches:
            raise CheckFailed(f"installs {spec}, expected {EXPECTED_SPEC}")


def check_template_guard_bounds() -> None:
    """Every guard compares against exactly the floor and the cap of the spec."""
    guards = _guard_versions(_template_text())
    if not guards:
        raise CheckFailed(f"no {DISTRIBUTION} guard found in {TEMPLATE}")
    expected = _expected_bounds()
    for versions in guards:
        if versions != expected:
            raise CheckFailed(
                f"a guard compares against {sorted(versions)}, "
                f"expected {sorted(expected)}"
            )


@pytest.mark.parametrize("path", REQUIREMENTS)
def test_requirements_declare_the_expected_spec(path: str) -> None:
    check_requirements_spec(path)


def test_the_spec_is_bounded_above() -> None:
    check_spec_is_bounded_above()


def test_template_installs_the_same_spec() -> None:
    check_template_install_spec()


def test_template_guards_on_the_same_bounds() -> None:
    check_template_guard_bounds()


def test_guards_are_read_per_command() -> None:
    """Another package's bounds in the same RUN must not count as Mooncake's."""
    template = (
        "RUN /usr/bin/python3 -c 'bounds = "
        '{"datasets": ("5.1.0", "5.1.1")}\' && \\\n'
        "    /usr/bin/python3 -c 'bounds = "
        f'{{"{DISTRIBUTION}": ("0.3.14", "0.4")}}\'\n'
    )
    assert _guard_versions(template) == [{"0.3.14", "0.4"}]


def test_requirements_missing_from_the_image_skip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_requirements_declare_the_expected_spec(REQUIREMENTS[0])
    with pytest.raises(pytest.skip.Exception):
        test_template_guards_on_the_same_bounds()


def _write_checkout(root: Path, vllm_spec: str) -> None:
    """Write the three files main() reads, in the shapes the real ones use."""
    files = {
        REQUIREMENTS[0]: f"{DISTRIBUTION}{vllm_spec}\n",
        REQUIREMENTS[1]: f"{DISTRIBUTION}{EXPECTED_SPEC}\n",
        TEMPLATE: (
            "RUN /usr/bin/python3 -m pip install --break-system-packages "
            f'"{DISTRIBUTION}{EXPECTED_SPEC}" && \\\n'
            "    /usr/bin/python3 -c 'bounds = "
            f'{{"{DISTRIBUTION}": ("0.3.14", "0.4")}}\'\n'
        ),
    }
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")


def test_script_mode_fails_under_optimize(tmp_path) -> None:
    """The hook must still fail under `python -O`, which removes `assert`."""
    _write_checkout(tmp_path, ">=0.3.13,<0.4")
    script = (
        "import importlib.util, pathlib, sys; "
        f"spec = importlib.util.spec_from_file_location('mooncake', {__file__!r}); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        f"module.ROOT = pathlib.Path({str(tmp_path)!r}); "
        "sys.exit(module.main())"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"requirements spec {REQUIREMENTS[0]}: declares" in result.stdout


def test_script_mode_lets_an_interrupt_through(tmp_path, monkeypatch) -> None:
    """main() reports failed checks, but it must not swallow Ctrl-C."""
    _write_checkout(tmp_path, EXPECTED_SPEC)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert main() == 0

    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "check_spec_is_bounded_above", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main()


def main() -> int:
    """Run every check outside pytest, for the `mooncake-floor-consistency` hook.

    A missing file is an error here, not the skip it is inside an image, because
    a checkout always has every file.
    """
    failures: list[str] = []

    def check(label: str, fn, *args) -> None:
        try:
            fn(*args)
        except CheckFailed as exc:
            failures.append(f"{label}: {exc}")

    for path in (*REQUIREMENTS, TEMPLATE):
        if not (ROOT / path).is_file():
            failures.append(f"{path}: file not found")
    if not failures:
        check("spec is bounded above", check_spec_is_bounded_above)
        for path in REQUIREMENTS:
            check(f"requirements spec {path}", check_requirements_spec, path)
        check("template install spec", check_template_install_spec)
        check("template guard bounds", check_template_guard_bounds)

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
