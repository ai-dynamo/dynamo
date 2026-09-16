# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the PyNvVideoCodec spec identical everywhere it is written down.

The spec is written in three requirements files, and its lower bound is repeated
as a guard constant in each of the three runtime templates. The templates cannot
parse the requirements file they are checking, since the point of those guards is
to fail when the installed version and the declared one disagree, so the
duplication is deliberate and needs a test rather than a comment.

A lower bound resolves to whatever is newest at build time, so on its own it
records nothing: the spec read `>=2.2.0` while the images had already moved to
2.2.3, and no check could tell. The declared version is a claim about what was
validated, and this is what keeps it one.

The template half only runs where the templates are on disk. `.dockerignore`
excludes `container/**/*.Dockerfile` from every build context except
`wheel_builder.Dockerfile`, so the component images that run this suite ship
`container/deps/` but not `container/templates/`, and those cases skip there --
same situation, and the same resolution, as `test_aisimulate_consistency.py`'s
planner-Dockerfile check. The `pynvvideocodec-spec` pre-commit hook runs this
module as a script against a real checkout, which is what keeps the template half
enforced in CI rather than only on a developer's machine.
"""

import re
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

# The spec every image installs. The lower bound is the version the multimodal
# suite was validated against; the upper bound is what stops the next major being
# taken silently at build time.
EXPECTED_SPEC = ">=2.2.3,<3"
# The lower bound on its own -- this is what the template guards compare against,
# since a shell `sort -V` test cannot express a range.
EXPECTED_FLOOR = "2.2.3"

DISTRIBUTION = "pynvvideocodec"

REQUIREMENTS = (
    "container/deps/requirements.vllm.txt",
    "container/deps/requirements.sglang.txt",
    "container/deps/requirements.trtllm.txt",
)

TEMPLATES = (
    "container/templates/vllm_runtime.Dockerfile",
    "container/templates/sglang_runtime.Dockerfile",
    "container/templates/trtllm_runtime.Dockerfile",
)

_SEMVER = re.compile(r"\b\d+\.\d+\.\d+\b")
_HEREDOC = re.compile(r"<<-?'?([A-Za-z_][A-Za-z0-9_]*)'?")
# A real Dockerfile instruction, used to bound an unterminated heredoc. Spelled out
# rather than matched as "an uppercase word": the heredoc bodies assign uppercase
# Python constants (`FLOOR = "2.2.3"`), and a pattern that treats those as a new
# instruction cuts the body off exactly where the constant being checked lives.
_INSTRUCTION = re.compile(
    r"^(FROM|RUN|CMD|LABEL|MAINTAINER|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME"
    r"|USER|WORKDIR|ARG|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\s"
)
# Diagnostic output does not constrain a guard -- it describes one, and it is where
# historical versions legitimately appear ("the base image's 2.1.0 copy survived the
# overlay"). Only the lines that compare or install are held to the floor.
_DIAGNOSTIC = re.compile(r'^\s*(\|\|\s*\{\s*)?(echo\b|print\(|sys\.exit\(|f?")')
# A PyNvVideoCodec requirement specifier written inline in a template, e.g. the
# `pip install 'PyNvVideoCodec>=2.2.3,<3'` in trtllm_runtime.Dockerfile.
_INLINE_SPEC = re.compile(
    r"(?i)pynvvideocodec\s*((?:[<>=!~]=?\s*[0-9][^\'\"\s;]*)(?:\s*,\s*[<>=!~]=?\s*[0-9][^\'\"\s;]*)*)"
)


def _instruction_blocks(text: str) -> list[list[str]]:
    """Split a Dockerfile into instructions, keeping continuations together.

    Handles both shapes the templates use: backslash-continued commands and
    BuildKit heredocs. Matching has to be per instruction, not per file, because
    the ``sort -V`` comparison idiom the TRT-LLM guard uses is also how the DALI
    pin in the same file is written -- a file-wide scan reads DALI's version as a
    PyNvVideoCodec floor and fails for the wrong reason.

    Comment lines are excluded from both the continuation and the heredoc tests: a
    comment that quotes ``<<'PYEOF'`` or ends in a backslash would otherwise swallow
    the instructions after it into one block and bring that cross-contamination
    straight back. An unterminated heredoc stops at the next line that starts a new
    instruction, for the same reason.
    """
    lines = text.splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        start = i

        def is_comment(n: int) -> bool:
            return lines[n].lstrip().startswith("#")

        heredoc = None if is_comment(i) else _HEREDOC.search(lines[i])
        while (
            not is_comment(i)
            and lines[i].rstrip().endswith("\\")
            and i + 1 < len(lines)
        ):
            i += 1
            if not is_comment(i):
                heredoc = heredoc or _HEREDOC.search(lines[i])
        if heredoc:
            tag = heredoc.group(1)
            while i + 1 < len(lines) and lines[i].strip() != tag:
                # An unterminated heredoc must not run to the end of the file.
                if i > start and _INSTRUCTION.match(lines[i + 1]):
                    break
                i += 1
        blocks.append(lines[start : i + 1])
        i += 1
    return blocks


def _requirement(path: str) -> Requirement:
    """Parse the one PyNvVideoCodec line, tolerating an inline `#` comment.

    Inline comments are the prevailing style in these files (`pillow>=12.3.0,<13
    # bounded: ...` sits three lines away), and `Requirement()` rejects them, so
    stripping first is what keeps a future annotation from turning this into an
    opaque parse error. Selection is by canonical distribution name rather than
    by prefix, so a differently-named sibling cannot be mistaken for this one.
    """
    found: list[Requirement] = []
    for raw in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            requirement = Requirement(line)
        except Exception:  # noqa: BLE001 - not a requirement line
            continue
        if canonicalize_name(requirement.name) == DISTRIBUTION:
            found.append(requirement)
    assert len(found) == 1, f"expected one PyNvVideoCodec line in {path}, got {found}"
    return found[0]


def _template_floor_versions(path: str) -> set[str]:
    """Every version literal the PyNvVideoCodec instructions in `path` compare against.

    Every semver on a comparing or installing line inside an instruction that names
    PyNvVideoCodec is collected, not just the ones adjacent to the package name. The
    guards are shell tests whose two sides sit on different lines --
    `newest=$(printf '%s\\n2.2.3\\n' ...)` and `if [ "$newest" != "2.2.3" ]` -- and
    pinning only the side that happens to carry the package name lets a half-finished
    bump invert the comparison: the guard then accepts the stale version and rejects
    the new one, with the test still green.

    Diagnostic lines are excluded, because that is where versions the guard is *not*
    comparing against legitimately appear -- the post-overlay check names the base
    image's 2.1.0 in its error text, and holding that to the floor would force an
    unrelated rewrite of the message on every bump.
    """
    versions: set[str] = set()
    for block in _instruction_blocks((ROOT / path).read_text(encoding="utf-8")):
        code = [line for line in block if not line.lstrip().startswith("#")]
        if not any(DISTRIBUTION in line.lower() for line in code):
            continue
        for line in code:
            if _DIAGNOSTIC.match(line):
                continue
            # `printf '%s\n2.2.3\n'` is the shape of the shell comparisons, and the
            # literal backslash-n leaves no word boundary in front of the version.
            # Without flattening the escape first, that constant matches nothing here
            # and is silently unpinned.
            versions.update(_SEMVER.findall(line.replace("\\n", " ")))
    return versions


def _template_install_specs(path: str) -> list[str]:
    """Requirement specifiers a template installs PyNvVideoCodec with."""
    specs: list[str] = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#") or _DIAGNOSTIC.match(line):
            continue
        specs += _INLINE_SPEC.findall(line)
    return specs


@pytest.mark.parametrize("path", REQUIREMENTS)
def test_requirements_declare_the_expected_spec(path: str) -> None:
    assert _requirement(path).specifier == SpecifierSet(EXPECTED_SPEC)


def test_the_spec_is_bounded_above() -> None:
    """`container/deps/README.md`: "Never use `>=`" -- it stops pinning anything.

    Distinct from the exact-spec cases above, which move with EXPECTED_SPEC: this
    one survives an edit that drops the cap and updates that constant to match.
    """
    specifier = _requirement(REQUIREMENTS[0]).specifier
    assert any(
        s.operator in ("<", "<=", "==", "~=") for s in specifier
    ), f"PyNvVideoCodec is specified as {specifier}, which has no upper bound"


def test_templates_install_the_same_spec() -> None:
    """A template that installs PyNvVideoCodec must use the declared specifier.

    The TRT-LLM image installs a system-site copy of its own, beside the venv copy
    `requirements.trtllm.txt` provides, and its comment says the two specifiers
    must stay identical. Nothing enforced that: the floor scan below reads lower
    bounds only, so a template left at `>=2.2.3` while the requirements file moved
    to `>=2.2.3,<3` would pass, and a 3.x release would then land in system site
    while the venv stayed on 2.x.

    One test over all templates rather than one per template: only TRT-LLM carries
    an inline specifier today, so a per-template parametrization would report two
    cases that assert nothing. Scanning all three still matters -- a specifier
    added to another template has to be caught -- so the scan is asserted to find
    at least one, which is what stops it going quietly blind.
    """
    staged = [path for path in TEMPLATES if (ROOT / path).is_file()]
    if not staged:
        pytest.skip("no runtime templates are staged in this component image")
    found = {path: _template_install_specs(path) for path in staged}
    present = {path: specs for path, specs in found.items() if specs}
    assert present, f"no template installs PyNvVideoCodec any more: {found}"
    for path, specs in present.items():
        for spec in specs:
            assert SpecifierSet(spec) == SpecifierSet(
                EXPECTED_SPEC
            ), f"{path} installs {spec}"


@pytest.mark.parametrize("path", TEMPLATES)
def test_templates_guard_on_the_same_floor(path: str) -> None:
    """Every floor constant in a template must equal the declared lower bound.

    Deliberately an assertion over the whole set, not "the floor appears
    somewhere": a guard still comparing against a superseded version passes that
    weaker check while letting through exactly the drift this file exists to catch.
    """
    template = ROOT / path
    if not template.is_file():
        pytest.skip(f"{path} is not staged in this component image (.dockerignore)")
    versions = _template_floor_versions(path)
    assert versions, f"no PyNvVideoCodec floor constant found in {path}"
    assert versions == {EXPECTED_FLOOR}, f"{path} guards on {sorted(versions)}"


def main() -> int:
    """Run every check outside pytest, for the `pynvvideocodec-spec` hook.

    The pre-commit environment has no repo conftest and no framework images, so it
    is both the only place the template half is guaranteed to run and the place
    that must not depend on the suite around it. A missing template is a hard error
    here, not the skip it is inside an image: in a checkout the file is always there.
    """
    failures: list[str] = []

    def check(label: str, fn, *args) -> None:
        try:
            fn(*args)
        except BaseException as exc:  # noqa: BLE001 - report, never abort the sweep
            failures.append(f"{label}: {exc}")

    check("spec is bounded above", test_the_spec_is_bounded_above)
    check("template install specs", test_templates_install_the_same_spec)
    for path in REQUIREMENTS:
        check(
            f"requirements spec {path}",
            test_requirements_declare_the_expected_spec,
            path,
        )
    for path in TEMPLATES:
        if not (ROOT / path).is_file():
            failures.append(f"template {path}: file not found")
            continue
        check(f"template floor {path}", test_templates_guard_on_the_same_floor, path)

    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        print(f"\n{len(failures)} PyNvVideoCodec spec check(s) failed.")
        return 1
    print(
        f"PyNvVideoCodec spec {EXPECTED_SPEC} is consistent across "
        f"{len(REQUIREMENTS)} requirements files and {len(TEMPLATES)} templates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
