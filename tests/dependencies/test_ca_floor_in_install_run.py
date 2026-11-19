# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the ca-certificates floor check in the RUN that installs the package.

BuildKit keys a RUN layer on its parent layer and its command text, not on the
package versions that the RUN installs. If the check is a RUN of its own, a
cached install layer that holds an older ca-certificates is reused, and the
check then fails on every build that gets that layer. If the check is the last
command of the install RUN, a layer below the floor fails before it is cached.

`.dockerignore` keeps the templates out of the image build contexts, so these
cases skip inside the images. The `ca-floor-in-install-run` pre-commit hook
runs this module as a script against a checkout.
"""

import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "container" / "templates"
FRAGMENT = TEMPLATES / "verify_ca_certificates.Dockerfile"

_INCLUDE = re.compile(
    r'\{%-?\s*include\s+"templates/verify_ca_certificates\.Dockerfile"'
)
_RUN = re.compile(r"^RUN\b")


def misplaced_checks(text: str) -> list[str]:
    """Return one problem per check include that is not the end of an install RUN."""
    lines = text.splitlines()
    problems = []
    for i, line in enumerate(lines):
        if not _INCLUDE.search(line):
            continue
        where = f"line {i + 1}"
        # Docker skips blank and comment lines inside a continued RUN.
        previous = next(
            (
                lines[j].rstrip()
                for j in range(i - 1, -1, -1)
                if lines[j].strip() and not lines[j].lstrip().startswith("#")
            ),
            "",
        )
        if not previous.endswith("\\"):
            problems.append(f"{where}: the check is not inside a RUN")
            continue
        if line.rstrip().endswith("\\"):
            problems.append(f"{where}: the check is not the last command of its RUN")
        start = next((j for j in range(i - 1, -1, -1) if _RUN.match(lines[j])), None)
        body = lines[start:i] if start is not None else []
        if not any(
            "ca-certificates" in body_line and not body_line.lstrip().startswith("#")
            for body_line in body
        ):
            problems.append(
                f"{where}: its RUN does not install or upgrade ca-certificates"
            )
    return problems


def _templates() -> list[Path]:
    return sorted(
        path
        for path in TEMPLATES.glob("*.Dockerfile")
        if path != FRAGMENT and _INCLUDE.search(path.read_text(encoding="utf-8"))
    )


def test_a_check_in_its_own_run_is_rejected() -> None:
    """The layout that let a stale install layer reach the check is flagged."""
    text = (
        "RUN apt-get update && apt-get install -y ca-certificates && \\\n"
        "    rm -rf /var/lib/apt/lists/*\n"
        "\n"
        '{% include "templates/verify_ca_certificates.Dockerfile" %}\n'
    )
    assert misplaced_checks(text) == ["line 4: the check is not inside a RUN"]


def test_a_check_at_the_end_of_the_install_run_passes() -> None:
    text = (
        "RUN apt-get update && apt-get install -y ca-certificates && \\\n"
        "    rm -rf /var/lib/apt/lists/* && \\\n"
        '    {% include "templates/verify_ca_certificates.Dockerfile" %}\n'
    )
    assert misplaced_checks(text) == []


def test_the_check_is_a_fragment_not_a_run() -> None:
    if not FRAGMENT.is_file():
        pytest.skip(f"{FRAGMENT} is not in this image (.dockerignore)")
    text = FRAGMENT.read_text(encoding="utf-8")
    assert not re.search(r"^RUN\b", text, re.MULTILINE)


def test_every_check_ends_an_install_run() -> None:
    if not FRAGMENT.is_file():
        pytest.skip(f"{FRAGMENT} is not in this image (.dockerignore)")
    templates = _templates()
    assert templates, "no template includes the ca-certificates check"
    problems = [
        f"{path.relative_to(ROOT)} {problem}"
        for path in templates
        for problem in misplaced_checks(path.read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)


def main() -> int:
    """Check every template outside pytest, for the pre-commit hook."""
    if not FRAGMENT.is_file():
        print(f"ERROR: {FRAGMENT.relative_to(ROOT)}: file not found")
        return 1
    templates = _templates()
    problems = [
        f"{path.relative_to(ROOT)} {problem}"
        for path in templates
        for problem in misplaced_checks(path.read_text(encoding="utf-8"))
    ]
    if not templates:
        problems.append("no template includes the ca-certificates check")
    for problem in problems:
        print(f"ERROR: {problem}")
    if problems:
        return 1
    print(f"{len(templates)} templates end their install RUN with the check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
