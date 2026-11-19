# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    r"""\{%[-+]?\s*include\s+["']templates/verify_ca_certificates\.Dockerfile["']"""
)
_RUN = re.compile(r"^RUN\b")
# A line that holds only a Jinja tag or comment renders to nothing.
_JINJA_ONLY = re.compile(r"\s*\{[%#].*[%#]\}\s*")
# An install or upgrade command, and the package name as a whole word, so that
# update-ca-certificates and /etc/ssl/certs/ca-certificates.crt do not count.
_INSTALL = re.compile(r"\b(?:apt-get|apt|dnf)\b[^;&|]*\b(?:install|upgrade)\b")
_PACKAGE = re.compile(r"(?<![\w./-])ca-certificates(?![\w.-])")
# Keywords that can close the RUN after the check, as in `if ...; then ...; fi`.
_CLOSER = re.compile(r"\s*(?:fi|done|esac|\}|\))\s*;?\s*\\?\s*")


def _skipped(line: str) -> bool:
    """Docker drops blank and comment lines in a RUN, and Jinja drops tag lines."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#") or bool(_JINJA_ONLY.fullmatch(line))


def _run_body(lines: list[str], i: int) -> list[str]:
    """Return the lines of the RUN that continues into line i, or [] if none does."""
    body = []
    for j in range(i - 1, -1, -1):
        if _skipped(lines[j]):
            continue
        if not lines[j].rstrip().endswith("\\"):
            return []
        body.append(lines[j])
        if _RUN.match(lines[j]):
            return body[::-1]
    return []


def _ends_run(lines: list[str], i: int) -> bool:
    """Return True if only closing keywords follow line i in its RUN."""
    if not lines[i].rstrip().endswith("\\"):
        return True
    for line in lines[i + 1 :]:
        if _skipped(line):
            continue
        if not _CLOSER.fullmatch(line):
            return False
        if not line.rstrip().endswith("\\"):
            return True
    return False


def misplaced_checks(text: str) -> list[str]:
    """Return one problem per check include that is not the end of an install RUN."""
    lines = text.splitlines()
    problems = []
    for i, line in enumerate(lines):
        if not _INCLUDE.search(line):
            continue
        where = f"line {i + 1}"
        body = _run_body(lines, i)
        if not body:
            problems.append(f"{where}: the check is not inside a RUN")
            continue
        previous = body[-1].rstrip()
        if not (
            re.search(r"(?:&&|;)\s*\\$", previous) or line.lstrip().startswith("&&")
        ):
            problems.append(f"{where}: the check does not follow && or ;")
        if not _ends_run(lines, i):
            problems.append(f"{where}: the check is not the last command of its RUN")
        joined = " ".join(body_line.rstrip().rstrip("\\") for body_line in body)
        if not (_INSTALL.search(joined) and _PACKAGE.search(joined)):
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


_INC = '{% include "templates/verify_ca_certificates.Dockerfile" %}'
_INSTALL_RUN = "RUN apt-get update && apt-get install -y ca-certificates && \\\n"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param(
            _INSTALL_RUN + "    rm -rf /x\n\n" + _INC + "\n",
            "not inside a RUN",
            id="own-run",
        ),
        pytest.param(
            _INSTALL_RUN + "    rm -rf /x\n\n"
            "RUN update-ca-certificates && \\\n    " + _INC + "\n",
            "does not install or upgrade ca-certificates",
            id="run-without-the-package",
        ),
        pytest.param(
            "RUN apt-get install -y ca-certificates || \\\n    " + _INC + "\n",
            "does not follow && or ;",
            id="after-or",
        ),
        pytest.param(
            _INSTALL_RUN + "    " + _INC + " && \\\n    true\n",
            "not the last command",
            id="not-last",
        ),
        pytest.param(
            _INSTALL_RUN + "    rm -rf /x\n\n" + _INC.replace('"', "'") + "\n",
            "not inside a RUN",
            id="own-run-single-quotes",
        ),
        pytest.param(
            _INSTALL_RUN + "    rm -rf /x\n\n" + _INC.replace("{%", "{%+") + "\n",
            "not inside a RUN",
            id="own-run-plus-tag",
        ),
    ],
)
def test_a_misplaced_check_is_flagged(text: str, problem: str) -> None:
    problems = misplaced_checks(text)
    assert len(problems) == 1 and problem in problems[0], problems


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            _INSTALL_RUN + "    rm -rf /x && \\\n    " + _INC + "\n", id="and"
        ),
        pytest.param(
            "RUN apt-get install -y ca-certificates \\\n"
            "    && rm -rf /x \\\n    && " + _INC + "\n",
            id="leading-and",
        ),
        pytest.param(
            "RUN mkdir -p /w \\\n    && if [ -n x ]; then \\\n"
            "        apt-get install -y ca-certificates \\\n"
            "        && " + _INC + "; \\\n    fi\n",
            id="inside-if",
        ),
        pytest.param(
            _INSTALL_RUN
            + "{% if x %}\n    true && \\\n{% endif %}\n    "
            + _INC
            + "\n",
            id="jinja-tag-lines",
        ),
        pytest.param(
            _INSTALL_RUN + "    # note\n    " + _INC + "\n", id="comment-line"
        ),
    ],
)
def test_a_check_that_ends_its_install_run_passes(text: str) -> None:
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
