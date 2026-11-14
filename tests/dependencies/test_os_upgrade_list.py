# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the OS package refresh lists carrying the packages they must upgrade.

Each Ubuntu-based image has one `apt-get install --only-upgrade` step with a
named package list. A package that is not in the list keeps the base image's
build even after Ubuntu publishes a newer one. The lists differ on purpose (only
the TensorRT-LLM list names ca-certificates, and the sidecar list has no
libbrotli1), so this checks that every list carries the shared set below instead
of comparing the lists with each other.

`.dockerignore` keeps `container/**/*.Dockerfile` out of the image build
contexts, so the template cases skip inside the images. The `os-upgrade-list`
pre-commit hook runs this module as a script against a checkout, where a missing
file is an error.
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

DOCKERFILES = (
    "container/templates/frontend.Dockerfile",
    "container/templates/sglang_runtime.Dockerfile",
    "container/templates/trtllm_runtime.Dockerfile",
    "container/templates/vllm_runtime.Dockerfile",
    "lib/sidecar/Dockerfile",
)

REQUIRED_PACKAGES = frozenset(
    {
        "apt",
        "apt-utils",
        "libapt-pkg6.0t64",
        "libglib2.0-0t64",
        "libgnutls30t64",
        "libperl5.38t64",
        "openssh-client",
        "openssh-server",
        "openssh-sftp-server",
        "perl",
        "perl-base",
        "perl-modules-5.38",
        "zlib1g",
        "zlib1g-dev",
    }
)

# One package per continuation line, as the lists are written: `pkg \`,
# `pkg && \` on the last line before the next command, or a bare `pkg` that ends
# the RUN.
_PACKAGE_LINE = re.compile(r"^\s+([a-z0-9][a-z0-9.+-]*)\s*(&&\s*)?(\\)?\s*$")


def upgrade_list(text: str) -> list[str]:
    """Return the packages of the one `--only-upgrade` install in a Dockerfile."""
    lines = text.splitlines()
    starts = [
        i
        for i, line in enumerate(lines)
        if "--only-upgrade" in line
        and "install" in line
        and not line.lstrip().startswith("#")
    ]
    if len(starts) != 1:
        raise AssertionError(
            f"expected one --only-upgrade install, found {len(starts)}"
        )
    packages: list[str] = []
    for line in lines[starts[0] + 1 :]:
        match = _PACKAGE_LINE.match(line)
        if not match:
            break
        packages.append(match.group(1))
        if match.group(2) or not match.group(3):
            break
    return packages


def _missing(path: str) -> list[str]:
    packages = set(upgrade_list((ROOT / path).read_text(encoding="utf-8")))
    return sorted(REQUIRED_PACKAGES - packages)


@pytest.mark.parametrize("path", DOCKERFILES)
def test_os_upgrade_lists_carry_required_packages(path: str) -> None:
    if not (ROOT / path).is_file():
        pytest.skip(f"{path} is not in this image (.dockerignore)")
    missing = _missing(path)
    assert not missing, f"{path}: --only-upgrade list lacks {missing}"


def main() -> int:
    """Check every file outside pytest, for the `os-upgrade-list` hook."""
    failures = []
    for path in DOCKERFILES:
        if not (ROOT / path).is_file():
            failures.append(f"{path}: file not found")
            continue
        try:
            missing = _missing(path)
        except AssertionError as exc:
            failures.append(f"{path}: {exc}")
            continue
        if missing:
            failures.append(f"{path}: --only-upgrade list lacks {missing}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print(
        f"{len(DOCKERFILES)} --only-upgrade lists carry all "
        f"{len(REQUIRED_PACKAGES)} required packages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
