# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the planner and operator on distroless bases at v4.2.1 or newer.

The planner runtime stage starts from nvcr.io/nvidia/distroless/python at
`planner_runtime_image_tag` in `container/context.yaml`. The operator image
starts from nvcr.io/nvidia/distroless/go in `deploy/operator/Dockerfile`.
v4.2.1 is the first release of both bases that ships zlib1g
1:1.3.dfsg+really1.3.1-1+deb13u1, so neither tag may go below it.

`container/context.yaml` is excluded by `.dockerignore`, so the cases skip in
the images that run this suite. The `distroless-base-tags` pre-commit hook runs
this module as a script against a checkout, where a missing file is an error,
as `test_ffmpeg_version_consistency.py` does. The checks raise `CheckFailed`
explicitly instead of using `assert`, so the hook also fails under `python -O`.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]

CONTEXT = "container/context.yaml"
OPERATOR_DOCKERFILE = "deploy/operator/Dockerfile"

# The first distroless release of both bases with the zlib1g build above.
FLOOR = (4, 2, 1)

# The release part of a distroless tag: "3.13-v4.2.1" and "v4.2.1" both end in it.
_RELEASE = re.compile(r"(?:^|-)v(\d+)\.(\d+)\.(\d+)$")
_OPERATOR_BASE = re.compile(
    r"^FROM\s+nvcr\.io/nvidia/distroless/go:(\S+)\s*$", re.MULTILINE
)


class CheckFailed(Exception):
    """A distroless base tag is below FLOOR or cannot be read."""


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not staged in this image (.dockerignore)")
    return path.read_text(encoding="utf-8")


def _release(tag: str) -> tuple[int, int, int]:
    match = _RELEASE.search(tag)
    if not match:
        raise CheckFailed(f"cannot read a vX.Y.Z release from the tag {tag!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def planner_tag() -> str:
    try:
        tag = yaml.safe_load(_read(CONTEXT))["dynamo"]["planner_runtime_image_tag"]
    except (KeyError, TypeError) as exc:
        raise CheckFailed(f"{CONTEXT} has no dynamo.planner_runtime_image_tag") from exc
    return str(tag)


def operator_tag() -> str:
    tags = _OPERATOR_BASE.findall(_read(OPERATOR_DOCKERFILE))
    if len(tags) != 1:
        raise CheckFailed(
            f"expected one distroless/go FROM in {OPERATOR_DOCKERFILE}, found {tags}"
        )
    return tags[0]


def check_floor() -> None:
    low = [
        f"{name} {tag}"
        for name, tag in (
            ("distroless/python", planner_tag()),
            ("distroless/go", operator_tag()),
        )
        if _release(tag) < FLOOR
    ]
    if low:
        floor = ".".join(str(part) for part in FLOOR)
        raise CheckFailed(f"below the v{floor} floor: {', '.join(low)}")


def test_distroless_bases_meet_the_zlib_floor() -> None:
    check_floor()


def test_release_parsing() -> None:
    assert _release("3.13-v4.2.1") == (4, 2, 1)
    assert _release("v4.10.0") == (4, 10, 0)
    assert _release("v4.10.0") > FLOOR
    assert _release("3.13-v4.1.10") < FLOOR
    with pytest.raises(CheckFailed):
        _release("latest")


def test_files_missing_from_the_image_skip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_distroless_bases_meet_the_zlib_floor()


def main() -> int:
    """Script entry point for the `distroless-base-tags` pre-commit hook."""
    for rel in (CONTEXT, OPERATOR_DOCKERFILE):
        if not (ROOT / rel).is_file():
            print(f"ERROR: {rel} not found")
            return 1
    try:
        check_floor()
    except CheckFailed as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"distroless/python {planner_tag()} and distroless/go {operator_tag()} "
        "are at or above the floor."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
