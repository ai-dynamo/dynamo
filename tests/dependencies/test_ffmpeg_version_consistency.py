# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the in-tree FFmpeg version identical in the three files that declare it.

`container/context.yaml` decides what `wheel_builder.Dockerfile` actually
compiles. `container/compliance/native_packages.yaml` is what the generated SBOM
claims the image contains. The `deny_components` floor in
`container/compliance/policy/codec_policy.yaml` is what that SBOM is then
measured against. Nothing compared them, and only one direction fails loudly: an
SBOM *below* the floor is flagged, while an SBOM left above the version really
built simply makes the gate agree with a document that is wrong.

All three read "8.1.2" before the 9.0.1 bump, so this pins an invariant the repo
already held rather than imposing a new one.

Run as a script by the `ffmpeg-version-consistency` pre-commit hook, because
`container/context.yaml` is excluded by `.dockerignore` and so is absent from the
component images that run this suite -- the pytest cases skip there, and the hook
is what keeps the check enforced in CI. Same situation and same resolution as
`test_pynvvideocodec_spec.py`.
"""

import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]

CONTEXT = "container/context.yaml"
NATIVE_PACKAGES = "container/compliance/native_packages.yaml"
CODEC_POLICY = "container/compliance/policy/codec_policy.yaml"


def _load(rel: str) -> dict:
    path = ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not staged in this component image (.dockerignore)")
    return yaml.safe_load(path.read_text())


def _built_version() -> str:
    return _load(CONTEXT)["dynamo"]["ffmpeg_version"]


def _declared_versions() -> list[str]:
    packages = _load(NATIVE_PACKAGES)["packages"]
    return [p["version"] for p in packages if p["name"] == "ffmpeg"]


def _policy_floors() -> list[str]:
    components = _load(CODEC_POLICY).get("deny_components") or []
    return [
        c["min_fixed_version"]
        for c in components
        if c.get("name") == "ffmpeg" and "min_fixed_version" in c
    ]


def test_sbom_declares_the_version_that_is_built() -> None:
    built = _built_version()
    assert _declared_versions() == [built], (
        f"{NATIVE_PACKAGES} declares ffmpeg {_declared_versions()} but {CONTEXT} "
        f"builds {built}; the SBOM would describe an image that was never built"
    )


def test_policy_floor_matches_the_version_that_is_built() -> None:
    built = _built_version()
    assert _policy_floors() == [built], (
        f"{CODEC_POLICY} floors ffmpeg at {_policy_floors()} but {CONTEXT} builds "
        f"{built}; the SBOM gate would accept a stale bundled ffmpeg"
    )


def main() -> int:
    """Script entry point for the pre-commit hook."""
    for rel in (CONTEXT, NATIVE_PACKAGES, CODEC_POLICY):
        if not (ROOT / rel).is_file():
            print(f"ERROR: {rel} not found")
            return 1

    built = _built_version()
    failures = []
    if _declared_versions() != [built]:
        failures.append(
            f"{NATIVE_PACKAGES} declares ffmpeg {_declared_versions()}, "
            f"{CONTEXT} builds {built}"
        )
    if _policy_floors() != [built]:
        failures.append(
            f"{CODEC_POLICY} floors ffmpeg at {_policy_floors()}, "
            f"{CONTEXT} builds {built}"
        )

    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print(f"In-tree FFmpeg {built} is declared consistently across 3 files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
