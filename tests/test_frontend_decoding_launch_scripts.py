# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every launch script that enables frontend decoding must put the NIXL wheel's
native libs on the loader path.

A frontend started with ``--frontend-decoding`` builds a media pipeline that
needs a real NIXL agent. nixl-sys resolves ``libnixl_capi.so`` with a plain
``dlopen()``, and the SGLang image keeps that library in the wheel's private
directory rather than on the default loader path. Without
``export_nixl_wheel_libs`` the frontend reports "NIXL is not supported in stub
mode", never adds the model from discovery, and ``/v1/models`` stays empty
forever -- a deployment that hangs at startup with no failing process.

This shipped once: ``agg_vision.sh`` carried the setup inline, and when
``--frontend-decoding`` was extended to the E/PD and disaggregated scripts they
did not get it, so ``multimodal_e_pd_fd_qwen`` failed every post-merge run.
"""

import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"
_HELPER = "export_nixl_wheel_libs"

# An uncommented call / definition. Matching bare text instead would accept a
# commented-out call -- which is the likeliest way this regresses, and exactly
# what this test exists to catch.
_HELPER_CALL = re.compile(rf"^\s*{_HELPER}\b", re.MULTILINE)
_HELPER_DEF = re.compile(rf"^\s*{_HELPER}\s*\(\)", re.MULTILINE)


def _forwards_frontend_decoding(text: str) -> bool:
    """Whether a script hands ``--frontend-decoding`` to a worker.

    Skips the places the flag appears without being forwarded: comments, help
    text, and the ``--frontend-decoding)`` case label (which parses the flag
    rather than passing it on). A script that only documents the flag does not
    launch a decoding frontend and so needs no library-path setup.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", "echo", "printf")):
            continue
        if "--frontend-decoding" not in stripped:
            continue
        if re.fullmatch(r"--frontend-decoding\)", stripped):
            continue
        return True
    return False


def _scripts_enabling_frontend_decoding() -> list[Path]:
    """Launch scripts that hand ``--frontend-decoding`` to a worker."""
    return [
        path
        for path in sorted(_EXAMPLES.rglob("*.sh"))
        if _forwards_frontend_decoding(path.read_text(encoding="utf-8"))
    ]


def test_frontend_decoding_scripts_export_nixl_wheel_libs():
    scripts = _scripts_enabling_frontend_decoding()
    # Guard the guard: if the flag is renamed this test must fail loudly rather
    # than silently pass over an empty set.
    assert scripts, "no launch script passes --frontend-decoding; update this test"

    missing = [
        str(p.relative_to(_REPO_ROOT))
        for p in scripts
        if not _HELPER_CALL.search(p.read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"launch scripts enable --frontend-decoding without calling {_HELPER}(): "
        f"{missing}. Their frontend will fail with 'NIXL is not supported in "
        f"stub mode' and never serve the model."
    )


def test_helper_is_defined_in_shared_launch_utils():
    """The scripts above only source launch_utils.sh, so the helper must live
    there -- otherwise they call an undefined function and, under `set -e`,
    die at startup."""
    utils = (_EXAMPLES / "common" / "launch_utils.sh").read_text(encoding="utf-8")
    assert _HELPER_DEF.search(utils), f"{_HELPER}() is not defined in launch_utils.sh"
