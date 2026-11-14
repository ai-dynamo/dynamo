# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every system-site pin in the TensorRT-LLM template needs a pre_runtime whiteout.

The template raises packages in the base image's system site in `runtime_full`.
`pre_runtime` starts from the base image again and copies `runtime_full` over it
with one `COPY`, which adds and replaces files but never deletes them. An upgrade
renames the version-stamped dist-info directory, so without a whiteout of the old
one the image ships both, and an image inventory reads the old version. The
post-overlay checks catch this only for the packages they name. This test covers
every pin, including one added later without a check.

The layout follows test_pynvvideocodec_spec.py. The template is not in the image
build contexts, so the test skips inside an image, and the
`trtllm-system-site-pins` pre-commit hook runs this module as a script against a
checkout.
"""

import fnmatch
import shlex
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = "container/templates/trtllm_runtime.Dockerfile"
SITE = "/usr/local/lib/python3.12/dist-packages/"

_SYSTEM_PIP_INSTALL = "/usr/bin/python3 -m pip install --break-system-packages"
_WHITEOUT = "RUN rm -rf /workspace /home/ubuntu"
# pip options in these steps that take a value.
_OPTIONS_WITH_VALUE = {"--report", "-c", "--constraint", "-r", "--requirement"}


def _instructions(text: str) -> list[str]:
    """Split a Dockerfile into instructions, joining backslash continuations."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        current.append(line.rstrip().removesuffix("\\"))
        if not line.rstrip().endswith("\\"):
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def _template_text() -> str:
    template = ROOT / TEMPLATE
    if not template.is_file():
        pytest.skip(f"{TEMPLATE} is not staged in this image (.dockerignore)")
    return template.read_text(encoding="utf-8")


def _system_site_pins(text: str) -> set[str]:
    """Canonical names of every distribution a system-site pip step installs."""
    pins: set[str] = set()
    for block in _instructions(text):
        start = block.find(_SYSTEM_PIP_INSTALL)
        while start != -1:
            command = block[start + len(_SYSTEM_PIP_INSTALL) :]
            for end in ("&&", ";", "||"):
                command = command.split(end, 1)[0]
            tokens = shlex.split(command)
            skip_next = False
            for token in tokens:
                if skip_next:
                    skip_next = False
                    continue
                if token.startswith("-"):
                    skip_next = token in _OPTIONS_WITH_VALUE
                    continue
                pins.add(canonicalize_name(Requirement(token).name))
            start = block.find(_SYSTEM_PIP_INSTALL, start + 1)
    return pins


def _whiteout_patterns(text: str) -> list[str]:
    """The site-packages entries of the pre_runtime whiteout, as glob patterns."""
    whiteouts = [b for b in _instructions(text) if b.lstrip().startswith(_WHITEOUT)]
    assert (
        len(whiteouts) == 1
    ), f"expected one pre_runtime whiteout, found {len(whiteouts)}"
    command = whiteouts[0].split("&&", 1)[0]
    return [
        token[len(SITE) :] for token in shlex.split(command) if token.startswith(SITE)
    ]


def _uncovered(text: str) -> list[str]:
    """Pins whose dist-info directory no whiteout pattern matches."""
    patterns = _whiteout_patterns(text)
    missing = []
    for name in sorted(_system_site_pins(text)):
        # pip names the directory <name with "_" for "-">-<version>.dist-info.
        sample = f"{name.replace('-', '_')}-0.0.0.dist-info"
        if not any(fnmatch.fnmatchcase(sample, pattern) for pattern in patterns):
            missing.append(name)
    return missing


def test_the_parser_finds_the_system_site_pins() -> None:
    """A parser that finds nothing would pass the coverage test below."""
    pins = _system_site_pins(_template_text())
    assert {"certifi", "pip", "pillow"} <= pins, f"found {sorted(pins)}"


def test_every_system_site_pin_has_a_dist_info_whiteout() -> None:
    missing = _uncovered(_template_text())
    assert not missing, (
        f"{TEMPLATE} raises {missing} in system site, but the pre_runtime "
        "whiteout has no entry for their dist-info directories"
    )


def test_template_missing_from_the_image_skips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_every_system_site_pin_has_a_dist_info_whiteout()


def main() -> int:
    """Run the checks outside pytest, for the `trtllm-system-site-pins` hook."""
    if not (ROOT / TEMPLATE).is_file():
        print(f"ERROR: {TEMPLATE}: file not found")
        return 1
    failures: list[str] = []
    for check in (
        test_the_parser_finds_the_system_site_pins,
        test_every_system_site_pin_has_a_dist_info_whiteout,
    ):
        try:
            check()
        except BaseException as exc:  # noqa: BLE001 - report, never abort the sweep
            failures.append(f"{check.__name__}: {exc}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    pins = sorted(_system_site_pins((ROOT / TEMPLATE).read_text(encoding="utf-8")))
    print(f"{len(pins)} system-site pins in {TEMPLATE} have a whiteout: {pins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
