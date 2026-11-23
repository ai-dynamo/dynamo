# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the pins that the TensorRT-LLM template raises in system site.

The template repeats each pin as version bounds in the checks after the install.
This test keeps those bounds equal to each other and around the installed
version, so a bump cannot leave one of them behind.

A template that overlays `runtime_full` onto a fresh base with
`COPY --from=runtime_full / /`, as release branches before #18850 do, also needs
a pre_runtime whiteout for each pin. That `COPY` adds and replaces files but
never deletes them. An upgrade renames the version-stamped dist-info directory,
so without a whiteout of the old one the image ships both, and an image inventory
reads the old version. Without a whiteout of the package directory, every module
that the new version removed stays in the image. Without the overlay, pip
removes the old files in place, and the whiteout check does not apply.

The template is not in the image build contexts, so the template tests skip
inside an image. The `trtllm-system-site-pins` pre-commit hook runs this module
as a script against a checkout. The checks raise `CheckFailed` explicitly instead
of using `assert`, so they also fail under `python -O`.

The file name has no `test_trtllm_` prefix on purpose. pyproject.toml registers
`dynamo.trtllm.tests.conftest` as a pytest plugin, and its `pytest_ignore_collect`
drops every `test_trtllm_*.py` file where tensorrt_llm is not installed. This
module carries no `trtllm` marker either, so with that prefix no CI test job
would run it.
"""

import ast
import fnmatch
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

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
_OVERLAY = "COPY --from=runtime_full / /"
# pip options in these steps that take a value.
_OPTIONS_WITH_VALUE = {"--report", "-c", "--constraint", "-r", "--requirement"}
# A check names its version bounds in a dict literal, for example
# bounds = {"transformers": ("5.10.4", "5.10.5")}.
_BOUNDS = re.compile(r"\bbounds = (\{[^{}]*\})")

# The top-level site-packages folders that each pin's wheel installs, besides its
# dist-info directory: the import packages and the vendored library folders. The
# wheel file lists on PyPI give these names. PyNvVideoCodec also installs loose
# `samples/` and `benchmarks/` folders, which this table leaves out. A pin that is
# missing here fails the test, so a new pin must name its folders.
PACKAGE_DIRS: dict[str, tuple[str, ...]] = {
    "aiohttp": ("aiohttp",),
    "certifi": ("certifi",),
    "datasets": ("datasets",),
    "filelock": ("filelock",),
    "fsspec": ("fsspec",),
    "jupyter-server": ("jupyter_server",),
    "jupyterlab": ("jupyterlab",),
    "mooncake-transfer-engine-cuda13": (
        "mooncake",
        "mooncake_transfer_engine_cuda13.libs",
    ),
    "multidict": ("multidict",),
    "notebook": ("notebook",),
    "onnx": ("onnx",),
    "pillow": ("PIL", "pillow.libs"),
    "pip": ("pip",),
    "pyasn1": ("pyasn1",),
    "pynvvideocodec": ("PyNvVideoCodec", "pynvvideocodec."),
    "soupsieve": ("soupsieve",),
    "transformers": ("transformers",),
    "urllib3": ("urllib3",),
    "werkzeug": ("werkzeug",),
}


class CheckFailed(Exception):
    """The template breaks one of the rules in this module."""


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


def _system_site_pins(text: str) -> dict[str, SpecifierSet]:
    """The specifier of every distribution that a system-site pip step installs."""
    pins: dict[str, SpecifierSet] = {}
    for block in _instructions(text):
        start = block.find(_SYSTEM_PIP_INSTALL)
        while start != -1:
            command = block[start + len(_SYSTEM_PIP_INSTALL) :]
            for end in ("&&", ";", "||"):
                command = command.split(end, 1)[0]
            skip_next = False
            for token in shlex.split(command):
                if skip_next:
                    skip_next = False
                    continue
                if token.startswith("-"):
                    skip_next = token in _OPTIONS_WITH_VALUE
                    continue
                try:
                    requirement = Requirement(token)
                except InvalidRequirement as exc:
                    raise CheckFailed(
                        f"cannot parse {token!r} in a system-site pip step: {exc}"
                    ) from exc
                pins[canonicalize_name(requirement.name)] = requirement.specifier
            start = block.find(_SYSTEM_PIP_INSTALL, start + 1)
    return pins


def _whiteout_patterns(text: str) -> list[str]:
    """The site-packages entries of the pre_runtime whiteout, as glob patterns."""
    whiteouts = [b for b in _instructions(text) if b.lstrip().startswith(_WHITEOUT)]
    if len(whiteouts) != 1:
        raise CheckFailed(f"expected one pre_runtime whiteout, found {len(whiteouts)}")
    command = whiteouts[0].split("&&", 1)[0]
    return [
        token[len(SITE) :] for token in shlex.split(command) if token.startswith(SITE)
    ]


def _bounds(text: str) -> dict[str, set[tuple[str, str]]]:
    """Every (floor, cap) pair that the checks use, by canonical name."""
    found: dict[str, set[tuple[str, str]]] = {}
    for block in _instructions(text):
        for literal in _BOUNDS.findall(block):
            try:
                bounds = ast.literal_eval(literal)
                pairs = {
                    canonicalize_name(name): (str(floor), str(cap))
                    for name, (floor, cap) in bounds.items()
                }
            except (ValueError, TypeError, SyntaxError) as exc:
                raise CheckFailed(f"cannot parse bounds {literal}: {exc}") from exc
            for name, pair in pairs.items():
                found.setdefault(name, set()).add(pair)
    return found


def _version(text: str) -> Version:
    try:
        return Version(text)
    except InvalidVersion as exc:
        raise CheckFailed(f"{text!r} is not a version") from exc


def check_parsers(text: str) -> None:
    """A parser that finds nothing would pass the checks below."""
    pins = _system_site_pins(text)
    if not {"certifi", "pip", "pillow"} <= pins.keys():
        raise CheckFailed(f"the pin parser found only {sorted(pins)}")
    if not _bounds(text).keys() & pins.keys():
        raise CheckFailed("the bounds parser found no bounds for any pin")


def _has_overlay(text: str) -> bool:
    return any(b.strip().startswith(_OVERLAY) for b in _instructions(text))


def check_whiteouts(text: str) -> None:
    """With an overlay, each pin's dist-info and package folders need a whiteout."""
    if not _has_overlay(text):
        return
    patterns = _whiteout_patterns(text)
    problems = []
    for name in sorted(_system_site_pins(text)):
        if name not in PACKAGE_DIRS:
            problems.append(f"{name} is not in PACKAGE_DIRS; add its folders there")
            continue
        # pip names the directory <name with "_" for "-">-<version>.dist-info.
        entries = (f"{name.replace('-', '_')}-0.0.0.dist-info", *PACKAGE_DIRS[name])
        missing = [
            entry
            for entry in entries
            if not any(fnmatch.fnmatchcase(entry, pattern) for pattern in patterns)
        ]
        if missing:
            problems.append(f"{name} has no whiteout entry for {missing}")
    if problems:
        raise CheckFailed(
            f"{TEMPLATE} raises these pins in system site: " + "; ".join(problems)
        )


def check_bounds(text: str) -> None:
    """The checks of a pin use one pair of bounds, and it holds the installed version.

    For an exact pin that version is the pinned one. For a range it is the floor,
    and the cap of the range must equal the cap of the bounds.
    """
    pins = _system_site_pins(text)
    problems = []
    for name, pairs in sorted(_bounds(text).items()):
        if name not in pins:
            continue
        if len(pairs) != 1:
            problems.append(f"{name}: the checks use different bounds {sorted(pairs)}")
            continue
        ((floor, cap),) = pairs
        specifier = pins[name]
        exact = [s.version for s in specifier if s.operator == "=="]
        lower = [s.version for s in specifier if s.operator == ">="]
        upper = [s.version for s in specifier if s.operator == "<"]
        installed = exact or lower
        if len(installed) != 1:
            problems.append(f"{name}: cannot tell the installed version of {specifier}")
            continue
        if not _version(floor) <= _version(installed[0]) < _version(cap):
            problems.append(
                f"{name}: installed as {specifier}, outside the bounds {floor}..{cap}"
            )
        elif upper and [_version(v) for v in upper] != [_version(cap)]:
            problems.append(
                f"{name}: installed as {specifier}, but the bounds cap is {cap}"
            )
    if problems:
        raise CheckFailed(f"{TEMPLATE}: " + "; ".join(problems))


CHECKS = (check_parsers, check_whiteouts, check_bounds)


def test_the_parsers_find_pins_and_bounds() -> None:
    check_parsers(_template_text())


def test_every_system_site_pin_has_whiteouts() -> None:
    check_whiteouts(_template_text())


def test_the_check_bounds_match_the_installed_versions() -> None:
    check_bounds(_template_text())


def test_template_missing_from_the_image_skips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        test_every_system_site_pin_has_whiteouts()


# A small template in the shapes the real one uses, for the cases below.
_SAMPLE = r"""
RUN /usr/bin/python3 -m pip install --break-system-packages --upgrade pip==26.2.2 \
        certifi==2026.10.23 "pillow>=13.0.0,<14"
RUN /usr/bin/python3 -m pip install --break-system-packages --upgrade \
        --report /tmp/upgrade.json "transformers==5.10.4" && \
    /usr/bin/python3 -c 'bounds = {"transformers": ("5.10.4", "5.10.5")}'
RUN /usr/bin/python3 -m pip install --break-system-packages --upgrade "filelock>=4.2.1,<5"
RUN rm -rf /workspace /home/ubuntu \
    /usr/local/lib/python3.12/dist-packages/pip \
    /usr/local/lib/python3.12/dist-packages/pip-*.dist-info \
    /usr/local/lib/python3.12/dist-packages/certifi \
    /usr/local/lib/python3.12/dist-packages/certifi-* \
    /usr/local/lib/python3.12/dist-packages/PIL \
    /usr/local/lib/python3.12/dist-packages/pillow-* \
    /usr/local/lib/python3.12/dist-packages/pillow.libs \
    /usr/local/lib/python3.12/dist-packages/transformers \
    /usr/local/lib/python3.12/dist-packages/transformers-*.dist-info \
    /usr/local/lib/python3.12/dist-packages/filelock \
    /usr/local/lib/python3.12/dist-packages/filelock-*.dist-info && \
    true
COPY --from=runtime_full / /
RUN /usr/bin/python3 -c 'bounds = {"transformers": ("5.10.4", "5.10.5"), "filelock": ("4.2.1", "5")}'
"""


def _mutate(old: str, new: str) -> str:
    if _SAMPLE.count(old) != 1:
        raise ValueError(f"{old!r} occurs {_SAMPLE.count(old)} times in the sample")
    return _SAMPLE.replace(old, new)


def test_the_sample_passes() -> None:
    for check in CHECKS:
        check(_SAMPLE)


def _without_overlay(text: str) -> str:
    """The sample in the single-stage layout: no whiteout and no overlay."""
    start = text.index("RUN rm -rf /workspace /home/ubuntu")
    end = text.index(_OVERLAY) + len(_OVERLAY) + 1
    return text[:start] + text[end:]


def test_without_an_overlay_no_whiteout_is_needed() -> None:
    sample = _without_overlay(_SAMPLE)
    assert _OVERLAY not in sample and _WHITEOUT not in sample
    for check in CHECKS:
        check(sample)


def test_an_overlay_without_a_whiteout_fails() -> None:
    sample = _without_overlay(_SAMPLE) + _OVERLAY + "\n"
    with pytest.raises(CheckFailed, match="expected one pre_runtime whiteout, found 0"):
        check_whiteouts(sample)


def test_bounds_are_checked_without_an_overlay() -> None:
    sample = _without_overlay(_SAMPLE).replace(
        '"transformers==5.10.4"', '"transformers==5.10.5"'
    )
    with pytest.raises(CheckFailed, match="outside the bounds 5.10.4..5.10.5"):
        check_bounds(sample)


@pytest.mark.parametrize(
    "mutant, message",
    [
        pytest.param(
            _mutate(
                "    /usr/local/lib/python3.12/dist-packages/transformers \\\n", ""
            ),
            "transformers has no whiteout entry for ['transformers']",
            id="package-folder-whiteout-missing",
        ),
        pytest.param(
            _mutate(
                "    /usr/local/lib/python3.12/dist-packages/filelock-*.dist-info && \\",
                "    true && \\",
            ),
            "filelock has no whiteout entry for ['filelock-0.0.0.dist-info']",
            id="dist-info-whiteout-missing",
        ),
        pytest.param(
            _mutate('"filelock>=4.2.1,<5"', '"filelock>=4.2.1,<5" "wheel==0.45.1"'),
            "wheel is not in PACKAGE_DIRS",
            id="unknown-pin",
        ),
    ],
)
def test_whiteout_mutants_fail(mutant: str, message: str) -> None:
    with pytest.raises(CheckFailed, match=re.escape(message)):
        check_whiteouts(mutant)


@pytest.mark.parametrize(
    "mutant, message",
    [
        pytest.param(
            _mutate(
                '"transformers": ("5.10.4", "5.10.5"), "filelock"',
                '"transformers": ("5.10.3", "5.10.5"), "filelock"',
            ),
            "transformers: the checks use different bounds",
            id="one-check-left-behind",
        ),
        pytest.param(
            _mutate('"transformers==5.10.4"', '"transformers==5.10.5"'),
            "transformers: installed as ==5.10.5, outside the bounds 5.10.4..5.10.5",
            id="pin-bumped-without-the-checks",
        ),
        pytest.param(
            _mutate('"filelock>=4.2.1,<5"', '"filelock>=4.2.1,<6"'),
            "filelock: installed as <6,>=4.2.1, but the bounds cap is 5",
            id="range-cap-moved",
        ),
    ],
)
def test_bounds_mutants_fail(mutant: str, message: str) -> None:
    with pytest.raises(CheckFailed, match=re.escape(message)):
        check_bounds(mutant)


def _write_template(root: Path, text: str) -> None:
    template = root / TEMPLATE
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(text, encoding="utf-8")


def test_script_mode_fails_under_optimize(tmp_path) -> None:
    """The hook must still fail under `python -O`, which removes `assert`."""
    _write_template(
        tmp_path,
        _mutate("    /usr/local/lib/python3.12/dist-packages/transformers \\\n", ""),
    )
    script = (
        "import importlib.util, pathlib, sys; "
        f"spec = importlib.util.spec_from_file_location('pins', {__file__!r}); "
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
    assert "transformers has no whiteout entry for ['transformers']" in result.stdout


def test_script_mode_lets_an_interrupt_through(tmp_path, monkeypatch) -> None:
    """main() reports failed checks, but it must not swallow Ctrl-C."""
    _write_template(tmp_path, _SAMPLE)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert main() == 0

    def interrupted(text: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "CHECKS", (interrupted,))
    with pytest.raises(KeyboardInterrupt):
        main()


def main() -> int:
    """Run the checks outside pytest, for the `trtllm-system-site-pins` hook."""
    template = ROOT / TEMPLATE
    if not template.is_file():
        print(f"ERROR: {TEMPLATE}: file not found")
        return 1
    text = template.read_text(encoding="utf-8")
    failures: list[str] = []
    for check in CHECKS:
        try:
            check(text)
        except CheckFailed as exc:
            failures.append(f"{check.__name__}: {exc}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    pins = sorted(_system_site_pins(text))
    whiteouts = (
        "whiteouts and " if _has_overlay(text) else "no overlay to white out, and "
    )
    print(
        f"{len(pins)} system-site pins in {TEMPLATE}: {whiteouts}"
        f"consistent check bounds: {pins}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
