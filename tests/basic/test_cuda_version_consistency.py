# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test to check CUDA version consistency across various packages.

Every CUDA signal readable from a running container -- environment variables,
nvcc, dpkg packages, pip distributions -- must report the CUDA major this
repository ships. Asserting against a fixed expected major (rather than merely
asserting the signals agree with each other) catches both a mix of majors
within one image and a stray wheel built for a major we no longer support.
"""

import re
import subprocess

import pytest

# Mark this with every framework to test every container. framework_agnostic
# stops tests/conftest.py from skipping it in containers that ship only one of
# them: this test reads the image itself and imports no framework module.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.post_merge,
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.trtllm,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.framework_agnostic,
    # Runs in well under a second, but every signal is a subprocess: bound the
    # whole test so a wedged nvcc, dpkg or pip cannot hang CI.
    pytest.mark.timeout(60),
]

# Per-command bound, so one stuck tool fails fast instead of consuming the
# test-wide budget.
COMMAND_TIMEOUT_S = 20

# The CUDA major every signal must report. Bump this when the images migrate to
# the next major; the scan patterns and the assertion both derive from it.
EXPECTED_MAJOR = 13

# Majors the scanner recognizes. Keep majors we have already left behind, so a
# stray package is reported rather than silently unmatched, and add the next one
# here when a migration starts (nothing else needs editing).
RECOGNIZED_MAJORS = (12, 13)

# Optional floor as (major, minor), applied to every signal that carries a minor
# version. Set to (13, 1) to require CUDA >= 13.1 image-wide. None checks the
# major only.
MIN_VERSION: tuple[int, int] | None = None

# pip distributions permitted to report a major other than EXPECTED_MAJOR, keyed
# by exact distribution name and carrying the reason it cannot be removed.
#
# Exact names, never prefixes: the prefix list this replaced ignored "cupy" and
# "nixl" wholesale, which hid a cupy-cuda12x pin in pyproject.toml and every
# nixl-cu12 wheel. An entry here means "upstream forces this on us", so it needs
# a reason; it does not mean "this package is uninteresting".
PIP_MAJOR_EXCEPTIONS = {
    # nvidia-cutlass-dsl requires nvidia-cutlass-dsl-libs-cu12 unconditionally
    # and puts only the cu13 libs behind an extra, so the cu12 wheel always
    # lands alongside it.
    "nvidia-cutlass-dsl-libs-cu12": "unconditional dependency of nvidia-cutlass-dsl",
    # The nixl meta package requires both nixl-cu12 and nixl-cu13
    # unconditionally; nixl[cu13] narrows nothing.
    "nixl-cu12": "unconditional dependency of the nixl meta package",
}

# Cap how much of a long signal (dpkg listings) the failure report prints. Only
# the report is truncated; every line is still scanned.
MAX_REPORTED_LINES = 50

_MAJORS = "|".join(str(m) for m in RECOGNIZED_MAJORS)

# Group 1 is the CUDA major; group 2, where the signal carries one, is the minor.
# fmt: off
PATTERNS = [
    rf"\bCUDA_VERSION=({_MAJORS})\.(\d+)",             # CUDA_VERSION=13.0.2
    rf"\bNV_CUDA_.*?_VERSION=({_MAJORS})\.(\d+)",      # NV_CUDA_CUDART_VERSION=13.0.96-1
    rf"\+cuda({_MAJORS})\.(\d+)",                      # ...+cuda13.0
    rf"\bcuda\s*>=\s*({_MAJORS})\.(\d+)",              # cuda>=13.0 ...
    rf"\brelease\s+({_MAJORS})\.(\d+)",                # nvcc: release 13.0
    rf"-({_MAJORS})-(\d)\b",                           # dpkg: ...-13-0
    rf"\bcuda({_MAJORS})x\b",                          # cupy-cuda13x (from name)
    rf"[-+]cu({_MAJORS})(\d)?",                        # -cu13, +cu130
    rf"\bcuda({_MAJORS})(\d)?\b",                      # ...-cuda13, nvidia-dali-cuda130
    rf"^(?:nvidia-)?cuda[\w-]*==({_MAJORS})\.(\d+)",   # cuda-toolkit==13.0.2
]
# fmt: on

# Targeted pip listing: anything whose name or version could carry a CUDA major.
PIP_LIST_CMD = (
    "python -m pip list --format=freeze | grep -Ei "
    rf"'(cuda|cudnn|nccl|nvshmem|\+cu({_MAJORS})[0-9]{{1,2}}|-cu({_MAJORS})|"
    r"^(torch|torchaudio|torchvision)==)'"
)


def sh(cmd: str) -> str:
    """
    Run command and return stdout only.
    We intentionally drop stderr to avoid noisy tools (pip warnings, etc.).
    A timeout is deliberately left to propagate: a signal we cannot read is a
    failure, not a signal that reports nothing.
    """
    p = subprocess.run(
        ["bash", "-lc", f"{cmd} 2>/dev/null"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
        timeout=COMMAND_TIMEOUT_S,
    )
    return (p.stdout or "").strip()


def cuda_version_from_text(text: str) -> tuple[int, int | None] | None:
    """
    Extract a CUDA (major, minor) from a single line of text. minor is None when
    the signal carries only a major (e.g. a '-cu13' wheel tag). Returns None when
    no recognized CUDA version is present.

    Call this per line, never on a multi-line blob: it returns the first match,
    so scanning combined output would let a stray major hide behind an expected
    one earlier in the same output.
    """
    if not text:
        return None

    for pat in PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m is None:
            continue
        minor = None
        if m.re.groups >= 2 and m.group(2) is not None:
            minor = int(m.group(2))
        return int(m.group(1)), minor
    return None


def pip_distribution_name(line: str) -> str:
    """Distribution name from a pip freeze line, e.g. 'nixl-cu12==1.3.1'."""
    return line.split("==", 1)[0].strip().lower()


def format_version(version: tuple[int, int | None] | None) -> str:
    """Render a parsed version for the report: '13.0', '13', or '-'."""
    if version is None:
        return "-"
    major, minor = version
    return f"{major}.{minor}" if minor is not None else str(major)


def test_cuda_version_consistency() -> None:
    """
    Collect CUDA versions from predefined signals and assert every one of them
    reports EXPECTED_MAJOR. Prints a readable report with full relevant output
    when failing.
    """

    signals = [
        ("env:CUDA_VERSION", "env | grep -i '^CUDA_VERSION='"),
        ("env:NV_CUDA_CUDART_VERSION", "env | grep -i '^NV_CUDA_CUDART_VERSION='"),
        ("env:NV_CUDA_LIB_VERSION", "env | grep -i '^NV_CUDA_LIB_VERSION='"),
        ("env:NV_LIBNCCL_PACKAGE", "env | grep -i '^NV_LIBNCCL_PACKAGE='"),
        ("env:NVIDIA_REQUIRE_CUDA", "env | grep -i '^NVIDIA_REQUIRE_CUDA='"),
        ("nvcc", "nvcc --version | grep -i 'release' || nvcc --version"),
        ("dpkg:cuda-*", rf"dpkg -l | grep -E '^(ii|hi)\s+cuda-.*-({_MAJORS})-'"),
        (
            "dpkg:libcublas/libnccl",
            rf"dpkg -l | grep -E '^(ii|hi)\s+lib(cublas|nccl).*-({_MAJORS})-'",
        ),
        ("pip:selected", PIP_LIST_CMD),
    ]

    # (signal label, the line to name in a failure, parsed version)
    detected: list[tuple[str, str, tuple[int, int | None]]] = []
    excused: list[tuple[str, tuple[int, int | None], str]] = []
    report: list[str] = [f"CUDA version signals (expected major {EXPECTED_MAJOR}):"]

    for label, cmd in signals:
        lines = [ln.strip() for ln in sh(cmd).splitlines() if ln.strip()]
        report.append(f"  {label}")

        if not lines:
            report.append("        <no output>")
            continue

        for index, line in enumerate(lines):
            version = cuda_version_from_text(line)
            reason = (
                PIP_MAJOR_EXCEPTIONS.get(pip_distribution_name(line))
                if label.startswith("pip:")
                else None
            )

            note = ""
            if version is not None and reason is not None:
                excused.append((line, version, reason))
                note = f"   (allowed: {reason})"
            elif version is not None:
                detected.append((label, line, version))

            if index < MAX_REPORTED_LINES:
                report.append(f"        {format_version(version):>5}  {line}{note}")

        if len(lines) > MAX_REPORTED_LINES:
            report.append(f"        ... ({len(lines) - MAX_REPORTED_LINES} more lines)")

    # Excused packages are tolerated noise, not evidence that the scan worked:
    # an image whose only CUDA signal is an exception entry has told us nothing
    # about the major it ships, so treat it as unevaluable rather than green.
    if not detected:
        pytest.skip("No non-excused CUDA version detected from any signal.")

    violations: list[str] = []
    for label, line, (major, minor) in detected:
        if major != EXPECTED_MAJOR:
            violations.append(
                f"  {label}: {line} reports CUDA {format_version((major, minor))}, "
                f"expected major {EXPECTED_MAJOR}"
            )
        elif (
            MIN_VERSION is not None
            and minor is not None
            and (major, minor) < MIN_VERSION
        ):
            violations.append(
                f"  {label}: {line} reports CUDA {format_version((major, minor))}, "
                f"below the required minimum {MIN_VERSION[0]}.{MIN_VERSION[1]}"
            )

    assert not violations, "\n".join(
        [*report, "", "Unexpected CUDA versions:", *violations]
    )
