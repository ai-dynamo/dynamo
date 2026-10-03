# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install guard for PyNvVideoCodec, run beside the install in the runtime images.

A requirements specifier constrains what pip installs. It cannot say "exactly one
copy on disk" or "no libavcodec", and those are what the codec gate depends on.
Checking here names the cause of a regression instead of leaving it to surface as
an unattributed scan violation later.

Run from a directory holding the ``compliance`` package:

    PYTHONPATH=/tmp/compliance python3 -m compliance.check_pynvvideocodec --pinned 2.2.3

``--pinned`` is passed by each template rather than read from the requirements
file: this stage must not parse the file it is checking, and
tests/dependencies/test_pynvvideocodec_spec.py asserts the two agree.

What counts as denied comes from policy/codec_policy.yaml, the same verdict the
image-wide scan gives, so a library family added to the policy reaches this guard
too. The PyNvVideoCodec waivers there for libavutil and libavformat are what let
those two through.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import os
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

from .scan_codecs import CodecPolicy

DISTRIBUTION = "pynvvideocodec"
PACKAGE = "PyNvVideoCodec"
# The wheel demuxes with these two. A package directory that bundles neither
# satisfies every negative check while shipping no demuxer at all.
REQUIRED = ("libavformat", "libavutil")
DEFAULT_POLICY = Path(__file__).resolve().parent / "policy" / "codec_policy.yaml"


class GuardError(Exception):
    """A check failed; the message is what the build log shows."""


def _canonical(name: str | None) -> str:
    return re.sub(r"[-_.]+", "-", name or "").lower()


def check(pinned: str, policy: CodecPolicy, path: list[str] | None = None) -> None:
    """Raise GuardError unless exactly one pinned, policy-clean install is found.

    ``path`` replaces sys.path as the place distributions are looked up.
    """
    # Enumerated over sys.path rather than one scheme directory: these images carry
    # both /usr/local/lib/python3.12/dist-packages and /usr/lib/python3/dist-packages,
    # and the wheel declares Root-Is-Purelib: false, so neither purelib nor platlib
    # alone is guaranteed to be the install target or the only place a copy can hide.
    # A surviving base copy beside the new one is the failure being looked for, which
    # is also why this counts distributions instead of asking for one version.
    installed = [
        d
        for d in distributions(**({} if path is None else {"path": path}))
        if _canonical(d.metadata["Name"]) == DISTRIBUTION
    ]
    versions = sorted(d.version for d in installed)
    print(f"{PACKAGE} distributions on sys.path:", versions)
    if len(installed) != 1:
        raise GuardError(f"expected exactly one {PACKAGE}, found {versions}")
    if versions[0] != pinned:
        raise GuardError(
            f"{PACKAGE} is {versions[0]}, but the requirements file pins {pinned}"
        )

    site = os.path.normpath(str(installed[0].locate_file("")))
    pkg = os.path.join(site, PACKAGE)
    # Walked rather than globbed: ``**`` skips hidden directories, and auditwheel
    # grafts libraries under ``.libs/``.
    bundled = sorted(
        os.path.relpath(os.path.join(d, f), pkg)
        for d, _, files in os.walk(pkg)
        for f in fnmatch.filter(files, "lib*.so*")
    )
    print(f"{PACKAGE} bundles:", bundled)
    # Positive first: an empty package directory passes the policy check vacuously.
    for required in REQUIRED:
        if not any(os.path.basename(n).startswith(required) for n in bundled):
            raise GuardError(
                f"{PACKAGE} bundles no {required}, so the checks below would "
                f"pass vacuously; found {bundled}"
            )
    denied = [n for n in bundled if policy.violates(Path(pkg, n).as_posix())]
    if denied:
        raise GuardError(f"{PACKAGE} bundles libraries the codec gate denies: {denied}")

    # The FFmpeg source tarball lands outside site-packages, so its directory is read
    # from the wheel's own RECORD rather than guessed from a sysconfig path -- the
    # RECORD is what the installer actually wrote, and it moves if the layout does.
    record = installed[0].read_text("RECORD") or ""
    declared = [
        row[0]
        for row in csv.reader(record.splitlines())
        if row and row[0].endswith((".tar.xz", ".tar.gz", ".tar.bz2"))
    ]
    if len(declared) != 1:
        raise GuardError(f"expected one source tarball in the RECORD, found {declared}")
    external = os.path.dirname(os.path.normpath(os.path.join(site, declared[0])))
    tarballs = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(external, "ffmpeg-*.tar.*"))
    )
    print("bundled FFmpeg source tarballs in", external, "->", tarballs)
    if len(tarballs) != 1:
        raise GuardError(
            f"expected exactly one bundled FFmpeg source tarball, found {tarballs}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pinned", required=True, help="the version that must be installed"
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)
    try:
        check(args.pinned, CodecPolicy.load(args.policy))
    except GuardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
