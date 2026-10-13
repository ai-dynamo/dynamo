# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install guard for PyNvVideoCodec, run beside the install in the runtime images.

    PYTHONPATH=/tmp/compliance python3 -m compliance.check_pynvvideocodec --pinned 2.2.3

The pin is passed in, not read from the requirements file, and a test asserts the
two agree. What counts as denied comes from policy/codec_policy.yaml.
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
# An empty package directory passes the negative checks, so these must be present.
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
    # All of sys.path, so a surviving base-image copy is counted too.
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
    # Walked, not globbed: ``**`` skips hidden dirs such as auditwheel's ``.libs/``.
    bundled = sorted(
        os.path.relpath(os.path.join(d, f), pkg)
        for d, _, files in os.walk(pkg)
        for f in fnmatch.filter(files, "lib*.so*")
    )
    print(f"{PACKAGE} bundles:", bundled)
    for required in REQUIRED:
        if not any(os.path.basename(n).startswith(required) for n in bundled):
            raise GuardError(
                f"{PACKAGE} bundles no {required}, so the checks below would "
                f"pass vacuously; found {bundled}"
            )
    # Match each denied family by prefix, as the old heredocs did, so a
    # build-suffixed name such as libavcodec_x.so.63 is caught too. The policy
    # waivers still apply.
    families = tuple(
        m.group(1)
        for g in policy.deny_globs
        if (m := re.fullmatch(r"\*\*/(lib[\w-]+?)(?:-\*)?\.so\*", g))
    )
    denied = [
        n
        for n in bundled
        if policy.violates(Path(pkg, n).as_posix())
        or (
            os.path.basename(n).startswith(families)
            and policy.classify(Path(pkg, n).as_posix())[0] == "violation"
        )
    ]
    if denied:
        raise GuardError(f"{PACKAGE} bundles libraries the codec gate denies: {denied}")

    # The tarball lands outside site-packages; its directory comes from the RECORD.
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
