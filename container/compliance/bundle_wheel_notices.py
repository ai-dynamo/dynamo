# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundle a human-readable third-party Rust NOTICES into a maturin wheel.

A maturin wheel statically links a graph of Rust crates into its compiled
extension, so redistributing the wheel triggers those crates' attribution
clauses (MIT/BSD/Apache require reproducing the copyright + license text). The
wheel already ships the machine-readable CycloneDX SBOM maturin embeds at
``<dist-info>/sboms/*.cyclonedx.json`` — but that records license *identifiers*,
not texts. Best practice is to ship BOTH: the SBOM and a human-readable
third-party license file.

This reads the embedded SBOM (reusing the rust NOTICES generator + the
cargo-registry license harvest), renders the same NOTICES-Rust text that the
container licenses stage produces, and injects it into the wheel at PEP 639's
``<dist-info>/licenses/THIRD-PARTY-RUST-LICENSES.txt`` (updating RECORD so the
wheel stays valid). No-op for wheels with no cargo SBOM (e.g. pure-Python).

Usage:
    python3 -m compliance.bundle_wheel_notices \\
        --wheel dist/ai_dynamo_runtime-*.whl \\
        --licenses-dir /opt/dynamo/rust-licenses
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from .generators import common, rust

logger = logging.getLogger(__name__)

_NOTICES_ARCNAME = "licenses/THIRD-PARTY-RUST-LICENSES.txt"


def _record_hash(data: bytes) -> str:
    """RECORD's `sha256=<urlsafe-b64-nopad>` digest form (PEP 427)."""
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _dist_info_dir(names: list[str]) -> str | None:
    for n in names:
        top = n.split("/", 1)[0]
        if top.endswith(".dist-info"):
            return top
    return None


def _render_wheel_rust_notices(wheel: Path, licenses_dir: Path | None) -> str | None:
    """Extract the wheel's embedded SBOM(s) and render NOTICES-Rust text.

    Returns None when the wheel carries no cargo components (nothing to bundle).
    """
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(wheel) as z:
            sbom_members = [
                n
                for n in z.namelist()
                if "/sboms/" in n and n.endswith(".cyclonedx.json")
            ]
            if not sbom_members:
                logger.info("%s: no embedded SBOM; nothing to bundle", wheel.name)
                return None
            for n in sbom_members:
                z.extract(n, td)
        # collect_components globs "*.dist-info/sboms/*.cyclonedx.json" under the
        # search path, so the extracted layout is exactly what it expects.
        components = rust.collect_components([Path(td)], licenses_dir)
    if not components:
        logger.info("%s: SBOM carried no cargo crates; nothing to bundle", wheel.name)
        return None
    return common.render_notices("rust", components)


def inject(wheel: Path, arcname: str, content: bytes) -> None:
    """Add `arcname` to the wheel and append its RECORD entry, atomically."""
    with zipfile.ZipFile(wheel) as zin:
        names = zin.namelist()
        di = _dist_info_dir(names)
        if di is None:
            raise ValueError(f"{wheel}: no .dist-info dir found")
        record_name = f"{di}/RECORD"
        if record_name not in names:
            raise ValueError(f"{wheel}: no {record_name}")
        full_arcname = f"{di}/{arcname}"
        record_lines = [
            ln
            for ln in zin.read(record_name).decode("utf-8").splitlines()
            if ln.strip()
        ]
        members = {n: zin.read(n) for n in names}

    # RECORD: keep every original line (hashes unchanged — those files are
    # untouched), add a line for the new file, keep the RECORD self-line last.
    # Idempotent: drop any prior RECORD entry for the file we're (re)adding so a
    # re-run replaces it instead of appending a duplicate line.
    self_line = f"{record_name},,"
    body = [
        ln
        for ln in record_lines
        if ln != self_line and not ln.startswith(f"{full_arcname},")
    ]
    body.append(f"{full_arcname},{_record_hash(content)},{len(content)}")
    new_record = ("\n".join(body) + "\n" + self_line + "\n").encode("utf-8")

    tmp = wheel.with_suffix(wheel.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        # Skip the prior copy of the injected file too, so re-running doesn't
        # leave two zip members with the same name.
        for n, data in members.items():
            if n in (record_name, full_arcname):
                continue
            zout.writestr(n, data)
        zout.writestr(full_arcname, content)
        zout.writestr(record_name, new_record)
    os.replace(tmp, wheel)
    logger.info("Bundled %s into %s", full_arcname, wheel.name)


def process(wheel: Path, licenses_dir: Path | None) -> int:
    notices = _render_wheel_rust_notices(wheel, licenses_dir)
    if notices is None:
        return 0
    inject(wheel, _NOTICES_ARCNAME, notices.encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bundle_wheel_notices",
        description="Inject a third-party Rust NOTICES file into maturin wheels.",
    )
    parser.add_argument(
        "--wheel",
        dest="wheels",
        type=Path,
        action="append",
        required=True,
        help="Wheel to process (repeatable).",
    )
    parser.add_argument(
        "--licenses-dir",
        type=Path,
        default=None,
        help=(
            "Harvested crate-LICENSE dir (keyed <name>-<version>); when present "
            "the real upstream texts are used, else canonical SPDX text."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    rc = 0
    for wheel in args.wheels:
        if not wheel.is_file():
            logger.error("wheel not found: %s", wheel)
            rc = 1
            continue
        rc = process(wheel, args.licenses_dir) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
