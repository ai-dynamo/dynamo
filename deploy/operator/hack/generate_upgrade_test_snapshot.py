# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the frozen pre-v1.6.0 test snapshot; --check rejects stale output.

The archive is generated, not hand edited. Its source is the inventoryRevision
in capture-v1.6.0.json. Only this maintenance command needs that Git history;
the lifecycle tests consume the checked-in archive in source-only images.
"""

import argparse
import gzip
import io
import json
import subprocess
from pathlib import Path

import generate_upgrade_fixtures as fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = json.loads((fixtures.FIXTURES / "capture-v1.6.0.json").read_text())
    archive = subprocess.check_output(
        [
            "git",
            "-c",
            "tar.umask=0022",
            "archive",
            "--format=tar",
            inventory["inventoryRevision"],
            fixtures.FIXTURES.relative_to(fixtures.ROOT).as_posix(),
        ],
        cwd=fixtures.ROOT,
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        compressed.write(archive)
    output = Path(__file__).parent / "testdata/upgrade-pre-v1.6.0.tar.gz"
    if args.check:
        if output.read_bytes() != buffer.getvalue():
            raise SystemExit(f"Stale generated snapshot: run {Path(__file__).name}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(buffer.getvalue())


if __name__ == "__main__":
    main()
