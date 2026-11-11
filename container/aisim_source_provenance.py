# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the final staged AISim wheel bytes to installed Git source provenance."""

import hashlib
import importlib.metadata
import json
import re
import sys
import zipfile
from email import message_from_bytes
from pathlib import Path

from packaging.requirements import Requirement


def write_provenance(requirements: Path, wheelhouse: Path) -> None:
    lines = [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError("expected one AISim source requirement")
    pin = Requirement(lines[0])
    match = re.fullmatch(
        r"git\+https://github\.com/ai-dynamo/aisimulate\.git@([0-9a-f]{40})#subdirectory=python/aisim",
        pin.url or "",
    )
    if pin.name != "aisim" or match is None:
        raise ValueError("expected the reviewed immutable AISim source pin")
    distribution = importlib.metadata.distribution("aisim")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (
        origin.get("vcs_info", {}).get("commit_id") != match[1]
        or origin.get("subdirectory") != "python/aisim"
    ):
        raise ValueError(
            "installed AISim source differs from the staged source requirement"
        )
    wheels = list(wheelhouse.glob("aisim-*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected one staged AISim wheel")
    wheel = wheels[0]
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("expected one staged AISim wheel metadata record")
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
        if (
            metadata.get("Name") != "aisim"
            or metadata.get("Version") != distribution.version
        ):
            raise ValueError("staged AISim metadata differs from installed Git source")
        for name in archive.namelist():
            if name.endswith("/") or not name.startswith(
                (
                    "aisim/",
                    "aisim_core/",
                    "aisimulate/",
                    "aisimulate_core/",
                    "collector/",
                )
            ):
                continue
            checked += 1
            if archive.read(name) != Path(distribution.locate_file(name)).read_bytes():
                raise ValueError(
                    f"staged AISim payload differs from installed Git source: {name}"
                )
    if not checked:
        raise ValueError("staged AISim wheel contains no runtime payload")
    with wheel.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "source": origin,
        "wheel": wheel.name,
        "sha256": checksum,
        "version": distribution.version,
    }
    (wheelhouse / "aisim-override.txt").write_text(
        f"aisim @ file:///opt/dynamo/wheelhouse/{wheel.name}\n"
    )
    (wheelhouse / "aisim-source.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    write_provenance(Path(sys.argv[1]), Path(sys.argv[2]))
