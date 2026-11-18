# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the supplied specification separate from derived comparison documents.

load_input returns a parsed document alongside its original bytes and SHA-256.
Snapshot.save writes those bytes and acquisition.json so findings can be traced
to an exact input. Metadata records local read time and caller labels, not schema
generation time or verified server identity. Loading checks the version prefix;
assessment modules validate structure and comparison support separately.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


def write_json(path: Path, value) -> None:
    """Overwrite UTF-8 JSON with sorted keys; parents must exist.

    Serialization and I/O errors propagate; a failed write may truncate the file.
    """
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def server_name(value: str) -> str:
    """Return a 1–64 character slug matching [a-z][a-z0-9_-]*, or raise ValueError."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Server name must be a lowercase slug of 1 to 64 characters")
    return value


def parse_document(raw: bytes) -> dict:
    """Parse JSON/YAML bytes with an OpenAPI 3.1/3.2 version prefix.

    Reject other prefixes or non-object documents with ValueError; parsing errors
    propagate. This does not validate the full document or resolve references.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = yaml.safe_load(raw)
    if not isinstance(document, dict) or not str(
        document.get("openapi", "")
    ).startswith(("3.1.", "3.2.")):
        raise ValueError("Expected an OpenAPI 3.1 or 3.2 JSON or YAML document")
    return document


@dataclass
class Snapshot:
    """Parsed local schema, exact bytes, and input metadata; metadata is not attestation."""

    document: dict
    metadata: dict
    raw: bytes

    def save(self, directory: Path) -> None:
        """Create directory with openapi.raw and acquisition.json; parents may be created.

        An existing directory raises FileExistsError. I/O or serialization failures
        propagate and can leave partial output; the snapshot is unchanged.
        """
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "openapi.raw").write_bytes(self.raw)
        write_json(directory / "acquisition.json", self.metadata)


def load_input(
    source: Path, server: str, metadata_path: Path | None = None
) -> Snapshot:
    """Read a local OpenAPI JSON/YAML file and optional unverified metadata JSON object.

    Return parsed content, exact bytes, checksum and read time. Metadata labels do
    not filter declarations. Invalid versions, identities or metadata raise ValueError;
    parsing and I/O errors propagate. No network access or input mutation occurs.
    """
    metadata = json.loads(metadata_path.read_text()) if metadata_path else None
    if metadata_path is not None and not isinstance(metadata, dict):
        raise ValueError("Caller metadata must be a JSON object")
    raw = source.read_bytes()
    return Snapshot(
        parse_document(raw),
        {
            "schema": "protocol-schema-acquisition/v2",
            "server": server_name(server),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": {"kind": "file", "path": str(source.resolve())},
            "sha256": hashlib.sha256(raw).hexdigest(),
            "caller_metadata": metadata,
            "caller_metadata_verified": False,
        },
        raw,
    )
