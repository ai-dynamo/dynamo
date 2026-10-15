# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load local OpenAPI files and retain exact bytes and unverified input metadata."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


def write_json(path: Path, value) -> None:
    """Overwrite a UTF-8 JSON file with sorted keys; parents must exist and I/O errors propagate."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def server_name(value: str) -> str:
    """Validate a caller-selected identity, not an attested server identity."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Server name must be a lowercase slug of 1 to 64 characters")
    return value


def parse_document(raw: bytes) -> dict:
    """Parse retained JSON/YAML bytes; reject non-3.1/3.2 documents, without resolving refs."""
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
        """Create a new directory and save bytes/metadata; failures may leave partial output."""
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
