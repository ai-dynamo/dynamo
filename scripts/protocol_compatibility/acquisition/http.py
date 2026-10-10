# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain exact OpenAPI bytes from files or HTTP(S), without deployment inspection."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

import yaml

MAX_BYTES = 32 * 1024 * 1024
SCHEMA = "protocol-schema-acquisition/v2"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, stream, code, message, headers, url):
        return None


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def server_name(value: str) -> str:
    """Validate a caller-selected identity, not an attested server identity."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Server name must be a lowercase slug of 1 to 64 characters")
    return value


def read_limited(stream) -> bytes:
    raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("OpenAPI document exceeds the 32 MiB capture limit")
    return raw


def parse_document(raw: bytes) -> dict:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = yaml.safe_load(raw)
    if not isinstance(document, dict) or not str(
        document.get("openapi", "")
    ).startswith("3.1."):
        raise ValueError("Expected an OpenAPI 3.1 JSON or YAML document")
    return document


@dataclass
class Snapshot:
    document: dict
    metadata: dict
    raw: bytes
    filename: str = "openapi.raw"

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / self.filename).write_bytes(self.raw)
        write_json(directory / "acquisition.json", self.metadata)


def snapshot(raw: bytes, server: str, source: dict, caller_metadata=None) -> Snapshot:
    return Snapshot(
        parse_document(raw),
        {
            "schema": SCHEMA,
            "server": server_name(server),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "caller_metadata": caller_metadata,
            "caller_metadata_verified": False,
        },
        raw,
    )


def acquire(args) -> int:
    server_name(args.server)
    caller_metadata = (
        json.loads(args.metadata.read_text()) if args.metadata is not None else None
    )
    if args.metadata is not None and not isinstance(caller_metadata, dict):
        raise ValueError("Caller metadata must be a JSON object")
    if args.url is not None:
        parsed = urlsplit(args.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Use an HTTP or HTTPS OpenAPI URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("URL must not contain credentials, query or fragment")
        with build_opener(ProxyHandler({}), NoRedirect).open(
            args.url, timeout=30
        ) as response:
            if response.url != args.url or response.status != 200:
                raise ValueError("OpenAPI URL must return HTTP 200 without redirects")
            raw = read_limited(response)
            source = {"kind": "url", "url": args.url, "http_status": response.status}
    else:
        with args.file.open("rb") as stream:
            raw = read_limited(stream)
        source = {"kind": "file", "path": str(args.file.resolve())}
    snapshot(raw, args.server, source, caller_metadata).save(args.output_dir)
    return 0


def load_capture(source: Path, server: str) -> Snapshot:
    """Load a bare JSON/YAML file or a checksummed v1/v2 acquisition directory.

    Capture annotations (including legacy deployment provenance) are retained,
    not verified. The raw bytes, not a later reread, are saved with the report.
    """
    server_name(server)
    if not source.is_dir():
        with source.open("rb") as stream:
            raw = read_limited(stream)
        return snapshot(raw, server, {"kind": "file", "path": str(source.resolve())})
    metadata = json.loads((source / "acquisition.json").read_text())
    filenames = {
        "protocol-http-acquisition/v1": "openapi.raw.json",
        SCHEMA: "openapi.raw",
    }
    if metadata["schema"] not in filenames or metadata["server"] != server:
        raise ValueError(f"Wrong acquisition kind for {server}")
    filename = filenames[metadata["schema"]]
    with (source / filename).open("rb") as stream:
        raw = read_limited(stream)
    if metadata["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"Raw {server} export changed after acquisition")
    return Snapshot(parse_document(raw), metadata, raw, filename)
