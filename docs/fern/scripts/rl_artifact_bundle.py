#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build or verify a digest-closed local RL publication artifact bundle.

The bundle closes every artifact://bundle/ URI found in the framework,
program, and clean-room records against a regular file below one artifact
root. Publication mode also runs all three record publication checkers and
verifies the clean-room record's linked record URIs and digests. This proves
local existence and immutability, not the semantic truth of an artifact or
the identity and independence of its human reviewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import check_rl_clean_room_record
import check_rl_program_record
import check_rl_validation_record

SCHEMA = "dynamo.rl.artifact-bundle.v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
BUNDLE_SCHEME = "artifact"
BUNDLE_AUTHORITY = "bundle"
RECORD_SCHEMAS = {
    "framework_validation": "dynamo.rl.validation.v1",
    "program_evidence": "dynamo.rl.program-evidence.v1",
    "clean_room_review": "dynamo.rl.clean-room-review.v1",
}
PUBLICATION_CHECKERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    RECORD_SCHEMAS["framework_validation"]: (
        check_rl_validation_record.publication_findings
    ),
    RECORD_SCHEMAS["program_evidence"]: check_rl_program_record.publication_findings,
    RECORD_SCHEMAS["clean_room_review"]: (
        check_rl_clean_room_record.publication_findings
    ),
}


class BundleError(ValueError):
    """Raised when a bundle path, URI, index, or record is unsafe or invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} {path} must contain a JSON object")
    return value


def _artifact_root(path: Path) -> Path:
    if path.is_symlink():
        raise BundleError(f"artifact root must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"artifact root does not exist: {path}") from exc
    if not resolved.is_dir():
        raise BundleError(f"artifact root must be a directory: {path}")
    return resolved


def _relative_input_path(path: Path) -> None:
    if ".." in path.parts:
        raise BundleError(f"bundle member path must not contain '..': {path}")


def _lexical_member(root: Path, candidate: Path, label: str) -> Path:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise BundleError(f"{label} escapes artifact root: {candidate}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BundleError(f"{label} must not traverse a symbolic link: {candidate}")
    return relative


def _member(root: Path, path: Path, label: str) -> tuple[Path, str]:
    _relative_input_path(path)
    candidate = path if path.is_absolute() else root / path
    lexical_relative = _lexical_member(root, candidate, label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"{label} does not exist: {path}") from exc
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise BundleError(f"{label} escapes artifact root: {path}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BundleError(f"{label} must not traverse a symbolic link: {path}")
    if not resolved.is_file():
        raise BundleError(f"{label} must be a regular file: {path}")
    if relative != lexical_relative:
        raise BundleError(f"{label} path changed during resolution: {path}")
    return resolved, relative.as_posix()


def _output_member(root: Path, path: Path, label: str) -> Path:
    _relative_input_path(path)
    candidate = path if path.is_absolute() else root / path
    _lexical_member(root, candidate, label)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
        relative_parent = resolved_parent.relative_to(root)
    except OSError as exc:
        raise BundleError(f"{label} parent does not exist: {path}") from exc
    except ValueError as exc:
        raise BundleError(f"{label} escapes artifact root: {path}") from exc
    cursor = root
    for part in relative_parent.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BundleError(f"{label} must not traverse a symbolic link: {path}")
    if candidate.exists() and candidate.is_symlink():
        raise BundleError(f"{label} must not be a symbolic link: {path}")
    return resolved_parent / candidate.name


def _uri_for_path(relative_path: str) -> str:
    encoded = quote(relative_path, safe="/._-")
    return f"{BUNDLE_SCHEME}://{BUNDLE_AUTHORITY}/{encoded}"


def _path_for_uri(uri: str) -> str | None:
    parts = urlsplit(uri)
    if parts.scheme != BUNDLE_SCHEME:
        return None
    if parts.netloc != BUNDLE_AUTHORITY:
        return None
    if parts.query or parts.fragment:
        raise BundleError(f"bundle URI must not contain a query or fragment: {uri}")
    decoded = unquote(parts.path)
    if not decoded.startswith("/") or decoded.startswith("//"):
        raise BundleError(f"bundle URI must contain one absolute URI path: {uri}")
    relative = PurePosixPath(decoded[1:])
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise BundleError(f"bundle URI contains an unsafe path: {uri}")
    canonical = relative.as_posix()
    if uri != _uri_for_path(canonical):
        raise BundleError(f"bundle URI is not canonical: {uri}")
    return canonical


def _artifact_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"artifact", "checker_output_artifact", "uri"}:
                if isinstance(child, str) and child:
                    references.add(child)
            elif key == "artifacts":
                if isinstance(child, list):
                    references.update(
                        item for item in child if isinstance(item, str) and item
                    )
            else:
                references.update(_artifact_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_artifact_references(child))
    return references


def _record_entry(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    resolved, relative = _member(root, path, "record")
    record = _load_json(resolved, "record")
    schema = record.get("schema")
    if not isinstance(schema, str) or not schema:
        raise BundleError(f"record {relative} must have a non-empty schema")
    checker = PUBLICATION_CHECKERS.get(schema)
    publication_findings = (
        checker(record) if checker is not None else [f"unsupported record schema {schema}"]
    )
    entry = {
        "schema": schema,
        "uri": _uri_for_path(relative),
        "path": relative,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "publication_gate_passed": not publication_findings,
        "publication_finding_count": len(publication_findings),
    }
    return entry, record, publication_findings


def _artifact_entry(root: Path, uri: str) -> dict[str, Any]:
    relative = _path_for_uri(uri)
    if relative is None:
        raise BundleError(f"not a local bundle URI: {uri}")
    resolved, actual_relative = _member(root, Path(relative), f"artifact {uri}")
    if actual_relative != relative:
        raise BundleError(
            f"artifact URI path is not canonical after resolution: {uri} -> {actual_relative}"
        )
    return {
        "uri": uri,
        "path": relative,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _record_roles(entries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    findings: list[str] = []
    by_schema: dict[str, dict[str, Any]] = {}
    expected = set(RECORD_SCHEMAS.values())
    for entry in entries:
        schema = entry["schema"]
        if schema in by_schema:
            findings.append(f"record schema appears more than once: {schema}")
        by_schema[schema] = entry
        if schema not in expected:
            findings.append(f"unsupported record schema: {schema}")
    missing = sorted(expected - set(by_schema))
    if missing:
        findings.append("missing required record schemas: " + ", ".join(missing))
    return by_schema, findings


def _clean_room_links(
    entries_by_schema: dict[str, dict[str, Any]],
    records_by_schema: dict[str, dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    clean_schema = RECORD_SCHEMAS["clean_room_review"]
    clean_room = records_by_schema.get(clean_schema)
    if clean_room is None:
        return findings
    linked = clean_room.get("linked_records")
    if not isinstance(linked, dict):
        return ["clean-room record linked_records must be an object"]
    for role in ("framework_validation", "program_evidence"):
        schema = RECORD_SCHEMAS[role]
        expected = entries_by_schema.get(schema)
        actual = linked.get(role)
        if expected is None or not isinstance(actual, dict):
            findings.append(f"clean-room linked_records.{role} is unavailable")
            continue
        if actual.get("uri") != expected["uri"]:
            findings.append(
                f"clean-room linked_records.{role}.uri must equal {expected['uri']}"
            )
        if actual.get("sha256") != expected["sha256"]:
            findings.append(
                f"clean-room linked_records.{role}.sha256 does not match the bundled record"
            )
    return findings


def build_index(root_path: Path, record_paths: Iterable[Path]) -> tuple[dict[str, Any], list[str]]:
    root = _artifact_root(root_path)
    paths = list(record_paths)
    if not paths:
        raise BundleError("at least one record path is required")

    record_entries: list[dict[str, Any]] = []
    records_by_schema: dict[str, dict[str, Any]] = {}
    publication_findings_by_path: dict[str, list[str]] = {}
    referenced_uris: set[str] = set()
    for path in paths:
        entry, record, publication_findings = _record_entry(root, path)
        record_entries.append(entry)
        if entry["schema"] not in records_by_schema:
            records_by_schema[entry["schema"]] = record
        publication_findings_by_path[entry["path"]] = publication_findings
        referenced_uris.update(_artifact_references(record))

    record_entries.sort(key=lambda item: item["path"])
    entries_by_schema, findings = _record_roles(record_entries)
    findings.extend(_clean_room_links(entries_by_schema, records_by_schema))
    for entry in record_entries:
        for finding in publication_findings_by_path[entry["path"]]:
            findings.append(f"{entry['path']} publication gate: {finding}")

    artifacts: list[dict[str, Any]] = []
    missing_uris: list[str] = []
    external_uris: list[str] = []
    for uri in sorted(referenced_uris):
        relative = _path_for_uri(uri)
        if relative is None:
            external_uris.append(uri)
            continue
        try:
            artifacts.append(_artifact_entry(root, uri))
        except BundleError:
            missing_uris.append(uri)

    if not referenced_uris:
        findings.append("records do not reference any artifact URIs")
    if missing_uris:
        findings.append("missing local artifacts: " + ", ".join(missing_uris))
    if external_uris:
        findings.append(
            "external artifact URIs are not closed by this bundle: "
            + ", ".join(external_uris)
        )
    artifacts.sort(key=lambda item: item["uri"])
    index = {
        "schema": SCHEMA,
        "closure_status": "complete" if not findings else "incomplete",
        "artifact_root": ".",
        "records": record_entries,
        "artifacts": artifacts,
        "referenced_uri_count": len(referenced_uris),
        "missing_uris": missing_uris,
        "external_uris": external_uris,
        "finding_count": len(findings),
    }
    return index, findings


def _validate_index_shape(index: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if index.get("schema") != SCHEMA:
        findings.append(f"index schema must be {SCHEMA}")
    if index.get("closure_status") != "complete":
        findings.append("index closure_status must be complete")
    if index.get("artifact_root") != ".":
        findings.append("index artifact_root must be '.'")
    for name in ("records", "artifacts", "missing_uris", "external_uris"):
        if not isinstance(index.get(name), list):
            findings.append(f"index {name} must be a list")
    for name in ("referenced_uri_count", "finding_count"):
        value = index.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            findings.append(f"index {name} must be a nonnegative integer")
    if findings:
        return findings
    if index["missing_uris"]:
        findings.append("index missing_uris must be empty")
    if index["external_uris"]:
        findings.append("index external_uris must be empty")
    if index["finding_count"] != 0:
        findings.append("index finding_count must be zero")
    if index["referenced_uri_count"] != len(index["artifacts"]):
        findings.append("index referenced_uri_count must equal artifact entry count")
    return findings


def verify_index(
    root_path: Path, index_path: Path, expected_index_sha256: str
) -> list[str]:
    root = _artifact_root(root_path)
    resolved_index, _ = _member(root, index_path, "index")
    if not SHA256.fullmatch(expected_index_sha256):
        return ["expected index SHA-256 must be a full lowercase digest"]
    actual_index_sha256 = _sha256(resolved_index)
    if actual_index_sha256 != expected_index_sha256:
        return ["index SHA-256 does not match the externally anchored digest"]
    index = _load_json(resolved_index, "index")
    findings = _validate_index_shape(index)
    if findings:
        return findings

    record_paths: list[Path] = []
    indexed_records: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(index["records"]):
        location = f"index records[{position}]"
        if not isinstance(entry, dict):
            findings.append(f"{location} must be an object")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            findings.append(f"{location}.path must be a non-empty string")
            continue
        record_paths.append(Path(path))
        indexed_records[path] = entry
    if findings:
        return findings
    if len(indexed_records) != len(index["records"]):
        findings.append("index record entries must have unique paths")
        return findings

    try:
        rebuilt, rebuild_findings = build_index(root, record_paths)
    except BundleError as exc:
        return [str(exc)]
    findings.extend(rebuild_findings)
    rebuilt_records = {entry["path"]: entry for entry in rebuilt["records"]}
    rebuilt_artifacts = {entry["uri"]: entry for entry in rebuilt["artifacts"]}
    indexed_artifacts = {
        entry.get("uri"): entry
        for entry in index["artifacts"]
        if isinstance(entry, dict) and isinstance(entry.get("uri"), str)
    }
    if len(indexed_artifacts) != len(index["artifacts"]):
        findings.append("index artifact entries must be objects with unique string URIs")
    if set(rebuilt_records) != set(indexed_records):
        findings.append("indexed record path set does not match rebuilt bundle")
    if set(rebuilt_artifacts) != set(indexed_artifacts):
        findings.append("indexed artifact URI set does not match rebuilt bundle")
    for path in sorted(set(rebuilt_records).intersection(indexed_records)):
        for field in (
            "schema",
            "uri",
            "sha256",
            "size_bytes",
            "publication_gate_passed",
            "publication_finding_count",
        ):
            if indexed_records[path].get(field) != rebuilt_records[path].get(field):
                findings.append(f"record {path} {field} does not match current bundle")
    for uri in sorted(set(rebuilt_artifacts).intersection(indexed_artifacts)):
        for field in ("path", "sha256", "size_bytes"):
            if indexed_artifacts[uri].get(field) != rebuilt_artifacts[uri].get(field):
                findings.append(f"artifact {uri} {field} does not match current bundle")
    return findings


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build an artifact bundle index")
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--record", type=Path, action="append", required=True)
    build.add_argument("--index-json", type=Path, required=True)
    build.add_argument("--strict", action="store_true")

    verify = subparsers.add_parser("verify", help="verify a checked bundle index")
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--index-json", type=Path, required=True)
    verify.add_argument("--expected-index-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        try:
            root = _artifact_root(args.artifact_root)
            output = _output_member(root, args.index_json, "index output")
            index, findings = build_index(root, args.record)
            _write_json(output, index)
        except BundleError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            f"RL artifact bundle indexed ({index['closure_status']}; "
            f"{len(index['records'])} records; {len(index['artifacts'])} artifacts; "
            f"index_sha256={_sha256(output)})."
        )
        if args.strict and findings:
            for finding in findings:
                print(f"ERROR: {finding}", file=sys.stderr)
            return 1
        return 0

    try:
        findings = verify_index(
            args.artifact_root, args.index_json, args.expected_index_sha256
        )
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    index = _load_json(
        _member(_artifact_root(args.artifact_root), args.index_json, "index")[0],
        "index",
    )
    print(
        f"RL artifact bundle verified ({len(index['records'])} records; "
        f"{len(index['artifacts'])} artifacts)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
