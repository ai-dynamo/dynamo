#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the exact campaign source tree synced to the eval runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_BUNDLE_CONTENTS = ["campaign.env", "eval"]
IGNORED_PARTS = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


class SourceProvenanceError(ValueError):
    """Raised when the synced evaluator source cannot be proven exact."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignored(relative: PurePosixPath) -> bool:
    return any(part in IGNORED_PARTS for part in relative.parts) or (
        relative.suffix in IGNORED_SUFFIXES
    )


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "sha256": file_sha256(path),
        "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
    }


def source_files(root: Path) -> dict[str, dict[str, Any]]:
    """Return logical-path hashes for campaign.env and the complete eval tree."""

    root = root.resolve()
    campaign_env = root / "campaign.env"
    eval_dir = root / "eval"
    if not campaign_env.is_file() or campaign_env.is_symlink():
        raise SourceProvenanceError("campaign.env is missing or is not a regular file")
    if not eval_dir.is_dir() or eval_dir.is_symlink():
        raise SourceProvenanceError("eval is missing or is not a regular directory")

    files = {"campaign.env": file_identity(campaign_env)}
    for path in sorted(eval_dir.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _ignored(relative):
            continue
        if path.is_symlink():
            raise SourceProvenanceError(
                f"campaign source contains a symlink: {relative.as_posix()}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise SourceProvenanceError(
                f"campaign source contains a non-regular file: {relative.as_posix()}"
            )
        files[relative.as_posix()] = file_identity(path)
    if len(files) < 2:
        raise SourceProvenanceError("campaign eval source tree is empty")
    return dict(sorted(files.items()))


def _aggregate(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eval_files = {key: value for key, value in files.items() if key.startswith("eval/")}
    if not eval_files:
        raise SourceProvenanceError("campaign source manifest has no eval files")
    return {
        "source_tree_sha256": canonical_sha256(files),
        "eval_tree_sha256": canonical_sha256(eval_files),
        "campaign_env_sha256": files["campaign.env"]["sha256"],
        "source_file_count": len(files),
        "eval_file_count": len(eval_files),
    }


def build_source_provenance(
    source_root: Path,
    *,
    source_commit: str,
    source_branch: str | None,
    bundle_sha256: str,
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise SourceProvenanceError("source_commit must be a lowercase 40-hex commit")
    if SHA256_RE.fullmatch(bundle_sha256) is None:
        raise SourceProvenanceError("bundle_sha256 must be a lowercase SHA-256 digest")
    files = source_files(source_root)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "source_branch": source_branch or None,
        "source_clean": True,
        "source_changed_path_count": 0,
        "bundle_sha256": bundle_sha256,
        "bundle_contents": EXPECTED_BUNDLE_CONTENTS,
        **_aggregate(files),
        "source_files": files,
    }


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SourceProvenanceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validated_manifest(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise SourceProvenanceError("source_files must be a non-empty object")
    result: dict[str, dict[str, Any]] = {}
    for raw_path, raw_identity in value.items():
        if not isinstance(raw_path, str):
            raise SourceProvenanceError("source_files paths must be strings")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or raw_path != path.as_posix()
            or ".." in path.parts
            or "." in path.parts
            or not path.parts
        ):
            raise SourceProvenanceError(f"invalid logical source path: {raw_path!r}")
        if raw_path != "campaign.env" and not raw_path.startswith("eval/"):
            raise SourceProvenanceError(
                f"source path is outside campaign.env/eval: {raw_path!r}"
            )
        if not isinstance(raw_identity, dict) or set(raw_identity) != {
            "sha256",
            "mode",
        }:
            raise SourceProvenanceError(
                f"source_files[{raw_path!r}] must contain sha256 and mode"
            )
        mode = raw_identity["mode"]
        if not isinstance(mode, str) or re.fullmatch(r"0[0-7]{3}", mode) is None:
            raise SourceProvenanceError(
                f"source_files[{raw_path!r}].mode must be a four-digit octal mode"
            )
        result[raw_path] = {
            "sha256": _require_digest(
                raw_identity["sha256"], f"source_files[{raw_path!r}].sha256"
            ),
            "mode": mode,
        }
    if "campaign.env" not in result:
        raise SourceProvenanceError("source_files is missing campaign.env")
    return dict(sorted(result.items()))


def verify_source_provenance(
    provenance_path: Path,
    source_root: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Verify provenance and current files, returning a path-free compact identity."""

    if COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise SourceProvenanceError("expected source commit is invalid")
    document = json.loads(provenance_path.read_text())
    expected_fields = {
        "schema_version",
        "generated_at",
        "source_commit",
        "source_branch",
        "source_clean",
        "source_changed_path_count",
        "bundle_sha256",
        "bundle_contents",
        "source_tree_sha256",
        "eval_tree_sha256",
        "campaign_env_sha256",
        "source_file_count",
        "eval_file_count",
        "source_files",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        actual = (
            sorted(document) if isinstance(document, dict) else type(document).__name__
        )
        raise SourceProvenanceError(
            f"source provenance fields differ: expected {sorted(expected_fields)}, got {actual}"
        )
    if document["schema_version"] != 2 or isinstance(document["schema_version"], bool):
        raise SourceProvenanceError("source provenance schema_version must be 2")
    if not isinstance(document["generated_at"], str):
        raise SourceProvenanceError("source provenance generated_at is invalid")
    try:
        generated_at = datetime.fromisoformat(
            document["generated_at"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise SourceProvenanceError(
            "source provenance generated_at is not ISO-8601"
        ) from error
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise SourceProvenanceError("source provenance generated_at has no UTC offset")
    source_branch = document["source_branch"]
    if source_branch is not None and (
        not isinstance(source_branch, str)
        or not source_branch
        or any(character in source_branch for character in "\r\n\0")
    ):
        raise SourceProvenanceError("source provenance source_branch is invalid")
    if document["source_commit"] != expected_source_commit:
        raise SourceProvenanceError(
            "source provenance commit differs from the serving deployment recipe"
        )
    if (
        document["source_clean"] is not True
        or isinstance(document["source_changed_path_count"], bool)
        or document["source_changed_path_count"] != 0
    ):
        raise SourceProvenanceError(
            "source provenance does not describe a clean bundle"
        )
    if document["bundle_contents"] != EXPECTED_BUNDLE_CONTENTS:
        raise SourceProvenanceError("source provenance bundle contents are invalid")
    _require_digest(document["bundle_sha256"], "bundle_sha256")

    expected_files = _validated_manifest(document["source_files"])
    aggregate = _aggregate(expected_files)
    for field, expected in aggregate.items():
        if document[field] != expected:
            raise SourceProvenanceError(f"source provenance {field} is inconsistent")

    actual_files = source_files(source_root)
    if actual_files != expected_files:
        expected_paths = set(expected_files)
        actual_paths = set(actual_files)
        changed = sorted(
            path
            for path in expected_paths & actual_paths
            if expected_files[path] != actual_files[path]
        )
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise SourceProvenanceError(
            "current campaign source differs from the synced manifest: "
            f"changed={changed}, missing={missing}, unexpected={unexpected}"
        )

    return {
        "schema_version": 1,
        "source_commit": document["source_commit"],
        "source_clean": True,
        "source_changed_path_count": 0,
        "bundle_sha256": document["bundle_sha256"],
        **aggregate,
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True, type=Path)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--source-branch", default="")
    build.add_argument("--bundle-sha256", required=True)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--provenance", required=True, type=Path)
    verify.add_argument("--source-root", required=True, type=Path)
    verify.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()

    try:
        if args.action == "build":
            document = build_source_provenance(
                args.source_root,
                source_commit=args.source_commit,
                source_branch=args.source_branch or None,
                bundle_sha256=args.bundle_sha256,
            )
            _write_exclusive(args.output, document)
        else:
            compact = verify_source_provenance(
                args.provenance, args.source_root, args.expected_source_commit
            )
            print(json.dumps(compact, indent=2, sort_keys=True))
    except (OSError, json.JSONDecodeError, SourceProvenanceError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
