# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked, durable checkpoints for resuming compatible match executions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
        + b"\n"
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite float range")
    return number


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"checkpoint path must not be a symlink: {path.name}")
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read checkpoint {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path.name} must contain a JSON object")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported checkpoint schema in {path.name}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    if path.is_symlink():
        raise ValueError(f"checkpoint path must not be a symlink: {path.name}")
    content = _json_bytes(value)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="wb", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_ARTIFACT_KEYS = frozenset(
    {
        "directory",
        "trace_report",
        "worker_log",
        "requests_jsonl",
        "telemetry",
        "preparation_manifest",
        "jev_decisions",
    }
)
_ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".csv", ".html"})


def _artifact_references(root: Path, artifacts: Any) -> list[str]:
    """Select local session-owned references, ignoring URLs and metadata."""
    references = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and value and "://" not in value:
            candidate = Path(value)
            if ".." in candidate.parts:
                return
            if candidate.is_absolute():
                if not candidate.is_relative_to(root):
                    return
                relative = candidate.relative_to(root)
            else:
                relative = candidate
                candidate = root / relative
                if not (
                    key in _ARTIFACT_KEYS
                    or "/" in value
                    or candidate.suffix in _ARTIFACT_SUFFIXES
                    or candidate.exists()
                ):
                    return
            if not relative.parts or relative.parts[0] in {
                "checkpoints",
                "attempts",
                "session-manifest.json",
                ".runs.lock",
            }:
                raise ValueError("artifact references must point to session output")
            references.add(relative.as_posix())

    visit(artifacts)
    return sorted(references)


def _artifact_fingerprint(root: Path, relative: str) -> dict[str, Any]:
    """Hash output files and directory contents without following symlinks."""
    path = root / relative
    unavailable = {"path": relative, "kind": "unavailable", "sha256": None}

    def file_digest(file_path: Path) -> str:
        with file_path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def walk_error(error: OSError) -> None:
        raise error

    try:
        current = root
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                return unavailable
        mode = path.stat().st_mode
        if stat.S_ISREG(mode):
            kind, digest = "file", file_digest(path)
        elif stat.S_ISDIR(mode):
            kind = "directory"
            tree = hashlib.sha256()
            for directory, directories, files in os.walk(
                path, followlinks=False, onerror=walk_error
            ):
                directories.sort()
                files.sort()
                for name in directories + files:
                    child = Path(directory) / name
                    mode = child.lstat().st_mode
                    child_name = child.relative_to(path).as_posix()
                    if stat.S_ISREG(mode):
                        tree.update(
                            _json_bytes([child_name, "file", file_digest(child)])
                        )
                    elif stat.S_ISDIR(mode):
                        tree.update(_json_bytes([child_name, "directory"]))
                    else:
                        return unavailable
            digest = tree.hexdigest()
        else:
            return unavailable
    except OSError:
        # Missing, unreadable or changing output is not reusable evidence.
        return unavailable
    return {"path": relative, "kind": kind, "sha256": digest}


def _validate_artifact_manifest(manifest: Any, references: list[str]) -> None:
    if not isinstance(manifest, list):
        raise ValueError("checkpoint artifact manifest must be a list")
    paths = []
    for entry in manifest:
        if not isinstance(entry, dict) or set(entry) != {"path", "kind", "sha256"}:
            raise ValueError("malformed checkpoint artifact manifest entry")
        path, kind, digest = entry["path"], entry["kind"], entry["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or Path(path).as_posix() != path
            or not isinstance(kind, str)
            or kind not in {"file", "directory", "unavailable"}
        ):
            raise ValueError("invalid checkpoint artifact manifest path or kind")
        if kind == "unavailable":
            if digest is not None:
                raise ValueError(
                    "unavailable checkpoint artifact must not have a digest"
                )
        elif not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid checkpoint artifact manifest digest")
        paths.append(path)
    if paths != references:
        raise ValueError(
            "checkpoint artifact manifest does not match result references"
        )


class RunJournal:
    """Hold one session lock while validating and recording individual runs.

    The caller creates a fresh empty session directory, or explicitly requests
    resume of an existing session. Compatibility comes from the supplied input,
    replay configuration and software signature. Failed results are retained
    for diagnosis but are never reused as successful executions. Sessions are
    bound to their original absolute directory: copying or moving a session
    does not rewrite stored artifact references and cannot be resumed.
    """

    def __init__(
        self, session_root: Path, signature: dict[str, Any], resume: bool = False
    ):
        if not isinstance(signature, dict):
            raise ValueError("run journal signature must be a JSON object")
        if not isinstance(resume, bool):
            raise ValueError("run journal resume must be a boolean")
        self.session_root = session_root
        # A canonical round trip both validates JSON compatibility and detaches
        # the identity from later mutations by the caller.
        self._signature_bytes = _json_bytes(signature)
        self._signature = json.loads(self._signature_bytes)
        self._signature_sha256 = hashlib.sha256(self._signature_bytes).hexdigest()
        self._resume = resume
        self._stack = ExitStack()
        self._active = False

    def __enter__(self) -> RunJournal:
        if self._active:
            raise RuntimeError("run journal is already open")
        if self.session_root.is_symlink() or not self.session_root.is_dir():
            raise ValueError("run journal requires an existing real session directory")
        lock_path = self.session_root / ".runs.lock"
        if lock_path.is_symlink():
            raise ValueError("run journal lock must not be a symlink")
        try:
            lock = self._stack.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "another process is already executing this session"
                ) from exc
            self._stack.callback(fcntl.flock, lock, fcntl.LOCK_UN)
            manifest_path = self.session_root / "session-manifest.json"
            if self._resume:
                manifest = _read_json(manifest_path)
                if (
                    set(manifest) != {"schema_version", "signature", "session_root"}
                    or _json_bytes(manifest.get("signature")) != self._signature_bytes
                ):
                    raise ValueError(
                        "session checkpoint signature does not match current inputs, configuration or software"
                    )
                if manifest["session_root"] != str(self.session_root.resolve()):
                    raise ValueError(
                        "session belongs to a different artifact root; relocated sessions cannot be resumed"
                    )
            else:
                if any(path != lock_path for path in self.session_root.iterdir()):
                    raise ValueError(
                        "fresh run journal requires an empty session; use resume for a known session"
                    )
                _atomic_json(
                    manifest_path,
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "signature": self._signature,
                        "session_root": str(self.session_root.resolve()),
                    },
                )
            for name in ("runs", "checkpoints", "attempts"):
                self._directory(name)
            _fsync_directory(self.session_root)
            self._active = True
            return self
        except BaseException:
            # __exit__ is not invoked when __enter__ fails, including interrupts.
            self._stack.close()
            raise

    def __exit__(self, *exc_info) -> None:
        self._active = False
        self._stack.__exit__(*exc_info)

    def _directory(self, name: str) -> Path:
        path = self.session_root / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"run journal {name} must be a real directory")
        path.mkdir(exist_ok=True)
        return path

    def _validate_run_id(self, run_id: str) -> None:
        if not self._active:
            raise RuntimeError("run journal must be used inside its context manager")
        if (
            not isinstance(run_id, str)
            or len(run_id) > 240
            or not _SAFE_RUN_ID.fullmatch(run_id)
        ):
            raise ValueError("run_id must be a safe filename of at most 240 characters")

    def load_success(self, run_id: str) -> dict[str, Any] | None:
        """Return a compatible successful result; never reuse failed work."""
        self._validate_run_id(run_id)
        checkpoint = self._directory("checkpoints") / f"{run_id}.json"
        if not os.path.lexists(checkpoint):
            return None
        value = _read_json(checkpoint)
        if (
            set(value)
            != {
                "schema_version",
                "signature_sha256",
                "run_id",
                "result",
                "artifact_manifest",
            }
            or value.get("signature_sha256") != self._signature_sha256
            or value.get("run_id") != run_id
        ):
            raise ValueError(f"checkpoint identity mismatch for {run_id}")
        result = value.get("result")
        self._validate_result(run_id, result)
        root = self.session_root.resolve()
        references = _artifact_references(root, result.get("artifacts", {}))
        manifest = value["artifact_manifest"]
        _validate_artifact_manifest(manifest, references)
        if result["status"] != "ok":
            return None
        for entry in manifest:
            if (
                entry["kind"] == "unavailable"
                or _artifact_fingerprint(root, entry["path"]) != entry
            ):
                return None
        return result

    def prepare_run(self, run_id: str) -> None:
        """Archive artifacts and checkpoint from any earlier attempt before retry."""
        self._validate_run_id(run_id)
        run_dir = self._directory("runs") / run_id
        checkpoint = self._directory("checkpoints") / f"{run_id}.json"
        if run_dir.is_symlink() or checkpoint.is_symlink():
            raise ValueError("run artifacts and checkpoints must not be symlinks")
        if not run_dir.exists() and not checkpoint.exists():
            return
        attempt = Path(
            tempfile.mkdtemp(prefix=run_id + "-", dir=self._directory("attempts"))
        )
        # Retire the checkpoint first so interruption cannot leave a reusable
        # success pointing at artifacts that have already moved.
        if checkpoint.exists():
            checkpoint.rename(attempt / "checkpoint.json")
            _fsync_directory(checkpoint.parent)
            _fsync_directory(attempt)
        if run_dir.exists():
            run_dir.rename(attempt / "run")
            _fsync_directory(run_dir.parent)
            _fsync_directory(attempt)
        _fsync_directory(attempt.parent)

    def record(self, run_id: str, result: dict[str, Any]) -> None:
        """Atomically save one terminal result after validating its identity."""
        self._validate_run_id(run_id)
        self._validate_result(run_id, result)
        root = self.session_root.resolve()
        artifact_manifest = [
            _artifact_fingerprint(root, relative)
            for relative in _artifact_references(root, result.get("artifacts", {}))
        ]
        _atomic_json(
            self._directory("checkpoints") / f"{run_id}.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "signature_sha256": self._signature_sha256,
                "run_id": run_id,
                "result": result,
                "artifact_manifest": artifact_manifest,
            },
        )

    @staticmethod
    def _validate_result(run_id: str, result: Any) -> None:
        if not isinstance(result, dict) or result.get("run_id") != run_id:
            raise ValueError(f"checkpoint result identity mismatch for {run_id}")
        status = result.get("status")
        if not isinstance(status, str) or status not in {"ok", "failed"}:
            raise ValueError(
                f"checkpoint result has invalid terminal status for {run_id}"
            )
