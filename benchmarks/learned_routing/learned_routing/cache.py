# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed result cache (``CR/runs/cache/results/``).

The key is the SHA-256 of the canonical JSON of::

    {policy_sha, cell_id, cell_sha, repeat, protocol, harness_version, build_id}

- ``policy_sha``: seedless canonical policy identity (:mod:`learned_routing.policy`); the policy
  seed is ``repeat + 1`` and therefore implied by ``repeat`` (Amendment A1);
- ``cell_sha``: cell content SHA (:mod:`learned_routing.cells`);
- ``protocol``: the replicate protocol (``crn-order-v1``);
- ``build_id``: the replay build, i.e. the SHA-256 of the ``dynamo._core`` extension module plus
  the ``aisimulate`` version that supplies AIS data. A rebuilt binding invalidates every entry.

Writes are atomic (temp file + ``os.replace``). Errored evaluations are never cached, so a rerun
retries them.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

from learned_routing.canon import atomic_write_text, sha256_json


def cache_key(
    *,
    policy_sha: str,
    cell_id: str,
    cell_sha: str,
    repeat: int,
    protocol: str,
    harness_version: str,
    build_id: str,
) -> str:
    return sha256_json(
        {
            "policy_sha": policy_sha,
            "cell_id": cell_id,
            "cell_sha": cell_sha,
            "repeat": int(repeat),
            "protocol": protocol,
            "harness_version": harness_version,
            "build_id": build_id,
        }
    )


class ResultCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir) / "results"

    def path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def per_request_path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.per_request.jsonl.gz"

    def get(self, key: str) -> dict | None:
        path = self.path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        return (
            record
            if record.get("cache_key") == key and not record.get("error")
            else None
        )

    def put(self, key: str, record: dict) -> None:
        if record.get("error"):
            raise ValueError("errored records are not cached")
        if record.get("cache_key") != key:
            raise ValueError("record cache_key does not match its key")
        atomic_write_text(self.path(key), json.dumps(record, sort_keys=True))

    def keys(self):
        if not self.dir.exists():
            return
        for path in self.dir.glob("*/*.json"):
            yield path.stem


def _core_extension() -> Path:
    spec = importlib.util.find_spec("dynamo._core")
    if spec is None or not spec.origin:
        raise RuntimeError("dynamo._core is not importable; build the bindings first")
    return Path(spec.origin)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bindings_build_id(cache_dir: Path, core_path: Path | None = None) -> dict:
    """``{"build_id", "core_so", "core_so_sha256", "aisimulate"}``, memoized by file stat."""
    core = core_path or _core_extension()
    stat = core.stat()
    memo_path = Path(cache_dir) / "build_ids.json"
    memo_key = f"{core.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    memo: dict = {}
    if memo_path.exists():
        memo = json.loads(memo_path.read_text())
    so_sha = memo.get(memo_key)
    if so_sha is None:
        so_sha = _file_sha256(core)
        memo_path.parent.mkdir(parents=True, exist_ok=True)
        lock = os.open(memo_path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(memo_path.read_text()) if memo_path.exists() else {}
            current[memo_key] = so_sha
            atomic_write_text(memo_path, json.dumps(current, indent=1, sort_keys=True))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)
    try:
        aisimulate = importlib.metadata.version("aisimulate")
    except importlib.metadata.PackageNotFoundError:
        aisimulate = None
    identity = {"core_so_sha256": so_sha, "aisimulate": aisimulate}
    return {"build_id": sha256_json(identity), "core_so": str(core), **identity}
