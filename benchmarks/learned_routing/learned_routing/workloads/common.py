# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers: hashing, canonical JSON, JSONL IO, seeded draws and the campaign root."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

CAMPAIGN_ROOT_ENV = "LR_CAMPAIGN_ROOT"
MAX_MODEL_LEN = 131072
MOONCAKE_BLOCK_SIZE = 512
WEKA_BLOCK_SIZE = 64


def campaign_root(explicit: str | os.PathLike | None = None) -> Path:
    """The campaign root ``CR``: an explicit path, else ``$LR_CAMPAIGN_ROOT``."""
    value = explicit if explicit is not None else os.environ.get(CAMPAIGN_ROOT_ENV)
    if not value:
        raise SystemExit(
            f"campaign root unknown: pass --campaign-root or set ${CAMPAIGN_ROOT_ENV}"
        )
    return Path(value).resolve()


def canonical_json(obj) -> str:
    """Compact JSON with sorted keys; floats keep Python's shortest repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl_lines(path: Path) -> Iterator[bytes]:
    with Path(path).open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield raw


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    """Rows as JSONL with sorted keys, so equal rows serialize to equal bytes."""
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def write_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_atomic(
        path,
        (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def unit_draw(seed: int, label: str, key) -> float:
    """Counter-based uniform draw in [0, 1), independent of Python's ``random`` implementation."""
    digest = hashlib.blake2b(f"{seed}|{label}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def index_draw(seed: int, label: str, key, bound: int) -> int:
    digest = hashlib.blake2b(f"{seed}|{label}|{key}".encode(), digest_size=16).digest()
    return int.from_bytes(digest, "big") % bound


def quantile(values: list[float], q: float) -> float | None:
    """Nearest-rank quantile; ``None`` for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[rank]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p99": quantile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }
