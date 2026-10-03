# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common-random-numbers (CRN) replicates for ``lr-eval --repeats K``.

Offline replay is deterministic for a fixed (policy, workload, policy seed), so re-running a
deterministic policy (round-robin, lmetric, learned-choice at temperature 0) has zero spread, and
a noise floor built from identical re-runs is false. The arbitrary part of a recorded workload is
the order of simultaneous arrivals: Mooncake and toolagent arrive in 3 s bursts of up to 47 rows,
and the router places a burst in file order. Reordering a burst moves per-cell goodput by about
1.5% sd for every policy (setup audit F1, ``audits/setup/determinism-cost-context-r0.md``).

Replicate ``k`` of a cell is therefore:

- a seeded permutation of that arbitrary order, applied to the cell's materialized trace and
  shared by every policy evaluated on the cell (CRN); and
- policy seed ``k + 1`` for every policy that takes a ``seed`` parameter.

Per trace format:

- ``mooncake``: sessions (single-turn rows are one-row sessions) that share a first-arrival
  timestamp are permuted; every session keeps its turns in order and all arrival times are
  unchanged. Rows with no timestamp arrive at 0, as in the replay driver.
- ``weka``: the play order is permuted. Agentic lanes take plays in source order, so this
  re-draws which plays share lanes; in timestamp mode it only reorders plays that tie.
- ``agentic_mooncake`` (recycled AgentX copies, :mod:`learned_routing.workloads.agentx_lowered`):
  the copy order is permuted, the analogue of the Weka play order. Copy ``j`` of the replicate is
  source copy ``pi[j]`` relabeled exactly as the generator would have written it at position
  ``j``: id prefix ``lrx:<j>``, ``source_play_ordinal = j``, hash ids moved to the ``j``-th
  consecutive range, rows sorted by ``request_id``. The file is byte-identical to generating the
  permuted draw list, so lanes (``ordinal % lanes``) run a re-dealt sequence of the same copies.
  Each copy keeps its own ``not_before_ms``, so in open mode only tie order changes.
- synthetic sessions: no trace file; pass :func:`synthetic_arrival_seed` as ``arrival_seed``.

Arrival spread (``spread_ms > 0``, Mooncake format only; protocol ``crn-spread-v1``). Mooncake and
FAST25-conversation timestamps are logged on a ~3 s grid and the synthetic sessions on a 1 s grid,
so up to 47 sessions arrive at the same instant. Replay delivers such a burst at once whatever the
arrival speedup, so a worker's share of a burst grows as 1/N and queueing inside bursts dominates
small-N cells (calibration, ``facts/calibration.json``). With ``spread_ms`` set to the logging
quantum, replicate ``k`` moves every session's first arrival to ``t + spread_ms * u`` with ``u``
uniform in ``[0, 1)`` drawn from the replicate seed and the session's position, and stably re-sorts
sessions by the new arrival (later turns keep their completion-relative ``delay``). Replicates then
differ in where inside its logging slot each session arrives, which subsumes the tie permutation.

Every replicate, including ``k = 0``, is permuted, so all replicates are exchangeable draws and the
recorded file order has no special status. The replicate seed depends only on the source trace's
content and ``k``, never on the cell id, so cells that share a trace (different N or load level)
also share their replicate workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL = "crn-order-v1"
SPREAD_PROTOCOL = "crn-spread-v1"
WEKA_SUFFIXES = (".json", ".jsonl")


@dataclass(frozen=True)
class PermutationStats:
    """How much of a trace a replicate permutation can and did move."""

    units: int  # sessions (mooncake) or plays (weka)
    tie_groups: int  # groups of >= 2 units sharing an arrival (mooncake) or the play list (weka)
    units_in_ties: int  # units that the permutation may move
    moved: int  # units whose position changed

    @property
    def degenerate(self) -> bool:
        """True when every replicate equals the source, so replicates cannot measure noise."""
        return self.units_in_ties == 0


@dataclass(frozen=True)
class Replicate:
    k: int
    protocol: str
    trace_format: str
    source_sha256: str
    replicate_seed: int
    policy_seed: int
    path: Path
    sha256: str
    stats: PermutationStats

    def to_record(self) -> dict:
        record = asdict(self)
        record["path"] = str(self.path)
        record["degenerate"] = self.stats.degenerate
        return record


def replicate_seed(source_key: str, k: int) -> int:
    """Seed for replicate ``k`` of a source; fits in a signed 64-bit integer."""
    if k < 0:
        raise ValueError(f"replicate index must be >= 0, got {k}")
    digest = hashlib.sha256(f"{PROTOCOL}|{source_key}|{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def policy_seed(k: int) -> int:
    """Policy ``seed`` parameter for replicate ``k``; replicate 0 uses setup's seed 1."""
    if k < 0:
        raise ValueError(f"replicate index must be >= 0, got {k}")
    return k + 1


def synthetic_arrival_seed(spec_key: str, k: int) -> int:
    """``arrival_seed`` for replicate ``k`` of a synthetic-session cell keyed by ``spec_key``."""
    return replicate_seed(f"synthetic|{spec_key}", k)


def _uniform_below(seed: int, label: str, index: int, bound: int) -> int:
    # Counter-based draw, so the permutation does not depend on Python's `random` implementation.
    digest = hashlib.blake2b(
        f"{seed}|{label}|{index}".encode(), digest_size=16
    ).digest()
    return int.from_bytes(digest, "big") % bound


def _shuffle(items: Sequence, seed: int, label: str) -> list:
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = _uniform_below(seed, label, i, i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def permute_mooncake_ties(
    lines: Sequence[str], seed: int
) -> tuple[list[str], PermutationStats]:
    """Permute sessions that share a first-arrival timestamp, keeping each session's turns."""
    sessions: dict[tuple, list[str]] = {}
    arrival: dict[tuple, float] = {}
    for index, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        row = json.loads(line)
        if "wait_for" in row:
            raise ValueError(
                f"line {index + 1}: wait_for dependencies are not supported by {PROTOCOL}"
            )
        session_id = row.get("session_id")
        key = ("session", session_id) if session_id is not None else ("row", index)
        turns = sessions.setdefault(key, [])
        if not turns:
            timestamp = row.get("timestamp", row.get("created_time"))
            arrival[key] = 0.0 if timestamp is None else float(timestamp)
        turns.append(line)

    order = list(sessions)
    groups: dict[float, list[int]] = {}
    for position, key in enumerate(order):
        groups.setdefault(arrival[key], []).append(position)

    new_order = list(order)
    tie_groups = units_in_ties = 0
    for timestamp, positions in groups.items():
        if len(positions) < 2:
            continue
        tie_groups += 1
        units_in_ties += len(positions)
        shuffled = _shuffle([order[p] for p in positions], seed, f"t={timestamp!r}")
        for position, key in zip(positions, shuffled):
            new_order[position] = key

    stats = PermutationStats(
        units=len(order),
        tie_groups=tie_groups,
        units_in_ties=units_in_ties,
        moved=sum(old != new for old, new in zip(order, new_order)),
    )
    return [line for key in new_order for line in sessions[key]], stats


def _unit(seed: int, label: str, index: int) -> float:
    digest = hashlib.blake2b(f"{seed}|{label}|{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2.0**64


def spread_mooncake_arrivals(
    lines: Sequence[str], seed: int, spread_ms: float
) -> tuple[list[str], PermutationStats]:
    """Re-draw each session's first arrival uniformly inside its logging slot (module docstring)."""
    if not spread_ms > 0:
        raise ValueError(f"spread_ms must be > 0, got {spread_ms!r}")
    sessions: dict[tuple, list[dict]] = {}
    arrival: dict[tuple, float] = {}
    for index, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        row = json.loads(line)
        if "wait_for" in row:
            raise ValueError(
                f"line {index + 1}: wait_for dependencies are not supported by {SPREAD_PROTOCOL}"
            )
        session_id = row.get("session_id")
        key = ("session", session_id) if session_id is not None else ("row", index)
        turns = sessions.setdefault(key, [])
        if not turns:
            stamp = row.get("timestamp", row.get("created_time"))
            arrival[key] = 0.0 if stamp is None else float(stamp)
        turns.append(row)
    order = list(sessions)
    counts: dict[float, int] = {}
    for key in order:
        counts[arrival[key]] = counts.get(arrival[key], 0) + 1
    moved_at: dict[tuple, int] = {}
    for position, key in enumerate(order):
        first = sessions[key][0]
        field = "timestamp" if "timestamp" in first else "created_time"
        if field not in first:
            raise ValueError(
                f"session {key!r} has no first-arrival timestamp to spread"
            )
        new = arrival[key] + spread_ms * _unit(seed, "spread", position)
        first[field] = int(new) if isinstance(first[field], int) else new
        moved_at[key] = position
    new_order = sorted(
        order,
        key=lambda key: (
            float(
                sessions[key][0].get("timestamp", sessions[key][0].get("created_time"))
            ),
            moved_at[key],
        ),
    )
    stats = PermutationStats(
        units=len(order),
        tie_groups=sum(1 for c in counts.values() if c >= 2),
        units_in_ties=len(order),
        moved=sum(old != new for old, new in zip(order, new_order)),
    )
    out = [
        json.dumps(row, separators=(",", ":"))
        for key in new_order
        for row in sessions[key]
    ]
    return out, stats


def _weka_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in WEKA_SUFFIXES
    ]
    # The replay loader orders a directory's files by relative path.
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def load_weka_plays(source: Path) -> list[str]:
    """Plays in the replay loader's source order, each as one compact JSON line."""
    decoder = json.JSONDecoder()
    plays: list[str] = []
    for path in _weka_files(source):
        text = path.read_text()
        objects = []
        position = 0
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                break
            obj, position = decoder.raw_decode(text, position)
            objects.append(obj)
        if path.suffix.lower() == ".json" and len(objects) != 1:
            raise ValueError(f"{path}: a .json Weka source must hold exactly one play")
        plays.extend(json.dumps(obj, separators=(",", ":")) for obj in objects)
    if not plays:
        raise ValueError(f"{source}: no Weka plays found")
    return plays


def permute_weka_plays(
    plays: Sequence[str], seed: int
) -> tuple[list[str], PermutationStats]:
    """Permute the play order of a play-per-line Weka trace."""
    order = list(range(len(plays)))
    shuffled = _shuffle(order, seed, "plays")
    stats = PermutationStats(
        units=len(plays),
        tie_groups=1 if len(plays) >= 2 else 0,
        units_in_ties=len(plays) if len(plays) >= 2 else 0,
        moved=sum(old != new for old, new in zip(order, shuffled)),
    )
    return [plays[i] for i in shuffled], stats


AGENTIC_LABEL = re.compile(r"^(lrx:(\d{6})):")


def _agentic_relabel(value, old: str, new: str):
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old + ":") else value
    if isinstance(value, list):
        return [_agentic_relabel(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _agentic_relabel(v, old, new) for k, v in value.items()}
    return value


def permute_agentic_copies(
    lines: Sequence[str], seed: int
) -> tuple[list[str], PermutationStats]:
    """Permute the copies of a recycled Agentic Mooncake v2 trace (module docstring)."""
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows or "schema" not in rows[0]:
        raise ValueError("agentic_mooncake trace needs a header line")
    header, body = rows[0], rows[1:]
    copies: dict[int, list[dict]] = {}
    for row in body:
        match = AGENTIC_LABEL.match(row.get("request_id", ""))
        if match is None:
            raise ValueError(
                f"row {row.get('request_id')!r} has no lrx:<copy> label; only recycled "
                "AgentX traces are supported"
            )
        copies.setdefault(int(match[2]), []).append(row)
    order = sorted(copies)
    if order != list(range(len(order))):
        raise ValueError("copy labels must be contiguous from lrx:000000")
    shuffled = _shuffle(order, seed, "copies")
    out_rows: list[dict] = []
    next_hash = 1
    for position, source in enumerate(shuffled):
        copy_rows = copies[source]
        ids = [h for row in copy_rows for h in row["hash_ids"]]
        low = min(ids) if ids else next_hash
        old, new = f"lrx:{source:06d}", f"lrx:{position:06d}"
        for row in copy_rows:
            moved = _agentic_relabel(row, old, new)
            moved["source_play_ordinal"] = position
            moved["hash_ids"] = [h - low + next_hash for h in row["hash_ids"]]
            out_rows.append(moved)
        next_hash += (max(ids) - low + 1) if ids else 0
    out_rows.sort(key=lambda row: row["request_id"])
    stats = PermutationStats(
        units=len(order),
        tie_groups=1 if len(order) >= 2 else 0,
        units_in_ties=len(order) if len(order) >= 2 else 0,
        moved=sum(old != new for old, new in zip(order, shuffled)),
    )
    compact = [json.dumps(header, separators=(",", ":"))]
    compact += [json.dumps(row, separators=(",", ":")) for row in out_rows]
    return compact, stats


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(source: Path, trace_format: str) -> str:
    """Content key of a trace source: the file's SHA-256, or a manifest hash for a directory."""
    if source.is_file():
        return _sha256_file(source)
    if trace_format != "weka":
        raise ValueError(f"{source}: only weka sources may be directories")
    manifest = "".join(
        f"{path.relative_to(source).as_posix()}\0{_sha256_file(path)}\n"
        for path in _weka_files(source)
    )
    return _sha256_bytes(manifest.encode())


def materialize_replicate(
    source: Path, trace_format: str, k: int, out_dir: Path, spread_ms: float = 0.0
) -> Replicate:
    """Write replicate ``k`` of ``source`` under ``out_dir`` (idempotent) and describe it.

    ``spread_ms > 0`` (Mooncake format) selects protocol ``crn-spread-v1``.
    """
    source = Path(source)
    out_dir = Path(out_dir)
    if source.is_dir() and out_dir.resolve().is_relative_to(source.resolve()):
        raise ValueError(
            f"out_dir {out_dir} must not be inside the source directory {source}"
        )
    source_key = source_sha256(source, trace_format)
    seed = replicate_seed(source_key, k)
    protocol = PROTOCOL
    if spread_ms:
        if trace_format != "mooncake":
            raise ValueError(
                f"arrival spread applies to mooncake traces, not {trace_format}"
            )
        protocol = SPREAD_PROTOCOL
    tag = f"-s{float(spread_ms):g}" if spread_ms else ""
    path = out_dir / f"{protocol}-{trace_format}-{source_key[:16]}{tag}-k{k}.jsonl"
    # A replicate is a pure function of (source content, protocol, k, spread): reuse a complete
    # earlier materialization instead of regenerating it (large traces take seconds each).
    identity = {
        "record_version": 1,
        "protocol": protocol,
        "trace_format": trace_format,
        "source_sha256": source_key,
        "k": int(k),
        "replicate_seed": seed,
        "spread_ms": float(spread_ms or 0.0),
    }
    record_path = path.with_name(path.name + ".rep.json")
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if {key: record.get(key) for key in identity} == identity and (
            path.stat().st_size == record.get("bytes")
        ):
            return Replicate(
                k=k,
                protocol=protocol,
                trace_format=trace_format,
                source_sha256=source_key,
                replicate_seed=seed,
                policy_seed=policy_seed(k),
                path=path,
                sha256=record["sha256"],
                stats=PermutationStats(**record["stats"]),
            )
    if spread_ms:
        lines, stats = spread_mooncake_arrivals(
            source.read_text().splitlines(), seed, float(spread_ms)
        )
    elif trace_format == "mooncake":
        lines, stats = permute_mooncake_ties(source.read_text().splitlines(), seed)
    elif trace_format == "weka":
        lines, stats = permute_weka_plays(load_weka_plays(source), seed)
    elif trace_format == "agentic_mooncake":
        lines, stats = permute_agentic_copies(source.read_text().splitlines(), seed)
    else:
        raise ValueError(f"unsupported trace_format for {PROTOCOL}: {trace_format}")

    content = "".join(line + "\n" for line in lines).encode()
    sha = _sha256_bytes(content)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists() or _sha256_file(path) != sha:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
    record = {**identity, "sha256": sha, "bytes": len(content), "stats": asdict(stats)}
    tmp = record_path.with_name(f".{record_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, sort_keys=True))
    os.replace(tmp, record_path)
    return Replicate(
        k=k,
        protocol=protocol,
        trace_format=trace_format,
        source_sha256=source_key,
        replicate_seed=seed,
        policy_seed=policy_seed(k),
        path=path,
        sha256=sha,
        stats=stats,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {PROTOCOL} replicates.")
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--format", choices=("mooncake", "weka", "agentic_mooncake"), required=True
    )
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for k in range(args.repeats):
        replicate = materialize_replicate(args.source, args.format, k, args.out_dir)
        print(json.dumps(replicate.to_record(), sort_keys=True))


if __name__ == "__main__":
    main()
