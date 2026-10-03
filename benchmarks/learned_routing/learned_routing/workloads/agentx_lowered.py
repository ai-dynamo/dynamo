# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX as recycled Agentic Mooncake v2 traces, built on AISimulate's own Weka lowering.

README
======

Why (CONTRACT Amendment A3)
---------------------------
Native Weka replay cannot hold a steady AgentX load at N = 16/32: timestamp mode starts every
play's root at t = 0 (aisimulate-core ``weka.rs`` sets ``not_before_ms = t - root_time``), and
``agentic_lanes`` deals the plays to lanes once (``play_index % lanes``) with no recycling. This
module lowers each selected play once with AISim's own lowering, then recycles copies of the
lowered rows into arbitrarily long ``trace_format="agentic_mooncake"`` files.

Pipeline
--------
1. ``build-base`` (once): for each of the 82 selected plays (``CR/traces/agentx/plays``), apply the
   campaign's AgentX transform (``transform.transform_weka`` with ``think_cap_s`` = 300 s and the
   131072 context cap, byte-identical to the plays inside B2's AgentX cells), write the capped play
   to ``CR/traces/agentx_lowered/weka_cap300/<play>.json`` and lower it with
   ``benchmarks/learned_routing/tools/agentx_lower`` (a small Rust CLI over the public
   ``aisimulate_core::replay::loadgen::WekaImporter``, pinned to the crate the bindings link) into
   ``CR/traces/agentx_lowered/base_cap300/<play stem>.jsonl``. The rows are therefore AISim's
   lowering itself, not a reimplementation. For every play the helper also proves that the native
   Weka graph and the base file compile to the same canonical graph digest. ``MANIFEST.json``
   records every hash.
2. ``generate`` (the default command): draw copies, relabel them, and write one agentic file.

Play transforms (cells with ``think_mult`` or ``osl_mult``)
-----------------------------------------------------------
B2's AgentX cells use six play transforms: the plain one (``think_cap_s`` = 300 s) and
``think_mult`` 0.5/2/4 or ``osl_mult`` 1.5/2.5 on top of it. ``PlayTransform`` names one of them
and ``PlayTransform.from_cell(cell["transform"])`` maps a cell to it (keys Weka cannot express are
rejected). Each transform is applied to the raw play by ``transform.transform_weka`` (cap, then
time dilation about the first request, then output scaling and the context clamp; per play, so
byte-identical to the play's line in a B2 cell trace with that transform) and lowered by AISim
into its own directories ``weka_<tag>`` and ``base_<tag>`` (``cap300``, ``cap300_think4``,
``cap300_osl2.5``, ...), each with its own 82-play graph-digest proof. ``GenSpec(think_mult=...,
osl_mult=...)`` and the ``--think-mult`` / ``--osl-mult`` flags select the base; draws, labels,
hash ranges and arrivals do not depend on the transform, so equal specs that differ only in the
transform replay the same plays in the same order. Identity multipliers are left out of the spec
key and of the manifest, so untransformed traces keep their v1 bytes. ``build-base
--all-cell-transforms`` builds every transform the cells use.

Usage::

    export LR_ROOT=CR PYTHONDONTWRITEBYTECODE=1
    PY=WT/.venv/bin/python
    # once (needs the helper: cargo build --release --offline in tools/agentx_lower)
    $PY -m learned_routing.workloads.agentx_lowered build-base --all-cell-transforms
    # open loop: Poisson play arrivals at a TOTAL rate (= per-worker rate x N)
    $PY -m learned_routing.workloads.agentx_lowered --split train --mode open \\
        --num-copies 600 --seed 0 --play-rate-per-s 0.032 --out CR/traces/agentx_lowered/gen/x.jsonl
    # closed loop: replay with agentic_lanes = lanes per worker x N; copies >> lanes
    $PY -m learned_routing.workloads.agentx_lowered --split val --mode closed \\
        --num-copies 400 --seed 0 --out CR/traces/agentx_lowered/gen/y.jsonl
    # a transformed cell, e.g. agentx-T1-think4.0 (test pool, think_mult 4)
    $PY -m learned_routing.workloads.agentx_lowered --split test --mode closed \\
        --num-copies 160 --seed 0 --think-mult 4
    # evidence
    $PY -m learned_routing.workloads.agentx_lowered parity [--think-mult M | --osl-mult M]
    $PY -m learned_routing.workloads.agentx_lowered validate --runs-json SPEC.json

Replay a generated file with ``run_trace_replay(path, trace_format="agentic_mooncake",
execution_model="Qwen/Qwen3-32B", ...)`` and ``arrival_speedup_ratio=1`` (a speedup also divides
dependency delays, i.e. think and tool time). ``trace_block_size`` is ignored for this format: the
header's ``block_size`` (64, the Weka source unit) is used, and the engine re-chunks each prompt
into its own 16-token blocks (``ReplayRequestHashes::from_tokens``), exactly as for native Weka.

Hash-namespace semantics (native, reused unchanged)
---------------------------------------------------
- The lowering namespaces a play by its source name: ``ns = "weka:" + blake3(name)[:16]`` where
  ``name`` is the play's path relative to the corpus root (``<file>#<line:06>`` inside ``.jsonl``).
  Request, session and play ids are ``{ns}:request:<source id>``, ``{ns}:session:<stream>`` and
  ``{ns}:play:<trace id>``.
- A full 64-token source block with source hash ``h`` becomes
  ``u64_le(blake3("weka-full-block\\0source:{name}:{h}\\0{nonce}")[:8])``: the pair
  (relative path, source hash) is the identity, so equal source hashes collide only inside one
  play, never across plays. A missing source hash gets ``private:missing:{request_id}:{i}``; a
  partial tail block gets ``weka-partial-tail\\0{request_id}:{blocks}:{input_length}``, private to
  its request. ``nonce`` resolves u64 collisions within one play.
- The header says ``hash_id_scope: "local"``, but replay interns hash ids GLOBALLY across the
  file (``WorkloadDriver`` maps each distinct u64 to a dense u32 in node order). Isolation between
  plays therefore comes only from the per-play namespace, and copies of one play must get new
  hash ids or they would share KV prefixes.

Recycling semantics (this module)
---------------------------------
- Copy ``i`` (``0 <= i < num_copies``) draws play ``pool[index_draw(seed, "agentx-lowered-draw|"
  + split, i, len(pool))]`` with replacement, where ``pool`` is the sorted play list of the split
  (``CR/cells/SPLIT_MANIFEST.json`` ``agentx_play_subsets``: train 33, val 21, test 28 plays,
  disjoint).
- Copy ``i`` prefixes every ``request_id``, ``session_id``, ``play_id`` and dependency target with
  ``lrx:{i:06d}:`` (the native id stays as the suffix, so the subagent session structure and the
  within-play sort order are untouched), sets ``source_play_ordinal = i``, and maps the k-th
  distinct hash id of the play (first appearance in row order) to ``hash_base_i + k``. The ranges
  ``[hash_base_i, hash_base_i + distinct_i)`` are consecutive and disjoint, so equal hashes stay
  equal within a copy and never collide across copies or plays.
- Dependency edges (relation, trigger, ``delay_ms``), lengths, models and
  ``recorded_api_time_ms`` are copied unchanged.
- Closed mode keeps ``not_before_ms`` as lowered. Replay sorts plays by ``source_play_ordinal``
  and deals them ``ordinal % lanes``; lane ``j`` runs copies ``j, j + L, j + 2L, ...`` one after
  another and rebases each play's timing to its activation (``start + max(nb - root_nb, 0)``), so
  ``num_copies >> lanes`` keeps every lane busy until its list runs out. The draw order fixes the
  deal; nothing depends on play ids.
- Open mode sets ``not_before_ms = a_i + max(nb - root_nb, 0)`` for every row of copy ``i`` (the
  same rebase a lane applies when it activates a play at ``a_i``; descendants keep their offsets,
  which replay treats as floors under the dependency times). ``a_0 = 0`` and
  ``a_i = a_{i-1} - ln(1 - u_i) / rate`` with ``u_i = unit_draw(seed, "agentx-lowered-arrival",
  i)``, rounded to whole nanoseconds like the lowering. The arrival draws do not depend on the split
  or on the rate, so files for different N share their normalized arrival sequence.
- Node order: replay sorts nodes by ``request_id``. That order breaks ties between requests that
  become ready at the same instant and fixes the dense u32 hash interning. Prefixing keeps the
  order inside a copy; across copies it is the copy index, while a native corpus orders plays by
  blake3 of their file names. Neither order is privileged (the effect is that of a crn-order-v1
  replicate), so parity runs pass ``label_ranks`` that reproduce the native order.

Evidence (``CR/facts/agentx_lowered.json``)
-------------------------------------------
- Graph parity: for all 82 plays and each of the six transforms, the native Weka graph and the
  base file compile to the same canonical graph digest; every transformed play that appears in a
  B2 AgentX cell with that transform is byte-identical to its line there. Against ``cap300``, the
  transformed bases keep every id, hash id, input length and dependency edge; ``think_mult``
  scales every offset, delay and api time by exactly ``m``, and ``osl_mult`` changes only output
  lengths.
- Replay parity (``parity``, per transform): native Weka vs lowered agentic files give identical
  ``per_request`` records (every field, including ttft, e2e, worker, lengths and reused tokens)
  in 84 of 84 cases for each transform. The cases are 6 plays covering explicit subagents, joins,
  dispatch-triggered spawns and the zero-output request: each play alone (timestamp mode only),
  the plays as one corpus (timestamp mode and lanes 2, 3), and a 12-copy recycled corpus with
  replacement (timestamp mode and lanes 3, 5), at N = 2 and 4, with seeded
  ``dynamo-default-cost-fn`` and ``round_robin``. The same recycled corpus in copy-index order
  differs (node order only).
- Validation (``validate``): open and closed traces of the train pool run at N = 8, 16, 32 (val
  and test pools spot-checked at N = 16) with every request and play completed, warm-ups of 25 to
  82 minutes of simulated time and a stationary window afterwards at sub-cliff loads. Closed mode
  is stable up to 6 lanes per worker; open mode has a cliff between 0.002 and 0.0025 plays/s per
  worker, where queues and KV thrash grow.
- Suggested loads (provisional SLA I = 40 ms, S = 3; calibration finalizes): closed 4/5/6 lanes
  per worker for L1/L2/L3; open 0.0015/0.002 plays/s per worker for L1/L2 and no stationary open
  L3.
- Cost: generation about 6 s for 56K rows (474 MB); replay about 3-5 ms per row in one process.
- Harness gap: ``learned_routing.cells`` accepts ``agentic_lanes`` only for ``weka`` traces and
  ``replicates`` rejects ``agentic_mooncake``; replicate ``k`` of a lowered cell is the same
  ``GenSpec`` with ``seed = k``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import goodput
from ..e0 import E0Table
from ..slots import SlotPool
from .common import (
    MAX_MODEL_LEN,
    canonical_json,
    index_draw,
    sha256_bytes,
    sha256_file,
    sha256_text,
    unit_draw,
    write_atomic,
    write_json,
)
from .transform import TransformSpec, transform_weka

VERSION = "lr-agentx-lowered-v1"
SPLITS = ("train", "val", "test")
MODES = ("open", "closed")
DEFAULT_THINK_CAP_S = 300.0
LABEL_PREFIX = "lrx"
MAX_COPIES = 1_000_000
AGENTIC_SCHEMA = "dynamo.agentic_mooncake"
AGENTIC_VERSION = 2
DEFAULT_ROOT = Path.home() / "learned-routing"
HELPER_REL = Path("tools/agentx_lower/target/release/agentx-lower")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------
# Paths and play pools


def root_dir(explicit: str | os.PathLike | None = None) -> Path:
    value = explicit or os.environ.get("LR_ROOT") or DEFAULT_ROOT
    return Path(value).resolve()


def cap_tag(think_cap_s: float | None) -> str:
    if think_cap_s is None:
        return "nocap"
    return f"cap{think_cap_s:g}"


# Cell transform keys that change the play graph or its lengths. ``plays``, ``seed`` and
# ``max_model_len`` (fixed at MAX_MODEL_LEN) do not select a base lowering; every other key of
# ``TransformSpec`` is Mooncake-only and must sit at its identity value.
_WEKA_MULTS = ("think_mult", "osl_mult")
_MOONCAKE_ONLY_IDENTITY = {
    "window": None,
    "isl_unique_mult": 1.0,
    "isl_prefix_mult": 1.0,
    "prefix_root_mult": 1,
}


@dataclass(frozen=True)
class PlayTransform:
    """The per-play Weka transform of B2's AgentX cells (``transform.transform_weka``).

    Each distinct transform is lowered into its own base directory, ``base_<tag>``. The tag of
    the plain AgentX transform (``think_cap_s`` only) is ``cap300``; non-identity multipliers
    append ``_think<m>`` and ``_osl<m>`` (e.g. ``cap300_think4``, ``cap300_osl2.5``).
    """

    think_cap_s: float | None = DEFAULT_THINK_CAP_S
    think_mult: float = 1.0
    osl_mult: float = 1.0

    def __post_init__(self) -> None:
        self.spec().validate()

    @classmethod
    def from_cell(cls, transform: dict) -> PlayTransform:
        """The play transform of a cell's ``transform`` object; rejects keys Weka cannot express."""
        for key, identity in _MOONCAKE_ONLY_IDENTITY.items():
            if transform.get(key, identity) != identity:
                raise ValueError(
                    f"{key}={transform[key]!r} is not a Weka play transform"
                )
        max_len = transform.get("max_model_len", MAX_MODEL_LEN)
        if max_len != MAX_MODEL_LEN:
            raise ValueError(f"max_model_len {max_len} != campaign {MAX_MODEL_LEN}")
        cap = transform.get("think_cap_s")
        return cls(
            think_cap_s=None if cap is None else float(cap),
            think_mult=float(transform.get("think_mult", 1.0)),
            osl_mult=float(transform.get("osl_mult", 1.0)),
        )

    def spec(self) -> TransformSpec:
        return TransformSpec(
            think_cap_s=self.think_cap_s,
            think_mult=self.think_mult,
            osl_mult=self.osl_mult,
            max_model_len=MAX_MODEL_LEN,
        )

    def tag(self) -> str:
        tag = cap_tag(self.think_cap_s)
        if self.think_mult != 1.0:
            tag += f"_think{self.think_mult:g}"
        if self.osl_mult != 1.0:
            tag += f"_osl{self.osl_mult:g}"
        return tag

    def mults(self) -> dict:
        """Non-identity multipliers only, so identity keys and digests keep their v1 form."""
        return {k: getattr(self, k) for k in _WEKA_MULTS if getattr(self, k) != 1.0}

    def to_dict(self) -> dict:
        return {**asdict(self), "max_model_len": MAX_MODEL_LEN, "tag": self.tag()}


def lowered_dir(root: Path) -> Path:
    return root / "traces" / "agentx_lowered"


def weka_dir(root: Path, transform: PlayTransform) -> Path:
    return lowered_dir(root) / f"weka_{transform.tag()}"


def base_dir(root: Path, transform: PlayTransform) -> Path:
    return lowered_dir(root) / f"base_{transform.tag()}"


def default_helper() -> Path:
    return PACKAGE_ROOT / HELPER_REL


def split_play_pools(split_manifest: Path) -> dict[str, list[str]]:
    """Sorted, disjoint per-split play pools from B2's ``agentx_play_subsets``."""
    manifest = json.loads(Path(split_manifest).read_text())
    pools: dict[str, list[str]] = {split: [] for split in SPLITS}
    owner: dict[str, str] = {}
    for subset, info in sorted(manifest["agentx_play_subsets"].items()):
        split = info["split"]
        if split not in pools:
            raise ValueError(f"subset {subset}: unknown split {split!r}")
        for play in info["plays"]:
            if owner.setdefault(play, split) != split:
                raise ValueError(f"play {play} is in splits {owner[play]} and {split}")
            pools[split].append(play)
    return {split: sorted(set(plays)) for split, plays in pools.items()}


def subset_plays(
    split_manifest: Path, split: str, subsets: Sequence[str]
) -> tuple[str, ...]:
    """Sorted plays of B2 play subsets (``agentx_play_subsets``), all from ``split``."""
    info = json.loads(Path(split_manifest).read_text())["agentx_play_subsets"]
    plays: set[str] = set()
    for name in subsets:
        if name not in info:
            raise ValueError(f"unknown AgentX play subset {name!r}")
        if info[name]["split"] != split:
            raise ValueError(
                f"subset {name} belongs to split {info[name]['split']!r}, not {split!r}"
            )
        plays.update(info[name]["plays"])
    return tuple(sorted(plays))


def play_stem(play: str) -> str:
    return play[: -len(".json")] if play.endswith(".json") else play


# ---------------------------------------------------------------------------------------------
# Base lowering (AISim's own, through the Rust helper)


def capped_play_bytes(raw_play: dict, name: str, transform: PlayTransform) -> bytes:
    """The play exactly as B2's AgentX cells replay it (one line of their derived ``.jsonl``).

    ``transform_weka`` transforms each play independently, so one play alone gives the same
    bytes as its line inside a multi-play cell trace with the same transform.
    """
    plays, _ = transform_weka([(name, raw_play)], transform.spec())
    return json.dumps(plays[0], separators=(",", ":")).encode()


def _helper_json(helper: Path, *args: str) -> dict:
    out = subprocess.run(
        [str(helper), *args], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def base_stats(rows: Sequence[dict]) -> dict:
    relations = Counter()
    for row in rows:
        for dep in row.get("dependencies") or []:
            relations[f"{dep['relation']}/{dep['trigger']}"] += 1
    distinct = len({h for row in rows for h in row["hash_ids"]})
    return {
        "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "subagent_sessions": len(
            {
                row["session_id"]
                for row in rows
                if ":session:subagent:" in row["session_id"]
            }
        ),
        "roots": sum(1 for row in rows if not row.get("dependencies")),
        "zero_outputs": sum(1 for row in rows if row["output_length"] == 0),
        "distinct_hashes": distinct,
        "dependency_edges": dict(sorted(relations.items())),
        "span_ms": max(row["not_before_ms"] for row in rows),
    }


def build_base(
    root: Path,
    helper: Path,
    transform: PlayTransform = PlayTransform(),
    plays: Sequence[str] | None = None,
) -> dict:
    """Write transformed Weka plays, AISim-lowered base rows and ``MANIFEST.json``; return the
    manifest. Every transform gets its own ``weka_<tag>`` and ``base_<tag>`` directories.
    """
    pools = split_play_pools(root / "cells" / "SPLIT_MANIFEST.json")
    split_of = {play: split for split, names in pools.items() for play in names}
    names = sorted(plays) if plays is not None else sorted(split_of)
    wdir, bdir = weka_dir(root, transform), base_dir(root, transform)
    wdir.mkdir(parents=True, exist_ok=True)
    bdir.mkdir(parents=True, exist_ok=True)
    source_dir = root / "traces" / "agentx" / "plays"
    entries = {}
    for name in names:
        raw_path = source_dir / name
        capped = capped_play_bytes(json.loads(raw_path.read_text()), name, transform)
        weka_path = wdir / name
        if not weka_path.exists() or weka_path.read_bytes() != capped:
            write_atomic(weka_path, capped)
        base_path = bdir / f"{play_stem(name)}.jsonl"
        lowered = _helper_json(helper, "lower", str(weka_path), str(base_path))
        native = _helper_json(helper, "digest-weka", str(weka_path))
        agentic = _helper_json(helper, "digest-agentic", str(base_path))
        header, rows = read_agentic(base_path)
        entries[name] = {
            "split": split_of.get(name),
            "raw_sha256": sha256_file(raw_path),
            "weka_sha256": sha256_bytes(capped),
            "base_sha256": sha256_file(base_path),
            "namespace": rows[0]["request_id"].split(":request:")[0],
            "header": header,
            "importer": {
                k: lowered[k]
                for k in ("requests", "raw_zero_outputs", "nested_timestamp_basis")
            },
            "graph_digest_weka": native["graph_digest"],
            "graph_digest_agentic": agentic["graph_digest"],
            "graph_digest_equal": native["graph_digest"] == agentic["graph_digest"]
            and native["nodes"] == agentic["nodes"],
            **base_stats(rows),
        }
    manifest = {
        "schema": "learned-routing.agentx-lowered-base.v1",
        "version": VERSION,
        "think_cap_s": transform.think_cap_s,
        # absent for identity multipliers, so the plain cap300 manifest keeps its v1 bytes
        **transform.mults(),
        "b2_cell_trace_check": check_against_cells(root, transform),
        "max_model_len": MAX_MODEL_LEN,
        "helper": {
            "binary": str(helper),
            "binary_sha256": sha256_file(helper),
            "crate": "aisimulate-core =0.13.0-dev.202609300000000061",
            "source": "benchmarks/learned_routing/tools/agentx_lower",
        },
        "weka_dir": str(wdir),
        "base_dir": str(bdir),
        "plays": entries,
        "all_graph_digests_equal": all(
            e["graph_digest_equal"] for e in entries.values()
        ),
    }
    write_json(bdir / "MANIFEST.json", manifest)
    return manifest


def cell_transforms(root: Path) -> dict[PlayTransform, list[str]]:
    """Every play transform used by B2's native AgentX cells, with the cell ids using it."""
    out: dict[PlayTransform, list[str]] = defaultdict(list)
    for cells_file in sorted((root / "cells").glob("*.jsonl")):
        for line in cells_file.read_text().splitlines():
            if not line.strip():
                continue
            cell = json.loads(line)
            if cell.get("family") != "agentx" or cell.get("trace_format") != "weka":
                continue
            out[PlayTransform.from_cell(cell.get("transform") or {})].append(
                cell["cell_id"]
            )
    return dict(out)


def check_against_cells(root: Path, transform: PlayTransform) -> dict:
    """Compare each transformed play with its line in B2's AgentX cell traces that use exactly
    this transform."""
    wdir = weka_dir(root, transform)
    checked, identical, seen = 0, 0, set()
    for cells_file in sorted((root / "cells").glob("*.jsonl")):
        for line in cells_file.read_text().splitlines():
            if not line.strip():
                continue
            cell = json.loads(line)
            spec = cell.get("transform") or {}
            if cell.get("family") != "agentx" or cell.get("trace_format") != "weka":
                continue
            if PlayTransform.from_cell(spec) != transform:
                continue
            trace = root / cell["trace_files"][0].removeprefix("CR/")
            lines = trace.read_bytes().splitlines()
            for index, play in enumerate(spec.get("plays") or []):
                if play in seen or not (wdir / play).exists():
                    continue
                seen.add(play)
                checked += 1
                identical += (wdir / play).read_bytes() == lines[index]
    return {"plays_checked": checked, "byte_identical": identical}


def read_agentic(path: Path) -> tuple[dict, list[dict]]:
    with Path(path).open() as handle:
        lines = [line for line in handle if line.strip()]
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


@dataclass(frozen=True)
class BasePlay:
    name: str
    header: dict
    rows: tuple[dict, ...]


_base_cache: dict[tuple[str, int, int], BasePlay] = {}


def load_base(bdir: Path, name: str) -> BasePlay:
    path = Path(bdir) / f"{play_stem(name)}.jsonl"
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _base_cache:
        header, rows = read_agentic(path)
        _base_cache[key] = BasePlay(name=name, header=header, rows=tuple(rows))
    return _base_cache[key]


# ---------------------------------------------------------------------------------------------
# Relabeling and generation


def copy_label(rank: int) -> str:
    if not 0 <= rank < MAX_COPIES:
        raise ValueError(f"copy rank {rank} outside [0, {MAX_COPIES})")
    return f"{LABEL_PREFIX}:{rank:06d}"


def root_not_before_ms(rows: Sequence[dict]) -> float:
    roots = [row["not_before_ms"] for row in rows if not row.get("dependencies")]
    if not roots:
        raise ValueError("play has no root row")
    return min(roots)


def relabel_copy(
    base: BasePlay,
    *,
    label: str,
    ordinal: int,
    hash_base: int,
    start_ms: float | None,
) -> tuple[list[dict], int]:
    """One isolated copy of ``base``: new ids under ``label``, hash ids ``hash_base + k``."""
    prefix = f"{label}:"
    hash_map: dict[int, int] = {}
    root_nb = root_not_before_ms(base.rows)
    out = []
    for row in base.rows:
        new = dict(row)
        new["request_id"] = prefix + row["request_id"]
        new["play_id"] = prefix + row["play_id"]
        new["session_id"] = prefix + row["session_id"]
        new["source_play_ordinal"] = ordinal
        new["hash_ids"] = [
            hash_map.setdefault(h, hash_base + len(hash_map)) for h in row["hash_ids"]
        ]
        if row.get("dependencies"):
            new["dependencies"] = [
                {**dep, "request_id": prefix + dep["request_id"]}
                for dep in row["dependencies"]
            ]
        if start_ms is not None:
            new["not_before_ms"] = start_ms + max(row["not_before_ms"] - root_nb, 0.0)
        out.append(new)
    return out, len(hash_map)


def draw_plays(
    pool: Sequence[str], split: str, num_copies: int, seed: int
) -> list[str]:
    if not pool:
        raise ValueError(f"empty play pool for split {split!r}")
    label = f"agentx-lowered-draw|{split}"
    return [pool[index_draw(seed, label, i, len(pool))] for i in range(num_copies)]


def poisson_arrivals_ms(num_copies: int, seed: int, rate_per_s: float) -> list[float]:
    """``a_0 = 0``; exponential gaps of mean ``1 / rate``; whole nanoseconds, in milliseconds."""
    if not (math.isfinite(rate_per_s) and rate_per_s > 0):
        raise ValueError(f"play rate must be finite and > 0, got {rate_per_s}")
    arrivals, t_s = [], 0.0
    for i in range(num_copies):
        if i:
            u = unit_draw(seed, "agentx-lowered-arrival", i)
            t_s += -math.log1p(-u) / rate_per_s
        arrivals.append(round(t_s * 1e9) / 1e6)
    return arrivals


def assemble(
    bases: Sequence[BasePlay],
    *,
    starts_ms: Sequence[float | None],
    label_ranks: Sequence[int] | None = None,
    hash_start: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Rows of all copies (copy ``i`` has ordinal ``i``) and one record per copy."""
    if len(starts_ms) != len(bases):
        raise ValueError("starts_ms must have one entry per copy")
    if label_ranks is None:
        label_ranks = range(len(bases))
    if sorted(label_ranks) != list(range(len(bases))):
        raise ValueError("label_ranks must be a permutation of the copy indices")
    rows, copies, next_hash = [], [], hash_start
    for i, (base, start, rank) in enumerate(zip(bases, starts_ms, label_ranks)):
        label = copy_label(rank)
        copy_rows, distinct = relabel_copy(
            base, label=label, ordinal=i, hash_base=next_hash, start_ms=start
        )
        copies.append(
            {
                "ordinal": i,
                "label": label,
                "play": base.name,
                "rows": len(copy_rows),
                "hash_range": [next_hash, next_hash + distinct],
                "start_ms": start,
            }
        )
        rows.extend(copy_rows)
        next_hash += distinct
    rows.sort(key=lambda row: row["request_id"])
    return rows, copies


@dataclass(frozen=True)
class GenSpec:
    split: str
    mode: str
    num_copies: int
    seed: int
    play_rate_per_s: float | None = None
    think_cap_s: float | None = DEFAULT_THINK_CAP_S
    version: str = VERSION
    think_mult: float = 1.0
    osl_mult: float = 1.0
    # Draw from this subset of the split's pool (e.g. one B2 segment such as A1) instead of the
    # whole pool, so segments stay independent workloads (Amendment A4). None = the whole pool.
    plays: tuple[str, ...] | None = None

    @property
    def transform(self) -> PlayTransform:
        return PlayTransform(
            think_cap_s=self.think_cap_s,
            think_mult=self.think_mult,
            osl_mult=self.osl_mult,
        )

    def key_dict(self) -> dict:
        """The spec as hashed into the trace digest; identity multipliers are omitted, so an
        untransformed spec keeps the key (and the file bytes) it had before multipliers existed.
        """
        data = asdict(self)
        for key in _WEKA_MULTS:
            if data[key] == 1.0:
                del data[key]
        if data["plays"] is None:
            del data["plays"]
        else:
            data["plays"] = list(data["plays"])
        return data

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 1 <= self.num_copies <= MAX_COPIES:
            raise ValueError(f"num_copies must be in [1, {MAX_COPIES}]")
        if self.mode == "open" and self.play_rate_per_s is None:
            raise ValueError("open mode needs play_rate_per_s")
        if self.mode == "closed" and self.play_rate_per_s is not None:
            raise ValueError("play_rate_per_s applies to open mode only")
        if self.plays is not None and (
            not self.plays or list(self.plays) != sorted(set(self.plays))
        ):
            raise ValueError("plays must be a non-empty, sorted, duplicate-free subset")
        self.transform.spec().validate()


def generate(
    spec: GenSpec, root: Path, pools: dict[str, list[str]] | None = None
) -> tuple[dict, list[dict], dict]:
    """Header, rows and metadata of one recycled lowered trace (no IO besides reading bases)."""
    spec.validate()
    transform = spec.transform
    bdir = base_dir(root, transform)
    manifest_path = bdir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path}: no base lowering for transform {transform.tag()}; run "
            f"build-base --think-cap-s {transform.think_cap_s} --think-mult "
            f"{transform.think_mult:g} --osl-mult {transform.osl_mult:g}"
        )
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("all_graph_digests_equal"):
        raise ValueError(f"{manifest_path}: base lowering parity is not established")
    built = (
        manifest.get("think_cap_s"),
        *(manifest.get(k, 1.0) for k in _WEKA_MULTS),
    )
    if built != (transform.think_cap_s, transform.think_mult, transform.osl_mult):
        raise ValueError(f"{manifest_path}: built for {built}, spec asks {transform}")
    if pools is None:
        pools = split_play_pools(root / "cells" / "SPLIT_MANIFEST.json")
    pool = pools[spec.split]
    if spec.plays is not None:
        outside = [p for p in spec.plays if p not in pool]
        if outside:
            raise ValueError(
                f"plays {outside[:3]} are not in the {spec.split} pool (split purity, A4)"
            )
        pool = list(spec.plays)
    missing = [p for p in pool if p not in manifest["plays"]]
    if missing:
        raise ValueError(f"base rows missing for {missing[:3]}")
    names = draw_plays(pool, spec.split, spec.num_copies, spec.seed)
    bases = [load_base(bdir, name) for name in names]
    if spec.mode == "open":
        starts: list[float | None] = list(
            poisson_arrivals_ms(spec.num_copies, spec.seed, spec.play_rate_per_s)
        )
    else:
        starts = [None] * spec.num_copies
    rows, copies = assemble(bases, starts_ms=starts)
    first = bases[0].header
    if {(b.header["block_size"], b.header["hash_id_scope"]) for b in bases} != {
        (first["block_size"], first["hash_id_scope"])
    }:
        raise ValueError("base plays disagree on block_size or hash_id_scope")
    spec_key = sha256_text(
        canonical_json(
            {"spec": spec.key_dict(), "base_manifest": sha256_file(manifest_path)}
        )
    )
    header = {
        "schema": AGENTIC_SCHEMA,
        "version": AGENTIC_VERSION,
        "block_size": first["block_size"],
        "hash_id_scope": first["hash_id_scope"],
        "source": {"format": "weka", "digest": f"{VERSION}:{spec_key}"},
    }
    meta = {
        "schema": "learned-routing.agentx-lowered-trace.v1",
        "version": VERSION,
        "spec": asdict(spec),
        "spec_key": spec_key,
        "transform": transform.to_dict(),
        "base_manifest": str(manifest_path),
        "base_manifest_sha256": sha256_file(manifest_path),
        "pool": pool,
        "pool_size": len(pool),
        "draw_counts": dict(sorted(Counter(names).items())),
        "rows": len(rows),
        "copies": copies,
        "hash_ids_used": copies[-1]["hash_range"][1] - copies[0]["hash_range"][0],
        "last_start_ms": starts[-1] if spec.mode == "open" else None,
        "replay": {
            "trace_format": "agentic_mooncake",
            "arrival_speedup_ratio": 1.0,
            "agentic_lanes": "lanes per worker x N (closed mode only)",
            "header_block_size": first["block_size"],
        },
    }
    return header, rows, meta


def agentic_bytes(header: dict, rows: Sequence[dict]) -> bytes:
    lines = [json.dumps(header, separators=(",", ":"))]
    lines += [json.dumps(row, separators=(",", ":")) for row in rows]
    return ("\n".join(lines) + "\n").encode()


def write_generated(spec: GenSpec, root: Path, out: Path | None = None) -> dict:
    header, rows, meta = generate(spec, root)
    data = agentic_bytes(header, rows)
    sha = sha256_bytes(data)
    out = Path(out) if out is not None else lowered_dir(root) / "gen" / f"{sha}.jsonl"
    write_atomic(out, data)
    meta["path"] = str(out)
    meta["sha256"] = sha
    write_json(out.with_name(out.name + ".meta.json"), meta)
    return meta


# ---------------------------------------------------------------------------------------------
# Replay helpers (parity and validation; dynamo imported lazily)


@dataclass
class Replayer:
    root: Path
    out_dir: Path

    def __post_init__(self) -> None:
        self.engine = json.loads((self.root / "config" / "engine.json").read_text())
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.default_yaml = self.out_dir / "policy-dynamo-default-cost-fn-seed1.yaml"
        write_atomic(
            self.default_yaml,
            b"worker_selection:\n  aggregated: candidate\n  instances:\n"
            b"    - name: candidate\n      type: dynamo-default-cost-fn\n"
            b"      parameters:\n        seed: 1\n",
        )

    def run(
        self,
        trace: Path,
        trace_format: str,
        num_workers: int,
        policy: str,
        *,
        lanes: int | None = None,
        telemetry_ms: float | None = None,
        label: str = "agentx_lowered",
    ) -> dict:
        # Generation must work without the bindings; only replays import them.
        from dynamo.llm import KvRouterConfig
        from dynamo.mocker import MockEngineArgs
        from dynamo.replay import TelemetryOptions, run_trace_replay

        kwargs = {
            "extra_engine_args": MockEngineArgs.from_json(
                json.dumps(self.engine["mock_engine_args"])
            ),
            "num_workers": num_workers,
            "trace_format": trace_format,
            "execution_model": self.engine["model"],
            "capture_per_request": True,
        }
        if policy == "round_robin":
            kwargs.update(router_mode="round_robin", router_config=None)
        elif policy == "default-seed1":
            kwargs.update(
                router_mode="kv_router",
                router_config=KvRouterConfig(
                    router_policy_config=str(self.default_yaml)
                ),
            )
        else:
            raise ValueError(f"unknown policy {policy!r}")
        if lanes is not None:
            kwargs["agentic_lanes"] = lanes
        if telemetry_ms is not None:
            kwargs["telemetry_options"] = TelemetryOptions(
                sample_interval_ms=telemetry_ms
            )
        with SlotPool(self.root / "slots").slot(label=label):
            started = time.monotonic()
            report = run_trace_replay(str(trace), **kwargs)
            wall_s = time.monotonic() - started
        result = {
            "summary": dict(report.summary),
            "per_request": [dict(r) for r in report.per_request or []],
            "wall_s": wall_s,
        }
        if telemetry_ms is not None and report.telemetry is not None:
            result["telemetry"] = list(report.telemetry.samples)
        return result


def _map_strings(value, rename: Callable[[str], str]):
    if isinstance(value, str):
        return rename(value)
    if isinstance(value, list):
        return [_map_strings(v, rename) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, rename) for k, v in value.items()}
    return value


def prefix_renamer(pairs: dict[str, str]) -> Callable[[str], str]:
    """Rewrite strings starting with a known prefix (longest first)."""
    ordered = sorted(pairs.items(), key=lambda kv: -len(kv[0]))

    def rename(text: str) -> str:
        for old, new in ordered:
            if text.startswith(old):
                return new + text[len(old) :]
        return text

    return rename


def compare_per_request(
    reference: Sequence[dict],
    candidate: Sequence[dict],
    rename: Callable[[str], str] | None = None,
    ignore_fields: Sequence[str] = (),
) -> dict:
    """Exact comparison of every per_request field after mapping candidate ids to reference ids."""
    rename = rename or (lambda text: text)
    ref = {
        row["request_id"]: {k: v for k, v in row.items() if k not in ignore_fields}
        for row in reference
    }
    cand = {}
    for row in candidate:
        mapped = _map_strings(row, rename)
        cand[mapped["request_id"]] = {
            k: v for k, v in mapped.items() if k not in ignore_fields
        }
    differing, fields = 0, Counter()
    for request_id in ref.keys() & cand.keys():
        a, b = ref[request_id], cand[request_id]
        if a != b:
            differing += 1
            fields.update(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    return {
        "reference_rows": len(ref),
        "candidate_rows": len(cand),
        "missing": len(ref.keys() - cand.keys()),
        "extra": len(cand.keys() - ref.keys()),
        "differing_rows": differing,
        "differing_fields": dict(fields.most_common()),
        "identical": len(ref) == len(cand)
        and not (ref.keys() ^ cand.keys())
        and differing == 0,
        "fields_compared": sorted({k for row in reference for k in row}),
    }


# ---------------------------------------------------------------------------------------------
# Parity


DEFAULT_PARITY_PLAYS = (
    # 1 explicit subagent lowered to 19 subagent streams: join, replay barriers, both spawn triggers
    "0098-3e8b931fd474375ea08ae58872aea7a8e0f5.json",
    # 6 explicit subagents, 5 dispatch-triggered (concurrent, non-blocking) spawns
    "0073-2a2da059b7425d9dc1f999fca1177bc1cdb9.json",
    # 5 explicit subagents with joins and dispatch spawns
    "0390-fe1121fb09adada530be4b8c8568a7ee6758.json",
    # 3 explicit subagents, 119 rows
    "0013-07dd40536557a1d6440a923557c3129dc929.json",
    # the only authored zero-output (prefill-only) request, plus a dispatch spawn
    "0021-0bed163ac9566086e14ffd85c7481096d49d.json",
    # no explicit subagent, 10 sessions from 4 concurrent detected streams
    "0315-c7cdef4e9eb74c966d9e8dfe982319c9806d.json",
)


def _native_namespaces_by_ordinal(rows: Sequence[dict]) -> dict[int, str]:
    out = {}
    for row in rows:
        out[row["source_play_ordinal"]] = row["request_id"].split(":request:")[0]
    return out


def _ranks(keys: Sequence[str]) -> list[int]:
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = [0] * len(keys)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def run_parity(
    root: Path,
    helper: Path,
    plays: Sequence[str],
    out_dir: Path,
    *,
    workers: Sequence[int] = (2, 4),
    policies: Sequence[str] = ("default-seed1", "round_robin"),
    recycle_copies: int = 12,
    recycle_seed: int = 0,
    transform: PlayTransform = PlayTransform(),
) -> dict:
    """Native Weka vs lowered agentic replays, per_request field for field, for the plays of
    one transform (``weka_<tag>`` against ``base_<tag>``).

    - ``single``: each play alone, timestamp mode only: native transformed play vs the helper's
      base file (ids identical) vs one relabeled copy (ids mapped back).
    - ``corpus``: the plays as one native directory vs its helper lowering vs relabeled copies
      ordered like the native ids; timestamp mode and lanes < plays (sequential lanes).
    - ``recycle``: ``recycle_copies`` seeded draws with replacement. The native reference is a
      directory holding one file per copy (distinct names, so distinct namespaces, which is
      native copy isolation); the candidate is this module's recycling of the base rows with
      disjoint hash ranges; lanes < copies, so lanes run several copies each.
    """
    replayer = Replayer(root, out_dir)
    wdir, bdir = weka_dir(root, transform), base_dir(root, transform)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    fields_seen: set[str] = set()

    def check(
        case: dict,
        ref_path: Path,
        ref_fmt: str,
        candidates: list[tuple[str, Path, Callable | None]],
        lanes_list,
    ):
        for n in workers:
            for policy in policies:
                for lanes in lanes_list:
                    ref = replayer.run(
                        ref_path,
                        ref_fmt,
                        n,
                        policy,
                        lanes=lanes,
                        label="agentx_lowered-parity",
                    )
                    for cand_name, cand_path, rename in candidates:
                        res = replayer.run(
                            cand_path,
                            "agentic_mooncake",
                            n,
                            policy,
                            lanes=lanes,
                            label="agentx_lowered-parity",
                        )
                        cmp = compare_per_request(
                            ref["per_request"], res["per_request"], rename
                        )
                        fields_seen.update(cmp["fields_compared"])
                        loose = compare_per_request(
                            ref["per_request"],
                            res["per_request"],
                            rename,
                            ignore_fields=("uuid",),
                        )
                        cases.append(
                            {
                                **case,
                                "differing_rows_ignoring_uuid": loose["differing_rows"],
                                "differing_fields_ignoring_uuid": loose[
                                    "differing_fields"
                                ],
                                **{
                                    f"{side}_{key}": run["summary"].get(key)
                                    for side, run in (
                                        ("reference", ref),
                                        ("candidate", res),
                                    )
                                    for key in (
                                        "mean_ttft_ms",
                                        "mean_e2e_latency_ms",
                                        "mean_itl_ms",
                                    )
                                },
                                "candidate": cand_name,
                                "num_workers": n,
                                "policy": policy,
                                "agentic_lanes": lanes,
                                "reference_wall_s": ref["wall_s"],
                                "candidate_wall_s": res["wall_s"],
                                "reference_completed": ref["summary"].get(
                                    "completed_requests"
                                ),
                                "candidate_completed": res["summary"].get(
                                    "completed_requests"
                                ),
                                "reference_duration_ms": ref["summary"].get(
                                    "duration_ms"
                                ),
                                "candidate_duration_ms": res["summary"].get(
                                    "duration_ms"
                                ),
                                **{
                                    k: cmp[k]
                                    for k in (
                                        "reference_rows",
                                        "candidate_rows",
                                        "missing",
                                        "extra",
                                        "differing_rows",
                                        "differing_fields",
                                        "identical",
                                    )
                                },
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    k: cases[-1][k]
                                    for k in (
                                        "tier",
                                        "name",
                                        "candidate",
                                        "num_workers",
                                        "policy",
                                        "agentic_lanes",
                                        "identical",
                                        "reference_rows",
                                    )
                                }
                            ),
                            flush=True,
                        )

    # single plays
    for name in plays:
        base = load_base(bdir, name)
        relabeled_rows, _ = assemble([base], starts_ms=[None])
        header = dict(base.header)
        single_path = out_dir / "single" / f"{play_stem(name)}.relabeled.jsonl"
        write_atomic(single_path, agentic_bytes(header, relabeled_rows))
        rename = prefix_renamer({f"{copy_label(0)}:": ""})
        check(
            {"tier": "single", "name": name},
            wdir / name,
            "weka",
            [
                ("base", bdir / f"{play_stem(name)}.jsonl", None),
                ("relabeled", single_path, rename),
            ],
            [None],
        )
    # corpus of the plays (native directory)
    corpus_dir = out_dir / "corpus" / "native"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for name in plays:
        shutil.copyfile(wdir / name, corpus_dir / name)
    corpus_lowered = out_dir / "corpus" / "helper_lowered.jsonl"
    _helper_json(helper, "lower", str(corpus_dir), str(corpus_lowered))
    _, corpus_rows = read_agentic(corpus_lowered)
    native_ns = _native_namespaces_by_ordinal(corpus_rows)
    ordered = sorted(plays)  # native corpus order = sorted relative paths
    bases = [load_base(bdir, name) for name in ordered]
    ranks = _ranks([native_ns[i] for i in range(len(ordered))])
    rows, copies = assemble(bases, starts_ms=[None] * len(bases), label_ranks=ranks)
    corpus_relabeled = out_dir / "corpus" / "relabeled.jsonl"
    write_atomic(corpus_relabeled, agentic_bytes(bases[0].header, rows))
    rename = prefix_renamer({f"{c['label']}:": "" for c in copies})
    lanes_list = [None] + [lanes for lanes in (2, 3) if lanes < len(plays)]
    check(
        {"tier": "corpus", "name": f"{len(plays)} plays"},
        corpus_dir,
        "weka",
        [
            ("helper_lowered", corpus_lowered, None),
            ("relabeled", corpus_relabeled, rename),
        ],
        lanes_list,
    )
    # recycling with replacement: native corpus of per-copy files vs module recycling
    draws = draw_plays(sorted(plays), "parity", recycle_copies, recycle_seed)
    recycle_dir = out_dir / "recycle" / "native"
    recycle_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(draws):
        shutil.copyfile(wdir / name, recycle_dir / f"c{i:04d}-{name}")
    recycle_lowered = out_dir / "recycle" / "helper_lowered.jsonl"
    _helper_json(helper, "lower", str(recycle_dir), str(recycle_lowered))
    _, recycle_rows = read_agentic(recycle_lowered)
    native_ns = _native_namespaces_by_ordinal(recycle_rows)
    bases = [load_base(bdir, name) for name in draws]
    ranks = _ranks([native_ns[i] for i in range(len(draws))])
    rows, copies = assemble(bases, starts_ms=[None] * len(bases), label_ranks=ranks)
    recycle_relabeled = out_dir / "recycle" / "relabeled.jsonl"
    write_atomic(recycle_relabeled, agentic_bytes(bases[0].header, rows))
    pairs = {}
    for copy in copies:
        base_ns = (
            load_base(bdir, copy["play"]).rows[0]["request_id"].split(":request:")[0]
        )
        pairs[f"{copy['label']}:{base_ns}"] = native_ns[copy["ordinal"]]
    rename = prefix_renamer(pairs)
    # also the generator's own label order (copy index), to measure tie-order sensitivity
    rows_own, _ = assemble(bases, starts_ms=[None] * len(bases))
    recycle_own = out_dir / "recycle" / "relabeled_copy_order.jsonl"
    write_atomic(recycle_own, agentic_bytes(bases[0].header, rows_own))
    own_pairs = {}
    for i, name in enumerate(draws):
        base_ns = load_base(bdir, name).rows[0]["request_id"].split(":request:")[0]
        own_pairs[f"{copy_label(i)}:{base_ns}"] = native_ns[i]
    check(
        {
            "tier": "recycle",
            "name": f"{recycle_copies} copies of {len(set(draws))} plays (seed {recycle_seed})",
            "draws": draws,
        },
        recycle_dir,
        "weka",
        [
            ("relabeled", recycle_relabeled, rename),
            ("relabeled_copy_order", recycle_own, prefix_renamer(own_pairs)),
        ],
        [None, 3, 5],
    )
    exact = [c for c in cases if c["candidate"] != "relabeled_copy_order"]
    result = {
        "transform": transform.to_dict(),
        "plays": list(plays),
        "workers": list(workers),
        "policies": list(policies),
        "cases": cases,
        "exact_cases": len(exact),
        "exact_identical": sum(c["identical"] for c in exact),
        "all_exact_identical": all(c["identical"] for c in exact),
        "copy_order_cases": len(cases) - len(exact),
        "copy_order_identical": sum(
            c["identical"] for c in cases if c["candidate"] == "relabeled_copy_order"
        ),
        "fields_compared": sorted(fields_seen),
    }
    write_json(out_dir / "parity.json", result)
    return result


# ---------------------------------------------------------------------------------------------
# Steady-state analysis of one replay


def _occupancy(
    intervals: Sequence[tuple[float, float]], bin_ms: float, n_bins: int
) -> list[float]:
    """Time-averaged number of open intervals per bin (exact, partial bins included)."""
    diff = [0.0] * (n_bins + 2)
    partial = [0.0] * (n_bins + 1)
    for start, end in intervals:
        if end <= start:
            continue
        b0, b1 = int(start // bin_ms), int(end // bin_ms)
        b0, b1 = min(b0, n_bins), min(b1, n_bins)
        if b0 == b1:
            if b0 < n_bins:
                partial[b0] += (end - start) / bin_ms
            continue
        if b0 < n_bins:
            partial[b0] += ((b0 + 1) * bin_ms - start) / bin_ms
        if b1 < n_bins:
            partial[b1] += (end - b1 * bin_ms) / bin_ms
        diff[b0 + 1] += 1.0
        diff[b1] -= 1.0
    out, running = [], 0.0
    for b in range(n_bins):
        running += diff[b]
        out.append(running + partial[b])
    return out


def _merge(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) / mean


def steady_state(
    per_request: Sequence[dict],
    *,
    num_workers: int,
    mode: str,
    bin_s: float = 60.0,
    telemetry: Sequence[dict] | None = None,
    play_span_ms: Callable[[str], float | None] | None = None,
) -> dict:
    """In-flight plays/requests and per-worker utilization over time, plus a steady window.

    Window end: open mode, the last play-root arrival (arrivals stop); closed mode, the first time
    a lane runs out of copies (``min`` over lanes of its last play's end).

    Warm-up end (suggested rules; calibration fixes the final one):

    - open: the first bin whose 5-bin trailing mean of in-flight plays reaches 90% of the median
      bin over the second half of ``[0, window end)`` (Little's-law plateau);
    - closed: every lane starts its first copy at t = 0 in the same phase; warm-up ends when 90%
      of the lanes have finished their first copy (p90 over lanes of the first play's end), after
      which the lanes are desynchronized.

    ``stationary`` compares the first and last thirds of the window: in-flight plays (open) or
    in-flight requests (closed) must agree within ``max(15%, 2 * noise)``, where ``noise`` is the
    relative sd of a third's mean for a Poisson count of ``m`` items renewed every mean sojourn
    (``1 / sqrt(m * third / sojourn)``); otherwise the load is drifting (overload or too short).
    With ``play_span_ms`` (recorded span of a play, by play id), the play stretch
    ``sum(sojourn) / sum(recorded span)`` of plays starting in the last third must also stay
    within the same tolerance of the first third's: a growing stretch means queues are growing.
    """
    bin_ms = bin_s * 1000.0
    done = [r for r in per_request if r.get("terminal_time_ms") is not None]
    makespan = max(r["terminal_time_ms"] for r in done)
    n_bins = int(math.ceil(makespan / bin_ms)) or 1
    plays: dict[str, list[dict]] = defaultdict(list)
    for r in per_request:
        plays[r["play_id"]].append(r)
    play_iv, root_arrivals, lane_of, play_end = [], [], {}, {}
    for pid, rows in plays.items():
        start = min(r["arrival_time_ms"] for r in rows)
        end = max((r["terminal_time_ms"] or start) for r in rows)
        play_iv.append((start, end))
        play_end[pid] = end
        root_arrivals.append(start)
        lane = (rows[0].get("agentic") or {}).get("lane_id")
        if lane is not None:
            lane_of[pid] = lane
    req_iv = [(r["arrival_time_ms"], r["terminal_time_ms"]) for r in done]
    run_iv = defaultdict(list)
    for r in done:
        if (
            r.get("first_admit_ms") is not None
            and r.get("decode_worker_idx") is not None
        ):
            run_iv[r["decode_worker_idx"]].append(
                (r["first_admit_ms"], r["terminal_time_ms"])
            )
    plays_bins = _occupancy(play_iv, bin_ms, n_bins)
    reqs_bins = _occupancy(req_iv, bin_ms, n_bins)
    worker_active = {
        w: _occupancy(run_iv.get(w, []), bin_ms, n_bins) for w in range(num_workers)
    }
    worker_busy = {
        w: _occupancy(_merge(run_iv.get(w, [])), bin_ms, n_bins)
        for w in range(num_workers)
    }
    if mode == "open":
        window_end = max(root_arrivals)
    elif lane_of:
        lane_last = defaultdict(float)
        for pid, lane in lane_of.items():
            lane_last[lane] = max(lane_last[lane], play_end[pid])
        window_end = min(lane_last.values())
    else:
        window_end = min(play_end.values())
    end_bin = max(1, min(n_bins, int(window_end // bin_ms)))

    def plateau_bin(series: Sequence[float]) -> int | None:
        second = sorted(series[end_bin // 2 : end_bin])
        if not second:
            return None
        target = 0.9 * second[len(second) // 2]
        for b in range(end_bin):
            if _mean(series[max(0, b - 4) : b + 1]) >= target:
                return b
        return None

    if mode == "open":
        warm_end_bin = plateau_bin(plays_bins)
    else:
        first_end = defaultdict(lambda: math.inf)
        first_start = defaultdict(lambda: math.inf)
        for pid, lane in lane_of.items():
            start = min(r["arrival_time_ms"] for r in plays[pid])
            if start < first_start[lane]:
                first_start[lane], first_end[lane] = start, play_end[pid]
        ends = sorted(first_end.values())
        warm_end_bin = (
            int(
                math.ceil(
                    ends[min(len(ends) - 1, int(math.ceil(0.9 * len(ends))) - 1)]
                    / bin_ms
                )
            )
            if ends
            else None
        )
    window = None
    if warm_end_bin is not None and warm_end_bin < end_bin:
        lo, hi = warm_end_bin, end_bin
        thirds = [lo + (hi - lo) * k // 3 for k in range(4)]
        per_worker_active = [_mean(worker_active[w][lo:hi]) for w in range(num_workers)]
        per_worker_busy = [_mean(worker_busy[w][lo:hi]) for w in range(num_workers)]
        drift_series = plays_bins if mode == "open" else reqs_bins
        first_third, last_third = _mean(drift_series[thirds[0] : thirds[1]]), _mean(
            drift_series[thirds[2] : thirds[3]]
        )
        drift = last_third / first_third if first_third else None
        started = [
            play_end[pid] - min(r["arrival_time_ms"] for r in plays[pid])
            for pid in plays
            if lo * bin_ms
            <= min(r["arrival_time_ms"] for r in plays[pid])
            < hi * bin_ms
        ]
        sojourn_ms = _mean(started)
        level = _mean(drift_series[lo:hi]) or 0.0
        third_ms = (hi - lo) * bin_ms / 3.0
        renewals = max(1.0, third_ms / sojourn_ms) if sojourn_ms else 1.0
        noise = 1.0 / math.sqrt(level * renewals) if level > 0 else None
        tolerance = max(0.15, 2.0 * noise) if noise is not None else 0.15

        def stretch(a: int, b: int) -> float | None:
            if play_span_ms is None:
                return None
            sojourn = spans = 0.0
            for pid, rows in plays.items():
                start = min(r["arrival_time_ms"] for r in rows)
                span = play_span_ms(pid)
                if a * bin_ms <= start < b * bin_ms and span:
                    sojourn += play_end[pid] - start
                    spans += span
            return sojourn / spans if spans else None

        stretch_thirds = [stretch(a, b) for a, b in zip(thirds, thirds[1:])]
        stretch_drift = (
            stretch_thirds[2] / stretch_thirds[0]
            if None not in stretch_thirds and stretch_thirds[0]
            else None
        )
        stationary = drift is not None and abs(drift - 1.0) <= tolerance
        if stretch_drift is not None:
            stationary = stationary and abs(stretch_drift - 1.0) <= tolerance
        window = {
            "stationary": stationary,
            "drift_last_over_first_third": drift,
            "play_stretch": stretch(lo, hi),
            "play_stretch_by_third": stretch_thirds,
            "play_stretch_drift": stretch_drift,
            "drift_tolerance": tolerance,
            "plays_started_in_window": len(started),
            "play_sojourn_mean_s": None if sojourn_ms is None else sojourn_ms / 1000.0,
            "littles_law_plays": (
                None
                if sojourn_ms is None
                else len(started) / ((hi - lo) * bin_ms) * sojourn_ms
            ),
            "start_s": lo * bin_s,
            "end_s": hi * bin_s,
            "length_s": (hi - lo) * bin_s,
            "in_flight_plays_mean": _mean(plays_bins[lo:hi]),
            "in_flight_requests_mean": _mean(reqs_bins[lo:hi]),
            "in_flight_requests_cv_bins": _cv(reqs_bins[lo:hi]),
            "in_flight_requests_by_third": [
                _mean(reqs_bins[a:b]) for a, b in zip(thirds, thirds[1:])
            ],
            "in_flight_plays_by_third": [
                _mean(plays_bins[a:b]) for a, b in zip(thirds, thirds[1:])
            ],
            "worker_active_requests_mean": _mean(per_worker_active),
            "worker_active_requests_min_max": [
                min(per_worker_active),
                max(per_worker_active),
            ],
            "worker_busy_frac_mean": _mean(per_worker_busy),
            "worker_busy_frac_min_max": [min(per_worker_busy), max(per_worker_busy)],
            "requests_arriving_in_window": sum(
                1
                for r in per_request
                if lo * bin_ms <= r["arrival_time_ms"] < hi * bin_ms
            ),
            "requests_completing_in_window": sum(
                1 for r in done if lo * bin_ms <= r["terminal_time_ms"] < hi * bin_ms
            ),
        }
        if telemetry:
            usage = defaultdict(list)
            running = defaultdict(list)
            for sample in telemetry:
                at = sample.get("sampled_at_ms", 0.0)
                if (
                    sample.get("kind") != "periodic"
                    or not lo * bin_ms <= at < hi * bin_ms
                ):
                    continue
                for m in sample.get("decode_scheduler_metrics") or []:
                    usage[m["worker_id"]].append(m["active_cache_usage"])
                    running[m["worker_id"]].append(m["running_requests"])
            if usage:
                means = [_mean(v) for v in usage.values()]
                window["telemetry_active_kv_usage_mean"] = _mean(means)
                window["telemetry_active_kv_usage_min_max_worker"] = [
                    min(means),
                    max(means),
                ]
                window["telemetry_running_requests_mean"] = _mean(
                    [_mean(v) for v in running.values()]
                )
                window["telemetry_samples_per_worker"] = min(
                    len(v) for v in usage.values()
                )
    stride = max(1, n_bins // 60)
    return {
        "bin_s": bin_s,
        "makespan_s": makespan / 1000.0,
        "window_end_rule": "last root arrival"
        if mode == "open"
        else "first lane exhausted",
        "window_end_s": window_end / 1000.0,
        "warmup_end_s": None if warm_end_bin is None else warm_end_bin * bin_s,
        "steady_window": window,
        "plays": len(plays),
        "requests": len(per_request),
        "completed": sum(
            1 for r in per_request if r.get("terminal_status") == "completed"
        ),
        "series_every_bins": stride,
        "series": {
            "t_s": [b * bin_s for b in range(0, n_bins, stride)],
            "in_flight_plays": [
                round(plays_bins[b], 3) for b in range(0, n_bins, stride)
            ],
            "in_flight_requests": [
                round(reqs_bins[b], 3) for b in range(0, n_bins, stride)
            ],
            "worker_active_mean": [
                round(_mean([worker_active[w][b] for w in range(num_workers)]), 3)
                for b in range(0, n_bins, stride)
            ],
        },
    }


SLA_GRID_ITL_MS = (25.0, 30.0, 40.0, 50.0, 60.0)
SLA_GRID_SLOWDOWN = (1.5, 2.0, 3.0, None)


def band_table(
    per_request: Sequence[dict], window: dict | None, mode: str, e0
) -> dict | None:
    """A2 good fraction in the steady window over a grid of provisional (I, S) values.

    Open mode scores requests that become ready inside the window (arrival basis), closed mode
    requests that complete inside it (completion basis), following ``learned_routing.goodput``.
    """
    if window is None:
        return None
    lo, hi = window["start_s"] * 1000.0, window["end_s"] * 1000.0
    key = "arrival_time_ms" if mode == "open" else "terminal_time_ms"
    rows = [
        goodput.compact_row(r)
        for r in per_request
        if r.get(key) is not None and lo <= r[key] < hi
    ]
    if not rows:
        return None
    done = [r for r in rows if goodput.completed(r)]
    e0.ensure((r["input_length"], r["output_length"]) for r in done)
    itl = sorted(v for r in done if (v := goodput.mean_itl_ms(r)) is not None)
    ratio = sorted(
        r["e2e_latency_ms"] / e0(r["input_length"], r["output_length"]) for r in done
    )
    table = {}
    for itl_ms in SLA_GRID_ITL_MS:
        for slowdown in SLA_GRID_SLOWDOWN:
            sla = {"itl_ms": itl_ms, "e2e_slowdown": slowdown}
            good = sum(goodput.a2_good(r, sla, e0) for r in rows)
            table[
                f"I{itl_ms:g}_S{slowdown if slowdown is not None else 'none'}"
            ] = good / len(rows)
    return {
        "basis": "arrival" if mode == "open" else "completion",
        "scored_requests": len(rows),
        "completed": len(done),
        "mean_itl_ms_p50_p90_p99": [goodput.percentile(itl, q) for q in (50, 90, 99)],
        "e2e_over_e0_p50_p90_p99": [goodput.percentile(ratio, q) for q in (50, 90, 99)],
        "good_frac": table,
    }


def open_mode_arrival_check(per_request: Sequence[dict], meta: dict) -> dict:
    """Roots become ready exactly at their copy's arrival; descendants never before their floor."""
    starts = {c["label"]: c["start_ms"] for c in meta["copies"]}
    roots_checked = roots_exact = early = 0
    max_abs_err = 0.0
    for row in per_request:
        label = ":".join(row["request_id"].split(":")[:2])
        start = starts.get(label)
        if start is None:
            continue
        if row["request_id"] == (row.get("agentic") or {}).get("root_id"):
            roots_checked += 1
            err = abs(row["arrival_time_ms"] - start)
            max_abs_err = max(max_abs_err, err)
            roots_exact += err <= 1e-6
        if row["arrival_time_ms"] < start - 1e-6:
            early += 1
    return {
        "roots_checked": roots_checked,
        "roots_ready_at_arrival": roots_exact,
        "max_abs_root_error_ms": max_abs_err,
        "requests_before_copy_arrival": early,
    }


# ---------------------------------------------------------------------------------------------
# CLI


def _add_transform_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "play transform (B2 AgentX cell transform; each has its own base_<tag> directory)"
    )
    group.add_argument("--think-cap-s", type=float, default=DEFAULT_THINK_CAP_S)
    group.add_argument("--think-mult", type=float, default=1.0)
    group.add_argument("--osl-mult", type=float, default=1.0)


def _transform_from_args(args: argparse.Namespace) -> PlayTransform:
    return PlayTransform(
        think_cap_s=args.think_cap_s,
        think_mult=args.think_mult,
        osl_mult=args.osl_mult,
    )


def _gen_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m learned_routing.workloads.agentx_lowered",
        description="Write a recycled AgentX Agentic Mooncake v2 trace (see module README).",
    )
    parser.add_argument(
        "--root", default=None, help="campaign root (default $LR_ROOT or CR)"
    )
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--num-copies", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--play-rate-per-s",
        type=float,
        default=None,
        help="open mode: TOTAL play arrival rate (per-worker rate x N)",
    )
    _add_transform_args(parser)
    parser.add_argument(
        "--subsets",
        default=None,
        help="draw only from these comma-separated B2 play subsets of the split (e.g. A1,A2)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default CR/traces/agentx_lowered/gen/<sha>.jsonl",
    )
    args = parser.parse_args(argv)
    root = root_dir(args.root)
    plays = None
    if args.subsets:
        plays = subset_plays(
            root / "cells" / "SPLIT_MANIFEST.json", args.split, args.subsets.split(",")
        )
    spec = GenSpec(
        split=args.split,
        mode=args.mode,
        num_copies=args.num_copies,
        seed=args.seed,
        play_rate_per_s=args.play_rate_per_s,
        think_cap_s=args.think_cap_s,
        think_mult=args.think_mult,
        osl_mult=args.osl_mult,
        plays=plays,
    )
    meta = write_generated(spec, root, args.out)
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "path",
                    "sha256",
                    "rows",
                    "pool_size",
                    "last_start_ms",
                    "hash_ids_used",
                )
            },
            sort_keys=True,
        )
    )


def _build_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="agentx_lowered build-base")
    parser.add_argument("--root", default=None)
    parser.add_argument("--helper", type=Path, default=default_helper())
    _add_transform_args(parser)
    parser.add_argument(
        "--all-cell-transforms",
        action="store_true",
        help="build every play transform used by B2's AgentX cells (ignores the flags above)",
    )
    args = parser.parse_args(argv)
    root = root_dir(args.root)
    if args.all_cell_transforms:
        transforms = sorted(cell_transforms(root), key=PlayTransform.tag)
    else:
        transforms = [_transform_from_args(args)]
    for transform in transforms:
        manifest = build_base(root, args.helper, transform)
        print(
            json.dumps(
                {
                    "transform": transform.tag(),
                    "plays": len(manifest["plays"]),
                    "all_graph_digests_equal": manifest["all_graph_digests_equal"],
                    "b2_cell_trace_check": manifest["b2_cell_trace_check"],
                    "rows": sum(p["rows"] for p in manifest["plays"].values()),
                }
            ),
            flush=True,
        )


def _parity_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(prog="agentx_lowered parity")
    parser.add_argument("--root", default=None)
    parser.add_argument("--helper", type=Path, default=default_helper())
    parser.add_argument("--plays", nargs="*", default=list(DEFAULT_PARITY_PLAYS))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--recycle-copies", type=int, default=12)
    parser.add_argument("--recycle-seed", type=int, default=0)
    _add_transform_args(parser)
    args = parser.parse_args(argv)
    root = root_dir(args.root)
    transform = _transform_from_args(args)
    default_out = (
        "parity" if transform == PlayTransform() else f"parity_{transform.tag()}"
    )
    out_dir = args.out_dir or root / "runs" / "agentx_lowered" / default_out
    result = run_parity(
        root,
        args.helper,
        args.plays,
        out_dir,
        recycle_copies=args.recycle_copies,
        recycle_seed=args.recycle_seed,
        transform=transform,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "exact_cases",
                    "exact_identical",
                    "all_exact_identical",
                    "copy_order_cases",
                    "copy_order_identical",
                )
            }
        )
    )


def _validate_main(argv: Sequence[str]) -> None:
    """Replay generated traces and write steady-state reports.

    ``--runs-json`` lists ``{"name", "trace", "num_workers", "policy", "mode", "lanes"?}``.
    """
    parser = argparse.ArgumentParser(prog="agentx_lowered validate")
    parser.add_argument("--root", default=None)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--telemetry-ms", type=float, default=30000.0)
    parser.add_argument("--keep-per-request", action="store_true")
    parser.add_argument(
        "--e0-dir",
        type=Path,
        default=None,
        help="private E0 cache dir (seeded read-only from CR/runs/cache/e0); default OUT/e0",
    )
    args = parser.parse_args(argv)
    root = root_dir(args.root)
    replayer = Replayer(root, args.out_dir)
    e0 = E0Table(replayer.engine, args.e0_dir or args.out_dir / "e0")
    shared = root / "runs" / "cache" / "e0" / e0.path.name
    if shared.exists():
        e0.values.update(json.loads(shared.read_text()))
    for spec in json.loads(args.runs_json.read_text()):
        out = args.out_dir / f"{spec['name']}.json"
        if out.exists():
            continue
        trace = Path(spec["trace"])
        meta = json.loads(trace.with_name(trace.name + ".meta.json").read_text())
        span_of = copy_span_lookup(meta)
        res = replayer.run(
            trace,
            "agentic_mooncake",
            spec["num_workers"],
            spec["policy"],
            lanes=spec.get("lanes"),
            telemetry_ms=args.telemetry_ms,
            label="agentx_lowered-validate",
        )
        report = {
            "spec": spec,
            "trace_sha256": meta["sha256"],
            "gen_spec": meta["spec"],
            "wall_s": res["wall_s"],
            "summary": {
                k: res["summary"].get(k)
                for k in (
                    "num_requests",
                    "completed_requests",
                    "duration_ms",
                    "wall_time_ms",
                    "mean_ttft_ms",
                    "p90_ttft_ms",
                    "mean_itl_ms",
                    "p90_itl_ms",
                    "prefix_cache_reused_ratio",
                    "total_trajectories",
                    "completed_trajectories",
                )
            },
            "steady_state": steady_state(
                res["per_request"],
                num_workers=spec["num_workers"],
                mode=spec["mode"],
                bin_s=args.bin_s,
                telemetry=res.get("telemetry"),
                play_span_ms=span_of,
            ),
        }
        report["bands"] = band_table(
            res["per_request"],
            report["steady_state"]["steady_window"],
            spec["mode"],
            e0,
        )
        e0.persist()
        if spec["mode"] == "open":
            report["open_mode_arrivals"] = open_mode_arrival_check(
                res["per_request"], meta
            )
        if args.keep_per_request:
            rows = [
                goodput.compact_row(r)
                | {
                    "play_id": r["play_id"],
                    "request_id": r["request_id"],
                    "agentic": {
                        k: (r.get("agentic") or {}).get(k)
                        for k in ("lane_id", "root_id")
                    },
                }
                for r in res["per_request"]
            ]
            data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
            write_atomic(
                args.out_dir / f"{spec['name']}.per_request.jsonl.gz",
                gzip.compress(data, mtime=0),
            )
        write_json(out, report)
        print(
            json.dumps(
                {
                    "name": spec["name"],
                    "wall_s": round(res["wall_s"], 2),
                    "window": report["steady_state"]["steady_window"] is not None,
                }
            ),
            flush=True,
        )


def copy_span_lookup(meta: dict) -> Callable[[str], float | None]:
    """Play id -> recorded span (last request's not_before offset) of the copy's base play."""
    manifest = json.loads(Path(meta["base_manifest"]).read_text())["plays"]
    by_label = {c["label"]: manifest[c["play"]]["span_ms"] for c in meta["copies"]}

    def lookup(play_id: str) -> float | None:
        return by_label.get(":".join(play_id.split(":")[:2]))

    return lookup


def _reanalyze_main(argv: Sequence[str]) -> None:
    """Recompute steady_state, bands and arrival checks of finished validate runs from their
    saved per_request rows (telemetry fields are kept when the window is unchanged)."""
    parser = argparse.ArgumentParser(prog="agentx_lowered reanalyze")
    parser.add_argument("--root", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bin-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    root = root_dir(args.root)
    engine = json.loads((root / "config" / "engine.json").read_text())
    e0 = E0Table(engine, args.out_dir / "e0")
    for path in sorted(args.out_dir.glob("*.json")):
        report = json.loads(path.read_text())
        if "steady_state" not in report:
            continue
        gz = path.with_name(path.stem + ".per_request.jsonl.gz")
        if not gz.exists():
            continue
        rows = [
            json.loads(line) for line in gzip.decompress(gz.read_bytes()).splitlines()
        ]
        spec = report["spec"]
        trace = Path(spec["trace"])
        meta = json.loads(trace.with_name(trace.name + ".meta.json").read_text())
        old_window = report["steady_state"]["steady_window"] or {}
        state = steady_state(
            rows,
            num_workers=spec["num_workers"],
            mode=spec["mode"],
            bin_s=args.bin_s,
            play_span_ms=copy_span_lookup(meta),
        )
        window = state["steady_window"]
        if (
            window
            and old_window
            and (window["start_s"], window["end_s"])
            == (
                old_window["start_s"],
                old_window["end_s"],
            )
        ):
            window.update(
                {k: v for k, v in old_window.items() if k.startswith("telemetry_")}
            )
        report["steady_state"] = state
        report["bands"] = band_table(rows, window, spec["mode"], e0)
        if spec["mode"] == "open":
            report["open_mode_arrivals"] = open_mode_arrival_check(rows, meta)
        report["reanalyzed"] = True
        write_json(path, report)
        print(
            json.dumps(
                {"name": spec["name"], "stationary": window and window["stationary"]}
            )
        )
    e0.persist()


COMMANDS = {
    "reanalyze": _reanalyze_main,
    "build-base": _build_main,
    "parity": _parity_main,
    "validate": _validate_main,
    "generate": _gen_main,
}


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        COMMANDS[argv[0]](argv[1:])
    else:
        _gen_main(argv)


if __name__ == "__main__":
    main()
