#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a learned-routing cell and CRN replicate ``k`` into exact AIPerf inputs (CONTRACT A13).

The live lane replays the same workload offline replay runs, request for request. This tool reads
the replicate trace the harness materializes (:func:`learned_routing.cells.resolve_replicate`,
protocols ``crn-order-v1`` and ``crn-spread-v1``) and parses it with the replay's own Mooncake
rules (aisimulate-core ``replay/loadgen/trace.rs`` ``MooncakeTraceBuilder::push``). It writes an
AIPerf 0.13.0 ``mooncake_trace`` file with one verbatim ``payload`` per request.

**Arrivals.**

- Open loop (``open_speedup``, ``open_rate``): replay subtracts the earliest first arrival and
  divides first arrivals and think delays by the speedup (``normalize_session_starts`` then
  ``speed_up_timing``). The generator applies the same two float operations in the same order, so
  every first-turn ``timestamp`` is bit-identical to replay's ``arrival_time_ms``. AIPerf replays
  them with ``--fixed-schedule``.
- Closed loop (``closed_concurrency``): no timestamps. Sessions appear in the replicate's session
  order, which is the order in which replay activates them. AIPerf runs them with session
  concurrency ``C``, ``--dataset-sampling-strategy sequential`` and ``--num-sessions S``. A session
  holds its slot across turns and think time, as in replay's ``ConcurrencyState``. Think delays
  stay at their trace values, since replay applies no speedup in concurrency mode.
- Later turns carry ``delay``: milliseconds after the previous turn of the session completes. That
  is replay's ``delay_after_previous_ms`` (an explicit ``delay``, otherwise the timestamp
  difference to the session's previous timestamped row).

**Prompts.** Each prompt is a token-ID list sent as ``/v1/completions`` ``prompt``. Dynamo's
integer-array path passes it through untouched, and Qwen3 adds no BOS, so the model sees exactly
``input_length`` tokens. Replay synthesizes trace block ``j`` as interned ``hash_ids[j]`` repeated
``trace_block_size`` times and truncates the result to ``input_length``. The live prompt keys
every trace block by its full hash path from the root (a prefix trie), not by the hash ID alone:

- equal hash prefixes give byte-equal token prefixes;
- a block's first token is the salt offset plus its rank among its siblings (children of the same
  path), so sibling blocks differ at their first token;
- the other ``trace_block_size - 1`` tokens come from SHAKE-256 of the salted path digest, drawn
  from the Qwen3 regular vocabulary ``[0, 151643)``, which excludes every special token.

So the token longest common prefix (LCP) of any two requests equals replay's,
``min(ISL_a, ISL_b, m * trace_block_size)``, where ``m`` is the length of the common hash prefix.
Every trace block size is a multiple of the engine block size (16), so the engine-block prefix
structure, which the KV router and vLLM's prefix cache see, is the same as well. A fresh ``--salt``
per live run keeps runs from sharing any KV block. That reproduces replay's cold start without
restarting the engines.

**Output length.** ``max_tokens = min_tokens = OSL`` and ``ignore_eos`` force exact OSL. A row
whose ``ISL + OSL`` exceeds ``max_model_len`` gets ``max_model_len - ISL`` output tokens, which is
how replay truncates. The transforms already keep every cell row within the cap.

**Sessions.** Replay passes session metadata to the router, as ``SessionContext(session_id)``,
except for all-single-turn traces in trace (open-loop) mode. The manifest's ``session_header``
says whether AIPerf must send ``X-Dynamo-Session-ID``. AIPerf takes the header value from each
session's stable correlation ID: a different label from replay's, with the same equivalence
classes.

**AgentX (``agentic_lanes``) is not supported.** The live lane needs typed dispatch and completion
edges, ``replay_barrier`` joins, play-relative release floors, and lanes dealt ``ordinal % L``
that recycle on play quiescence. Neither upstream AIPerf nor the prior finite AgentX fork
implements that. See ``README.md``.

Subcommands:

- ``cell``: a cell plus replicate ``k`` gives ``aiperf_input.jsonl``, ``requests.jsonl`` (one line
  per request, no tokens) and ``manifest.json`` (identity, AIPerf argv and env, E0 values).
- ``idle``: an idle-calibration input for the live-fitted E0. Requests run one at a time with
  unique prompts over an ISL grid.
- ``verify-replay``: check a generated schedule against a cached replay ``per_request`` file of
  the same cell and replicate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from learned_routing.canon import sha256_json
from learned_routing.cells import Cell, load_cells, resolve_replicate
from learned_routing.paths import Layout

SCHEMA = "learned-routing.live-inputs.v1"
GENERATOR_VERSION = "lr-live-gen-v1"
AIPERF_VERSION = "0.13.0"
AIPERF_WHEEL_SHA256 = "a20100fd127f1d10ea45d4946b1658a186030e86770344c80dfd256a60192f34"
# Qwen3 tokenizer.json: regular BPE vocabulary 0..151642; special/added tokens are 151643..151668.
QWEN3_REGULAR_VOCAB = 151643
DEFAULT_MODEL = "Qwen/Qwen3-32B"
OPEN_MODES = ("open_speedup", "open_rate")
CLOSED_MODES = ("closed_concurrency",)
UNSUPPORTED_ROW_KEYS = (
    "output_token_ids",
    "priority",
    "strict_priority",
    "policy_class",
    "wait_for",
)
INPUT_FILE = "aiperf_input.jsonl"
REQUESTS_FILE = "requests.jsonl"
MANIFEST_FILE = "manifest.json"
SUBSET_TRACE_FILE = "subset_trace.jsonl"


class LiveInputError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# Replay's Mooncake parsing
# --------------------------------------------------------------------------------------------


@dataclass
class TraceTurn:
    line: int  # 1-based line number in the trace file (blank lines count, as in replay)
    input_length: int
    output_length: int
    hash_ids: list[int]
    delay_ms: float  # replay's delay_after_previous_ms, before any speedup
    timestamp_ms: float | None


@dataclass
class TraceSession:
    key: str  # replay's session ID: the row's session_id, else request_<line>
    explicit_id: bool
    first_timestamp_ms: float | None
    turns: list[TraceTurn]


def _first(row: dict, *names: str):
    for name in names:
        if name in row:
            return row[name]
    return None


def parse_mooncake(lines: Sequence[str], block_size: int) -> list[TraceSession]:
    """Sessions in replay order, by aisimulate-core ``MooncakeTraceBuilder::push`` rules."""
    if block_size <= 0:
        raise LiveInputError(f"trace_block_size must be > 0, got {block_size}")
    sessions: dict[str, TraceSession] = {}
    last_timestamp: dict[str, float | None] = {}
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        line = index + 1
        row = json.loads(raw)
        unsupported = sorted(set(row) & set(UNSUPPORTED_ROW_KEYS))
        if unsupported:
            raise LiveInputError(f"line {line}: unsupported trace fields {unsupported}")
        session_id = row.get("session_id")
        key = f"request_{line}" if session_id is None else str(session_id)
        hash_ids = row.get("hash_ids")
        if hash_ids is None:
            raise LiveInputError(f"line {line}: missing hash_ids")
        capacity = len(hash_ids) * block_size
        isl = _first(row, "input_length", "input_tokens")
        isl = capacity if isl is None else int(isl)
        if isl > capacity:
            raise LiveInputError(
                f"line {line}: input_length {isl} exceeds hash_ids capacity {capacity}"
            )
        if isl < 1:
            raise LiveInputError(f"line {line}: input_length must be >= 1, got {isl}")
        osl = _first(row, "output_length", "output_tokens")
        if osl is None:
            raise LiveInputError(f"line {line}: missing output_length")
        osl = int(osl)
        if osl < 1:
            raise LiveInputError(f"line {line}: output_length must be >= 1, got {osl}")
        timestamp = _first(row, "timestamp", "created_time")
        timestamp = None if timestamp is None else float(timestamp)
        explicit_delay = _first(row, "delay", "delay_ms")
        session = sessions.get(key)
        if session is None:
            session = TraceSession(
                key=key,
                explicit_id=session_id is not None,
                first_timestamp_ms=timestamp,
                turns=[],
            )
            sessions[key] = session
            last_timestamp[key] = timestamp
            if explicit_delay not in (None, 0, 0.0):
                raise LiveInputError(f"line {line}: delay on the first turn of {key}")
            delay = 0.0
        elif explicit_delay is not None:
            delay = float(explicit_delay)
        elif timestamp is not None:
            previous = last_timestamp[key]
            if previous is None:
                raise LiveInputError(
                    f"line {line}: cannot infer the delay of {key} without a previous timestamp"
                )
            delay = timestamp - previous
        else:
            delay = 0.0
        if not math.isfinite(delay) or delay < 0:
            raise LiveInputError(f"line {line}: invalid delay {delay!r}")
        used = -(-isl // block_size)
        session.turns.append(
            TraceTurn(
                line=line,
                input_length=isl,
                output_length=osl,
                hash_ids=[int(h) for h in hash_ids[:used]],
                delay_ms=delay,
                timestamp_ms=timestamp,
            )
        )
        if timestamp is not None:
            last_timestamp[key] = timestamp
    if not sessions:
        raise LiveInputError("trace has no requests")
    return list(sessions.values())


def replay_tokens(
    sessions: Sequence[TraceSession], block_size: int
) -> list[tuple[str, int, np.ndarray]]:
    """Replay's own synthesized prompt tokens: interned hash ID repeated per trace block.

    Mirrors ``HashIdInterner`` (first-seen order over the file) and
    ``synthesize_validated_trace_tokens``. Used to prove structural equivalence in tests.
    """
    interned: dict[int, int] = {}
    order = sorted(
        (
            (turn.line, session.key, t)
            for session in sessions
            for t, turn in enumerate(session.turns)
        ),
    )
    by_key = {session.key: session for session in sessions}
    out = []
    for _, key, t in order:
        turn = by_key[key].turns[t]
        ids = [interned.setdefault(h, len(interned)) for h in turn.hash_ids]
        tokens = np.repeat(np.asarray(ids, dtype=np.uint32), block_size)[
            : turn.input_length
        ]
        out.append((key, t, tokens))
    return out


# --------------------------------------------------------------------------------------------
# Prompt synthesis
# --------------------------------------------------------------------------------------------


class PromptSynth:
    """Prefix-trie prompt synthesis (module docstring, "Prompts")."""

    def __init__(self, block_size: int, salt: str, vocab: int = QWEN3_REGULAR_VOCAB):
        if block_size <= 0:
            raise LiveInputError(f"block_size must be > 0, got {block_size}")
        self.block_size = block_size
        self.vocab = vocab
        self.salt = salt.encode()
        self.offset = (
            int.from_bytes(
                hashlib.blake2b(b"lr-live-offset|" + self.salt, digest_size=8).digest(),
                "little",
            )
            % vocab
        )
        self._root_digest = hashlib.blake2b(
            b"lr-live-root|" + self.salt, digest_size=16
        ).digest()
        self._node: dict[tuple[int, int], int] = {}
        self._parent: list[int] = []
        self._hash: list[int] = []
        self._first: list[int] | None = None
        self._digest: list[bytes] | None = None
        self._blocks: dict[int, np.ndarray] = {}

    @property
    def num_blocks(self) -> int:
        return len(self._parent)

    def add(self, hash_ids: Sequence[int]) -> list[int]:
        """Insert a hash path; returns its node IDs (one per trace block)."""
        if self._first is not None:
            raise RuntimeError("add() after finalize()")
        parent = -1
        path = []
        for h in hash_ids:
            if not 0 <= h < 2**64:
                raise LiveInputError(f"hash id {h} is not an unsigned 64-bit integer")
            node = self._node.get((parent, h))
            if node is None:
                node = len(self._parent)
                self._node[(parent, h)] = node
                self._parent.append(parent)
                self._hash.append(h)
            path.append(node)
            parent = node
        return path

    def finalize(self) -> None:
        children: dict[int, list[int]] = defaultdict(list)
        for node, parent in enumerate(self._parent):
            children[parent].append(node)
        first = [0] * len(self._parent)
        for parent, nodes in children.items():
            if len(nodes) > self.vocab:
                raise LiveInputError(
                    f"{len(nodes)} sibling blocks under one prefix exceed the vocabulary "
                    f"({self.vocab}); sibling first tokens could not stay distinct"
                )
            for rank, node in enumerate(sorted(nodes, key=lambda n: self._hash[n])):
                first[node] = (self.offset + rank) % self.vocab
        # Parents are always created before their children, so one forward pass suffices.
        digest: list[bytes] = []
        for node, parent in enumerate(self._parent):
            base = self._root_digest if parent < 0 else digest[parent]
            digest.append(
                hashlib.blake2b(
                    base + self._hash[node].to_bytes(8, "little"), digest_size=16
                ).digest()
            )
        self._first = first
        self._digest = digest

    def block(self, node: int) -> np.ndarray:
        if self._first is None or self._digest is None:
            raise RuntimeError("finalize() first")
        tokens = self._blocks.get(node)
        if tokens is None:
            tokens = np.empty(self.block_size, dtype=np.uint32)
            tokens[0] = self._first[node]
            if self.block_size > 1:
                stream = hashlib.shake_256(
                    b"lr-live-body|" + self.salt + self._digest[node]
                ).digest(4 * (self.block_size - 1))
                tokens[1:] = np.frombuffer(stream, dtype="<u4") % np.uint32(self.vocab)
            self._blocks[node] = tokens
        return tokens

    def tokens(self, path: Sequence[int], length: int) -> np.ndarray:
        if length > len(path) * self.block_size:
            raise LiveInputError(f"length {length} exceeds {len(path)} blocks")
        return np.concatenate([self.block(node) for node in path])[:length]


def token_digest(tokens: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Cell planning
# --------------------------------------------------------------------------------------------


@dataclass
class LiveRequest:
    index: int  # position in aiperf_input.jsonl
    conversation_id: str  # AIPerf session_id == replay's session key
    turn_index: int
    line: int  # 1-based line in the replicate trace
    input_length: int
    output_length: int  # trace OSL (replay's requested_output_length)
    osl_sent: int  # max_tokens = min_tokens actually requested
    arrival_ms: float | None  # open-loop first turns: replay's arrival_time_ms
    delay_ms: float | None  # later turns: the delay AIPerf applies (after speedup if open loop)
    trace_delay_ms: float | None  # later turns: replay's delay before speedup
    prompt_sha256: str | None = None
    e0_ms: float | None = None


def cell_speedup(cell: Cell, trace_path: Path) -> float:
    kwargs = cell.load_kwargs(trace_path)
    if "arrival_speedup_ratio" not in kwargs:
        raise LiveInputError(
            f"{cell.cell_id}: load mode {cell.load_mode} has no speedup"
        )
    return float(kwargs["arrival_speedup_ratio"])


def plan_requests(
    sessions: Sequence[TraceSession],
    *,
    open_loop: bool,
    speedup: float | None,
    max_model_len: int | None,
) -> list[LiveRequest]:
    """The live schedule, request for request (module docstring, "Arrivals")."""
    if open_loop:
        if not (speedup and speedup > 0 and math.isfinite(speedup)):
            raise LiveInputError(
                f"open loop needs a finite speedup > 0, got {speedup!r}"
            )
        missing = [s.key for s in sessions if s.first_timestamp_ms is None]
        if missing:
            raise LiveInputError(
                f"open loop needs a first-turn timestamp on every session; {len(missing)} "
                f"have none (first: {missing[0]})"
            )
        # normalize_session_starts: subtract the minimum first arrival (unwrap_or(0.0)).
        origin = min(s.first_timestamp_ms or 0.0 for s in sessions)
    requests: list[LiveRequest] = []
    for session in sessions:
        for t, turn in enumerate(session.turns):
            osl_sent = turn.output_length
            if max_model_len is not None:
                if turn.input_length >= max_model_len:
                    raise LiveInputError(
                        f"line {turn.line}: input_length {turn.input_length} >= max_model_len "
                        f"{max_model_len} (replay rejects it)"
                    )
                osl_sent = min(osl_sent, max_model_len - turn.input_length)
            arrival = delay = trace_delay = None
            if t == 0:
                if open_loop:
                    # speed_up_timing: (t - min) / ratio, the same two float ops as replay.
                    arrival = (session.first_timestamp_ms - origin) / speedup
            else:
                trace_delay = turn.delay_ms
                delay = turn.delay_ms / speedup if open_loop else turn.delay_ms
            requests.append(
                LiveRequest(
                    index=len(requests),
                    conversation_id=session.key,
                    turn_index=t,
                    line=turn.line,
                    input_length=turn.input_length,
                    output_length=turn.output_length,
                    osl_sent=osl_sent,
                    arrival_ms=arrival,
                    delay_ms=delay,
                    trace_delay_ms=trace_delay,
                )
            )
    return requests


def session_metadata_emitted(sessions: Sequence[TraceSession], open_loop: bool) -> bool:
    """Whether replay passes ``SessionContext`` to the router (module docstring, "Sessions")."""
    single_turn = all(len(s.turns) == 1 for s in sessions)
    return not (open_loop and single_turn)


def build_payload(
    tokens: np.ndarray, osl_sent: int, model: str, worker_id_field: bool
) -> dict:
    payload = {
        "model": model,
        "prompt": tokens.tolist(),
        "max_tokens": osl_sent,
        "min_tokens": osl_sent,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if worker_id_field:
        payload["nvext"] = {"extra_fields": ["worker_id"]}
    return payload


def aiperf_row(request: LiveRequest, payload: dict) -> dict:
    row: dict = {"session_id": request.conversation_id}
    if request.arrival_ms is not None:
        row["timestamp"] = request.arrival_ms
    if request.delay_ms is not None:
        row["delay"] = request.delay_ms
    row["output_length"] = request.osl_sent
    row["payload"] = payload
    return row


def aiperf_spec(
    *,
    open_loop: bool,
    concurrency: int | None,
    num_sessions: int,
    num_requests: int,
    session_header: bool,
    model: str,
) -> dict:
    """AIPerf 0.13.0 argv and environment for this input (``${URL}``, ``${ARTIFACT_DIR}`` filled
    in by the launcher)."""
    argv = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        "builtin",
        "--url",
        "${URL}",
        "--endpoint-type",
        "completions",
        "--streaming",
        "--input-file",
        INPUT_FILE,
        "--custom-dataset-type",
        "mooncake_trace",
        "--use-server-token-count",
        "--export-level",
        "raw",
        "--request-timeout-seconds",
        "7200",
        "--random-seed",
        "0",
        "--ui-type",
        "none",
        "--artifact-dir",
        "${ARTIFACT_DIR}",
    ]
    if open_loop:
        argv += ["--fixed-schedule", "--fixed-schedule-auto-offset"]
    else:
        argv += [
            "--no-fixed-schedule",
            "--concurrency",
            str(concurrency),
            "--num-sessions",
            str(num_sessions),
            "--dataset-sampling-strategy",
            "sequential",
        ]
    env = {
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": (
            "true" if session_header else "false"
        ),
        "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID": "false",
    }
    return {
        "version": AIPERF_VERSION,
        "wheel_sha256": AIPERF_WHEEL_SHA256,
        "argv": argv,
        "env": env,
        "expected_sessions": num_sessions,
        "expected_requests": num_requests,
        "notes": [
            "Run from the directory that holds aiperf_input.jsonl, or make --input-file absolute.",
            "The Dynamo frontend must not set --router-session-affinity-ttl-secs: replay never "
            "pins sessions, the session ID only reaches policies as SessionContext.",
            "Start every run from idle engines; the salt keeps runs from sharing KV blocks.",
        ],
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    """Write rows (compact JSON) atomically; returns the file's SHA-256."""
    digest = hashlib.sha256()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        for row in rows:
            line = json.dumps(row, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            handle.write(line)
    os.replace(tmp, path)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def e0_values(engine: dict, layout: Layout, pairs: Iterable[tuple[int, int]]) -> dict:
    """AIS E0 (``learned_routing.e0``) for every (ISL, OSL) pair; persists new values."""
    from learned_routing.e0 import METHOD, E0Table

    table = E0Table(engine, layout.e0_dir)
    values = {f"{isl},{osl}": table(isl, osl) for isl, osl in sorted(set(pairs))}
    table.persist()
    return {"method": METHOD, "engine_sha": table.engine_sha, "values": values}


@dataclass
class CellPlan:
    """Everything about (cell, k) the live run reproduces, before any prompt is built."""

    cell: Cell
    k: int
    rep: object  # learned_routing.cells.ResolvedReplicate
    engine: dict
    open_loop: bool
    speedup: float | None
    concurrency: int | None
    block_size: int
    engine_block: int
    max_model_len: int | None
    sessions: list[TraceSession]
    total_sessions: int
    requests: list[LiveRequest]
    session_header: bool


def plan_cell(cell: Cell, k: int, max_sessions: int | None = None) -> CellPlan:
    """Resolve replicate ``k`` of ``cell`` and plan its live schedule (no prompts)."""
    if cell.load_mode == "agentic_lanes" or cell.trace_format != "mooncake":
        raise LiveInputError(
            f"{cell.cell_id}: {cell.trace_format}/{cell.load_mode} is not supported by the live "
            "lane (AgentX lanes need typed dependency edges and lane recycling; see README.md)"
        )
    if cell.load_mode not in OPEN_MODES + CLOSED_MODES:
        raise LiveInputError(f"{cell.cell_id}: unsupported load mode {cell.load_mode}")
    open_loop = cell.load_mode in OPEN_MODES
    rep = resolve_replicate(cell, k)
    if rep.trace_path is None:
        raise LiveInputError(f"{cell.cell_id}: no replicate trace")
    engine = cell.engine()
    block_size = int(cell.raw.get("trace_block_size") or 512)
    engine_block = int(engine["mock_engine_args"]["block_size"])
    if block_size % engine_block:
        raise LiveInputError(
            f"trace_block_size {block_size} is not a multiple of the engine block {engine_block}"
        )
    max_model_len = int(engine["mock_engine_args"].get("max_model_len") or 0) or None
    sessions = parse_mooncake(rep.trace_path.read_text().splitlines(), block_size)
    total_sessions = len(sessions)
    if max_sessions is not None:
        # Smoke subsets keep the replicate's first sessions (a prefix of replay's order).
        sessions = sessions[:max_sessions]
    speedup = cell_speedup(cell, rep.trace_path) if open_loop else None
    requests = plan_requests(
        sessions, open_loop=open_loop, speedup=speedup, max_model_len=max_model_len
    )
    return CellPlan(
        cell=cell,
        k=k,
        rep=rep,
        engine=engine,
        open_loop=open_loop,
        speedup=speedup,
        concurrency=None if open_loop else int(cell.load["value"]),
        block_size=block_size,
        engine_block=engine_block,
        max_model_len=max_model_len,
        sessions=sessions,
        total_sessions=total_sessions,
        requests=requests,
        session_header=session_metadata_emitted(sessions, open_loop),
    )


def generate_cell(
    cell: Cell,
    k: int,
    out_dir: Path,
    *,
    salt: str,
    model: str = DEFAULT_MODEL,
    worker_id_field: bool = True,
    with_e0: bool = True,
    max_sessions: int | None = None,
) -> dict:
    """Write the AIPerf input, the request table and the manifest for (cell, k)."""
    plan = plan_cell(cell, k, max_sessions)
    rep, requests = plan.rep, plan.requests
    synth = PromptSynth(plan.block_size, salt)
    paths = [synth.add(t.hash_ids) for session in plan.sessions for t in session.turns]
    synth.finalize()

    e0 = None
    if with_e0:
        e0 = e0_values(
            plan.engine, cell.layout, ((r.input_length, r.osl_sent) for r in requests)
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    def rows():
        for request, path in zip(requests, paths):
            tokens = synth.tokens(path, request.input_length)
            request.prompt_sha256 = token_digest(tokens)
            if e0 is not None:
                request.e0_ms = e0["values"][
                    f"{request.input_length},{request.osl_sent}"
                ]
            yield aiperf_row(
                request, build_payload(tokens, request.osl_sent, model, worker_id_field)
            )

    input_sha = _write_jsonl(out_dir / INPUT_FILE, rows())
    requests_sha = _write_jsonl(out_dir / REQUESTS_FILE, (asdict(r) for r in requests))
    files = {INPUT_FILE: input_sha, REQUESTS_FILE: requests_sha}
    if max_sessions is not None:
        # The subset's own replicate rows, in replicate order, so offline replay can run exactly
        # the requests of a smoke subset (pass it as a trace with no further arrival spread).
        keep = {r.line for r in requests}
        source = rep.trace_path.read_text().splitlines()
        subset = "".join(line + "\n" for n, line in enumerate(source, 1) if n in keep)
        (out_dir / SUBSET_TRACE_FILE).write_text(subset)
        files[SUBSET_TRACE_FILE] = hashlib.sha256(subset.encode()).hexdigest()
    arrivals = [r.arrival_ms for r in requests if r.arrival_ms is not None]
    manifest = {
        "schema": SCHEMA,
        "generator": GENERATOR_VERSION,
        "cell_id": cell.cell_id,
        "cell_content_sha256": cell.content_sha(),
        "split": cell.split,
        "family": cell.family,
        "num_workers": cell.num_workers,
        "load": dict(cell.load),
        "open_loop": plan.open_loop,
        "speedup": plan.speedup,
        "concurrency": plan.concurrency,
        "sla": cell.sla,
        "measure": cell.measure,
        "k": k,
        "replicate": {
            "protocol": rep.protocol,
            "path": str(rep.trace_path),
            "sha256": rep.trace_sha256,
            "replicate_seed": rep.replicate_seed,
            "policy_seed": rep.policy_seed,
        },
        "subset": None
        if max_sessions is None
        else {"max_sessions": max_sessions, "of_sessions": plan.total_sessions},
        "trace_block_size": plan.block_size,
        "engine_block_size": plan.engine_block,
        "max_model_len": plan.max_model_len,
        "model": model,
        "vocab": QWEN3_REGULAR_VOCAB,
        "salt": salt,
        "salt_offset": synth.offset,
        "distinct_blocks": synth.num_blocks,
        "session_header": plan.session_header,
        "num_sessions": len(plan.sessions),
        "num_requests": len(requests),
        "input_tokens": sum(r.input_length for r in requests),
        "output_tokens_sent": sum(r.osl_sent for r in requests),
        "osl_clamped": sum(r.osl_sent != r.output_length for r in requests),
        "first_arrival_ms": min(arrivals) if arrivals else None,
        "last_arrival_ms": max(arrivals) if arrivals else None,
        "worker_id_field": worker_id_field,
        "files": files,
        "e0": None if e0 is None else {k_: v for k_, v in e0.items() if k_ != "values"},
        "aiperf": aiperf_spec(
            open_loop=plan.open_loop,
            concurrency=plan.concurrency,
            num_sessions=len(plan.sessions),
            num_requests=len(requests),
            session_header=plan.session_header,
            model=model,
        ),
    }
    manifest["manifest_id"] = sha256_json(
        {
            key: manifest[key]
            for key in ("cell_content_sha256", "k", "salt", "files", "model")
        }
    )
    _write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


# --------------------------------------------------------------------------------------------
# Idle calibration (live-fitted E0)
# --------------------------------------------------------------------------------------------

DEFAULT_IDLE_ISL = (256, 1024, 4096, 8192, 16384, 32768, 65536, 98304, 129024)


def generate_idle(
    out_dir: Path,
    *,
    salt: str,
    isls: Sequence[int] = DEFAULT_IDLE_ISL,
    osl: int = 256,
    repeats: int = 3,
    model: str = DEFAULT_MODEL,
    max_model_len: int = 131072,
    worker_id_field: bool = True,
) -> dict:
    """Unique-prompt single requests, one at a time (concurrency 1), interleaved by repeat."""
    vocab = QWEN3_REGULAR_VOCAB
    offset = (
        int.from_bytes(
            hashlib.blake2b(b"lr-live-idle|" + salt.encode(), digest_size=8).digest(),
            "little",
        )
        % vocab
    )
    requests: list[LiveRequest] = []
    payloads: list[dict] = []
    for r in range(repeats):
        for isl in isls:
            if isl + osl > max_model_len:
                raise LiveInputError(f"ISL {isl} + OSL {osl} exceeds {max_model_len}")
            index = len(requests)
            stream = hashlib.shake_256(
                f"lr-live-idle-body|{salt}|{index}".encode()
            ).digest(4 * isl)
            tokens = np.frombuffer(stream, dtype="<u4") % np.uint32(vocab)
            tokens = tokens.copy()
            # Distinct first tokens: no two calibration prompts share even one engine block.
            tokens[0] = (offset + index) % vocab
            request = LiveRequest(
                index=index,
                conversation_id=f"idle-r{r}-isl{isl}",
                turn_index=0,
                line=index + 1,
                input_length=isl,
                output_length=osl,
                osl_sent=osl,
                arrival_ms=None,
                delay_ms=None,
                trace_delay_ms=None,
                prompt_sha256=token_digest(tokens),
            )
            requests.append(request)
            payloads.append(build_payload(tokens, osl, model, worker_id_field))
    out_dir.mkdir(parents=True, exist_ok=True)
    input_sha = _write_jsonl(
        out_dir / INPUT_FILE, (aiperf_row(q, p) for q, p in zip(requests, payloads))
    )
    requests_sha = _write_jsonl(out_dir / REQUESTS_FILE, (asdict(q) for q in requests))
    manifest = {
        "schema": SCHEMA,
        "generator": GENERATOR_VERSION,
        "kind": "idle-calibration",
        "cell_id": None,
        "open_loop": False,
        "concurrency": 1,
        "isl_grid": list(isls),
        "osl": osl,
        "repeats": repeats,
        "model": model,
        "salt": salt,
        "session_header": False,
        "num_sessions": len(requests),
        "num_requests": len(requests),
        "worker_id_field": worker_id_field,
        "files": {INPUT_FILE: input_sha, REQUESTS_FILE: requests_sha},
        "aiperf": aiperf_spec(
            open_loop=False,
            concurrency=1,
            num_sessions=len(requests),
            num_requests=len(requests),
            session_header=False,
            model=model,
        ),
    }
    manifest["manifest_id"] = sha256_json(
        {key: manifest[key] for key in ("salt", "files", "model", "isl_grid", "osl")}
    )
    _write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


# --------------------------------------------------------------------------------------------
# Verification against a replay per_request file
# --------------------------------------------------------------------------------------------

ABS_TOL_MS = 1e-6


def load_requests(inputs_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (inputs_dir / REQUESTS_FILE).read_text().splitlines()
        if line.strip()
    ]


def read_per_request(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_against_replay(
    manifest: dict, requests: Sequence[dict], rows: Sequence[dict]
) -> dict:
    """Compare the live schedule with replay's per_request rows of the same cell and replicate.

    - Open loop, no session metadata (single-turn): the multisets of
      ``(arrival_time_ms, input_length, requested_output_length)`` must be identical, bit for bit.
    - With session metadata: rows match by ``(session_id, turn_index)``. First turns of an open
      loop must arrive at exactly the generated timestamp. Later turns must arrive ``delay_ms``
      after the previous turn's terminal. Closed loop: sessions must start in the generated
      order, the first ``min(C, S)`` at time 0.
    """
    report: dict = {"requests": len(requests), "replay_rows": len(rows), "failures": []}
    fail = report["failures"]
    if len(rows) != len(requests):
        fail.append(f"row count {len(rows)} != generated {len(requests)}")
    if manifest["open_loop"] and not manifest["session_header"]:
        live = sorted(
            (r["arrival_ms"], r["input_length"], r["output_length"]) for r in requests
        )
        replay = sorted(
            (
                row["arrival_time_ms"],
                row["input_length"],
                row["requested_output_length"],
            )
            for row in rows
        )
        mismatched = sum(a != b for a, b in zip(live, replay))
        report.update(
            mode="open-single-turn", mismatched=mismatched, exact=mismatched == 0
        )
        if mismatched or len(live) != len(replay):
            fail.append(f"{mismatched} arrival/ISL/OSL tuples differ")
        return report

    by_key = {(row["session_id"], row["turn_index"]): row for row in rows}
    if len(by_key) != len(rows):
        fail.append("duplicate (session_id, turn_index) rows in per_request")
    first_errors, delay_errors, length_errors, missing = [], [], 0, 0
    sessions: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        sessions[request["conversation_id"]].append(request)
    for key, turns in sessions.items():
        prev = None
        for request in turns:
            row = by_key.get((key, request["turn_index"]))
            if row is None:
                missing += 1
                prev = None
                continue
            if (row["input_length"], row["requested_output_length"]) != (
                request["input_length"],
                request["output_length"],
            ):
                length_errors += 1
            if request["turn_index"] == 0:
                if manifest["open_loop"]:
                    first_errors.append(
                        abs(row["arrival_time_ms"] - request["arrival_ms"])
                    )
            elif prev is not None and prev.get("terminal_time_ms") is not None:
                gap = row["arrival_time_ms"] - prev["terminal_time_ms"]
                delay_errors.append(abs(gap - request["delay_ms"]))
            prev = row
    report.update(
        mode=("open" if manifest["open_loop"] else "closed") + "-sessions",
        missing=missing,
        length_mismatches=length_errors,
        first_arrival_max_abs_err_ms=max(first_errors) if first_errors else None,
        first_arrival_exact=all(e == 0.0 for e in first_errors),
        delay_checked=len(delay_errors),
        delay_max_abs_err_ms=max(delay_errors) if delay_errors else None,
    )
    if missing:
        fail.append(f"{missing} generated requests have no replay row")
    if length_errors:
        fail.append(f"{length_errors} ISL/OSL mismatches")
    if first_errors and max(first_errors) > 0.0:
        fail.append(f"first arrivals differ by up to {max(first_errors)!r} ms")
    if delay_errors and max(delay_errors) > ABS_TOL_MS:
        fail.append(f"think delays differ by up to {max(delay_errors)!r} ms")
    if not manifest["open_loop"]:
        order = list(sessions)
        starts = []
        for key in order:
            row = by_key.get((key, 0))
            if row is not None:
                starts.append(row["arrival_time_ms"])
        decreasing = sum(b < a for a, b in zip(starts, starts[1:]))
        concurrency = int(manifest["concurrency"])
        at_zero = sum(s == 0.0 for s in starts)
        report.update(
            session_start_inversions=decreasing,
            sessions_started_at_zero=at_zero,
            expected_at_zero=min(concurrency, len(order)),
        )
        if decreasing:
            fail.append(f"{decreasing} session starts out of generated order")
        if at_zero != min(concurrency, len(order)):
            fail.append(
                f"{at_zero} sessions start at 0, expected {min(concurrency, len(order))}"
            )
    report["exact"] = not fail
    return report


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _find_cell(cells_path: Path, cell_id: str, layout: Layout) -> Cell:
    for cell in load_cells(cells_path, layout):
        if cell.cell_id == cell_id:
            return cell
    raise LiveInputError(f"{cell_id} is not in {cells_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_cell = sub.add_parser("cell", help="AIPerf inputs for one cell and CRN replicate")
    p_cell.add_argument("--cells", type=Path, required=True)
    p_cell.add_argument("--cell-id", required=True)
    p_cell.add_argument("--k", type=int, required=True)
    p_cell.add_argument("--out-dir", type=Path, required=True)
    p_cell.add_argument(
        "--salt", required=True, help="unique per live run (e.g. the run ID)"
    )
    p_cell.add_argument("--root", type=Path, default=None, help="campaign root (CR)")
    p_cell.add_argument("--model", default=DEFAULT_MODEL)
    p_cell.add_argument("--no-worker-id-field", action="store_true")
    p_cell.add_argument(
        "--no-e0", action="store_true", help="skip the AIS E0 precompute"
    )
    p_cell.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="smoke subset: the first N sessions",
    )

    p_idle = sub.add_parser(
        "idle", help="idle calibration input for the live-fitted E0"
    )
    p_idle.add_argument("--out-dir", type=Path, required=True)
    p_idle.add_argument("--salt", required=True)
    p_idle.add_argument(
        "--isl",
        default=",".join(map(str, DEFAULT_IDLE_ISL)),
        help="comma-separated ISL grid",
    )
    p_idle.add_argument("--osl", type=int, default=256)
    p_idle.add_argument("--repeats", type=int, default=3)
    p_idle.add_argument("--model", default=DEFAULT_MODEL)
    p_idle.add_argument("--max-model-len", type=int, default=131072)

    p_ver = sub.add_parser(
        "verify-replay", help="compare a schedule with replay per_request"
    )
    p_ver.add_argument(
        "--inputs", type=Path, required=True, help="a `cell` output directory"
    )
    p_ver.add_argument("--per-request", type=Path, required=True)
    p_ver.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "cell":
        layout = Layout.resolve(args.root)
        cell = _find_cell(args.cells, args.cell_id, layout)
        manifest = generate_cell(
            cell,
            args.k,
            args.out_dir,
            salt=args.salt,
            model=args.model,
            worker_id_field=not args.no_worker_id_field,
            with_e0=not args.no_e0,
            max_sessions=args.max_sessions,
        )
        summary = {
            key: manifest[key]
            for key in (
                "cell_id",
                "k",
                "num_sessions",
                "num_requests",
                "input_tokens",
                "session_header",
                "speedup",
                "concurrency",
                "manifest_id",
            )
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "idle":
        manifest = generate_idle(
            args.out_dir,
            salt=args.salt,
            isls=[int(x) for x in args.isl.split(",") if x],
            osl=args.osl,
            repeats=args.repeats,
            model=args.model,
            max_model_len=args.max_model_len,
        )
        print(
            json.dumps(
                {
                    "num_requests": manifest["num_requests"],
                    "manifest_id": manifest["manifest_id"],
                }
            )
        )
        return 0
    manifest = json.loads((args.inputs / MANIFEST_FILE).read_text())
    report = verify_against_replay(
        manifest, load_requests(args.inputs), read_per_request(args.per_request)
    )
    report.update(
        cell_id=manifest["cell_id"], k=manifest["k"], per_request=str(args.per_request)
    )
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
