#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent audit: are the live AIPerf inputs semantically the same workload as offline replay?

CONTRACT A13 asks for proof that the live load generator (AIPerf 0.13.0 driven by
``gen_aiperf_inputs.py``) and offline replay run the same workload. This audit compares three
independent views of one (cell, CRN replicate ``k``):

- **AIPerf's own reading** of ``aiperf_input.jsonl``: ``audit_aiperf_view.py`` runs AIPerf's
  ``MooncakeTrace`` model and ``MooncakeTraceDatasetLoader`` in the AIPerf environment;
- **a reference written here from the replay source**, not from the generator: the replicate trace
  parsed by aisimulate-core ``MooncakeTraceBuilder::push`` rules, arrivals by
  ``normalize_session_starts`` then ``speed_up_timing``, prompts by ``HashIdInterner`` plus
  ``synthesize_validated_trace_tokens``;
- **replay's materialized requests**: the cached ``lr-eval`` ``per_request`` rows of the same cell
  and replicate, for one or more policies.

``static`` checks, per request: request set and order, arrival timestamps, ISL, the OSL target,
prefix-sharing structure (sampled token LCPs and the full 16-token block-hash structure),
session membership and think times, closed-loop admission, and the warm-up and window rule
(the live scorer applied to a perfect live run rebuilt from replay rows).

``stub-prep`` and ``stub-check`` support the runtime check: AIPerf against
``audit_stub_server.py``, which answers every request with replay's own TTFT and E2E for it, so
the live dispatch (arrivals, think times, closed-loop slot handoffs) can be compared with replay's
arrival times request for request.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LIVE_DIR = Path(__file__).resolve().parent
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

import gen_aiperf_inputs as gen  # noqa: E402
import score_live  # noqa: E402
from learned_routing import goodput  # noqa: E402
from learned_routing.cache import bindings_build_id  # noqa: E402
from learned_routing.cells import load_cells, resolve_replicate  # noqa: E402
from learned_routing.paths import Layout  # noqa: E402

SCHEMA = "learned-routing.live-loadgen-parity.v1"
ENGINE_BLOCK = 16
QWEN3_REGULAR_VOCAB = 151643
AIPERF_CONNECTION_LIMIT = (
    2500  # aiperf 0.13.0 common/environment.py HTTP CONNECTION_LIMIT
)
AIPERF_TIMEOUT_S = 7200.0  # the manifest's --request-timeout-seconds
ORIGIN_NS = 1_800_000_000_000_000_000


class AuditError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# Reference from the replay source (independent of gen_aiperf_inputs)
# --------------------------------------------------------------------------------------------


def reference_trace(path: Path, block: int) -> dict:
    """Replicate trace by aisimulate-core ``MooncakeTraceBuilder::push`` rules.

    Returns sessions in replay's order (first appearance), each turn with its 1-based line, ISL,
    OSL, the hash IDs of the used blocks, the raw delay (``delay``/``delay_ms``, else the timestamp
    difference) and the interned token IDs replay synthesizes (``HashIdInterner`` over every hash ID
    of every row, in file order).
    """
    sessions: dict[str, dict] = {}
    interned: dict[int, int] = {}
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("session_id")
            key = f"request_{idx + 1}" if sid is None else str(sid)
            hids = [int(h) for h in row["hash_ids"]]
            isl = row.get("input_length", row.get("input_tokens"))
            isl = len(hids) * block if isl is None else int(isl)
            osl = int(row.get("output_length", row.get("output_tokens")))
            ts = row.get("timestamp", row.get("created_time"))
            ts = None if ts is None else float(ts)
            delay = row.get("delay", row.get("delay_ms"))
            ids = [interned.setdefault(h, len(interned)) for h in hids]
            session = sessions.get(key)
            if session is None:
                session = {"key": key, "first_ts": ts, "last_ts": ts, "turns": []}
                sessions[key] = session
                raw_delay = 0.0
            elif delay is not None:
                raw_delay = float(delay)
            elif ts is not None:
                raw_delay = ts - session["last_ts"]
            else:
                raw_delay = 0.0
            used = -(-isl // block)
            session["turns"].append(
                {
                    "line": idx + 1,
                    "isl": isl,
                    "osl": osl,
                    "hash_ids": hids[:used],
                    "interned": ids[:used],
                    "raw_delay": raw_delay,
                }
            )
            if ts is not None:
                session["last_ts"] = ts
    return {"sessions": list(sessions.values()), "distinct_hash_ids": len(interned)}


def reference_requests(
    ref: dict, *, open_loop: bool, speedup: float | None, max_model_len: int
) -> list[dict]:
    sessions = ref["sessions"]
    origin = None
    if open_loop:
        origin = min(
            (s["first_ts"] if s["first_ts"] is not None else 0.0) for s in sessions
        )
    out = []
    for si, s in enumerate(sessions):
        for t, turn in enumerate(s["turns"]):
            arrival = delay = None
            if t == 0:
                if open_loop:
                    arrival = (s["first_ts"] - origin) / speedup
            else:
                delay = turn["raw_delay"] / speedup if open_loop else turn["raw_delay"]
            out.append(
                {
                    "key": (s["key"], t),
                    "session_index": si,
                    "line": turn["line"],
                    "isl": turn["isl"],
                    "osl": turn["osl"],
                    "osl_sent": min(turn["osl"], max_model_len - turn["isl"]),
                    "arrival": arrival,
                    "delay": delay,
                    "hash_ids": turn["hash_ids"],
                    "interned": turn["interned"],
                }
            )
    return out


def replay_tokens(req: dict, block: int) -> np.ndarray:
    ids = np.asarray(req["interned"], dtype=np.uint32)
    return np.repeat(ids, block)[: req["isl"]]


def chained_block_hashes(tokens: np.ndarray, block: int = ENGINE_BLOCK) -> np.ndarray:
    """Same hash as ``audit_aiperf_view.chained_block_hashes``."""
    full = len(tokens) // block
    out = np.empty(full, dtype=np.uint64)
    parent = b"\x00" * 8
    data = np.ascontiguousarray(tokens[: full * block], dtype="<u4").tobytes()
    step = 4 * block
    for i in range(full):
        parent = hashlib.blake2b(
            parent + data[i * step : (i + 1) * step], digest_size=8
        ).digest()
        out[i] = int.from_bytes(parent, "little")
    return out


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantiles(values) -> dict | None:
    values = [float(v) for v in values]
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "p50": goodput.percentile(values, 50),
        "p99": goodput.percentile(values, 99),
        "max": max(values),
    }


def find_cell(layout: Layout, cell_id: str):
    for split in ("train", "val", "test"):
        for cell in load_cells(layout.root / "cells" / f"{split}.jsonl", layout):
            if cell.cell_id == cell_id:
                return cell
    raise AuditError(f"{cell_id} not found in CR/cells")


def common_prefix(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def token_lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


def choose_pairs(
    reqs: list[dict], n_each: int, seed: int
) -> list[tuple[int, int, str]]:
    """Pairs of reference request indices, stratified so shared prefixes are well represented."""
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int, str]] = []
    n = len(reqs)

    def sample_groups(groups: dict, label: str) -> None:
        eligible = [idx for idx in groups.values() if len(idx) >= 2]
        if not eligible:
            return
        weights = np.asarray([len(g) for g in eligible], dtype=float)
        weights /= weights.sum()
        for _ in range(n_each):
            g = eligible[rng.choice(len(eligible), p=weights)]
            a, b = rng.choice(len(g), size=2, replace=False)
            pairs.append((g[a], g[b], label))

    by_root: dict = defaultdict(list)
    by_deep: dict = defaultdict(list)
    by_session: dict = defaultdict(list)
    for i, r in enumerate(reqs):
        by_root[r["hash_ids"][0]].append(i)
        by_deep[tuple(r["hash_ids"][:3])].append(i)
        by_session[r["key"][0]].append(i)
    sample_groups(by_root, "same_first_block")
    sample_groups(by_deep, "same_first_3_blocks")
    sample_groups(by_session, "same_session")
    for _ in range(n_each):
        a, b = rng.choice(n, size=2, replace=False)
        pairs.append((int(a), int(b), "uniform"))
    return pairs


def run_aiperf_view(
    aiperf_python: Path, view_script: Path, input_file: Path, out_dir: Path, pairs: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [
            str(aiperf_python),
            str(view_script),
            "--input",
            str(input_file),
            "--out-dir",
            str(out_dir),
            "--pairs",
            str(pairs),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (out_dir / "view.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise AuditError(f"audit_aiperf_view failed: {proc.stderr[-2000:]}")
    return json.loads((out_dir / "summary.json").read_text())


# --------------------------------------------------------------------------------------------
# Perfect live run (window check)
# --------------------------------------------------------------------------------------------


def replay_row_picker(replay_rows: list[dict], session_keys: bool):
    """Maps a generated request to its replay row.

    With session metadata the key is ``(session_id, turn_index)``. Without it (open-loop
    single-turn) it is ``(arrival, ISL, requested OSL)``; equal keys are interchangeable.
    """
    if session_keys:
        by_key = {(r["session_id"], r["turn_index"]): r for r in replay_rows}

        def pick(q: dict) -> dict:
            return by_key[(q["conversation_id"], q["turn_index"])]

        return pick
    pools: dict = defaultdict(list)
    for r in replay_rows:
        pools[
            (r["arrival_time_ms"], r["input_length"], r["requested_output_length"])
        ].append(r)

    def pick_pooled(q: dict) -> dict:
        return pools[(q["arrival_ms"], q["input_length"], q["output_length"])].pop()

    return pick_pooled


def perfect_aiperf_records(
    requests: list[dict], replay_rows: list[dict], session_keys: bool
):
    """AIPerf-shaped records of a live run that reproduced replay exactly (ns-rounded)."""
    pick = replay_row_picker(replay_rows, session_keys)
    records = []
    for q in requests:
        r = pick(q)
        done = goodput.completed(r)
        records.append(
            {
                "metadata": {
                    "conversation_id": q["conversation_id"],
                    "turn_index": q["turn_index"],
                    "credit_issued_ns": ORIGIN_NS + round(r["arrival_time_ms"] * 1e6),
                    "request_start_ns": ORIGIN_NS + round(r["arrival_time_ms"] * 1e6),
                    "request_end_ns": ORIGIN_NS + round(r["terminal_time_ms"] * 1e6),
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                },
                "metrics": {
                    "time_to_first_token": {"value": r["ttft_ms"], "unit": "ms"}
                    if done
                    else None,
                    "request_latency": {"value": r["e2e_latency_ms"], "unit": "ms"}
                    if done
                    else None,
                    "output_sequence_length": {
                        "value": r["output_length"],
                        "unit": "tokens",
                    },
                    "input_sequence_length": {
                        "value": r["input_length"],
                        "unit": "tokens",
                    },
                },
                "error": None,
            }
        )
    return records


WINDOW_FIELDS = (
    "window_start_ms",
    "window_end_ms",
    "window_requests",
    "window_good",
    "good_frac_window",
    "goodput_rps_window",
    "warmup_excluded_rows",
    "occupancy_peak",
    "window_below_cap_frac",
)


# --------------------------------------------------------------------------------------------
# static
# --------------------------------------------------------------------------------------------


def static_audit(args: argparse.Namespace) -> dict:
    layout = Layout.resolve(args.root)
    cell = find_cell(layout, args.cell_id)
    rep = resolve_replicate(cell, args.k)
    engine = cell.engine()
    block = int(cell.raw.get("trace_block_size") or 512)
    max_model_len = int(engine["mock_engine_args"]["max_model_len"])
    open_loop = cell.is_open_loop
    speedup = (
        float(cell.load_kwargs(rep.trace_path)["arrival_speedup_ratio"])
        if open_loop
        else None
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema": SCHEMA,
        "cell_id": cell.cell_id,
        "split": cell.split,
        "family": cell.family,
        "load": dict(cell.load),
        "num_workers": cell.num_workers,
        "k": args.k,
        "replicate": {
            "protocol": rep.protocol,
            "path": str(rep.trace_path),
            "sha256": rep.trace_sha256,
        },
        "trace_block_size": block,
        "open_loop": open_loop,
        "speedup": speedup,
        "failures": [],
    }
    fail = report["failures"]

    # Replay records (policy, k) of this exact cell content and build.
    build = bindings_build_id(layout.cache_dir)["build_id"]
    content_sha = cell.content_sha()
    policies: dict[str, list[dict]] = {}
    records: dict[str, dict] = {}
    for path in args.record or []:
        record = json.loads(Path(path).read_text())
        problems = []
        if record.get("cell_id") != cell.cell_id:
            problems.append("cell_id")
        if record.get("cell_sha") != content_sha:
            problems.append("cell_sha")
        if int(record.get("repeat")) != args.k:
            problems.append("repeat")
        if record.get("trace_sha256") != rep.trace_sha256:
            problems.append("trace_sha256")
        if record.get("build_id") != build:
            problems.append("build_id")
        if record.get("error"):
            problems.append("error")
        if problems:
            raise AuditError(
                f"{path}: record does not match this cell/replicate/build: {problems}"
            )
        name = record["policy_name"]
        policies[name] = read_jsonl(Path(record["per_request_path"]))
        records[name] = record
    report["replay_records"] = {
        name: {
            "cache_key": r["cache_key"],
            "policy_seed": r.get("policy_seed"),
            "per_request_path": r["per_request_path"],
            "num_requests": r.get("num_requests"),
        }
        for name, r in records.items()
    }

    # Generator output (the live inputs under audit).
    inputs = out_dir / "inputs"
    if not (inputs / gen.MANIFEST_FILE).exists():
        gen.generate_cell(cell, args.k, inputs, salt=args.salt, with_e0=not args.no_e0)
    manifest = json.loads((inputs / gen.MANIFEST_FILE).read_text())
    gen_requests = gen.load_requests(inputs)
    report["inputs"] = {
        "dir": str(inputs),
        "manifest_id": manifest["manifest_id"],
        "files": manifest["files"],
        "salt": manifest["salt"],
        "session_header": manifest["session_header"],
        "aiperf_argv": manifest["aiperf"]["argv"],
        "aiperf_env": manifest["aiperf"]["env"],
    }

    # Independent reference.
    ref_trace = reference_trace(rep.trace_path, block)
    ref = reference_requests(
        ref_trace, open_loop=open_loop, speedup=speedup, max_model_len=max_model_len
    )
    ref_by_key = {r["key"]: i for i, r in enumerate(ref)}
    single_turn = all(len(s["turns"]) == 1 for s in ref_trace["sessions"])
    emit_session = not (open_loop and single_turn)

    # AIPerf's reading, with sampled pairs for the LCP check.
    seed = int.from_bytes(
        hashlib.sha256(f"{cell.cell_id}|{args.k}".encode()).digest()[:4], "big"
    )
    pairs = choose_pairs(ref, args.pairs_per_stratum, seed)
    pairs_path = out_dir / "pairs.json"
    pairs_path.write_text(
        json.dumps([[list(ref[a]["key"]), list(ref[b]["key"])] for a, b, _ in pairs])
    )
    view_dir = out_dir / "aiperf_view"
    view_summary = run_aiperf_view(
        args.aiperf_python,
        LIVE_DIR / "audit_aiperf_view.py",
        inputs / gen.INPUT_FILE,
        view_dir,
        pairs_path,
    )
    view = read_jsonl(view_dir / "view.jsonl")
    view_by_key = {(v["conversation_id"], v["turn_index"]): v for v in view}

    # ---- request set and order ----
    ref_keys = [r["key"] for r in ref]
    view_keys = [(v["conversation_id"], v["turn_index"]) for v in view]
    conv_order = []
    for v in view:
        if v["turn_index"] == 0:
            conv_order.append(v["conversation_id"])
    ref_order = [s["key"] for s in ref_trace["sessions"]]
    request_set = {
        "reference_requests": len(ref),
        "aiperf_requests": len(view),
        "aiperf_conversations": view_summary["conversations"],
        "reference_sessions": len(ref_trace["sessions"]),
        "same_keys": set(ref_keys) == set(view_keys)
        and len(view_keys) == len(set(view_keys)),
        "aiperf_conversation_order_equals_replay_session_order": conv_order
        == ref_order,
        "aiperf_request_order_equals_reference": view_keys == ref_keys,
        "manifest_counts": [manifest["num_sessions"], manifest["num_requests"]],
    }
    if not request_set["same_keys"]:
        fail.append("AIPerf request keys differ from the replay reference")
    if not request_set["aiperf_conversation_order_equals_replay_session_order"]:
        fail.append("AIPerf conversation order differs from replay's session order")
    per_policy = {}
    for name, rows in policies.items():
        info = {"rows": len(rows), "count_equal": len(rows) == len(ref)}
        if emit_session:
            keys = {(r["session_id"], r["turn_index"]) for r in rows}
            info["keys_equal"] = keys == set(ref_keys)
        else:
            info["rows_without_session"] = sum(r["session_id"] is None for r in rows)
            live = sorted(
                (v["timestamp_ms"], v["prompt_len"], v["max_tokens"]) for v in view
            )
            rep_t = sorted(
                (r["arrival_time_ms"], r["input_length"], r["requested_output_length"])
                for r in rows
            )
            info["arrival_isl_osl_multiset_equal"] = live == rep_t
        per_policy[name] = info
        if (
            not info["count_equal"]
            or not info.get("keys_equal", True)
            or not info.get("arrival_isl_osl_multiset_equal", True)
        ):
            fail.append(f"{name}: replay request set differs")
    request_set["replay"] = per_policy
    report["request_set"] = request_set

    # Match replay rows to reference requests.
    def match_rows(rows: list[dict]) -> list[dict | None]:
        if emit_session:
            by = {(r["session_id"], r["turn_index"]): r for r in rows}
            return [by.get(r["key"]) for r in ref]
        pools: dict = defaultdict(list)
        for r in rows:
            pools[
                (r["arrival_time_ms"], r["input_length"], r["requested_output_length"])
            ].append(r)
        return [
            (pools.get((r["arrival"], r["isl"], r["osl"])) or [None]).pop() for r in ref
        ]

    matched = {name: match_rows(rows) for name, rows in policies.items()}

    # ---- arrivals ----
    arrivals: dict = {}
    if open_loop:
        firsts = [i for i, r in enumerate(ref) if r["key"][1] == 0]
        exact_ref = sum(
            view_by_key[ref[i]["key"]]["timestamp_ms"] == ref[i]["arrival"]
            for i in firsts
        )
        sched_err = [
            abs(view_by_key[ref[i]["key"]]["schedule_offset_ms"] - ref[i]["arrival"])
            for i in firsts
        ]
        arrivals.update(
            first_turns=len(firsts),
            aiperf_timestamp_equals_reference_bitwise=exact_ref,
            aiperf_fixed_schedule_offset_max_abs_err_ms=max(sched_err)
            if sched_err
            else None,
        )
        if exact_ref != len(firsts):
            fail.append(
                "AIPerf first-turn timestamps differ from the reference arrivals"
            )
        # Schedule order vs arrival order, and ties.
        ts = [view_by_key[ref[i]["key"]]["timestamp_ms"] for i in firsts]
        counts = Counter(ts)
        arrivals["tie_groups"] = sum(1 for c in counts.values() if c >= 2)
        arrivals["requests_in_ties"] = sum(c for c in counts.values() if c >= 2)
        ranks = [view_by_key[ref[i]["key"]]["schedule_rank"] for i in firsts]
        order = sorted(range(len(firsts)), key=lambda j: ranks[j])
        arrivals["schedule_order_inversions"] = sum(
            ts[order[j + 1]] < ts[order[j]] for j in range(len(order) - 1)
        )
        for name, rows in matched.items():
            got = [rows[i]["arrival_time_ms"] for i in firsts if rows[i] is not None]
            want = [ref[i]["arrival"] for i in firsts if rows[i] is not None]
            exact = sum(a == b for a, b in zip(got, want))
            arrivals[f"replay[{name}]_first_turns_bitwise_equal"] = exact
            arrivals[f"replay[{name}]_first_turns_max_abs_err_ms"] = max(
                (abs(a - b) for a, b in zip(got, want)), default=None
            )
            if exact != len(firsts):
                fail.append(
                    f"{name}: replay first-turn arrivals differ from AIPerf timestamps"
                )
    else:
        later_ts = sum(v["timestamp_ms"] is not None for v in view)
        arrivals["closed_loop_rows_with_timestamp"] = later_ts
        if later_ts:
            fail.append("closed loop: AIPerf rows carry timestamps")
    later_with_ts = sum(
        v["turn_index"] > 0 and v["timestamp_ms"] is not None for v in view
    )
    arrivals["later_turns_with_timestamp"] = later_with_ts
    if later_with_ts:
        fail.append(
            "later turns carry absolute timestamps (FixedSchedule would ignore delay)"
        )
    report["arrivals"] = arrivals

    # ---- ISL / OSL ----
    isl_bad = sum(view_by_key[r["key"]]["prompt_len"] != r["isl"] for r in ref)
    osl_bad = sum(
        not (
            view_by_key[r["key"]]["max_tokens"] == r["osl_sent"]
            and view_by_key[r["key"]]["min_tokens"] == r["osl_sent"]
            and view_by_key[r["key"]]["turn_max_tokens"] == r["osl_sent"]
        )
        for r in ref
    )
    flags_bad = sum(
        not (
            v["ignore_eos"] is True
            and v["stream"] is True
            and v["include_usage"] is True
        )
        for v in view
    )
    vocab_bad = sum(
        v["prompt_len"] > 0
        and not (0 <= v["prompt_min"] and v["prompt_max"] < QWEN3_REGULAR_VOCAB)
        for v in view
    )
    lengths = {
        "isl_mismatch_vs_reference": isl_bad,
        "osl_target_mismatch_vs_reference": osl_bad,
        "osl_clamped_by_max_model_len": sum(r["osl_sent"] != r["osl"] for r in ref),
        "stream_ignore_eos_usage_flag_errors": flags_bad,
        "prompt_tokens_outside_regular_vocab": vocab_bad,
        "max_isl_plus_osl": max(r["isl"] + r["osl_sent"] for r in ref),
        "max_model_len": max_model_len,
        "input_tokens": sum(r["isl"] for r in ref),
    }
    for name, rows in matched.items():
        isl_r = sum(
            row is not None and row["input_length"] != len_
            for row, len_ in zip(
                rows, [view_by_key[r["key"]]["prompt_len"] for r in ref]
            )
        )
        req_r = sum(
            row is not None and row["requested_output_length"] != r["osl"]
            for row, r in zip(rows, ref)
        )
        out_r = sum(
            row is not None
            and goodput.completed(row)
            and row["output_length"] != view_by_key[r["key"]]["max_tokens"]
            for row, r in zip(rows, ref)
        )
        lengths[f"replay[{name}]"] = {
            "isl_mismatch_vs_aiperf_prompt": isl_r,
            "requested_osl_mismatch_vs_trace": req_r,
            "generated_osl_mismatch_vs_aiperf_max_tokens": out_r,
            "incomplete_rows": sum(
                row is not None and not goodput.completed(row) for row in rows
            ),
        }
        if isl_r or out_r or req_r:
            fail.append(f"{name}: ISL/OSL differ between replay and the AIPerf payload")
    if isl_bad or osl_bad or flags_bad or vocab_bad:
        fail.append("ISL/OSL/flags/vocab errors in the AIPerf payloads")
    report["lengths"] = lengths

    # ---- prefix structure: sampled token LCPs ----
    lcp_live = json.loads((view_dir / "lcp.json").read_text())
    strata: dict = defaultdict(
        lambda: {
            "pairs": 0,
            "nonzero": 0,
            "mismatch_expected": 0,
            "mismatch_replay_tokens": 0,
            "partial_block": 0,
            "lcp_blocks_sum": 0,
        }
    )
    for (a, b, label), live in zip(pairs, lcp_live):
        ra, rb = ref[a], ref[b]
        m = common_prefix(ra["hash_ids"], rb["hash_ids"])
        expected = min(ra["isl"], rb["isl"], m * block)
        replay_lcp = token_lcp(replay_tokens(ra, block), replay_tokens(rb, block))
        st = strata[label]
        st["pairs"] += 1
        st["nonzero"] += expected > 0
        st["mismatch_expected"] += live != expected
        st["mismatch_replay_tokens"] += live != replay_lcp
        st["partial_block"] += expected % block != 0
        st["lcp_blocks_sum"] += m
    total = sum(s["pairs"] for s in strata.values())
    bad = sum(
        s["mismatch_expected"] + s["mismatch_replay_tokens"] for s in strata.values()
    )
    report["prefix_pairs"] = {
        "pairs": total,
        "nonzero_lcp_pairs": sum(s["nonzero"] for s in strata.values()),
        "mismatches": bad,
        "by_stratum": dict(strata),
        "rule": "token LCP of the AIPerf prompts == min(ISL_a, ISL_b, m * trace_block_size) == LCP of replay's synthesized tokens",
    }
    if bad:
        fail.append(f"{bad} sampled pair LCP mismatches")

    # ---- prefix structure: full 16-token block-hash isomorphism ----
    live_blocks = np.load(view_dir / "blocks.npy")
    live_offsets = np.load(view_dir / "block_offsets.npy")
    view_index = {(v["conversation_id"], v["turn_index"]): v["index"] for v in view}
    live_seq, replay_seq, length_bad = [], [], 0
    for r in ref:
        vi = view_index[r["key"]]
        lb = live_blocks[live_offsets[vi] : live_offsets[vi + 1]]
        rb = chained_block_hashes(replay_tokens(r, block))
        if len(lb) != len(rb):
            length_bad += 1
            continue
        live_seq.append(lb)
        replay_seq.append(rb)
    L = np.concatenate(live_seq) if live_seq else np.empty(0, np.uint64)
    R = np.concatenate(replay_seq) if replay_seq else np.empty(0, np.uint64)
    pair_view = np.stack([L, R], axis=1)
    unique_pairs = len(np.unique(pair_view, axis=0)) if len(L) else 0
    iso = {
        "engine_block_size": ENGINE_BLOCK,
        "total_full_blocks": int(len(L)),
        "distinct_live_blocks": int(len(np.unique(L))),
        "distinct_replay_blocks": int(len(np.unique(R))),
        "distinct_pairs": int(unique_pairs),
        "requests_with_block_count_mismatch": length_bad,
    }
    iso["bijection"] = (
        length_bad == 0
        and iso["distinct_live_blocks"]
        == iso["distinct_replay_blocks"]
        == iso["distinct_pairs"]
    )
    iso["reused_block_instances"] = int(len(L) - iso["distinct_live_blocks"])
    report["block_structure"] = iso
    if not iso["bijection"]:
        fail.append("16-token block-hash structure is not isomorphic to replay's")

    # ---- sessions and think times ----
    sessions: dict = {
        "replay_emits_session_context": emit_session,
        "manifest_session_header": manifest["session_header"],
        "aiperf_env_dynamo_session_header": manifest["aiperf"]["env"].get(
            "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"
        ),
        "multi_turn_sessions": sum(len(s["turns"]) > 1 for s in ref_trace["sessions"]),
        "later_turns": sum(r["key"][1] > 0 for r in ref),
    }
    if manifest["session_header"] != emit_session:
        fail.append("session header setting differs from replay's SessionContext rule")
    delay_err = [
        abs(view_by_key[r["key"]]["delay_ms"] - r["delay"])
        for r in ref
        if r["key"][1] > 0
    ]
    sessions["aiperf_delay_vs_reference_max_abs_err_ms"] = (
        max(delay_err) if delay_err else None
    )
    sessions["aiperf_delay_bitwise_equal_reference"] = sum(e == 0.0 for e in delay_err)
    first_delay = sum(
        view_by_key[r["key"]]["delay_ms"] not in (None, 0, 0.0)
        for r in ref
        if r["key"][1] == 0
    )
    sessions["first_turns_with_delay"] = first_delay
    if delay_err and max(delay_err) > 0.0:
        fail.append("AIPerf delays differ from the reference")
    for name, rows in matched.items():
        errs = []
        for i, r in enumerate(ref):
            if r["key"][1] == 0:
                continue
            prev = rows[ref_by_key[(r["key"][0], r["key"][1] - 1)]]
            row = rows[i]
            if row is None or prev is None or prev.get("terminal_time_ms") is None:
                continue
            gap = row["arrival_time_ms"] - prev["terminal_time_ms"]
            errs.append(abs(gap - view_by_key[r["key"]]["delay_ms"]))
        sessions[f"replay[{name}]_think_checked"] = len(errs)
        sessions[f"replay[{name}]_think_max_abs_err_ms"] = max(errs) if errs else None
        if errs and max(errs) > 1e-6:
            fail.append(f"{name}: replay think gaps differ from AIPerf delays")
        if emit_session:
            members = defaultdict(list)
            for row in policies[name]:
                members[row["session_id"]].append(row["turn_index"])
            ok = all(
                sorted(members[s["key"]]) == list(range(len(s["turns"])))
                for s in ref_trace["sessions"]
            )
            sessions[f"replay[{name}]_session_membership_equal"] = ok and len(
                members
            ) == len(ref_trace["sessions"])
            if not sessions[f"replay[{name}]_session_membership_equal"]:
                fail.append(f"{name}: session membership differs")
    report["sessions"] = sessions

    # ---- admission (closed loop) / open-loop capacity ----
    admission: dict = {}
    argv = manifest["aiperf"]["argv"]
    if open_loop:
        admission["aiperf_mode"] = (
            "fixed-schedule" if "--fixed-schedule" in argv else "?"
        )
        admission["aiperf_concurrency_flag"] = "--concurrency" in argv
        for name, rows in policies.items():
            events = []
            for row in rows:
                if row.get("terminal_time_ms") is not None:
                    events.append((row["arrival_time_ms"], 1))
                    events.append((row["terminal_time_ms"], -1))
            events.sort(key=lambda e: (e[0], e[1]))
            cur = peak = 0
            for _, d in events:
                cur += d
                peak = max(peak, cur)
            e2e = [row["e2e_latency_ms"] for row in rows if goodput.completed(row)]
            admission[f"replay[{name}]_peak_in_flight"] = peak
            admission[f"replay[{name}]_max_e2e_s"] = max(e2e) / 1000.0 if e2e else None
            if peak >= AIPERF_CONNECTION_LIMIT:
                fail.append(
                    f"{name}: replay in-flight {peak} reaches AIPerf's connection limit"
                )
            if e2e and max(e2e) / 1000.0 >= AIPERF_TIMEOUT_S:
                fail.append(f"{name}: replay e2e exceeds AIPerf's request timeout")
    else:
        c = int(cell.load["value"])
        num_sessions = len(ref_trace["sessions"])
        admission.update(
            concurrency=c,
            argv_concurrency=argv[argv.index("--concurrency") + 1]
            if "--concurrency" in argv
            else None,
            argv_num_sessions=argv[argv.index("--num-sessions") + 1]
            if "--num-sessions" in argv
            else None,
            argv_sampling=argv[argv.index("--dataset-sampling-strategy") + 1]
            if "--dataset-sampling-strategy" in argv
            else None,
            argv_fixed_schedule="--fixed-schedule" in argv,
            sessions=num_sessions,
        )
        if (
            admission["argv_concurrency"] != str(c)
            or admission["argv_num_sessions"] != str(num_sessions)
            or admission["argv_sampling"] != "sequential"
            or admission["argv_fixed_schedule"]
        ):
            fail.append(
                "closed-loop AIPerf argv does not encode C slots over S sequential sessions"
            )
        for name, rows in matched.items():
            spans = {}
            for r, row in zip(ref, rows):
                if row is None:
                    continue
                key = r["key"][0]
                lo, hi = spans.get(key, (math.inf, -math.inf))
                spans[key] = (
                    min(lo, row["arrival_time_ms"]),
                    max(hi, row["terminal_time_ms"]),
                )
            starts = [spans[s["key"]][0] for s in ref_trace["sessions"]]
            ends = sorted(spans[s["key"]][1] for s in ref_trace["sessions"])
            at_zero = sum(s == 0.0 for s in starts[:c])
            later = starts[c:]
            slot_exact = sum(a == b for a, b in zip(later, ends))
            slot_err = max((abs(a - b) for a, b in zip(later, ends)), default=0.0)
            inversions = sum(starts[j + 1] < starts[j] for j in range(len(starts) - 1))
            admission[f"replay[{name}]"] = {
                "first_c_sessions_start_at_0": at_zero,
                "expected_at_0": min(c, num_sessions),
                "later_sessions": len(later),
                "start_equals_kth_session_end_exact": slot_exact,
                "start_vs_kth_session_end_max_abs_err_ms": slot_err,
                "session_start_order_inversions": inversions,
            }
            if (
                at_zero != min(c, num_sessions)
                or slot_exact != len(later)
                or inversions
            ):
                fail.append(
                    f"{name}: replay closed-loop admission differs from the C-slot rule"
                )
    report["admission"] = admission

    # ---- warm-up and window: the live scorer on a perfect live run ----
    window: dict = {}
    for name, rows in policies.items():
        record = records[name]
        live_records = perfect_aiperf_records(gen_requests, rows, emit_session)
        live_rows, diag = score_live.build_rows(manifest, gen_requests, live_records)
        metrics = score_live.score_rows(
            manifest,
            live_rows,
            score_live.ais_e0(gen_requests, layout.root),
            warmup=score_live.warmup_ids(manifest),
        )
        cmp = {}
        for field in WINDOW_FIELDS:
            a, b = metrics.get(field), record.get(field)
            if a is None or b is None:
                cmp[field] = {"live": a, "replay": b, "equal": a == b}
            else:
                cmp[field] = {
                    "live": a,
                    "replay": b,
                    "abs_err": abs(float(a) - float(b)),
                }
        window[name] = {
            "measure": cell.measure,
            "fields": cmp,
            "live_origin_ns_offset": diag["origin_ns"] - ORIGIN_NS,
        }
        counts_ok = all(
            cmp[f].get("abs_err", 0.0) == 0.0
            for f in ("window_requests", "window_good", "warmup_excluded_rows")
        )
        times_ok = all(
            cmp[f].get("abs_err", 0.0) <= 1e-5
            for f in ("window_start_ms", "window_end_ms")
        )
        rate_ok = cmp["goodput_rps_window"].get("abs_err", 0.0) <= 1e-9 * max(
            1.0, abs(record.get("goodput_rps_window") or 0.0)
        )
        window[name]["equal"] = counts_ok and times_ok and rate_ok
        if not window[name]["equal"]:
            fail.append(
                f"{name}: live scorer window/goodput differs from replay's record"
            )
    report["window"] = window
    report["exact"] = not fail
    return report


# --------------------------------------------------------------------------------------------
# Stub-run support
# --------------------------------------------------------------------------------------------


def stub_prep(args: argparse.Namespace) -> dict:
    """Per generated request: replay's TTFT/E2E/OSL for the stub server to play back."""
    manifest = json.loads((args.inputs / gen.MANIFEST_FILE).read_text())
    requests = gen.load_requests(args.inputs)
    record = json.loads(Path(args.record).read_text())
    rows = read_jsonl(Path(record["per_request_path"]))
    if (
        record["cell_id"] != manifest["cell_id"]
        or int(record["repeat"]) != manifest["k"]
    ):
        raise AuditError("record and inputs disagree on (cell, k)")
    if record["trace_sha256"] != manifest["replicate"]["sha256"]:
        raise AuditError("record and inputs disagree on the replicate trace")
    pick = replay_row_picker(rows, bool(manifest["session_header"]))
    out = []
    for q in requests:
        r = pick(q)
        if not goodput.completed(r):
            raise AuditError(
                f"replay row for {q['conversation_id']}/{q['turn_index']} is incomplete"
            )
        out.append(
            {
                "index": q["index"],
                "conversation_id": q["conversation_id"],
                "turn_index": q["turn_index"],
                "prompt_sha256": q["prompt_sha256"],
                "max_tokens": q["osl_sent"],
                "arrival_ms": q["arrival_ms"],
                "replay_arrival_ms": r["arrival_time_ms"],
                "replay_terminal_ms": r["terminal_time_ms"],
                "ttft_ms": r["ttft_ms"],
                "e2e_ms": r["e2e_latency_ms"],
                "output_length": r["output_length"],
            }
        )
    with args.out.open("w") as handle:
        for row in out:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "requests": len(out),
        "record": record["cache_key"],
        "policy": record["policy_name"],
    }
    print(json.dumps(summary))
    return summary


def stub_check(args: argparse.Namespace) -> dict:
    """Compare a stub run (server ledger + AIPerf export) with replay, request for request."""
    manifest = json.loads((args.inputs / gen.MANIFEST_FILE).read_text())
    requests = gen.load_requests(args.inputs)
    latency = read_jsonl(args.latency)
    ledger = read_jsonl(args.ledger)
    lat_by_index = {r["index"]: r for r in latency}
    open_loop = bool(manifest["open_loop"])
    report: dict = {
        "schema": SCHEMA + ".stub",
        "cell_id": manifest["cell_id"],
        "k": manifest["k"],
        "open_loop": open_loop,
        "session_header": manifest["session_header"],
        "subset": manifest.get("subset"),
        "failures": [],
    }
    fail = report["failures"]
    got = Counter(e.get("index") for e in ledger)
    unmatched = sum(1 for e in ledger if e.get("index") is None)
    missing = [q["index"] for q in requests if got.get(q["index"], 0) == 0]
    dup = [i for i, c in got.items() if i is not None and c > 1]
    report["delivery"] = {
        "expected": len(requests),
        "received": len(ledger),
        "unmatched_prompts": unmatched,
        "missing": len(missing),
        "duplicates": len(dup),
        "ambiguous_prompt_matches": sum(1 for e in ledger if e.get("ambiguous")),
    }
    if missing or dup or unmatched:
        fail.append("requests were not received exactly once")
    by_index = {e["index"]: e for e in ledger if e.get("index") is not None}
    payload_bad = 0
    header_bad = 0
    header_values: dict = defaultdict(set)
    for q in requests:
        e = by_index.get(q["index"])
        if e is None:
            continue
        if not (
            e["prompt_len"] == q["input_length"]
            and e["max_tokens"] == q["osl_sent"]
            and e["min_tokens"] == q["osl_sent"]
            and e["ignore_eos"] is True
        ):
            payload_bad += 1
        has = e.get("session_header") is not None
        if has != bool(manifest["session_header"]):
            header_bad += 1
        if has:
            header_values[q["conversation_id"]].add(e["session_header"])
    distinct = len({next(iter(v)) for v in header_values.values() if len(v) == 1})
    report["payload"] = {
        "payload_mismatch": payload_bad,
        "session_header_presence_mismatch": header_bad,
        "sessions_with_one_header_value": sum(
            len(v) == 1 for v in header_values.values()
        ),
        "distinct_header_values": distinct,
        "sessions": len(header_values),
    }
    if payload_bad or header_bad or (header_values and distinct != len(header_values)):
        fail.append("payload or session header mismatch on the wire")

    # Server-side live times (ms) relative to an origin that corresponds to replay time 0.
    recv = {i: e["recv_ns"] for i, e in by_index.items()}
    last = {i: e["last_ns"] for i, e in by_index.items()}
    if open_loop:
        offsets = [
            recv[q["index"]] - round(q["arrival_ms"] * 1e6)
            for q in requests
            if q["arrival_ms"] is not None and q["index"] in recv
        ]
        origin = min(offsets)
    else:
        origin = min(recv.values())
    rel = lambda ns: (ns - origin) / 1e6  # noqa: E731
    arrival_err, hop, think_hop = [], [], []
    prev_of: dict = {}
    for q in requests:
        prev_of[(q["conversation_id"], q["turn_index"])] = q["index"]
    for q in requests:
        i = q["index"]
        if i not in recv:
            continue
        lat = lat_by_index[i]
        arrival_err.append(rel(recv[i]) - lat["replay_arrival_ms"])
        if q["turn_index"] > 0:
            p = prev_of[(q["conversation_id"], q["turn_index"] - 1)]
            if p in last:
                think_hop.append(rel(recv[i]) - rel(last[p]) - (q["delay_ms"] or 0.0))
        elif open_loop:
            hop.append(rel(recv[i]) - q["arrival_ms"])
    report["timing"] = {
        "origin": "min over first turns of (recv - arrival)"
        if open_loop
        else "first receive",
        "first_turn_lateness_ms": quantiles(hop) if open_loop else None,
        "think_residual_ms": quantiles(think_hop),
        "arrival_minus_replay_arrival_ms": quantiles(arrival_err),
        "server_service_error_ms": quantiles(
            [
                (by_index[i]["last_ns"] - by_index[i]["recv_ns"]) / 1e6
                - lat_by_index[i]["e2e_ms"]
                for i in by_index
            ]
        ),
    }
    if not open_loop:
        c = int(manifest["concurrency"])
        spans: dict = {}
        for q in requests:
            i = q["index"]
            if i not in recv:
                continue
            lo, hi = spans.get(q["conversation_id"], (math.inf, -math.inf))
            spans[q["conversation_id"]] = (min(lo, recv[i]), max(hi, last[i]))
        order = [
            q["conversation_id"]
            for q in requests
            if q["turn_index"] == 0 and q["conversation_id"] in spans
        ]
        starts = [spans[s][0] for s in order]
        ends = sorted(spans[s][1] for s in order)
        events = sorted(
            [(lo, 1) for lo, _ in spans.values()]
            + [(hi, -1) for _, hi in spans.values()],
            key=lambda e: (e[0], e[1]),
        )
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        handoff = [(a - b) / 1e6 for a, b in zip(starts[c:], ends)]
        inversions = sum(starts[j + 1] < starts[j] for j in range(len(starts) - 1))
        report["closed_loop"] = {
            "concurrency": c,
            "peak_concurrent_sessions": peak,
            "session_start_order_inversions": inversions,
            "slot_handoff_ms": quantiles(handoff),
            "negative_handoffs": sum(h < 0 for h in handoff),
        }
        if peak > c:
            fail.append(f"peak concurrent sessions {peak} > C {c}")
    if args.run_dir is not None:
        report["dispatch_order"] = dispatch_order(
            args.run_dir, requests, recv, open_loop
        )
    if args.run_dir is not None and args.record is not None:
        report["scoring"] = stub_scoring(args, manifest)
    report["exact_delivery"] = not fail
    return report


def dispatch_order(
    run_dir: Path, requests: list[dict], recv: dict, open_loop: bool
) -> dict:
    """Session starts in replay's order (open: arrival, then file order; closed: generated order)
    versus the order AIPerf issued credits, sent requests, and the server received them.
    """
    meta = {}
    for record in read_jsonl(run_dir / score_live.RECORDS_FILE):
        m = record["metadata"]
        meta[(m["conversation_id"], int(m["turn_index"]))] = m
    firsts = [q for q in requests if q["turn_index"] == 0]
    if open_loop:
        firsts.sort(key=lambda q: (q["arrival_ms"], q["index"]))

    def inversions(values: list[int]) -> dict:
        gaps = [(a - b) / 1e6 for a, b in zip(values, values[1:]) if b < a]
        return {"count": len(gaps), "max_ms": max(gaps) if gaps else 0.0}

    keyed = [(q["conversation_id"], 0) for q in firsts]
    out = {
        "starts": len(firsts),
        "credit_issued": inversions(
            [int(meta[k]["credit_issued_ns"]) for k in keyed if k in meta]
        ),
        "request_start": inversions(
            [int(meta[k]["request_start_ns"]) for k in keyed if k in meta]
        ),
        "server_receive": inversions(
            [recv[q["index"]] for q in firsts if q["index"] in recv]
        ),
    }
    if open_loop:
        ties = Counter(q["arrival_ms"] for q in firsts)
        out["scheduled_tie_groups"] = sum(1 for c in ties.values() if c >= 2)
    return out


SCORE_FIELDS = (
    "window_start_ms",
    "window_end_ms",
    "window_requests",
    "window_good",
    "good_frac_window",
    "goodput_rps_window",
    "warmup_excluded_rows",
    "window_below_cap_frac",
    "ttft_p50",
    "itl_p50",
    "slowdown_p50",
)


def stub_scoring(args: argparse.Namespace, manifest: dict) -> dict:
    """The live scorer on the stub run's AIPerf export, against replay's record of the same run.

    The stub reproduces replay's per-request latencies, so any difference here is what the load
    generator and the client add (dispatch lateness, think-time and slot-handoff residuals, the
    client-side latency base), with no engine difference at all.
    """
    record = json.loads(Path(args.record).read_text())
    live, rows = score_live.score_run(args.inputs, args.run_dir, e0_mode="ais")
    replay_rows = read_jsonl(Path(record["per_request_path"]))
    out = {
        "policy": record["policy_name"],
        "replay_cache_key": record["cache_key"],
        "fidelity_ok": live["fidelity_ok"],
        "live_diagnostics": {
            key: live["live"].get(key)
            for key in (
                "records",
                "matched",
                "missing",
                "statuses",
                "isl_mismatch",
                "osl_mismatch",
                "prompt_checked",
                "prompt_mismatch",
                "usage_isl_mismatch",
                "session_header_mismatch",
                "schedule_lateness_ms",
                "start_lag_ms",
                "think_residual_ms",
            )
        },
        "fields": {},
        "latency_live_over_replay": score_live.compare_with_replay(
            manifest, rows, replay_rows
        ),
    }
    for field in SCORE_FIELDS:
        a, b = live.get(field), record.get(field)
        entry = {"live": a, "replay": b}
        if a is not None and b is not None:
            entry["diff"] = float(a) - float(b)
            if b:
                entry["rel_diff"] = (float(a) - float(b)) / abs(float(b))
        out["fields"][field] = entry
    (args.out.parent / (args.out.stem + ".score.json")).write_text(
        json.dumps(live, indent=1, sort_keys=True, default=score_live._json_default)
    )
    return out


# --------------------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------------------


def summarize(audit_root: Path) -> dict:
    """One compact table over every ``static/*/report.json`` and ``stub/*/check.json``."""
    cells = []
    for path in sorted((audit_root / "static").glob("*/report.json")):
        r = json.loads(path.read_text())
        replay = sorted(r["replay_records"])
        arrivals = r["arrivals"]
        sessions = r["sessions"]
        entry = {
            "report": str(path),
            "cell_id": r["cell_id"],
            "split": r["split"],
            "family": r["family"],
            "mode": r["load"]["mode"],
            "num_workers": r["num_workers"],
            "k": r["k"],
            "protocol": r["replicate"]["protocol"],
            "replay_policies": replay,
            "reference_only": not replay,
            "requests": r["request_set"]["reference_requests"],
            "sessions": r["request_set"]["reference_sessions"],
            "exact": r["exact"],
            "failures": r["failures"],
            "checks": {
                "request_set_and_order": r["request_set"]["same_keys"]
                and r["request_set"][
                    "aiperf_conversation_order_equals_replay_session_order"
                ]
                and all(
                    v.get("count_equal")
                    and v.get("keys_equal", True)
                    and v.get("arrival_isl_osl_multiset_equal", True)
                    for v in r["request_set"]["replay"].values()
                ),
                "arrivals_bitwise": (
                    arrivals.get("aiperf_timestamp_equals_reference_bitwise")
                    == arrivals.get("first_turns")
                    and all(
                        arrivals.get(f"replay[{p}]_first_turns_bitwise_equal")
                        == arrivals.get("first_turns")
                        for p in replay
                    )
                )
                if r["open_loop"]
                else None,
                "isl_exact": r["lengths"]["isl_mismatch_vs_reference"] == 0
                and all(
                    r["lengths"][f"replay[{p}]"]["isl_mismatch_vs_aiperf_prompt"] == 0
                    for p in replay
                ),
                "osl_target_exact": r["lengths"]["osl_target_mismatch_vs_reference"]
                == 0
                and all(
                    r["lengths"][f"replay[{p}]"][
                        "generated_osl_mismatch_vs_aiperf_max_tokens"
                    ]
                    == 0
                    and r["lengths"][f"replay[{p}]"]["requested_osl_mismatch_vs_trace"]
                    == 0
                    for p in replay
                ),
                "prefix_pairs": f"{r['prefix_pairs']['pairs'] - r['prefix_pairs']['mismatches']}/{r['prefix_pairs']['pairs']} equal ({r['prefix_pairs']['nonzero_lcp_pairs']} with LCP > 0)",
                "block_bijection": r["block_structure"]["bijection"],
                "think_max_abs_err_ms": max(
                    [
                        sessions.get(f"replay[{p}]_think_max_abs_err_ms") or 0.0
                        for p in replay
                    ]
                    + [sessions.get("aiperf_delay_vs_reference_max_abs_err_ms") or 0.0]
                ),
                "session_header_rule": sessions["manifest_session_header"]
                == sessions["replay_emits_session_context"],
                "closed_loop_slot_rule": None
                if not replay
                else all(
                    r["admission"][f"replay[{p}]"]["start_equals_kth_session_end_exact"]
                    == r["admission"][f"replay[{p}]"]["later_sessions"]
                    for p in replay
                )
                if not r["open_loop"]
                else None,
                "window_scorer_equal": all(w["equal"] for w in r["window"].values())
                if r["window"]
                else None,
            },
            "tie_groups": arrivals.get("tie_groups"),
            "requests_in_ties": arrivals.get("requests_in_ties"),
            "full_blocks": r["block_structure"]["total_full_blocks"],
            "distinct_blocks": r["block_structure"]["distinct_live_blocks"],
            "later_turns": sessions["later_turns"],
        }
        cells.append(entry)
    stubs = []
    for path in sorted((audit_root / "stub").glob("*/check.json")):
        r = json.loads(path.read_text())
        entry = {
            "check": str(path),
            "run": path.parent.name,
            "cell_id": r["cell_id"],
            "k": r["k"],
            "open_loop": r["open_loop"],
            "exact_delivery": r["exact_delivery"],
            "failures": r["failures"],
            "delivery": r["delivery"],
            "payload": r["payload"],
            "timing": r["timing"],
            "closed_loop": r.get("closed_loop"),
            "dispatch_order": r.get("dispatch_order"),
        }
        scoring = r.get("scoring")
        if scoring:
            entry["scoring"] = {
                "policy": scoring["policy"],
                "fidelity_ok": scoring["fidelity_ok"],
                "fields": scoring["fields"],
                "latency_live_over_replay": scoring["latency_live_over_replay"],
                "aiperf_schedule_lateness_ms": scoring["live_diagnostics"][
                    "schedule_lateness_ms"
                ],
                "aiperf_start_lag_ms": scoring["live_diagnostics"]["start_lag_ms"],
                "aiperf_think_residual_ms": scoring["live_diagnostics"][
                    "think_residual_ms"
                ],
            }
        stubs.append(entry)
    return {
        "schema": SCHEMA + ".summary",
        "cells": cells,
        "static_all_exact": all(c["exact"] for c in cells),
        "stub_runs": stubs,
        "stub_all_exact_delivery": all(s["exact_delivery"] for s in stubs),
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("static")
    p.add_argument("--cell-id", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument(
        "--record", action="append", default=[], help="cached lr-eval record (.json)"
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--salt", default=None)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--no-e0", action="store_true")
    p.add_argument("--pairs-per-stratum", type=int, default=750)
    p.add_argument(
        "--aiperf-python",
        type=Path,
        default=Layout.resolve().root / "runs/live/env/aiperf-0.13.0/bin/python",
    )
    q = sub.add_parser("stub-prep")
    q.add_argument("--inputs", type=Path, required=True)
    q.add_argument("--record", required=True)
    q.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("stub-check")
    c.add_argument("--inputs", type=Path, required=True)
    c.add_argument("--latency", type=Path, required=True)
    c.add_argument("--ledger", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument(
        "--run-dir", type=Path, default=None, help="AIPerf artifact directory"
    )
    c.add_argument(
        "--record", default=None, help="the replay record the stub played back"
    )
    s = sub.add_parser("summarize")
    s.add_argument("--audit-root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summary = summarize(args.audit_root)
        args.out.write_text(json.dumps(summary, indent=1, sort_keys=True))
        print(
            json.dumps(
                {k: summary[k] for k in ("static_all_exact", "stub_all_exact_delivery")}
            )
        )
        return (
            0
            if summary["static_all_exact"] and summary["stub_all_exact_delivery"]
            else 1
        )
    if args.command == "static":
        args.salt = args.salt or f"lr-audit-{args.cell_id}-k{args.k}"
        report = static_audit(args)
        (args.out_dir / "report.json").write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str)
        )
        print(
            json.dumps(
                {
                    "cell_id": report["cell_id"],
                    "k": report["k"],
                    "exact": report["exact"],
                    "failures": report["failures"],
                }
            )
        )
        return 0 if report["exact"] else 1
    if args.command == "stub-prep":
        stub_prep(args)
        return 0
    report = stub_check(args)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        json.dumps(
            {k: report[k] for k in ("cell_id", "k", "exact_delivery", "failures")}
        )
    )
    return 0 if report["exact_delivery"] else 1


if __name__ == "__main__":
    sys.exit(main())
