# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side loadgen contract checks on synthetic ledgers."""

from __future__ import annotations

import json

import gen_aiperf_inputs as gen
import loadgen_contract as contract
import numpy as np
import pytest
from test_gen_aiperf_inputs import TINY, make_cell

T0 = 1_791_000_000_000_000_000


def ledger_for(out, *, session_header: bool, closed_cap: int | None = None):
    """A ledger of a perfect loadgen: exact payloads, on-time arrivals, sequential turns."""
    inputs = [json.loads(x) for x in (out / gen.INPUT_FILE).read_text().splitlines()]
    requests = gen.load_requests(out)
    entries, end_of = [], {}
    slots = [T0] * (closed_cap or 0)
    for row, request in zip(inputs, requests):
        key = request["conversation_id"]
        if request["turn_index"] == 0:
            if closed_cap:
                start = slots.pop(0)
            else:
                start = T0 + round(request["arrival_ms"] * 1e6)
        else:
            start = end_of[key] + round(request["delay_ms"] * 1e6) + 1_000_000
        done = start + 50_000_000
        end_of[key] = done
        if closed_cap and request["turn_index"] == max(
            r["turn_index"] for r in requests if r["conversation_id"] == key
        ):
            slots.append(done)
            slots.sort()
        payload = row["payload"]
        entries.append(
            {
                "recv_ns": start,
                "done_ns": done,
                "prompt_sha256": contract.prompt_digest(payload["prompt"]),
                "prompt_len": len(payload["prompt"]),
                "max_tokens": payload["max_tokens"],
                "min_tokens": payload["min_tokens"],
                "ignore_eos": True,
                "stream": True,
                "include_usage": True,
                "nvext": payload.get("nvext"),
                "model": payload["model"],
                "session_header": f"corr-{key}" if session_header else None,
            }
        )
    return entries


def write(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_prompt_digest_matches_the_generator():
    tokens = [3, 151642, 0, 77]
    assert contract.prompt_digest(tokens) == gen.token_digest(np.asarray(tokens))


def test_open_loop_contract_passes_and_catches_violations(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="open_speedup", value=1.0, spread=1000.0)
    out = tmp_path / "out"
    gen.generate_cell(cell, 0, out, salt="c", with_e0=False)
    entries = ledger_for(out, session_header=True)
    write(tmp_path / "ledger.jsonl", entries)
    report = contract.check(out, tmp_path / "ledger.jsonl")
    assert report["ok"], report["failures"]
    assert report["arrival_lateness_ms"]["max"] == 0.0
    assert report["think_residual_ms"]["min"] == pytest.approx(1.0, abs=1e-6)
    bad = [dict(e) for e in entries]
    bad.append(dict(bad[0]))  # received twice
    bad[1]["session_header"] = None  # header dropped
    bad[2]["min_tokens"] = bad[2]["max_tokens"] + 1  # OSL not forced to max_tokens
    write(tmp_path / "bad.jsonl", bad)
    failures = " | ".join(contract.check(out, tmp_path / "bad.jsonl")["failures"])
    assert "expected 1 receipts, got 2" in failures
    assert "session header" in failures
    assert "payload errors" in failures


def test_closed_loop_contract_checks_the_session_cap(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="closed_concurrency", value=2)
    out = tmp_path / "out"
    gen.generate_cell(cell, 0, out, salt="c", with_e0=False)
    entries = ledger_for(out, session_header=True, closed_cap=2)
    write(tmp_path / "ledger.jsonl", entries)
    report = contract.check(out, tmp_path / "ledger.jsonl")
    assert report["ok"], report["failures"]
    assert report["closed"] == {
        "cap": 2,
        "peak_sessions": 2,
        "start_order_inversions": 0,
    }
    requests = gen.load_requests(out)
    last = max(i for i, r in enumerate(requests) if r["turn_index"] == 0)
    entries[last][
        "recv_ns"
    ] = T0  # the last session jumps the queue: 3 sessions at once
    write(tmp_path / "bad.jsonl", entries)
    failures = " | ".join(contract.check(out, tmp_path / "bad.jsonl")["failures"])
    assert "exceed the cap 2" in failures
