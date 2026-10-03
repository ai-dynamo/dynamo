# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-lane input generator: replay parsing, prompt structure, schedules, AIPerf rows."""

from __future__ import annotations

import hashlib
import itertools
import json
import random

import gen_aiperf_inputs as gen
import numpy as np
import pytest
from learned_routing.cells import load_cells
from learned_routing.paths import Layout

ENGINE = {
    "model": "Qwen/Qwen3-32B",
    "ais_perf_config": {
        "backend": "vllm",
        "model": "Qwen/Qwen3-32B",
        "system": "h100_sxm",
    },
    "mock_engine_args": {
        "block_size": 16,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
    },
}


def lines(rows):
    return [json.dumps(row) for row in rows]


def lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) if len(diff) else n


def chain_hashes(tokens: np.ndarray, block: int) -> list[bytes]:
    """vLLM/router-style prefix-chained hashes of the full engine blocks."""
    out, parent = [], b""
    for start in range(0, len(tokens) - block + 1, block):
        parent = hashlib.sha256(
            parent + tokens[start : start + block].tobytes()
        ).digest()
        out.append(parent)
    return out


# --------------------------------------------------------------------------------------------
# Replay parsing rules
# --------------------------------------------------------------------------------------------


def test_parse_follows_replay_session_and_delay_rules():
    rows = [
        {"timestamp": 100, "input_length": 10, "output_length": 2, "hash_ids": [1]},
        {
            "session_id": "s",
            "timestamp": 200,
            "input_length": 70,
            "output_length": 3,
            "hash_ids": [2, 3],
        },
        {
            "session_id": "s",
            "timestamp": 1200,
            "input_length": 130,
            "output_length": 4,
            "hash_ids": [2, 3, 4],
        },
        {
            "session_id": "s",
            "delay": 7.5,
            "timestamp": 9999,
            "input_length": 64,
            "output_length": 1,
            "hash_ids": [2, 5, 6],
        },
        {"session_id": "s", "input_length": 1, "output_length": 1, "hash_ids": [9]},
        {"output_length": 5, "hash_ids": [7, 8]},
    ]
    text = lines(rows)
    text.insert(
        1, ""
    )  # a blank line still counts toward replay's request_<line> numbering
    sessions = gen.parse_mooncake(text, block_size=64)
    assert [s.key for s in sessions] == ["request_1", "s", "request_7"]
    turns = sessions[1].turns
    assert [t.line for t in turns] == [3, 4, 5, 6]
    # timestamp difference, then explicit delay wins over a timestamp, then no timing at all
    assert [t.delay_ms for t in turns] == [0.0, 1000.0, 7.5, 0.0]
    assert sessions[1].first_timestamp_ms == 200.0
    # only ceil(ISL / B) hash ids are used; ISL defaults to the hash capacity
    assert turns[2].hash_ids == [2]
    assert sessions[2].turns[0].input_length == 128
    assert sessions[2].first_timestamp_ms is None


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [{"input_length": 65, "output_length": 1, "hash_ids": [1]}],
            "exceeds hash_ids capacity",
        ),
        (
            [
                {
                    "session_id": "a",
                    "delay": 5,
                    "input_length": 1,
                    "output_length": 1,
                    "hash_ids": [1],
                }
            ],
            "first turn",
        ),
        (
            [{"input_length": 1, "output_length": 1, "hash_ids": [1], "priority": 3}],
            "unsupported",
        ),
        ([{"input_length": 1, "hash_ids": [1]}], "missing output_length"),
    ],
)
def test_parse_rejects_what_replay_rejects_or_live_cannot_express(rows, message):
    with pytest.raises(gen.LiveInputError, match=message):
        gen.parse_mooncake(lines(rows), block_size=64)


# --------------------------------------------------------------------------------------------
# Prompt structure
# --------------------------------------------------------------------------------------------


def random_trace(seed: int, n: int, block: int) -> list[dict]:
    """Requests over a small hash tree: shared prefixes, partial last blocks, divergences."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        depth = rng.randint(1, 6)
        ids, node = [], 0
        for level in range(depth):
            node = node * 4 + rng.randint(1, 3 if level < 3 else 40)
            ids.append(node)
        isl = rng.randint((depth - 1) * block + 1, depth * block)
        rows.append({"input_length": isl, "output_length": 1, "hash_ids": ids})
    return rows


@pytest.mark.parametrize("block", [16, 64])
def test_token_lcp_equals_replay_lcp_for_every_pair(block):
    sessions = gen.parse_mooncake(lines(random_trace(1, 60, block)), block)
    synth = gen.PromptSynth(block, salt="t")
    paths = [synth.add(s.turns[0].hash_ids) for s in sessions]
    synth.finalize()
    live = [synth.tokens(p, s.turns[0].input_length) for p, s in zip(paths, sessions)]
    replay = [tokens for _, _, tokens in gen.replay_tokens(sessions, block)]
    for i, j in itertools.combinations(range(len(sessions)), 2):
        a, b = sessions[i].turns[0], sessions[j].turns[0]
        common = sum(
            1
            for _ in itertools.takewhile(
                lambda p: p[0] == p[1], zip(a.hash_ids, b.hash_ids)
            )
        )
        expected = min(a.input_length, b.input_length, common * block)
        assert lcp(live[i], live[j]) == expected == lcp(replay[i], replay[j])
    assert all(len(t) == s.turns[0].input_length for t, s in zip(live, sessions))


def test_engine_block_prefix_structure_is_isomorphic_to_replay():
    block, engine_block = 64, 16
    sessions = gen.parse_mooncake(lines(random_trace(2, 80, block)), block)
    synth = gen.PromptSynth(block, salt="iso")
    paths = [synth.add(s.turns[0].hash_ids) for s in sessions]
    synth.finalize()
    forward, backward = {}, {}
    for path, session, (_, _, replay) in zip(
        paths, sessions, gen.replay_tokens(sessions, block)
    ):
        live = synth.tokens(path, session.turns[0].input_length)
        for sim_hash, live_hash in zip(
            chain_hashes(replay, engine_block),
            chain_hashes(live, engine_block),
            strict=True,
        ):
            assert forward.setdefault(sim_hash, live_hash) == live_hash
            assert backward.setdefault(live_hash, sim_hash) == sim_hash


def test_sibling_blocks_differ_at_their_first_token_and_stay_in_vocab():
    synth = gen.PromptSynth(8, salt="v", vocab=50)
    paths = [synth.add([1, h]) for h in range(40)] + [synth.add([2])]
    synth.finalize()
    firsts = [int(synth.block(p[-1])[0]) for p in paths[:40]]
    assert len(set(firsts)) == 40
    tokens = np.concatenate([synth.tokens(p, len(p) * 8) for p in paths])
    assert tokens.max() < 50
    too_many = gen.PromptSynth(8, salt="v", vocab=3)
    for h in range(4):
        too_many.add([h])
    with pytest.raises(gen.LiveInputError, match="sibling"):
        too_many.finalize()


def test_prompts_depend_on_hash_path_and_salt_only():
    trace = random_trace(3, 30, 16)
    sessions = gen.parse_mooncake(lines(trace), 16)

    def prompts(order, salt):
        synth = gen.PromptSynth(16, salt=salt)
        paths = {i: synth.add(sessions[i].turns[0].hash_ids) for i in order}
        synth.finalize()
        return {
            i: synth.tokens(paths[i], sessions[i].turns[0].input_length) for i in order
        }

    forward = prompts(range(30), "a")
    shuffled = prompts(random.Random(0).sample(range(30), 30), "a")
    assert all(np.array_equal(forward[i], shuffled[i]) for i in range(30))
    other = prompts(range(30), "b")
    # different salts never share a full engine block, so separate runs never share KV
    live_a = {h for t in forward.values() for h in chain_hashes(t, 16)}
    live_b = {h for t in other.values() for h in chain_hashes(t, 16)}
    assert not live_a & live_b
    assert all(int(t.max()) < gen.QWEN3_REGULAR_VOCAB for t in forward.values())


# --------------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------------


def test_open_loop_schedule_uses_replays_float_operations():
    rows = [
        {
            "session_id": "a",
            "timestamp": 3053,
            "input_length": 5,
            "output_length": 2,
            "hash_ids": [1],
        },
        {
            "session_id": "a",
            "delay": 1234.5,
            "input_length": 9,
            "output_length": 2,
            "hash_ids": [1, 2],
        },
        {
            "session_id": "b",
            "timestamp": 1001.7,
            "input_length": 5,
            "output_length": 2,
            "hash_ids": [3],
        },
    ]
    sessions = gen.parse_mooncake(lines(rows), 8)
    speedup = 0.579955
    plan = gen.plan_requests(
        sessions, open_loop=True, speedup=speedup, max_model_len=None
    )
    assert [r.conversation_id for r in plan] == ["a", "a", "b"]
    assert plan[0].arrival_ms == (3053.0 - 1001.7) / speedup
    assert plan[2].arrival_ms == 0.0
    assert plan[1].arrival_ms is None and plan[1].delay_ms == 1234.5 / speedup
    assert plan[1].trace_delay_ms == 1234.5
    closed = gen.plan_requests(
        sessions, open_loop=False, speedup=None, max_model_len=None
    )
    assert all(r.arrival_ms is None for r in closed)
    assert closed[1].delay_ms == 1234.5  # concurrency mode applies no speedup
    assert gen.session_metadata_emitted(sessions, open_loop=True)


def test_session_metadata_follows_replay():
    single = gen.parse_mooncake(
        lines([{"timestamp": 0, "output_length": 1, "hash_ids": [1]}]), 8
    )
    assert not gen.session_metadata_emitted(single, open_loop=True)
    assert gen.session_metadata_emitted(single, open_loop=False)


def test_output_length_is_truncated_like_replay_at_max_model_len():
    rows = [
        {
            "timestamp": 0,
            "input_length": 90,
            "output_length": 50,
            "hash_ids": list(range(12)),
        },
        {
            "timestamp": 0,
            "input_length": 100,
            "output_length": 5,
            "hash_ids": list(range(20, 33)),
        },
    ]
    sessions = gen.parse_mooncake(lines(rows), 8)
    plan = gen.plan_requests(
        sessions[:1], open_loop=True, speedup=1.0, max_model_len=100
    )
    assert (plan[0].output_length, plan[0].osl_sent) == (50, 10)
    with pytest.raises(gen.LiveInputError, match="max_model_len"):
        gen.plan_requests(sessions, open_loop=True, speedup=1.0, max_model_len=100)


# --------------------------------------------------------------------------------------------
# End to end on a tiny campaign layout
# --------------------------------------------------------------------------------------------


def make_cell(tmp_path, rows, *, mode, value, spread=0.0, block=16, measure=None):
    root = tmp_path / "cr"
    (root / "config").mkdir(parents=True)
    (root / "traces").mkdir()
    (root / "config" / "engine.json").write_text(json.dumps(ENGINE))
    trace = root / "traces" / "t.jsonl"
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cell = {
        "cell_id": f"tiny-{mode}",
        "family": "mooncake",
        "trace_files": ["CR/traces/t.jsonl"],
        "trace_format": "mooncake",
        "trace_block_size": block,
        "engine_ref": "CR/config/engine.json",
        "num_workers": 2,
        "load": {"mode": mode, "value": value},
        "sla": {"itl_ms": 50.0, "e2e_slowdown": 2.0, "ttft_ms": None},
        "measure": measure or {"basis": "arrival"},
    }
    if spread:
        cell["arrival_spread_ms"] = spread
    cells = root / "cells.jsonl"
    cells.write_text(json.dumps(cell) + "\n")
    return load_cells(cells, Layout.resolve(root))[0]


TINY = [
    {
        "session_id": "s1",
        "timestamp": 0,
        "input_length": 40,
        "output_length": 4,
        "hash_ids": [1, 2, 3],
    },
    {
        "session_id": "s1",
        "delay": 500.0,
        "input_length": 60,
        "output_length": 3,
        "hash_ids": [1, 2, 4, 5],
    },
    {
        "session_id": "s2",
        "timestamp": 1000,
        "input_length": 20,
        "output_length": 2,
        "hash_ids": [1, 6],
    },
    {
        "session_id": "s3",
        "timestamp": 1000,
        "input_length": 16,
        "output_length": 1,
        "hash_ids": [7],
    },
]


def read_out(out):
    manifest = json.loads((out / gen.MANIFEST_FILE).read_text())
    inputs = [json.loads(x) for x in (out / gen.INPUT_FILE).read_text().splitlines()]
    requests = gen.load_requests(out)
    return manifest, inputs, requests


def test_open_cell_writes_exact_aiperf_rows(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="open_speedup", value=2.0, spread=1000.0)
    out = tmp_path / "out"
    manifest = gen.generate_cell(cell, 1, out, salt="run-1", with_e0=False)
    manifest, inputs, requests = read_out(out)
    assert manifest["replicate"]["protocol"] == "crn-spread-v1"
    assert manifest["replicate"]["policy_seed"] == 2
    assert manifest["session_header"] is True
    assert manifest["num_requests"] == len(inputs) == len(requests) == 4
    assert (
        manifest["files"][gen.INPUT_FILE]
        == hashlib.sha256((out / gen.INPUT_FILE).read_bytes()).hexdigest()
    )
    argv = manifest["aiperf"]["argv"]
    assert "--fixed-schedule" in argv and "--concurrency" not in argv
    assert (
        manifest["aiperf"]["env"]["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"]
        == "true"
    )
    # the replicate the harness materialized drives the schedule
    rep = gen.parse_mooncake(
        open(manifest["replicate"]["path"]).read().splitlines(), 16
    )
    origin = min(s.first_timestamp_ms for s in rep)
    by_session = {s.key: s for s in rep}
    for row, request in zip(inputs, requests):
        payload = row["payload"]
        assert row["session_id"] == request["conversation_id"]
        assert set(payload) == {
            "model",
            "prompt",
            "max_tokens",
            "min_tokens",
            "ignore_eos",
            "stream",
            "stream_options",
            "nvext",
        }
        assert (
            payload["max_tokens"]
            == payload["min_tokens"]
            == request["osl_sent"]
            == row["output_length"]
        )
        assert payload["ignore_eos"] is True and payload["stream"] is True
        assert len(payload["prompt"]) == request["input_length"]
        assert (
            gen.token_digest(np.asarray(payload["prompt"])) == request["prompt_sha256"]
        )
        session = by_session[request["conversation_id"]]
        if request["turn_index"] == 0:
            assert row["timestamp"] == (session.first_timestamp_ms - origin) / 2.0
            assert "delay" not in row
        else:
            assert row["delay"] == 500.0 / 2.0 and "timestamp" not in row
    # turn 2 of s1 extends turn 1's two shared full blocks; s2 shares one block
    prompts = {
        (r["conversation_id"], r["turn_index"]): np.asarray(i["payload"]["prompt"])
        for r, i in zip(requests, inputs)
    }
    assert lcp(prompts[("s1", 0)], prompts[("s1", 1)]) == 32
    assert lcp(prompts[("s1", 0)], prompts[("s2", 0)]) == 16
    assert lcp(prompts[("s1", 0)], prompts[("s3", 0)]) == 0


def test_closed_cell_and_replay_verification(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="closed_concurrency", value=2)
    out = tmp_path / "out"
    gen.generate_cell(cell, 0, out, salt="run-2", with_e0=False, worker_id_field=False)
    manifest, inputs, requests = read_out(out)
    assert all("timestamp" not in row for row in inputs)
    assert "nvext" not in inputs[0]["payload"]
    argv = manifest["aiperf"]["argv"]
    assert argv[argv.index("--concurrency") + 1] == "2"
    assert argv[argv.index("--num-sessions") + 1] == "3"
    assert argv[argv.index("--dataset-sampling-strategy") + 1] == "sequential"
    order = [r["conversation_id"] for r in requests if r["turn_index"] == 0]
    # replay-shaped rows: the first C sessions start at 0, the third when one finishes
    terminal = {"s1": 100.0, "s2": 50.0, "s3": 80.0}
    start = {order[0]: 0.0, order[1]: 0.0, order[2]: 50.0}
    rows = []
    for r in requests:
        key = r["conversation_id"]
        arrival = start[key] if r["turn_index"] == 0 else terminal[key] + r["delay_ms"]
        rows.append(
            {
                "session_id": key,
                "turn_index": r["turn_index"],
                "arrival_time_ms": arrival,
                "terminal_time_ms": arrival + 100.0
                if r["turn_index"] == 0
                else arrival + 1.0,
                "input_length": r["input_length"],
                "requested_output_length": r["output_length"],
            }
        )
        if r["turn_index"] == 0:
            terminal[key] = arrival + 100.0
    report = gen.verify_against_replay(manifest, requests, rows)
    assert report["exact"], report
    second = next(
        r for r in rows if r["session_id"] == order[1] and r["turn_index"] == 0
    )
    second["arrival_time_ms"] += 1.0  # admitted late: only one session starts at 0
    report = gen.verify_against_replay(manifest, requests, rows)
    assert not report["exact"] and report["sessions_started_at_zero"] == 1


def test_smoke_subset_is_a_replay_order_prefix_with_its_own_trace(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="open_speedup", value=1.0, spread=1000.0)
    full = gen.plan_cell(cell, 0)
    out = tmp_path / "out"
    gen.generate_cell(cell, 0, out, salt="s", with_e0=False, max_sessions=2)
    manifest, inputs, requests = read_out(out)
    assert manifest["subset"] == {"max_sessions": 2, "of_sessions": 3}
    assert [r["conversation_id"] for r in requests] == [
        r.conversation_id for r in full.requests
    ][: len(requests)]
    assert [r["arrival_ms"] for r in requests] == [r.arrival_ms for r in full.requests][
        : len(requests)
    ]
    subset = (out / gen.SUBSET_TRACE_FILE).read_text()
    assert (
        manifest["files"][gen.SUBSET_TRACE_FILE]
        == hashlib.sha256(subset.encode()).hexdigest()
    )
    keys = [s.key for s in gen.parse_mooncake(subset.splitlines(), 16)]
    assert keys == [s.key for s in full.sessions[:2]]


def test_agentx_lanes_are_refused(tmp_path):
    cell = make_cell(tmp_path, TINY, mode="closed_concurrency", value=2)
    cell.raw["load"] = {"mode": "agentic_lanes", "value": 4}
    with pytest.raises(gen.LiveInputError, match="not supported"):
        gen.generate_cell(cell, 0, tmp_path / "out", salt="x", with_e0=False)


def test_idle_calibration_prompts_share_no_block(tmp_path):
    manifest = gen.generate_idle(tmp_path, salt="idle", isls=(32, 64), osl=8, repeats=2)
    inputs = [
        json.loads(x) for x in (tmp_path / gen.INPUT_FILE).read_text().splitlines()
    ]
    assert manifest["concurrency"] == 1 and manifest["num_requests"] == 4
    assert [len(r["payload"]["prompt"]) for r in inputs] == [32, 64, 32, 64]
    firsts = {r["payload"]["prompt"][0] for r in inputs}
    assert len(firsts) == 4
    assert all("timestamp" not in r and "delay" not in r for r in inputs)
