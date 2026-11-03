# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live retune window cutter.

The golden tests compare the cut window, decompressed, byte for byte with
``read_trace_window`` over the same clean segments. A few tests spawn the
real child process; they carry a timeout.
"""

import errno
import gzip
import io
import json
import os
import pickle
import random
import shutil
import stat
import threading
import time
from dataclasses import replace

import pytest

from dynamo.planner.retune.live import window_cut
from dynamo.planner.retune.live.contracts import WindowCutResult
from dynamo.planner.retune.live.window_cut import (
    MAX_DROPPED_MEMBER_RATIO,
    SEGMENT_MTIME_MARGIN_S,
    TEMP_NAME_RE,
    WINDOW_CUT_FAILURES,
    WindowCutError,
    _decode_exact_member,
    _Garbage,
    _GzipMemberReader,
    _Member,
    _run_in_child,
    _segment_files,
    cut_window,
    cut_window_inprocess,
    temp_file_name,
)
from dynamo.planner.retune.window import ROW_COUNT_KEYS, read_trace_window

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

SCHEMA = "dynamo.request.trace.v1"
START, END, AS_OF = 5_000, 15_000, 15_500
_UNSET = object()


# ------------------------------------------------------------------ helpers


def _row(
    request_id,
    received_ms=_UNSET,
    *,
    total_ms=10.0,
    event_ms=None,
    isl=128,
    osl=8,
    block_size=64,
    model="m",
    agent_context=None,
    output_tokens=_UNSET,
):
    request = {"request_id": request_id}
    if model is not None:
        request["model"] = model
    if received_ms is not _UNSET and received_ms is not None:
        request["request_received_ms"] = received_ms
    if total_ms is not None:
        request["total_time_ms"] = total_ms
    if output_tokens is not None:
        request["output_tokens"] = osl if output_tokens is _UNSET else output_tokens
    request["replay"] = {
        "trace_block_size": block_size,
        "input_length": isl,
        "input_sequence_hashes": [len(request_id), 7, 11_400_714_819_323_198_485],
    }
    if event_ms is None:
        base = received_ms if isinstance(received_ms, int) else 0
        event_ms = base + (round(total_ms) if total_ms is not None else 0)
    record = {
        "schema": SCHEMA,
        "event_type": "request_end",
        "event_time_unix_ms": event_ms,
        "request": request,
    }
    if agent_context is not None:
        record["agent_context"] = agent_context
    return record


def _envelope(record, timestamp=0):
    # The frontend's gzip sink wraps every record like this.
    return {"timestamp": timestamp, "event": record}


def _lines(records):
    return b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)


def _member(records):
    return gzip.compress(_lines(records))


def _write(path, *chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    return path


def _out(tmp_path, name="w.jsonl.gz"):
    return tmp_path / "windows" / "dgd" / name


def _cut(root, tmp_path, start=START, end=END, as_of=AS_OF, *, max_rows=10**6):
    out = _out(tmp_path)
    return cut_window_inprocess(root, start, end, as_of, out, max_rows=max_rows), out


def _golden(root, tmp_path, start=START, end=END, as_of=AS_OF):
    out = tmp_path / "golden" / "window.jsonl"
    window = read_trace_window([str(root)], start, end, as_of, out)
    return window, out.read_bytes()


def _assert_golden(result, out, window, golden_bytes):
    assert isinstance(result, WindowCutResult)
    assert result.num_requests == window.num_requests
    assert result.trace_block_size == window.trace_block_size
    assert result.row_counts == window.row_counts
    assert set(result.row_counts) == set(ROW_COUNT_KEYS)
    if window.num_requests:
        assert result.path == str(out)
        assert gzip.decompress(out.read_bytes()) == golden_bytes
    else:
        assert result.path is None
        assert not out.exists()


def _temp_files(directory):
    if not directory.exists():
        return []
    return [name for name in os.listdir(directory) if name.startswith(".tmp-")]


def _random_trace(root, seed, *, pods=3, segments=3, members=5):
    """Segments with every row shape ``read_trace_window`` handles.

    Arrivals and durations are coarse so (arrival, completion) pairs tie
    across files; some request IDs repeat (first in file order wins), some
    are not ASCII.
    """

    rng = random.Random(seed)
    issued = []
    serial = 0
    for pod in range(pods):
        for segment in range(segments):
            chunks = []
            for _ in range(members):
                records = []
                for _ in range(rng.randint(0, 14)):
                    roll = rng.random()
                    if roll < 0.04:
                        records.append(
                            {
                                "schema": SCHEMA,
                                "event_type": rng.choice(["tool_start", "tool_end"]),
                                "event_time_unix_ms": rng.randrange(20_000),
                                "agent_context": {"session_id": "s"},
                            }
                        )
                        continue
                    if issued and roll < 0.16:
                        request_id = rng.choice(issued)
                    else:
                        serial += 1
                        prefix = rng.choice(["req-", "req-", "réq-", "請求-", "Req-"])
                        request_id = f"{prefix}{serial % 97}-{serial}"
                        issued.append(request_id)
                    received = rng.randrange(0, 400) * 50
                    total = rng.choice([10.0, 10.5, 49.5, 99.4, 600.0, 1_200.25])
                    kwargs = {}
                    if rng.random() < 0.08:
                        kwargs["received_ms"] = None
                        kwargs["event_ms"] = received + round(total)
                    else:
                        kwargs["received_ms"] = received
                    if rng.random() < 0.05:
                        kwargs["total_ms"] = None
                        kwargs["event_ms"] = received + rng.randrange(0, 900)
                    else:
                        kwargs["total_ms"] = total
                    if rng.random() < 0.03:
                        kwargs["output_tokens"] = None
                    if rng.random() < 0.05:
                        kwargs["agent_context"] = {"session_id": f"s{serial}"}
                    if rng.random() < 0.1:
                        kwargs["model"] = None
                    received_arg = kwargs.pop("received_ms")
                    record = _row(
                        request_id,
                        received_arg,
                        isl=rng.randrange(1, 4096),
                        osl=rng.randrange(1, 512),
                        **kwargs,
                    )
                    if rng.random() < 0.5:
                        record = _envelope(record, timestamp=rng.randrange(10**6))
                    records.append(record)
                payload = _lines(records)
                if rng.random() < 0.1:
                    payload += b"\n   \n"  # blank lines are skipped
                chunks.append(payload)
            _write(
                root / f"frontend-{pod}" / f"req.{segment:06d}.jsonl.gz",
                *(gzip.compress(chunk) for chunk in chunks),
            )
    return root


def _items(data):
    return list(_GzipMemberReader(io.BytesIO(data)))


# ----------------------------------------------------------- gzip members


@pytest.fixture(params=["default", "tiny", "small"])
def reader_chunks(request, monkeypatch):
    """Run reader tests with reads and decode steps that split headers."""

    sizes = {"default": None, "tiny": (7, 5, 16), "small": (64, 3, 128)}[request.param]
    if sizes is not None:
        read_chunk, decode_step, compact_at = sizes
        monkeypatch.setattr(window_cut, "_READ_CHUNK", read_chunk)
        monkeypatch.setattr(window_cut, "_DECODE_STEP", decode_step)
        monkeypatch.setattr(window_cut, "_COMPACT_AT", compact_at)
    return request.param


def _spans(*members):
    offsets = [0]
    for member in members:
        offsets.append(offsets[-1] + len(member))
    return offsets


def test_clean_multi_member_file(reader_chunks):
    payloads = [b'{"a":1}\n', b"", b'{"b":2}\n{"c":3}\n', b"x" * 5000 + b"\n"]
    members = [gzip.compress(payload) for payload in payloads]
    offsets = _spans(*members)

    items = _items(b"".join(members))

    assert items == [
        _Member(offsets[i], offsets[i + 1], payloads[i]) for i in range(len(payloads))
    ]


@pytest.mark.parametrize("keep", ["header", "middle", "trailer"])
def test_truncated_last_member_is_dropped(reader_chunks, keep):
    first, last = gzip.compress(b"a\n" * 50), gzip.compress(b"b\n" * 50)
    cut = {"header": 5, "middle": len(last) // 2, "trailer": len(last) - 3}[keep]

    items = _items(first + last[:cut])

    assert items == [
        _Member(0, len(first), b"a\n" * 50),
        _Garbage(len(first), len(first) + cut, at_eof=True),
    ]


def test_partial_member_followed_by_complete_members(reader_chunks):
    first = gzip.compress(b"first\n" * 40)
    full = gzip.compress(b"".join(b"partial %d\n" % i for i in range(400)))
    partial = full[: len(full) // 2]
    third = gzip.compress(b"third\n")
    fourth = gzip.compress(b"fourth\n")
    offsets = _spans(first, partial, third, fourth)

    items = _items(first + partial + third + fourth)

    assert items == [
        _Member(offsets[0], offsets[1], b"first\n" * 40),
        _Garbage(offsets[1], offsets[2], at_eof=False),
        _Member(offsets[2], offsets[3], b"third\n"),
        _Member(offsets[3], offsets[4], b"fourth\n"),
    ]


@pytest.mark.parametrize("where", ["crc", "isize", "data"])
def test_flipped_member_is_dropped_and_reader_resyncs(reader_chunks, where):
    first = gzip.compress(b"first\n")
    middle = bytearray(gzip.compress(b"middle row\n" * 30))
    index = {"crc": len(middle) - 8, "isize": len(middle) - 2, "data": 14}[where]
    middle[index] ^= 0x40
    last = gzip.compress(b"last\n")
    offsets = _spans(first, middle, last)

    items = _items(first + bytes(middle) + last)

    assert items == [
        _Member(offsets[0], offsets[1], b"first\n"),
        _Garbage(offsets[1], offsets[2], at_eof=False),
        _Member(offsets[2], offsets[3], b"last\n"),
    ]


def test_zero_filled_hole_between_members(reader_chunks):
    first, last = gzip.compress(b"first\n"), gzip.compress(b"last\n")
    hole = bytes(4096)

    items = _items(first + hole + last)

    assert items == [
        _Member(0, len(first), b"first\n"),
        _Garbage(len(first), len(first) + len(hole), at_eof=False),
        _Member(len(first) + len(hole), len(first) + len(hole) + len(last), b"last\n"),
    ]


def test_false_gzip_magic_inside_garbage_is_one_run(reader_chunks):
    first, last = gzip.compress(b"first\n"), gzip.compress(b"last\n")
    junk = b"\x1f\x8b\x08\x00junk" * 20 + b"\x1f\x8b"

    items = _items(first + junk + last)

    assert items == [
        _Member(0, len(first), b"first\n"),
        _Garbage(len(first), len(first) + len(junk), at_eof=False),
        _Member(len(first) + len(junk), len(first) + len(junk) + len(last), b"last\n"),
    ]


def test_garbage_only_and_empty_files(reader_chunks):
    assert _items(b"") == []
    assert _items(b"plain text, not gzip\n") == [_Garbage(0, 21, at_eof=True)]
    member = gzip.compress(b"a\n")
    assert _items(member + bytes(10)) == [
        _Member(0, len(member), b"a\n"),
        _Garbage(len(member), len(member) + 10, at_eof=True),
    ]


def test_reader_starts_at_an_offset():
    first, second = gzip.compress(b"first\n"), gzip.compress(b"second\n")
    source = io.BytesIO(first + second)

    items = list(_GzipMemberReader(source, len(first)))

    assert items == [_Member(len(first), len(first) + len(second), b"second\n")]


def test_decode_exact_member():
    member = gzip.compress(b"row\n")
    assert _decode_exact_member(member) == b"row\n"
    assert _decode_exact_member(member + b"x") is None
    assert _decode_exact_member(member[:-1]) is None
    flipped = bytearray(member)
    flipped[-6] ^= 1
    assert _decode_exact_member(bytes(flipped)) is None
    assert _decode_exact_member(b"") is None


# ------------------------------------------------------------ golden cuts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cut_matches_read_trace_window(tmp_path, seed):
    root = _random_trace(tmp_path / "segments" / "dgd", seed)
    window, golden_bytes = _golden(root, tmp_path)
    assert window.num_requests > 50
    assert window.row_counts["duplicate"] > 0
    assert window.row_counts["censored"] > 0

    result, out = _cut(root, tmp_path)

    _assert_golden(result, out, window, golden_bytes)
    assert result.dropped_members == 0
    assert result.dropped_bytes == 0
    assert not result.reused
    assert stat.S_IMODE(out.stat().st_mode) == 0o644
    assert _temp_files(out.parent) == []


def test_plain_jsonl_files_are_ignored(tmp_path):
    # The reconciler never collects them, and the live sink writes gzip.
    root = _random_trace(tmp_path / "segments" / "dgd", 3)
    window, golden_bytes = _golden(root, tmp_path)
    for pod in ("frontend-0", "frontend-9"):
        _write(root / pod / "req.000000.jsonl", _lines([_row("plain", START + 1)]))

    result, out = _cut(root, tmp_path)

    _assert_golden(result, out, window, golden_bytes)


def test_cut_matches_golden_under_eviction_and_small_reads(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments" / "dgd", 4, members=8)
    window, golden_bytes = _golden(root, tmp_path)
    monkeypatch.setattr(window_cut, "_PASS2_CACHE_BYTES", 1)
    monkeypatch.setattr(window_cut, "_PASS2_BLOCK_ROWS", 7)
    monkeypatch.setattr(window_cut, "_READ_CHUNK", 97)
    monkeypatch.setattr(window_cut, "_DECODE_STEP", 13)
    monkeypatch.setattr(window_cut, "_COMPACT_AT", 64)
    monkeypatch.setattr(window_cut, "_WRITE_BUFFER", 10)
    monkeypatch.setattr(window_cut, "_OPEN_FILES", 2)
    loads = []
    original_load = window_cut._MemberLines._load

    def counting_load(self, member_id):
        loads.append(member_id)
        return original_load(self, member_id)

    monkeypatch.setattr(window_cut._MemberLines, "_load", counting_load)

    result, out = _cut(root, tmp_path)

    _assert_golden(result, out, window, golden_bytes)
    # Every member was evicted after each row and decoded again.
    assert len(loads) == window.num_requests
    assert len(set(loads)) < len(loads)


def test_dedup_compaction_keeps_first_rows_and_bounds_rows(tmp_path, monkeypatch):
    rows = [_row(f"r{i}", START + 10 * i) for i in range(10)]
    # Many repeats of the same IDs, in later files: the first row wins.
    repeats = [_row(f"r{i % 10}", START + 5 + i, isl=999) for i in range(40)]
    root = tmp_path / "segments"
    _write(root / "a" / "req.000000.jsonl.gz", _member(rows))
    _write(root / "b" / "req.000000.jsonl.gz", *(_member([row]) for row in repeats))
    window, golden_bytes = _golden(root, tmp_path)
    assert window.num_requests == 10
    monkeypatch.setattr(window_cut, "_DEDUP_SLACK_MIN", 1)
    monkeypatch.setattr(window_cut, "_DEDUP_SLACK_RATIO", 0.0)
    compactions = []
    original_dedup = window_cut._Selection._dedup

    def counting_dedup(self):
        compactions.append(self.num_rows)
        original_dedup(self)

    monkeypatch.setattr(window_cut._Selection, "_dedup", counting_dedup)

    result, out = _cut(root, tmp_path, max_rows=10)

    _assert_golden(result, out, window, golden_bytes)
    assert result.row_counts["duplicate"] == 40
    assert len(compactions) > 2  # during pass 1, not only at the end


def test_too_many_rows_is_window_too_large(tmp_path, monkeypatch):
    root = tmp_path / "segments"
    rows = [_row(f"r{i}", START + i) for i in range(30)]
    _write(root / "a" / "req.000000.jsonl.gz", _member(rows))
    monkeypatch.setattr(window_cut, "_DEDUP_SLACK_MIN", 1)
    monkeypatch.setattr(window_cut, "_DEDUP_SLACK_RATIO", 0.0)

    result, out = _cut(root, tmp_path, max_rows=30)
    assert result.num_requests == 30
    out.unlink()
    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path, max_rows=29)
    assert caught.value.reason == "window_too_large"


def test_empty_window_writes_no_file(tmp_path):
    root = tmp_path / "segments"
    _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("early", START - 100), _row("late", END + 1)]),
        _member([_row("no-output", START + 1, output_tokens=None)]),
    )
    window, golden_bytes = _golden(root, tmp_path)

    result, out = _cut(root, tmp_path)

    assert golden_bytes == b""
    _assert_golden(result, out, window, golden_bytes)
    assert result.row_counts["missing_output_tokens"] == 1
    assert not out.parent.exists()


def test_empty_segments_root_is_an_empty_window(tmp_path):
    root = tmp_path / "segments"
    (root / "frontend-0").mkdir(parents=True)

    result, out = _cut(root, tmp_path)

    assert result.path is None
    assert result.num_requests == 0
    assert result.trace_block_size is None


def test_missing_segments_root_fails(tmp_path):
    with pytest.raises(WindowCutError) as caught:
        _cut(tmp_path / "nowhere", tmp_path)
    assert caught.value.reason == "window_cut_failed"
    assert "does not exist" in str(caught.value)


def test_mixed_block_sizes_fail(tmp_path):
    root = tmp_path / "segments"
    _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("a", START, block_size=64), _row("b", START, block_size=16)]),
    )
    with pytest.raises(ValueError, match="mixes trace_block_size"):
        _golden(root, tmp_path)

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "window_cut_failed"
    assert "mixes trace_block_size values [16, 64]" in str(caught.value)


@pytest.mark.parametrize(
    "line, message",
    [
        (b"{not json", "invalid JSON"),
        (b"[1, 2]", "must be a JSON object"),
        (b'{"schema": "other", "event_type": "request_end"}', "expected"),
        (
            json.dumps(
                {
                    "schema": SCHEMA,
                    "event_type": "request_end",
                    "event_time_unix_ms": 1,
                    "request": {
                        "request_id": "x",
                        "output_tokens": 1,
                        "replay": {"input_length": 1, "trace_block_size": 64},
                    },
                }
            ).encode(),
            "input_sequence_hashes",
        ),
    ],
)
def test_malformed_row_in_a_valid_member_is_trace_corrupt(tmp_path, line, message):
    # The member's CRC is good: the writer wrote this, as replay would see it.
    root = tmp_path / "segments"
    _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("good", START)]),
        gzip.compress(line + b"\n"),
    )

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "trace_corrupt"
    assert message in str(caught.value)


def test_out_of_range_value_is_trace_corrupt(tmp_path):
    root = tmp_path / "segments"
    _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("huge", START, block_size=2**70)]),
    )

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "trace_corrupt"


# --------------------------------------------------- dropped gzip members


def _tiny_members(prefix, count, first_ms=START):
    return [
        _member([_envelope(_row(f"{prefix}{i}", first_ms + i), timestamp=i)])
        for i in range(count)
    ]


def _cut_and_twin(tmp_path, build):
    """Cut ``build(bad=True)`` and the golden of ``build(bad=False)``.

    ``build(root, bad)`` writes the segments with or without the bad bytes;
    the rows of a dropped member never reach the window.
    """

    clean = tmp_path / "clean"
    build(clean, bad=False)
    window, golden_bytes = _golden(clean, tmp_path)
    damaged = tmp_path / "damaged"
    build(damaged, bad=True)
    result, out = _cut(damaged, tmp_path)
    _assert_golden(result, out, window, golden_bytes)
    return result


@pytest.mark.parametrize("truncated", ["req.000001", "req.000000"])
def test_truncated_tail_of_a_segment_is_not_corruption(tmp_path, truncated):
    # req.000001's tail is the flush its writer is appending. req.000000's is
    # the last flush of a writer that crashed: after an in-place restart the
    # sink continues in req.000001.
    lost = _member([_row("lost", START + 900)])

    def build(root, bad):
        for name, prefix in (("req.000000", "old"), ("req.000001", "new")):
            tail = [lost[: len(lost) // 2]] if bad and name == truncated else []
            _write(
                root / "a" / f"{name}.jsonl.gz",
                *_tiny_members(prefix, 3),
                *tail,
            )

    result = _cut_and_twin(tmp_path, build)

    # 1 of 7 members is far above the ratio, but it is a writer's tail.
    assert result.dropped_members == 1
    assert result.dropped_bytes == len(lost) // 2


@pytest.mark.parametrize("damage", ["partial", "crc", "hole"])
def test_damaged_members_within_the_ratio_are_dropped(tmp_path, damage):
    # 1 bad member among 100 is exactly the limit: the cut goes on. Its bytes
    # are at most one member's, below the limit too.
    count = int(1 / MAX_DROPPED_MEMBER_RATIO) - 1
    bad_row = _member([_row("bad", START + 50)])
    if damage == "partial":
        damaged = bad_row[: len(bad_row) - 12]
    elif damage == "crc":
        flipped = bytearray(bad_row)
        flipped[-7] ^= 0x10
        damaged = bytes(flipped)
    else:
        damaged = bytes(len(bad_row))
    assert len(damaged) <= min(map(len, _tiny_members("m", count)))

    def build(root, bad):
        members = _tiny_members("m", count)
        if bad:
            members.insert(count // 2, damaged)
        _write(root / "a" / "req.000000.jsonl.gz", *members)

    result = _cut_and_twin(tmp_path, build)

    assert result.dropped_members == 1
    assert result.dropped_bytes == len(damaged)


def test_damaged_members_above_the_ratio_are_trace_corrupt(tmp_path):
    root = tmp_path / "segments"
    members = _tiny_members("m", 98)
    members.insert(10, bytes(64))
    members.insert(50, bytes(64))
    _write(root / "a" / "req.000000.jsonl.gz", *members)

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "trace_corrupt"
    assert "2 of 100 gzip members" in str(caught.value)


def test_hole_spanning_many_members_is_trace_corrupt(tmp_path):
    # A zero-filled hole over 20 of 200 members is one run of bad bytes: 1 of
    # 181 members is within the ratio, but 10% of the bytes is not.
    root = tmp_path / "segments"
    members = _tiny_members("m", 200)
    hole = sum(map(len, members[50:70]))
    data = b"".join([*members[:50], bytes(hole), *members[70:]])
    _write(root / "a" / "req.000000.jsonl.gz", data)

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)

    assert caught.value.reason == "trace_corrupt"
    message = str(caught.value)
    assert "1 of 181 gzip members" in message
    assert f"({hole} of " in message
    assert f"req.000000.jsonl.gz@{sum(map(len, members[:50]))}+{hole}" in message


def test_hole_within_the_byte_ratio_is_dropped(tmp_path):
    # A hole over 1 of 200 members: 1 run, under 1% of the bytes.
    members = _tiny_members("m", 200)

    def build(root, bad):
        kept = [*members[:50], *members[51:]]
        if bad:
            kept.insert(50, bytes(len(members[50])))
        _write(root / "a" / "req.000000.jsonl.gz", *kept)

    result = _cut_and_twin(tmp_path, build)

    assert result.dropped_members == 1
    assert result.dropped_bytes == len(members[50])


def test_segment_files_order(tmp_path):
    root = tmp_path / "segments"
    for name in (
        "b/req.000999.jsonl.gz",
        "b/req.1000000.jsonl.gz",
        "a/req.000000.jsonl.gz",
        "a/req.000001.jsonl",  # plain JSONL: ignored
        "a/notes.txt",
        "c/single.jsonl.gz",
    ):
        _write(root / name, b"")

    files = _segment_files(str(root))

    assert files == [
        str(root / "a/req.000000.jsonl.gz"),
        str(root / "b/req.000999.jsonl.gz"),
        str(root / "b/req.1000000.jsonl.gz"),
        str(root / "c/single.jsonl.gz"),
    ]


def test_a_segments_root_that_is_not_a_gzip_segment_fails(tmp_path):
    plain = _write(tmp_path / "req.000000.jsonl", _lines([_row("a", START)]))

    with pytest.raises(WindowCutError) as caught:
        _cut(plain, tmp_path)
    assert caught.value.reason == "window_cut_failed"
    single = _write(tmp_path / "req.000000.jsonl.gz", _member([_row("a", START)]))
    assert _cut(single, tmp_path)[0].num_requests == 1


def test_segments_last_written_before_the_window_are_not_read(tmp_path):
    # A row is written when its request completes, after it arrived: a
    # segment last modified before the window started holds none of its
    # rows. This one does (it cannot in production) to show it is not read,
    # and its hole would fail the cut as trace_corrupt if it were.
    start, end = 1_700_000_000_000, 1_700_000_010_000
    root = tmp_path / "segments"
    old = _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("old-0", start + 1)]),
        bytes(64),
        _member([_row("old-1", start + 2)]),
    )
    _write(root / "a" / "req.000001.jsonl.gz", _member([_row("new", start + 3)]))
    out = _out(tmp_path)

    def cut():
        return cut_window_inprocess(root, start, end, end + 500, out, max_rows=10)

    skipped = start / 1000 - SEGMENT_MTIME_MARGIN_S - 1
    os.utime(old, (skipped, skipped))
    result = cut()
    assert result.num_requests == 1
    assert result.dropped_members == 0
    out.unlink()

    within_margin = start / 1000 - SEGMENT_MTIME_MARGIN_S + 1
    os.utime(old, (within_margin, within_margin))
    with pytest.raises(WindowCutError) as caught:
        cut()
    assert caught.value.reason == "trace_corrupt"


def test_vanished_file_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "segments"
    _write(root / "a" / "req.000000.jsonl.gz", _member([_row("a", START)]))
    listed = _segment_files(str(root))
    gone = str(root / "a" / "req.000001.jsonl.gz")
    monkeypatch.setattr(window_cut, "_segment_files", lambda _: [*listed, gone])

    result, _ = _cut(root, tmp_path)

    assert result.num_requests == 1


def _two_pods(root):
    for pod in ("a", "b"):
        _write(
            root / pod / "req.000000.jsonl.gz",
            _member([_row(f"{pod}-{i}", START + i) for i in range(5)]),
        )
    return root


def _skip_as_root():
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")


@pytest.mark.parametrize("mode", [0o000, 0o600])
def test_unreadable_pod_directory_fails(tmp_path, mode):
    # 0o000 cannot be listed; 0o600 can, but its files cannot be stat-ed.
    # Either way that pod's rows would be missing from the window.
    _skip_as_root()
    root = _two_pods(tmp_path / "segments")
    pod = root / "b"
    pod.chmod(mode)
    try:
        with pytest.raises(WindowCutError) as caught:
            _cut(root, tmp_path)
    finally:
        pod.chmod(0o755)

    assert caught.value.reason == "window_cut_failed"
    assert str(pod) in str(caught.value)
    assert "PermissionError" in str(caught.value)


def test_unreadable_segments_root_fails(tmp_path):
    # Not an empty window: the cut cannot see any segment.
    _skip_as_root()
    root = _two_pods(tmp_path / "segments")
    root.chmod(0o000)
    try:
        with pytest.raises(WindowCutError) as caught:
            _cut(root, tmp_path)
    finally:
        root.chmod(0o755)

    assert caught.value.reason == "window_cut_failed"
    assert str(root) in str(caught.value)


@pytest.mark.parametrize("code", [errno.ENOENT, errno.EACCES])
def test_segment_walk_skips_only_vanished_entries(tmp_path, monkeypatch, code):
    root = _two_pods(tmp_path / "segments")
    pod = root / "b"
    real = os.scandir

    def failing(path, *args, **kwargs):
        if os.fspath(path) == str(pod):
            raise OSError(code, os.strerror(code), os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", failing)

    if code == errno.ENOENT:
        # The reconciler deleted pod b's segments under the walk.
        assert _segment_files(str(root)) == [str(root / "a" / "req.000000.jsonl.gz")]
        return
    with pytest.raises(WindowCutError) as caught:
        _segment_files(str(root))
    assert caught.value.reason == "window_cut_failed"
    assert str(pod) in str(caught.value)
    assert os.strerror(code) in str(caught.value)


# -------------------------------------------------------- atomic output


def _reused(result, path):
    return replace(
        result, path=str(path), reused=True, dropped_members=0, dropped_bytes=0
    )


def _header_comment(path):
    data = path.read_bytes()
    return json.loads(data[10 : data.index(b"\x00", 10)])


def test_published_window_is_reused_without_reading_the_trace(tmp_path):
    root = _random_trace(tmp_path / "segments", 5)
    first, out = _cut(root, tmp_path)
    published = out.read_bytes()
    assert _header_comment(out)["selection"] == {
        "start_ms": START,
        "end_ms": END,
        "as_of_ms": AS_OF,
    }
    # A later leader adopting the window must not fail it on segments that
    # changed, broke or were collected since it was published.
    shutil.rmtree(root)

    result, _ = _cut(root, tmp_path)

    assert result == _reused(first, out)
    assert out.read_bytes() == published
    assert _temp_files(out.parent) == []


@pytest.mark.parametrize(
    "selection",
    [{"start": START + 1000}, {"as_of": AS_OF + 100}],
    ids=["start", "as_of"],
)
def test_published_window_of_another_selection_is_not_reused(tmp_path, selection):
    root = _random_trace(tmp_path / "segments", 5)
    _, out = _cut(root, tmp_path)
    published = out.read_bytes()

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path, **selection)

    assert caught.value.reason == "window_cut_failed"
    assert f"was cut for [{START}, {END}) as of {AS_OF}, not [" in str(caught.value)
    assert out.read_bytes() == published
    assert _temp_files(out.parent) == []


def test_published_window_without_a_selection_is_not_reused(tmp_path):
    root = _random_trace(tmp_path / "segments", 5)
    _, out = _cut(root, tmp_path)
    data = out.read_bytes()
    end = data.index(b"\x00", 10)
    counts = _header_comment(out)
    del counts["selection"]
    out.write_bytes(data[:10] + json.dumps(counts).encode("ascii") + data[end:])

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "window_cut_failed"
    assert "holds no selection" in str(caught.value)


def test_a_window_file_without_cut_counts_fails(tmp_path):
    root = _random_trace(tmp_path / "segments", 5)
    out = _out(tmp_path)
    _write(out, gzip.compress(_lines([_row("a", START)])))

    with pytest.raises(WindowCutError) as caught:
        cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)
    assert caught.value.reason == "window_cut_failed"
    assert "holds no cut counts" in str(caught.value)


def _lose_link_race_to(monkeypatch, winner_out):
    """Make the cut's link find ``winner_out`` already published at its target."""

    real_link = os.link
    linked = []

    def racing_link(source, target):
        linked.append(os.path.basename(source))
        real_link(winner_out, target)  # another process published first
        return real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)
    return linked


def test_link_race_reports_the_winners_counts(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments", 6)
    winner_out = _out(tmp_path, "winner.jsonl.gz")
    winner = cut_window_inprocess(
        _random_trace(tmp_path / "other", 7),
        START,
        END,
        AS_OF,
        winner_out,
        max_rows=10**6,
    )
    assert winner.num_requests != _golden(root, tmp_path)[0].num_requests
    out = _out(tmp_path)
    linked = _lose_link_race_to(monkeypatch, winner_out)

    result = cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)

    assert result == _reused(winner, out)
    assert out.read_bytes() == winner_out.read_bytes()
    assert linked[0].startswith(f".tmp-{END}-")
    assert _temp_files(out.parent) == []


def test_link_race_won_by_another_selection_fails(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments", 6)
    winner_out = _out(tmp_path, "winner.jsonl.gz")
    cut_window_inprocess(root, START + 1000, END, AS_OF, winner_out, max_rows=10**6)
    out = _out(tmp_path)
    linked = _lose_link_race_to(monkeypatch, winner_out)

    with pytest.raises(WindowCutError) as caught:
        cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)

    assert caught.value.reason == "window_cut_failed"
    assert f"was cut for [{START + 1000}, {END}) as of {AS_OF}" in str(caught.value)
    assert linked and out.read_bytes() == winner_out.read_bytes()
    assert _temp_files(out.parent) == []


def test_link_enoent_is_window_cut_failed(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments", 7)
    out = _out(tmp_path)

    def vanished_link(source, target):
        raise FileNotFoundError(2, "No such file or directory", source)

    monkeypatch.setattr(os, "link", vanished_link)

    with pytest.raises(WindowCutError) as caught:
        cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)
    assert caught.value.reason == "window_cut_failed"
    assert not out.exists()
    assert _temp_files(out.parent) == []


def test_output_is_deterministic_and_0644_under_any_umask(tmp_path):
    root = _random_trace(tmp_path / "segments", 8)
    first, second = _out(tmp_path, "a.jsonl.gz"), _out(tmp_path, "b.jsonl.gz")
    previous = os.umask(0o077)
    try:
        cut_window_inprocess(root, START, END, AS_OF, first, max_rows=10**6)
        cut_window_inprocess(root, START, END, AS_OF, second, max_rows=10**6)
    finally:
        os.umask(previous)

    assert first.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o644
    with gzip.open(first, "rb") as source:
        assert source.read(1)
    # A comment (the counts) but no file name or mtime in the gzip header.
    header = first.read_bytes()[:10]
    assert header[3] == 0x10
    assert header[4:8] == b"\x00\x00\x00\x00"


@pytest.mark.parametrize("change", ["rewritten", "truncated", "deleted"])
def test_segment_changed_between_passes_fails(tmp_path, monkeypatch, change):
    root = tmp_path / "segments"
    segment = _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("a", START), _row("b", START + 1)]),
        _member([_row("c", START + 2)]),
    )
    original_write_rows = window_cut._write_rows

    def mutate_then_write(selection, paths, sink):
        if change == "rewritten":
            segment.write_bytes(
                _member([_row("x", START), _row("y", START + 1)])
                + _member([_row("z", START + 2)])
            )
        elif change == "truncated":
            segment.write_bytes(segment.read_bytes()[:40])
        else:
            segment.unlink()
        return original_write_rows(selection, paths, sink)

    monkeypatch.setattr(window_cut, "_write_rows", mutate_then_write)
    out = _out(tmp_path)

    with pytest.raises(WindowCutError) as caught:
        cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10)
    assert caught.value.reason == "window_cut_failed"
    assert not out.exists()
    assert _temp_files(out.parent) == []


def test_write_failure_is_window_cut_failed_and_leaves_no_files(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments", 10)
    out = _out(tmp_path)

    def failing_write(selection, paths, sink):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(window_cut, "_write_rows", failing_write)

    with pytest.raises(WindowCutError) as caught:
        cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)
    assert caught.value.reason == "window_cut_failed"
    assert "No space left" in str(caught.value)
    assert not out.exists()
    assert _temp_files(out.parent) == []


# ------------------------------------------------- arguments and helpers


@pytest.mark.parametrize(
    "start, end, as_of, out, max_rows",
    [
        (10, 10, 20, "w.jsonl.gz", 1),
        (10, 20, 19, "w.jsonl.gz", 1),
        (10, 20, 20, "w.jsonl", 1),
        (10, 20, 20, "w.jsonl.gz", 0),
        (10, 20, 20, "w.jsonl.gz", True),
        (float("nan"), 20, 20, "w.jsonl.gz", 1),
        ("10", 20, 20, "w.jsonl.gz", 1),
    ],
)
def test_invalid_arguments_raise_value_error(
    tmp_path, start, end, as_of, out, max_rows
):
    with pytest.raises(ValueError):
        cut_window_inprocess(
            tmp_path, start, end, as_of, tmp_path / out, max_rows=max_rows
        )
    with pytest.raises(ValueError):
        cut_window(
            tmp_path,
            start,
            end,
            as_of,
            tmp_path / out,
            max_rows=max_rows,
            deadline_s=time.time() + 60,
        )


def test_temp_path_must_be_next_to_out_path(tmp_path):
    with pytest.raises(ValueError, match="not next to"):
        cut_window_inprocess(
            tmp_path,
            START,
            END,
            AS_OF,
            tmp_path / "a" / "w.jsonl.gz",
            max_rows=1,
            temp_path=tmp_path / "b" / temp_file_name(END),
        )


def test_temp_file_name():
    assert temp_file_name(1_700_000_000_000.0, "ab12") == ".tmp-1700000000000-ab12"
    assert temp_file_name(5) != temp_file_name(5)
    # The reconciler finds a temp file's window with TEMP_NAME_RE.
    assert TEMP_NAME_RE.match(temp_file_name(5)).group(1) == "5"
    with pytest.raises(ValueError):
        temp_file_name(5, "../x")


def test_window_cut_error_reason_and_pickle():
    error = WindowCutError("trace_corrupt", "bad member")
    copy = pickle.loads(pickle.dumps(error))
    assert (copy.reason, str(copy)) == ("trace_corrupt", "bad member")
    assert set(WINDOW_CUT_FAILURES) == {
        "window_cut_timeout",
        "window_too_large",
        "trace_corrupt",
        "window_cut_failed",
        "cancelled",
    }
    with pytest.raises(ValueError):
        WindowCutError("nope", "x")


# ------------------------------------------------------- child process


def test_interrupted_before_spawn_does_not_start_a_child(tmp_path, monkeypatch):
    def no_spawn(*_):
        raise AssertionError("a child was started")

    monkeypatch.setattr(window_cut.multiprocessing, "get_context", no_spawn)
    out = _out(tmp_path)

    with pytest.raises(WindowCutError) as caught:
        cut_window(
            tmp_path, START, END, AS_OF, out, max_rows=1, deadline_s=time.time() - 1
        )
    assert caught.value.reason == "window_cut_timeout"

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(WindowCutError) as caught:
        cut_window(
            tmp_path,
            START,
            END,
            AS_OF,
            out,
            max_rows=1,
            deadline_s=time.time() + 60,
            cancel=cancel,
        )
    assert caught.value.reason == "cancelled"


@pytest.mark.timeout(120)
def test_cut_window_in_child_matches_golden(tmp_path):
    root = _random_trace(tmp_path / "segments" / "dgd", 9)
    window, golden_bytes = _golden(root, tmp_path)
    out = _out(tmp_path)

    result = cut_window(
        root, START, END, AS_OF, out, max_rows=10**6, deadline_s=time.time() + 100
    )

    _assert_golden(result, out, window, golden_bytes)
    assert _temp_files(out.parent) == []


@pytest.mark.timeout(120)
def test_cut_window_in_child_reports_the_failure_reason(tmp_path):
    root = tmp_path / "segments"
    _write(
        root / "a" / "req.000000.jsonl.gz",
        _member([_row("a", START), _row("b", START + 1)]),
    )

    with pytest.raises(WindowCutError) as caught:
        cut_window(
            root,
            START,
            END,
            AS_OF,
            _out(tmp_path),
            max_rows=1,
            deadline_s=time.time() + 100,
        )
    assert caught.value.reason == "window_too_large"
    assert "more than 1 requests" in str(caught.value)


@pytest.mark.timeout(120)
def test_child_that_dies_is_window_cut_failed():
    with pytest.raises(WindowCutError) as caught:
        _run_in_child(os._exit, (3,), {}, deadline=time.monotonic() + 100, cancel=None)
    assert caught.value.reason == "window_cut_failed"
    assert "exited with code 3" in str(caught.value)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("interrupt", ["cancel", "deadline"])
def test_interrupt_kills_child_and_removes_its_temp_file(tmp_path, interrupt):
    temp = tmp_path / temp_file_name(END)
    temp.write_bytes(b"partial window")
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    deadline = time.monotonic() + (100 if interrupt == "cancel" else 0.5)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(WindowCutError) as caught:
            _run_in_child(
                time.sleep,
                (600,),
                {},
                deadline=deadline,
                cancel=cancel if interrupt == "cancel" else None,
                cleanup=(str(temp),),
            )
    finally:
        timer.cancel()
        timer.join()
    expected = "cancelled" if interrupt == "cancel" else "window_cut_timeout"
    assert caught.value.reason == expected
    assert time.monotonic() - started < 20
    assert not temp.exists()
