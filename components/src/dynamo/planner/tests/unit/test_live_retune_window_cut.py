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
import stat
import threading
import time

import pytest

from dynamo.planner.retune.live import window_cut
from dynamo.planner.retune.live.contracts import WindowCutResult
from dynamo.planner.retune.live.window_cut import (
    MAX_DROPPED_MEMBER_RATIO,
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
    segment_index,
    temp_file_name,
    temp_file_window_end,
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


def _random_trace(root, seed, *, pods=3, segments=3, members=5, plain_pod=False):
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
            directory = root / f"frontend-{pod}"
            if plain_pod and pod == 0:
                _write(directory / f"req.{segment:06d}.jsonl", *chunks)
            else:
                _write(
                    directory / f"req.{segment:06d}.jsonl.gz",
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


def test_cut_matches_golden_with_plain_jsonl_files(tmp_path):
    root = _random_trace(tmp_path / "segments" / "dgd", 3, plain_pod=True)
    window, golden_bytes = _golden(root, tmp_path)

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

    result, _ = _cut(root, tmp_path, max_rows=30)
    assert result.num_requests == 30
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


def test_in_flight_tail_of_the_newest_segment_is_not_corruption(tmp_path):
    lost = _member([_row("in-flight", START + 900)])

    def build(root, bad):
        _write(root / "a" / "req.000000.jsonl.gz", *_tiny_members("old", 3))
        members = _tiny_members("new", 2, START + 100)
        _write(
            root / "a" / "req.000001.jsonl.gz",
            *members,
            *([lost[: len(lost) // 2]] if bad else []),
        )

    result = _cut_and_twin(tmp_path, build)

    # 1 of 6 members is far above the ratio, but it is the writer's tail.
    assert result.dropped_members == 1
    assert result.dropped_bytes == len(lost) // 2


def test_truncated_tail_of_an_older_segment_is_corruption(tmp_path):
    root = tmp_path / "segments"
    lost = _member([_row("lost", START + 900)])
    _write(root / "a" / "req.000000.jsonl.gz", *_tiny_members("old", 3), lost[:20])
    _write(root / "a" / "req.000001.jsonl.gz", *_tiny_members("new", 3))

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "trace_corrupt"
    assert "1 of 7 gzip members" in str(caught.value)
    assert "req.000000.jsonl.gz@" in str(caught.value)


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


@pytest.mark.parametrize("newest", [True, False])
def test_hole_spanning_many_members_is_trace_corrupt(tmp_path, newest):
    # A zero-filled hole over 20 of 200 members is one run of bad bytes: 1 of
    # 181 members is within the ratio, but 10% of the bytes is not.
    root = tmp_path / "segments"
    members = _tiny_members("m", 200)
    hole = sum(map(len, members[50:70]))
    data = b"".join([*members[:50], bytes(hole), *members[70:]])
    _write(root / "a" / "req.000000.jsonl.gz", data)
    if not newest:
        _write(root / "a" / "req.000001.jsonl.gz", *_tiny_members("next", 3))
    total_members = 181 + (0 if newest else 3)

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)

    assert caught.value.reason == "trace_corrupt"
    message = str(caught.value)
    assert f"1 of {total_members} gzip members" in message
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


@pytest.mark.parametrize("plain_block", [None, 64])
def test_plain_jsonl_unterminated_tail(tmp_path, monkeypatch, plain_block):
    if plain_block is not None:
        # Blocks of one line each: the tail is in a later block.
        monkeypatch.setattr(window_cut, "_PLAIN_BLOCK", plain_block)
    good = [_row("a", START), _row("b", START + 1)]

    def build(root, bad):
        _write(
            root / "a" / "req.000000.jsonl",
            _lines(good),
            *([b'{"schema": "dynamo.req'] if bad else []),
        )

    result = _cut_and_twin(tmp_path, build)
    assert result.dropped_bytes == len(b'{"schema": "dynamo.req')
    assert result.dropped_members == 0

    # An unterminated last line that parses is a row, as replay reads it.
    root = tmp_path / "parses"
    _write(
        root / "a" / "req.000000.jsonl",
        _lines(good) + json.dumps(_row("c", START + 2)).encode(),
    )
    window, golden_bytes = _golden(root, tmp_path)
    assert window.num_requests == 3
    out = _out(tmp_path, "parses.jsonl.gz")
    result = cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10)
    _assert_golden(result, out, window, golden_bytes)


@pytest.mark.parametrize("tail", ["partial", "parses"])
def test_plain_jsonl_stops_at_the_unterminated_tail(tmp_path, tail):
    # The writer finishes the last line (and writes another) right after the
    # first read: reading on would start inside that line.
    path = tmp_path / "req.000000.jsonl"
    first = _lines([_row("a", START)])
    if tail == "partial":
        line = _lines([_row("b", START + 1)])
        written, rest = line[:20], line[20:]
    else:
        written, rest = json.dumps(_row("b", START + 1)).encode(), b"\n"
    _write(path, first, written)

    class Appending:
        def __init__(self, source):
            self.source = source
            self.reads = 0

        def readlines(self, hint):
            lines = self.source.readlines(hint)
            self.reads += 1
            if self.reads == 1:
                with open(path, "ab") as writer:
                    writer.write(rest + _lines([_row("c", START + 2)]))
            return lines

    with open(path, "rb") as source:
        reader = Appending(source)
        blocks = list(window_cut._plain_blocks(reader))

    assert reader.reads == 1
    assert len(blocks) == 1
    if tail == "partial":
        assert blocks[0].payload == first
        assert blocks[0].dropped == len(written)
    else:
        assert blocks[0].payload == first + written
        assert blocks[0].dropped == 0
    assert blocks[0].end == len(blocks[0].payload)
    selection = window_cut._Selection(START, END, AS_OF, 10)
    selection.scan((0, str(path), True, block) for block in blocks)
    assert selection.num_rows == (1 if tail == "partial" else 2)


def test_segment_files_order_and_newest(tmp_path):
    root = tmp_path / "segments"
    for name in (
        "b/req.000999.jsonl.gz",
        "b/req.1000000.jsonl.gz",
        "a/req.000000.jsonl.gz",
        "a/req.000001.jsonl",
        "a/notes.txt",
        "c/single.jsonl.gz",
    ):
        _write(root / name, b"")

    files = _segment_files(str(root))

    assert files == [
        (str(root / "a/req.000000.jsonl.gz"), False),
        (str(root / "a/req.000001.jsonl"), True),
        (str(root / "b/req.000999.jsonl.gz"), False),
        (str(root / "b/req.1000000.jsonl.gz"), True),
        (str(root / "c/single.jsonl.gz"), True),
    ]


def test_compressed_and_plain_copies_of_one_segment_fail(tmp_path):
    root = tmp_path / "segments"
    _write(root / "a" / "req.000000.jsonl.gz", _member([_row("a", START)]))
    _write(root / "a" / "req.000000.jsonl", _lines([_row("a", START)]))

    with pytest.raises(WindowCutError) as caught:
        _cut(root, tmp_path)
    assert caught.value.reason == "window_cut_failed"


def test_vanished_file_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "segments"
    _write(root / "a" / "req.000000.jsonl.gz", _member([_row("a", START)]))
    listed = _segment_files(str(root))
    gone = str(root / "a" / "req.000001.jsonl.gz")
    monkeypatch.setattr(window_cut, "_segment_files", lambda _: [*listed, (gone, True)])

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


@pytest.mark.parametrize("where", ["pod-list", "file-stat", "root-list", "root-stat"])
@pytest.mark.parametrize(
    "code", [errno.ENOENT, errno.ESTALE, errno.EIO, errno.ETIMEDOUT, errno.EACCES]
)
def test_segment_walk_skips_only_vanished_entries(tmp_path, monkeypatch, where, code):
    root = _two_pods(tmp_path / "segments")
    target = {
        "pod-list": root / "b",
        "file-stat": root / "b" / "req.000000.jsonl.gz",
        "root-list": root,
        "root-stat": root,
    }[where]
    name = "scandir" if where.endswith("list") else "stat"
    real = getattr(os, name)

    def failing(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            raise OSError(code, os.strerror(code), os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, name, failing)
    vanished = code in (errno.ENOENT, errno.ESTALE)

    if vanished and not where.startswith("root"):
        # The reconciler deleted pod b's segments under the walk.
        assert _segment_files(str(root)) == [
            (str(root / "a" / "req.000000.jsonl.gz"), True)
        ]
        return
    with pytest.raises(WindowCutError) as caught:
        _segment_files(str(root))
    assert caught.value.reason == "window_cut_failed"
    if vanished and where == "root-stat":
        assert "does not exist" in str(caught.value)
    else:
        assert str(target) in str(caught.value)
        assert os.strerror(code) in str(caught.value)


# -------------------------------------------------------- atomic output


def test_existing_window_file_is_reused(tmp_path):
    root = _random_trace(tmp_path / "segments", 5)
    window, _ = _golden(root, tmp_path)
    out = _out(tmp_path)
    _write(out, b"published by another process")

    result = cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)

    assert result.reused
    assert result.path == str(out)
    assert result.num_requests == window.num_requests
    assert out.read_bytes() == b"published by another process"
    assert _temp_files(out.parent) == []


def test_link_race_reuses_the_winner(tmp_path, monkeypatch):
    root = _random_trace(tmp_path / "segments", 6)
    out = _out(tmp_path)
    real_link = os.link
    linked = []

    def racing_link(source, target):
        linked.append(os.path.basename(source))
        _write(out, b"winner")
        return real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)

    result = cut_window_inprocess(root, START, END, AS_OF, out, max_rows=10**6)

    assert result.reused
    assert out.read_bytes() == b"winner"
    assert temp_file_window_end(linked[0]) == END
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
    # No file name or mtime in the gzip header.
    header = first.read_bytes()[:10]
    assert header[3] & 0x08 == 0
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


def test_line_writer_writes_in_order_and_reraises_sink_errors(monkeypatch):
    monkeypatch.setattr(window_cut, "_WRITE_BUFFER", 8)
    sink = io.BytesIO()
    writer = window_cut._LineWriter(sink)
    for index in range(100):
        writer.write(f"row {index}")
    writer.close()
    assert sink.getvalue() == b"".join(b"row %d\n" % i for i in range(100))

    class FullDisk(io.BytesIO):
        def write(self, data):
            raise OSError(28, "No space left on device")

    writer = window_cut._LineWriter(FullDisk())
    with pytest.raises(OSError, match="No space left"):
        for index in range(100):
            writer.write(f"row {index}")
        writer.close()
    writer.abort()  # idempotent after a failure

    aborted = io.BytesIO()
    writer = window_cut._LineWriter(aborted)
    writer.write("x" * 100)
    writer.abort()
    writer.abort()
    assert writer._thread.is_alive() is False


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


def test_temp_and_segment_names():
    name = temp_file_name(1_700_000_000_000.0, "ab12")
    assert name == ".tmp-1700000000000-ab12"
    assert temp_file_window_end(name) == 1_700_000_000_000
    assert temp_file_window_end(temp_file_name(5)) == 5
    assert temp_file_name(5) != temp_file_name(5)
    assert temp_file_window_end("1700000000000.jsonl.gz") is None
    assert temp_file_window_end(".tmp-12-") is None
    with pytest.raises(ValueError):
        temp_file_name(5, "../x")
    assert segment_index("req.000042.jsonl.gz") == 42
    assert segment_index("req.1000000.jsonl") == 1_000_000
    assert segment_index("req.jsonl.gz") is None


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
