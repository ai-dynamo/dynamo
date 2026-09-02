# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The push-egress token-frame fast path, exercised through the real bridge.

``push_egress.rs`` encodes the common per-token frame -- a dict of exactly
``token_ids`` then ``index`` -- straight to msgpack, skipping serde and pythonize.

Rust unit tests in ``python_payload.rs`` already pin the resulting bytes against
the generic encoder.  What they cannot reach is the half that needs libpython,
which the extension-module build does not link into ``cargo test``:

* the gate that decides whether a dict is eligible for the fast path at all, and
* the reused encode buffer, including the rewind after a frame is rejected
  partway through being written.

Both are covered here against the built extension, end to end over the request
plane: whatever the handler pushes must be exactly what the client receives.

A gate that wrongly accepted a frame would corrupt that frame's value; a rewind
that left bytes behind would corrupt the *next* frame.  That second failure only
shows up when the two kinds of frame are interleaved, which is what
:func:`test_mixed_eligible_and_ineligible_frames_in_one_stream` does.

Comparisons are type-strict (see :func:`_identical`), because plain ``==`` would
miss the most interesting failure of all: ``True == 1`` in Python, so a ``bool``
wrongly encoded as the integer ``1`` still compares equal to what was sent.
"""

import asyncio
import contextlib

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Frame cases: what the handler pushes, and what the client must get back.
# ---------------------------------------------------------------------------
#
# Each case is a (sent, expected) pair. `sent` is passed to
# `response_sender.send()` verbatim; `expected` is what the client must get back,
# or None when it is identical to `sent`.
#
# The two differ only where msgpack has no equivalent of the Python type: a tuple
# has no msgpack form of its own and comes back as a list on either path.

# Eligible: the fast path must take these, and they must survive it unchanged.
_ELIGIBLE = [
    ({"token_ids": [5], "index": 0}, None),  # the steady-state frame
    ({"token_ids": [], "index": 0}, None),  # engine produced no new tokens
    ({"token_ids": [1, 2, 3], "index": 2}, None),  # multi-token chunk
    ({"token_ids": list(range(40)), "index": 0}, None),  # past the fixarray limit
    ({"token_ids": [0, 127, 128, 255, 256, 65535, 65536], "index": 1}, None),
    ({"token_ids": [4294967295], "index": 4294967295}, None),  # u32 bounds
]

# Ineligible: each trips a different clause of the gate and must fall back to
# the generic encoder, which preserves the value wherever msgpack can represent
# it at all.
_INELIGIBLE = [
    # In Python bool is a subclass of int, and pythonize checks PyBool before
    # PyInt, so the generic path encodes these as msgpack bools. The fast path
    # must not quietly flatten them to 1/0.
    ({"token_ids": [True], "index": 0}, None),
    ({"token_ids": [1], "index": True}, None),
    # Generically, negatives encode as signed ints and past-u64 values as bytes,
    # so neither may take the fast path.
    ({"token_ids": [-1], "index": 0}, None),
    ({"token_ids": [1], "index": -1}, None),
    # msgpack has no uint128, so rmp-serde writes a past-u64 int as a 16-byte
    # big-endian `bin`. That is pre-existing generic-path behaviour and is why
    # this frame does not come back as an int on *either* path -- what matters
    # here is only that the fast path declined it, which the bytes prove.
    (
        {"token_ids": [2**64], "index": 0},
        {"token_ids": [(2**64).to_bytes(16, "big")], "index": 0},
    ),
    # Not a list.
    ({"token_ids": (1, 2), "index": 0}, {"token_ids": [1, 2], "index": 0}),
    ({"token_ids": 5, "index": 0}, None),
    # Not exactly two keys, or not these two, or not in this order.
    ({"token_ids": [1], "index": 0, "finish_reason": "stop"}, None),
    ({"token_ids": [1]}, None),
    ({"index": 0, "token_ids": [1]}, None),
    ({"text": "hello", "index": 0}, None),
    # A non-int in the list, rejected partway through writing the frame.
    ({"token_ids": [1, "two", 3], "index": 0}, None),
    ({"token_ids": [1, None], "index": 0}, None),
    ({"token_ids": [1, 2.5], "index": 0}, None),
    # An annotated envelope must still be unwrapped, not encoded as raw data.
    (
        {"_dynamo_annotated": True, "data": {"token_ids": [1], "index": 0}},
        {"token_ids": [1], "index": 0},
    ),
    # Not a dict at all.
    ("just a string", None),
    ([1, 2, 3], None),
]

_CASES = {
    "eligible": _ELIGIBLE,
    "ineligible": _INELIGIBLE,
    # Interleaved, so every rejected frame is immediately followed by one that
    # would inherit its leftover bytes if the rewind were wrong.
    "mixed": [
        frame
        for pair in zip(_ELIGIBLE, _INELIGIBLE[: len(_ELIGIBLE)])
        for frame in pair
    ],
    # One long fast-path run, long enough that the encode buffer has to allocate
    # several chunks rather than serving every frame from its first one.
    "many": [({"token_ids": [i], "index": 0}, None) for i in range(512)],
}


def _sent(case: str):
    return [sent for sent, _ in _CASES[case]]


def _expected(case: str):
    return [sent if expected is None else expected for sent, expected in _CASES[case]]


def _identical(actual, expected) -> bool:
    """Equality that does not conflate ``bool`` with ``int``.

    In Python ``True == 1`` and ``False == 0``, so plain ``==`` cannot tell a
    bool that survived correctly from one the fast path flattened to an integer
    -- precisely the regression the gate's exact-type check exists to prevent.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _identical(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _identical(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


# ---------------------------------------------------------------------------
# Handler + endpoint
# ---------------------------------------------------------------------------


async def _push_handler(request, context, response_sender=None):
    """Push every frame of the requested case, then close.

    Simply declaring a ``response_sender`` parameter is what makes Rust pick the
    push engine (see ``push_egress.rs::handler_supports_push``).  The function
    must still be an async generator -- it just yields nothing on the push path,
    and Rust advances it exactly once per request.

    The pull arm below is reached by the health check and by in-process callers,
    which pass no sender.  It has to keep working, so it yields the same frames.
    """
    frames = _sent(request["case"])

    if response_sender is None:
        for frame in frames:
            yield frame
        return

    for frame in frames:
        response_sender.send(frame)
    response_sender.close()


@pytest.fixture
async def push_client(runtime):
    endpoint = runtime.endpoint("push-egress-token-frame.backend.generate")
    server_task = asyncio.ensure_future(
        endpoint.serve_endpoint(
            _push_handler,
            health_check_payload={"case": "eligible"},
        )
    )
    client = await endpoint.client()
    try:
        await client.wait_for_instances()
        yield client
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


async def _round_trip(client, case: str):
    stream = await client.generate({"case": case})
    return [response.data() async for response in stream]


def _assert_round_trip(actual, case: str):
    expected = _expected(case)
    count_message = f"case {case!r}: expected {len(expected)} frames, got {len(actual)}"
    assert len(actual) == len(expected), count_message
    for position, (got, want) in enumerate(zip(actual, expected)):
        assert _identical(got, want), (
            f"case {case!r}: frame {position} came back as {got!r} "
            f"(type {type(got).__name__}), expected {want!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_eligible_frames_survive_the_fast_path(push_client):
    """Frames the fast path accepts must reach the client unchanged.

    Covers the empty token list, multi-token chunks, and the list lengths and
    integer magnitudes that cross msgpack's fixarray and uint marker
    boundaries.
    """
    _assert_round_trip(await _round_trip(push_client, "eligible"), "eligible")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_ineligible_frames_fall_back_intact(push_client):
    """Every near-miss must fall back to the generic encoder, losing nothing.

    Each entry trips a different clause of the gate: a bool where an int is
    expected, a negative or oversized integer, a non-list, the wrong key set or
    the wrong key order, a non-int inside the list, an annotated envelope, and
    values that are not dicts at all.
    """
    _assert_round_trip(await _round_trip(push_client, "ineligible"), "ineligible")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_mixed_eligible_and_ineligible_frames_in_one_stream(push_client):
    """The reused encode buffer must not leak bytes between frames.

    A rejected frame can be abandoned partway through being written, into the
    very same buffer the next frame will use.  If the rewind were wrong, that
    next frame would carry the abandoned prefix and fail to decode -- a failure
    that only appears when the two kinds of frame are interleaved.
    """
    _assert_round_trip(await _round_trip(push_client, "mixed"), "mixed")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_long_fast_path_run_refills_the_encode_buffer(push_client):
    """A run long enough to exhaust and refill the buffer's chunk several times.

    Frames are cut out of a shared chunk with ``split``, so this covers the
    handover to a fresh chunk.  The frames are numbered, so it also proves none
    is duplicated or dropped across that boundary.
    """
    _assert_round_trip(await _round_trip(push_client, "many"), "many")
