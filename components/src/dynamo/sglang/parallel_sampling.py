# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel sampling (``n > 1``) fan-out for SGLang disaggregated serving.

SGLang cannot run parallel sampling across a prefill/decode handoff: its
scheduler clones the first sub-request, bootstrap room included, for the
prefix primer and every sample, so prefill registers one sender while decode
waits for ``n`` receivers and the request hangs (ai-dynamo/dynamo#14098; the
draft sgl-project/sglang#30723 would lift this on the SGLang side).

Dynamo keeps SGLang blind to ``n`` in PD mode: the prefill and decode handlers
run ``n`` independent ``n=1`` sub-requests, one bootstrap room each, and the
decode handler merges the sub-streams back by choice index. The frontend's
prefill router draws the rooms, so each keeps ``room % dp_size == dp_rank``
(SGLang's decode receiver derives the prefill DP rank from the room), and
carries them as ``bootstrap_info.bootstrap_rooms`` next to the single
``bootstrap_room`` older peers read. A frontend that predates the field sends
one room only; its ``n > 1`` requests are rejected with HTTP 400 instead of
hanging.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, TypeVar

from dynamo.llm import HttpError

logger = logging.getLogger(__name__)

BOOTSTRAP_ROOMS_KEY = "bootstrap_rooms"

T = TypeVar("T")


def requested_parallel_samples(sampling_params: Mapping[str, Any]) -> int:
    """Samples an SGLang ``sampling_params`` dict asks for.

    Anything that is not a positive integer is left for SGLang to validate and
    counts as a single sample here.
    """
    n = sampling_params.get("n")
    if isinstance(n, bool) or not isinstance(n, int):
        return 1
    return max(n, 1)


def single_sample_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``sampling_params`` for one fanned-out sub-request."""
    return {**sampling_params, "n": 1}


def choice_request_ids(
    request_id: str | None, fallback_id: str, num_choices: int
) -> list[str | None]:
    """SGLang ``rid`` per sub-request.

    A single sample keeps the caller's id (``None`` lets SGLang assign one).
    Fanned-out siblings need distinct ids that the decode worker also knows up
    front so it can abort them, derived from ``request_id`` or ``fallback_id``.
    """
    if num_choices == 1:
        return [request_id]
    base = request_id or fallback_id
    return [f"{base}-choice-{choice_index}" for choice_index in range(num_choices)]


def _unsupported_parallel_sampling_error(num_choices: int, reason: str) -> HttpError:
    return HttpError(
        400,
        f"SGLang disaggregated serving runs n={num_choices} as {num_choices} "
        f"single-sample sub-requests and needs one bootstrap room per choice, "
        f"but {reason}. Upgrade the Dynamo frontend so its prefill router "
        "draws per-choice bootstrap rooms, or send n=1.",
    )


def _validated_rooms(rooms: Any, num_choices: int) -> list[int] | None:
    if not isinstance(rooms, list) or len(rooms) != num_choices:
        return None
    if not all(isinstance(room, int) and not isinstance(room, bool) for room in rooms):
        return None
    if len(set(rooms)) != num_choices:
        return None
    return list(rooms)


def resolve_decode_bootstrap_rooms(
    bootstrap_info: Mapping[str, Any], num_choices: int
) -> list[int]:
    """Rooms the decode worker pairs on, one per choice.

    Raises HTTP 400 for ``n > 1`` when the frontend-supplied ``bootstrap_info``
    carries no usable per-choice rooms, which is what an older frontend sends.
    """
    if num_choices == 1:
        return [bootstrap_info["bootstrap_room"]]
    rooms = _validated_rooms(bootstrap_info.get(BOOTSTRAP_ROOMS_KEY), num_choices)
    if rooms is None:
        raise _unsupported_parallel_sampling_error(
            num_choices, "the request carried no usable per-choice room list"
        )
    return rooms


def resolve_prefill_bootstrap_rooms(
    bootstrap_info: Mapping[str, Any] | None,
    num_choices: int,
    generate_room: Callable[[], int],
) -> list[int]:
    """Rooms the prefill worker registers, one per choice.

    Router-drawn rooms win. Without any router room the worker draws its own
    (the existing single-sample fallback), one distinct room per choice. A
    single router room for ``n > 1`` means an older frontend: HTTP 400.
    """
    bootstrap_info = bootstrap_info or {}
    router_room = bootstrap_info.get("bootstrap_room")
    if num_choices == 1:
        return [router_room if router_room is not None else generate_room()]

    rooms = _validated_rooms(bootstrap_info.get(BOOTSTRAP_ROOMS_KEY), num_choices)
    if rooms is not None:
        return rooms
    if router_room is not None:
        raise _unsupported_parallel_sampling_error(
            num_choices, "the frontend supplied a single bootstrap room"
        )

    rooms = []
    while len(rooms) < num_choices:
        room = generate_room()
        if room not in rooms:
            rooms.append(room)
    return rooms


def raise_if_disagg_parallel_sampling(sampling_params: Mapping[str, Any]) -> None:
    """HTTP 400 for ``n > 1`` on a disaggregated path that does not fan out.

    The dedicated multimodal prefill/decode workers still hand SGLang one room
    for the whole request, so parallel sampling would hang there (#14098).
    """
    num_choices = requested_parallel_samples(sampling_params)
    if num_choices > 1:
        raise HttpError(
            400,
            f"n={num_choices} is not supported by SGLang multimodal "
            "disaggregated serving; send n=1.",
        )


_STREAM_DONE = object()


async def merge_choice_streams(
    streams: Sequence[AsyncIterator[T]],
    abort: Callable[[int], None] | None = None,
) -> AsyncIterator[tuple[int, T]]:
    """Interleave per-choice streams as they produce output.

    Every stream is driven concurrently: SGLang submits a sub-request when its
    generator is first iterated, and each decode sub-request must be in flight
    for its prefill counterpart to pair with it. Yields ``(choice_index, item)``
    pairs; a failure in one stream is re-raised and closing the merged stream
    closes them all.

    ``abort`` is called with the index of every stream that did not finish,
    before that stream's pump task is cancelled. Order matters: SGLang drops a
    cancelled request's state without telling its scheduler, so a later abort
    is a no-op and the sub-request would decode to ``max_tokens`` unconsumed.

    The queue is unbounded; it holds at most one request's output.
    """
    queue: asyncio.Queue[tuple[int, Any, BaseException | None]] = asyncio.Queue()
    finished: set[int] = set()

    async def pump(choice_index: int, stream: AsyncIterator[T]) -> None:
        try:
            async for item in stream:
                queue.put_nowait((choice_index, item, None))
        except Exception as error:  # noqa: BLE001 - forwarded to the consumer
            queue.put_nowait((choice_index, _STREAM_DONE, error))
            return
        queue.put_nowait((choice_index, _STREAM_DONE, None))

    tasks = [
        asyncio.create_task(pump(choice_index, stream))
        for choice_index, stream in enumerate(streams)
    ]
    try:
        while len(finished) < len(tasks):
            choice_index, item, error = await queue.get()
            if item is _STREAM_DONE:
                if error is not None:
                    raise error
                finished.add(choice_index)
                continue
            yield choice_index, item
    finally:
        if abort is not None:
            for choice_index in range(len(tasks)):
                if choice_index in finished:
                    continue
                try:
                    abort(choice_index)
                except Exception:  # noqa: BLE001 - best-effort sibling cleanup
                    logger.warning(
                        "Failed to abort fanned-out choice %d",
                        choice_index,
                        exc_info=True,
                    )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in streams:
            aclose = getattr(stream, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - best-effort sibling cleanup
                logger.debug(
                    "Failed to close a fanned-out choice stream", exc_info=True
                )
