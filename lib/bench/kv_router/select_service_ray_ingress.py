# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process ``SelectionService`` host that mirrors Ray Serve LLM's KV-aware ingress.

``select_service_bench --flow ingress`` connects over a Unix socket and plays the engine
replicas and clients; this process plays one ``LLMRouter`` ingress replica. The tracker
below follows ``KVAwareRouter.choose_replicas`` and ``KVTokenTracker`` from
ray-project/ray#67224 (python/ray/llm/_internal/serve/routing_policies/kv_aware/):
one asyncio loop owns a ``dynamo.llm.SelectionService``, scores every request with
``select_and_reserve``, and applies replica lifecycle batches with ``prefill_complete``,
one ``add_output_block`` per crossed KV block, and ``free_reservation``. Peer-ingress
reservation broadcasts are not emulated (single ingress replica).

Functions prefixed ``_ipc_`` exist only for the benchmark transport; Ray pays none of
their cost, and the profile analysis separates them.

Wire format (little endian): frames are ``u32 len | u8 type | body``.
"""

import argparse
import array
import asyncio
import hashlib
import math
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from dynamo.llm import SelectionService

_MODEL_NAME = "default"
_ROUTING_GROUP = "default"
# kv_aware/constants.py
REQUEST_TRACKING_TTL_S = 600

REGISTER, REGISTER_DONE, REGISTERED, SELECT, SELECT_REPLY, LIFECYCLE_BATCH = 1, 2, 3, 4, 5, 6
PREFILL_COMPLETE, DECODE_PROGRESS, COMPLETED = 1, 2, 3


def get_worker_id(replica_unique_id: str) -> int:
    """Deterministically derive a Dynamo worker id from a replica's unique id."""
    return int.from_bytes(
        hashlib.blake2b(replica_unique_id.encode(), digest_size=8).digest(), "big"
    )


@dataclass
class RequestLifecycle:
    worker_id: int
    prompt_tokens: int = 0
    expected_output_tokens: Optional[int] = None
    prefill_completed: bool = False
    output_tokens: int = 0
    total_blocks: int = 0
    created_at: float = field(default_factory=time.monotonic)


class Replica:
    """Stand-in for Serve's ``RunningReplica``: only the id the router hashes."""

    __slots__ = ("unique_id", "index")

    def __init__(self, unique_id: str, index: int):
        self.unique_id = unique_id
        self.index = index


class KVTokenTracker:
    """The select and lifecycle paths of Ray's ``KVTokenTracker``."""

    def __init__(self, svc: SelectionService, block_size: int):
        self._svc = svc
        self._block_size = block_size
        self._requests: "OrderedDict[str, RequestLifecycle]" = OrderedDict()
        self._request_ids_by_worker: Dict[int, Set[str]] = {}
        self._completed_request_ids: "OrderedDict[str, float]" = OrderedDict()

    async def upsert_worker(
        self, worker_id, replica_id, endpoint, replay_endpoint, max_num_batched_tokens
    ):
        await self._svc.upsert_worker(
            {
                "worker_id": worker_id,
                "model_name": _MODEL_NAME,
                "routing_group": _ROUTING_GROUP,
                "endpoint": f"ray://{replica_id}",
                "block_size": self._block_size,
                "max_num_batched_tokens": max_num_batched_tokens,
                "data_parallel_start_rank": 0,
                "data_parallel_size": 1,
                "kv_events_endpoints": {0: endpoint},
                "replay_endpoint": replay_endpoint,
            }
        )

    async def select_worker(
        self,
        request_id: str,
        token_ids: List[int],
        allowed_worker_ids: List[int],
        expected_output_tokens: Optional[int] = None,
        lora_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if token_ids is None or len(token_ids) == 0:
            raise ValueError("KV aware routing requires non-empty token_ids.")
        await self._evict_stale_requests()
        request = {
            "model_name": _MODEL_NAME,
            "routing_group": _ROUTING_GROUP,
            "selection_id": request_id,
            "token_ids": token_ids,
            "allowed_worker_ids": allowed_worker_ids,
            "expected_output_tokens": expected_output_tokens,
            "lora_name": lora_name,
            "session_id": session_id,
        }
        selection = await self._svc.select_and_reserve(request)
        self._track_request_state(
            request_id, selection["worker_id"], len(token_ids), expected_output_tokens
        )
        return {
            "worker_id": selection["worker_id"],
            "dp_rank": selection["dp_rank"],
            "overlap_tokens": selection["overlap"]["longest_matched"],
            "gpu_overlap_tokens": selection["overlap"]["gpu"],
            "effective_prefill_tokens": selection["effective_prefill_tokens"],
        }

    def _track_request_state(self, request_id, worker_id, prompt_tokens, expected_output_tokens):
        old = self._requests.pop(request_id, None)
        if old is not None:
            self._untrack_worker_request(request_id, old.worker_id)
        self._completed_request_ids.pop(request_id, None)
        self._requests[request_id] = RequestLifecycle(
            worker_id=worker_id,
            prompt_tokens=prompt_tokens,
            expected_output_tokens=expected_output_tokens,
            total_blocks=math.ceil(prompt_tokens / self._block_size),
        )
        self._request_ids_by_worker.setdefault(worker_id, set()).add(request_id)

    async def on_lifecycle_events(self, events) -> None:
        for hook_name, args in events:
            try:
                await getattr(self, hook_name)(*args)
            except Exception as error:  # noqa: BLE001 - mirrors Ray: log and continue
                print(f"lifecycle hook {hook_name} failed: {error}", flush=True)

    async def on_prefill_complete(self, request_id: str) -> None:
        state = self._requests.get(request_id)
        if state is None:
            return
        state.prefill_completed = True
        await self._svc.prefill_complete(request_id)

    async def on_decode_progress(self, request_id: str, cumulative_output_tokens: int) -> None:
        state = self._requests.get(request_id)
        if state is None:
            return
        state.output_tokens = cumulative_output_tokens
        new_total_blocks = math.ceil(
            (state.prompt_tokens + cumulative_output_tokens) / self._block_size
        )
        decay_fraction = self._get_decay_fraction(state)
        while new_total_blocks > state.total_blocks:
            state.total_blocks += 1
            self._svc.add_output_block(request_id, decay_fraction=decay_fraction)

    async def on_request_completed(self, request_id: str) -> None:
        state = self._requests.pop(request_id, None)
        self._completed_request_ids.pop(request_id, None)
        self._completed_request_ids[request_id] = time.monotonic()
        if state is None:
            return
        self._untrack_worker_request(request_id, state.worker_id)
        await self._svc.free_reservation(request_id)

    def _untrack_worker_request(self, request_id, worker_id):
        request_ids = self._request_ids_by_worker.get(worker_id)
        if request_ids is not None:
            request_ids.discard(request_id)
            if not request_ids:
                del self._request_ids_by_worker[worker_id]

    async def _evict_stale_requests(self) -> None:
        cutoff = time.monotonic() - REQUEST_TRACKING_TTL_S
        while self._completed_request_ids:
            request_id, completed_at = next(iter(self._completed_request_ids.items()))
            if completed_at > cutoff:
                break
            self._completed_request_ids.pop(request_id, None)
        while self._requests:
            request_id, state = next(iter(self._requests.items()))
            if state.created_at > cutoff:
                break
            self._requests.popitem(last=False)
            self._untrack_worker_request(request_id, state.worker_id)
            await self._svc.free_reservation(request_id)

    def _get_decay_fraction(self, state: RequestLifecycle) -> Optional[float]:
        if not state.expected_output_tokens:
            return None
        return max(0.0, 1.0 - state.output_tokens / state.expected_output_tokens)


async def choose_replicas(tracker, candidate_replicas, request_id, token_ids, max_tokens, session_id):
    """``KVAwareRouter.choose_replicas`` for a request with prompt token ids."""
    worker_id_to_replica = {
        get_worker_id(replica.unique_id): replica for replica in candidate_replicas
    }
    selection = await tracker.select_worker(
        request_id,
        token_ids,
        list(worker_id_to_replica),
        max_tokens,
        lora_name=None,
        session_id=session_id,
    )
    return worker_id_to_replica[selection["worker_id"]], selection


class Ingress:
    def __init__(self, args):
        self.args = args
        self.svc = SelectionService(
            indexer_threads=args.indexer_threads,
            session_affinity_ttl_secs=args.session_affinity_ttl_secs,
        )
        self.tracker: Optional[KVTokenTracker] = None
        self.replicas: List[Replica] = []
        self.pending_registrations = []
        self.lifecycle_queue: asyncio.Queue = asyncio.Queue()

    async def handle(self, reader, writer):
        lifecycle_task = asyncio.create_task(self._apply_lifecycle(writer))
        try:
            while True:
                kind, body = await _ipc_read_frame(reader)
                if kind == SELECT:
                    asyncio.create_task(self._select(writer, body))
                elif kind == LIFECYCLE_BATCH:
                    self.lifecycle_queue.put_nowait(body)
                elif kind == REGISTER:
                    self.pending_registrations.append(_ipc_decode_register(body))
                elif kind == REGISTER_DONE:
                    await self._register_all(writer)
        except asyncio.IncompleteReadError:
            pass
        finally:
            lifecycle_task.cancel()

    async def _register_all(self, writer):
        block_size = self.pending_registrations[0][1]
        self.tracker = KVTokenTracker(self.svc, block_size)
        for index, (_, _, max_batched, endpoint, replay_endpoint) in enumerate(
            self.pending_registrations
        ):
            replica = Replica(f"replica-{index}", index)
            self.replicas.append(replica)
            await self.tracker.upsert_worker(
                get_worker_id(replica.unique_id),
                replica.unique_id,
                endpoint,
                replay_endpoint,
                max_batched,
            )
        while self.svc.ready().get("schedulable_workers", 0) < len(self.replicas):
            await asyncio.sleep(0.2)
        _ipc_write_frame(writer, REGISTERED, struct.pack("<I", len(self.replicas)))
        print(f"registered {len(self.replicas)} workers", flush=True)

    async def _select(self, writer, body):
        request_id, session_id, max_tokens, token_ids = _ipc_decode_select(body)
        try:
            replica, selection = await choose_replicas(
                self.tracker, self.replicas, request_id, token_ids, max_tokens, session_id
            )
            reply = struct.pack(
                "<BII", 1, replica.index, min(selection["gpu_overlap_tokens"], 2**32 - 1)
            )
        except Exception as error:  # noqa: BLE001 - reported to the bench as a failure
            message = str(error).encode()[:65535]
            reply = struct.pack("<BH", 0, len(message)) + message
        _ipc_write_frame(writer, SELECT_REPLY, _ipc_encode_id(request_id) + reply)

    async def _apply_lifecycle(self, writer):
        while True:
            body = await self.lifecycle_queue.get()
            await self.tracker.on_lifecycle_events(_ipc_decode_lifecycle(body))


async def _ipc_read_frame(reader):
    header = await reader.readexactly(5)
    length, kind = struct.unpack("<IB", header)
    return kind, await reader.readexactly(length - 1)


def _ipc_write_frame(writer, kind, body):
    writer.write(struct.pack("<IB", len(body) + 1, kind) + body)


def _ipc_encode_id(request_id):
    raw = request_id.encode()
    return struct.pack("<H", len(raw)) + raw


def _ipc_decode_str(body, offset):
    (length,) = struct.unpack_from("<H", body, offset)
    offset += 2
    return body[offset : offset + length].decode(), offset + length


def _ipc_decode_register(body):
    worker, block_size, max_batched = struct.unpack_from("<QII", body, 0)
    endpoint, offset = _ipc_decode_str(body, 16)
    replay_endpoint, _ = _ipc_decode_str(body, offset)
    return worker, block_size, max_batched, endpoint, replay_endpoint or None


def _ipc_decode_select(body):
    request_id, offset = _ipc_decode_str(body, 0)
    session_id, offset = _ipc_decode_str(body, offset)
    max_tokens, count = struct.unpack_from("<II", body, offset)
    offset += 8
    tokens = array.array("I")
    tokens.frombytes(body[offset : offset + 4 * count])
    return request_id, session_id or None, max_tokens or None, tokens.tolist()


def _ipc_decode_lifecycle(body):
    (count,) = struct.unpack_from("<I", body, 0)
    offset = 4
    events = []
    for _ in range(count):
        kind = body[offset]
        request_id, offset = _ipc_decode_str(body, offset + 1)
        if kind == PREFILL_COMPLETE:
            events.append(("on_prefill_complete", (request_id,)))
        elif kind == DECODE_PROGRESS:
            (cumulative,) = struct.unpack_from("<I", body, offset)
            offset += 4
            events.append(("on_decode_progress", (request_id, cumulative)))
        else:
            events.append(("on_request_completed", (request_id,)))
    return events


async def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument("--indexer-threads", type=int, default=4)
    parser.add_argument("--session-affinity-ttl-secs", type=float, default=None)
    args = parser.parse_args()
    ingress = Ingress(args)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = await asyncio.start_unix_server(ingress.handle, path=args.socket, limit=2**26)
    print(f"ingress listening on {args.socket}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
