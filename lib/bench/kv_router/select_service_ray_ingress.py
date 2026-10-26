# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process ``SelectionService`` host that mirrors Ray Serve LLM's KV-aware ingress.

``select_service_bench --flow ingress`` connects over a Unix socket and plays the engine
replicas and clients; this process plays one ``LLMRouter`` ingress replica. The code
below follows ``KVAwareRouter.choose_replicas``, ``KVTokenTracker`` and
``PeerUpdateForwarder`` from ray-project/ray#67224 at 53b9942e4f
(python/ray/llm/_internal/serve/routing_policies/kv_aware/), on the session-binding API
of ai-dynamo/dynamo#16969:

- one asyncio loop owns a ``dynamo.llm.SelectionService`` built with soft session
  affinity and a host replication writer id;
- each request it receives is scored with ``select_and_reserve``;
- every replica lifecycle batch (the engines broadcast them to every ingress) is applied
  with ``prefill_complete``, ``add_output_block`` per crossed KV block when decode
  progress is reported, and ``free_reservation``;
- with ``--peer``, its load reservations and the bindings from
  ``session_binding_updates()`` are broadcast to the peer ingress processes, which apply
  them with ``create_reservation`` and ``apply_session_binding``.

Ray carries the peer broadcast as ``DeploymentHandle.broadcast("on_replica_updates")``;
here the batch is pickled once and written to each peer's Unix socket, and the sender
waits for every peer's acknowledgement before sending the next batch, as Ray waits on
``results_async``. Functions prefixed ``_ipc_`` (bench transport) and ``_peer_`` (the
stand-in for Ray's RPC) are harness code; the profile analysis separates them.

Wire format (little endian): frames are ``u32 len | u8 type | body``.
"""

import argparse
import array
import asyncio
import hashlib
import math
import os
import pickle
import random
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dynamo.llm import SelectionService

_MODEL_NAME = "default"
_ROUTING_GROUP = "default"
# kv_aware/constants.py
REQUEST_TRACKING_TTL_S = 3600
SESSION_AFFINITY_MODE = "soft"
LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S = 3
LIFECYCLE_HOOKS = frozenset(
    {"on_prefill_complete", "on_decode_progress", "on_request_completed"}
)

REGISTER, REGISTER_DONE, REGISTERED, SELECT, SELECT_REPLY, LIFECYCLE_BATCH = 1, 2, 3, 4, 5, 6
PEER_UPDATES, PEER_ACK = 7, 8
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


class PeerUpdateForwarder:
    """Ray's ``PeerUpdateForwarder``: batch pending updates into one broadcast."""

    def __init__(self, handle: "_PeerBroadcast", source_writer_id: int):
        self._handle = handle
        self._source_writer_id = source_writer_id
        self._updates: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None

    def report(self, kind: str, event: Dict[str, Any]) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._updates.put_nowait({"kind": kind, "event": event})

    async def _deliver(self) -> None:
        while True:
            batch = [await self._updates.get()]
            while not self._updates.empty():
                batch.append(self._updates.get_nowait())
            try:
                results = await self._handle.broadcast(
                    "on_replica_updates",
                    self._source_writer_id,
                    batch,
                    timeout_s=LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
                )
                errors = [r for r in results if isinstance(r, BaseException)]
                if errors:
                    print(
                        f"KV peer broadcasts dropped on {len(errors)}/{len(results)} "
                        f"ingress replicas: {errors[0]!r}",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001 - mirrors Ray: log and drop
                print(f"Dropping selection service peer broadcast: {e!r}", flush=True)
            finally:
                for _ in batch:
                    self._updates.task_done()


class KVTokenTracker:
    """The select, lifecycle and peer-update paths of Ray's ``KVTokenTracker``."""

    def __init__(self, svc: SelectionService, block_size: int, writer_id: int, ttl):
        self._svc = svc
        self._block_size = block_size
        self._session_affinity_ttl_s = ttl
        self._requests: "OrderedDict[str, RequestLifecycle]" = OrderedDict()
        self._request_ids_by_worker: Dict[int, Set[str]] = {}
        self._completed_request_ids: "OrderedDict[str, float]" = OrderedDict()
        self._replication_writer_id = writer_id
        self._replica_forwarder: Optional[PeerUpdateForwarder] = None
        self._session_affinity_task: Optional[asyncio.Task] = None
        self._replica_updates: asyncio.Queue = asyncio.Queue()
        self._replica_apply_task: Optional[asyncio.Task] = None

    def start_replica_broadcast(self, handle: "_PeerBroadcast") -> None:
        self._replica_forwarder = PeerUpdateForwarder(handle, self._replication_writer_id)
        if self._session_affinity_ttl_s:
            self._session_affinity_task = asyncio.create_task(
                self._forward_session_bindings()
            )

    async def _forward_session_bindings(self) -> None:
        async for binding in self._svc.session_binding_updates():
            self._replica_forwarder.report("session_affinity", binding)

    async def on_replica_updates(
        self, source_writer_id: int, updates: List[Dict[str, Any]]
    ) -> None:
        if source_writer_id == self._replication_writer_id:
            return
        if self._replica_apply_task is None or self._replica_apply_task.done():
            self._replica_apply_task = asyncio.create_task(self._apply_replica_updates())
        self._replica_updates.put_nowait(updates)

    async def _apply_replica_updates(self) -> None:
        while True:
            batches = [await self._replica_updates.get()]
            while not self._replica_updates.empty():
                batches.append(self._replica_updates.get_nowait())
            try:
                await self._evict_stale_requests()
                for update in [update for batch in batches for update in batch]:
                    try:
                        await self._apply_update(update)
                    except Exception as error:  # noqa: BLE001 - mirrors Ray
                        print(f"Failed to apply peer update: {error!r}", flush=True)
            finally:
                for _ in batches:
                    self._replica_updates.task_done()

    async def _apply_update(self, update: Dict[str, Any]) -> None:
        if update["kind"] == "session_affinity":
            await self._svc.apply_session_binding(update["event"])
            return
        booking = update["event"]
        request_id = booking["selection_id"]
        if request_id in self._requests or request_id in self._completed_request_ids:
            return
        await self._svc.create_reservation(booking)
        if request_id in self._completed_request_ids:
            await self._svc.free_reservation(request_id)
            return
        self._track_request_state(
            request_id,
            booking["worker_id"],
            booking["isl_tokens"],
            booking["expected_output_tokens"],
        )

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
        if self._replica_forwarder is not None:
            self._replica_forwarder.report(
                "load_reservation",
                {
                    "model_name": _MODEL_NAME,
                    "routing_group": _ROUTING_GROUP,
                    "selection_id": request_id,
                    "worker_id": selection["worker_id"],
                    "dp_rank": selection["dp_rank"],
                    "sequence_hashes": selection["sequence_hashes"],
                    "isl_tokens": selection["isl_tokens"],
                    "expected_output_tokens": expected_output_tokens,
                    "effective_prefill_tokens": selection["effective_prefill_tokens"],
                    "lora_name": lora_name,
                },
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
            if hook_name not in LIFECYCLE_HOOKS:
                continue
            try:
                await getattr(self, hook_name)(*args)
            except Exception as error:  # noqa: BLE001 - mirrors Ray: log and continue
                print(f"lifecycle hook {hook_name} failed: {error!r}", flush=True)

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
        self._mark_request_completed(request_id)
        if state is None:
            return
        self._untrack_worker_request(request_id, state.worker_id)
        await self._svc.free_reservation(request_id)

    def _mark_request_completed(self, request_id: str) -> None:
        self._completed_request_ids.pop(request_id, None)
        self._completed_request_ids[request_id] = time.monotonic()

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


class _PeerBroadcast:
    """Bench stand-in for ``DeploymentHandle.broadcast`` to the peer ingress replicas."""

    def __init__(self, paths: List[str]):
        self._paths = paths
        self._peers = []

    async def connect(self) -> None:
        for path in self._paths:
            while True:
                try:
                    reader, writer = await asyncio.open_unix_connection(path, limit=2**26)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    await asyncio.sleep(0.2)
            self._peers.append((reader, writer))

    async def broadcast(self, method, source_writer_id, batch, *, timeout_s):
        del method  # always on_replica_updates
        payload = pickle.dumps((source_writer_id, batch), protocol=5)
        for _, writer in self._peers:
            _ipc_write_frame(writer, PEER_UPDATES, payload)
        return await asyncio.gather(
            *(
                asyncio.wait_for(_peer_read_ack(reader), timeout_s)
                for reader, _ in self._peers
            ),
            return_exceptions=True,
        )


async def _peer_read_ack(reader):
    kind, _ = await _ipc_read_frame(reader)
    if kind != PEER_ACK:
        raise RuntimeError(f"unexpected peer frame {kind}")


class Ingress:
    def __init__(self, args):
        self.args = args
        ttl = args.session_affinity_ttl_secs
        # Ray: get_worker_id(uuid4().hex), one per ingress incarnation.
        self.writer_id = random.getrandbits(63)
        self.svc = SelectionService(
            indexer_threads=args.indexer_threads,
            session_affinity_ttl_secs=ttl,
            session_affinity_mode=SESSION_AFFINITY_MODE,
            session_replication_writer_id=self.writer_id if ttl else None,
        )
        self.peers = _PeerBroadcast(args.peer)
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

    async def handle_peer(self, reader, writer):
        """Ray's ``LLMRouter.on_replica_updates`` RPC from one peer ingress."""
        try:
            while True:
                kind, body = await _ipc_read_frame(reader)
                if kind != PEER_UPDATES:
                    continue
                source_writer_id, updates = _peer_decode(body)
                if self.tracker is not None:
                    await self.tracker.on_replica_updates(source_writer_id, updates)
                _ipc_write_frame(writer, PEER_ACK, b"")
        except asyncio.IncompleteReadError:
            pass

    async def _register_all(self, writer):
        block_size = self.pending_registrations[0][1]
        self.tracker = KVTokenTracker(
            self.svc, block_size, self.writer_id, self.args.session_affinity_ttl_secs
        )
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
        await self.peers.connect()
        self.tracker.start_replica_broadcast(self.peers)
        _ipc_write_frame(writer, REGISTERED, struct.pack("<I", len(self.replicas)))
        print(
            f"registered {len(self.replicas)} workers; {len(self.args.peer)} peers",
            flush=True,
        )

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


def _peer_decode(body):
    return pickle.loads(body)


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
    parser.add_argument(
        "--peer-socket",
        help="Unix socket this ingress accepts peer broadcasts on (multi-ingress runs).",
    )
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer ingress --peer-socket to broadcast to; repeat once per peer.",
    )
    args = parser.parse_args()
    ingress = Ingress(args)
    servers = []
    for path, handler in ((args.socket, ingress.handle), (args.peer_socket, ingress.handle_peer)):
        if path is None:
            continue
        if os.path.exists(path):
            os.unlink(path)
        servers.append(await asyncio.start_unix_server(handler, path=path, limit=2**26))
    print(f"ingress listening on {args.socket}", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
