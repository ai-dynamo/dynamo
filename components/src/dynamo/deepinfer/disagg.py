# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill/decode disaggregation: the handoff schema and the per-rank flows.

Dynamo's completed-handoff path: the prefill worker (``EngineType.PROMPT``)
prefills with ``max_tokens=1``, pins the KV and returns one terminal chunk
(``finish_reason="length"``, the first token, and
``disaggregated_params={"deepinfer": DeepInferHandoff}``). The decode worker
(``EngineType.COMPLETION``) allocates, pulls the KV through a ``KVTransfer``,
appends the first token and queues the seq for decode admission.

The flows run on the thread that owns the rank's engine. Rank worker
processes import this module: no ``dynamo._core``, torch or DeepInfer.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from dynamo.deepinfer.kv_cache.base import HeldKV, KVCacheManager, admission_reserve
from dynamo.deepinfer.kv_transfer.base import KVTransfer
from dynamo.deepinfer.protocol import Output

logger = logging.getLogger(__name__)

HANDOFF_KEY = "deepinfer"
HANDOFF_VERSION = 1
# Decode does not start a READ this close to the prefill lease deadline; the
# prefill side frees (and may reuse) the blocks once the lease lapses.
READ_MARGIN_S = 5.0
# A READ that completes this close to (or past) the deadline is discarded: the
# prefill side may already have reused the blocks. Covers clock skew.
EXPIRY_SKEW_S = 1.0


@dataclass
class DeepInferHandoff:
    version: int
    engine_id: str
    dp_rank: int
    # Prefill-side hold id ("{request id}#{n}"); the release notification names it.
    request_id: str
    agent_name: str
    # base64 transfer metadata (NIXL: agent metadata plus region layout).
    agent_metadata: str
    layout_hash: str
    # Prefill block ids covering the prompt's KV, [0, ceil(num_tokens / 128)).
    block_ids: list[int]
    state_id: int
    num_tokens: int
    first_token: int
    expiry_ts: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> DeepInferHandoff:
        if not isinstance(data, Mapping):
            raise ValueError(f"deepinfer handoff must be an object, got {type(data)}")
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                raise ValueError(f"deepinfer handoff is missing {f.name!r}")
            kwargs[f.name] = data[f.name]
        h = cls(**kwargs)
        _check_types(h)
        if h.version != HANDOFF_VERSION:
            raise ValueError(
                f"deepinfer handoff version {h.version}; expected {HANDOFF_VERSION}"
            )
        if h.num_tokens <= 0 or not h.block_ids:
            raise ValueError("deepinfer handoff has no KV to transfer")
        return h


_INT_FIELDS = ("version", "dp_rank", "state_id", "num_tokens", "first_token")
_STR_FIELDS = ("engine_id", "request_id", "agent_name", "agent_metadata", "layout_hash")


def _check_types(h: DeepInferHandoff) -> None:
    for name in _INT_FIELDS:
        value = getattr(h, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"deepinfer handoff {name} must be an int, got {value!r}")
    for name in _STR_FIELDS:
        if not isinstance(getattr(h, name), str):
            raise ValueError(f"deepinfer handoff {name} must be a string")
    if not all(isinstance(b, int) and not isinstance(b, bool) for b in h.block_ids):
        raise ValueError("deepinfer handoff block_ids must be ints")
    if not isinstance(h.expiry_ts, (int, float)):
        raise ValueError("deepinfer handoff expiry_ts must be a number")


def handoff_from_prefill_result(prefill_result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the decode request's ``prefill_result`` and return the handoff dict."""
    params = prefill_result.get("disaggregated_params") or {}
    if HANDOFF_KEY not in params:
        raise ValueError(
            "prefill_result.disaggregated_params has no 'deepinfer' handoff; "
            "the prefill peer must be a DeepInfer worker"
        )
    return DeepInferHandoff.from_dict(params[HANDOFF_KEY]).to_dict()


def _status(seq: Any) -> str:
    return seq.status.name


def _drop_from_running(scheduler: Any, request_id: str) -> None:
    scheduler.running = [s for s in scheduler.running if s.request_id != request_id]


class PrefillFlow:
    """Prefill role: hold finished prompts' KV until the decode peer pulls it."""

    def __init__(
        self,
        engine: Any,
        cache: KVCacheManager,
        transfer: Optional[KVTransfer],
        *,
        dp_rank: int,
        engine_id: str,
        layout_hash: str,
        lease_s: float,
    ) -> None:
        self.engine = engine
        self.cache = cache
        self.transfer = transfer
        self.dp_rank = dp_rank
        self.engine_id = engine_id
        self.layout_hash = layout_hash
        self.lease_s = lease_s
        self.block_size = int(engine.cache_config.block_size)
        self._agent = (
            transfer.agent_info()
            if transfer is not None
            else {"agent_name": "", "agent_metadata": ""}
        )
        self._expect: set[str] = set()
        # hold id -> (held KV, lease expiry)
        self._held: dict[str, tuple[HeldKV, float]] = {}
        self._hold_seq = itertools.count()
        self._handoffs: dict[str, dict[str, Any]] = {}
        # DYN_DEEPINFER_TRACE_HANDOFF=1: log the holds and the oldest hold's age every 10 s.
        self._trace = os.environ.get("DYN_DEEPINFER_TRACE_HANDOFF") == "1"
        self._next_hold_log = 0.0

    @property
    def num_held(self) -> int:
        return len(self._held)

    def expect(self, request_id: str) -> None:
        """Mark a request whose KV a decode peer will pull."""
        self._expect.add(request_id)

    def forget(self, request_id: str) -> None:
        self._expect.discard(request_id)

    def release(self, handoff: Mapping[str, Any]) -> None:
        """Drop the hold a handoff names; no decode peer received it."""
        entry = self._held.pop(str(handoff.get("request_id")), None)
        if entry is not None:
            logger.debug("Released undelivered hold %s", handoff.get("request_id"))
            self.cache.release_hold(entry[0])

    def on_kv_saved(self, scheduled: list[Any], finished: list[Any]) -> None:
        """``pe_do_save_kvcache`` hook. The pull transfer reads the KV after the
        handoff, so nothing needs saving here."""

    def on_processed_prompts(self) -> None:
        """``send_processed_prompts`` hook: prefilled seqs sit finished in
        ``running``; hold the expected ones and free the rest."""
        scheduler = self.engine.scheduler
        for seq in [s for s in scheduler.running if _status(s).startswith("FINISHED")]:
            rid = seq.request_id
            expected = rid in self._expect
            self._expect.discard(rid)
            if expected and _status(seq) == "FINISHED_LENGTH_CAPPED":
                self._hold(seq)
            else:
                # EOS as first token, aborts and health probes: nothing to hand off.
                self.cache.on_finished(seq)
                _drop_from_running(scheduler, rid)

    def _hold(self, seq: Any) -> None:
        held = self.cache.hold_for_transfer(seq)
        # The release notification names this id, not the request id, so a
        # reused request id (migration retries) cannot free another hold.
        hold_id = f"{held.request_id}#{next(self._hold_seq)}"
        expiry = time.time() + self.lease_s
        num_blocks = -(-held.num_tokens // self.block_size)
        self._held[hold_id] = (held, expiry)
        self._handoffs[held.request_id] = DeepInferHandoff(
            version=HANDOFF_VERSION,
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            request_id=hold_id,
            agent_name=self._agent["agent_name"],
            agent_metadata=self._agent["agent_metadata"],
            layout_hash=self.layout_hash,
            block_ids=held.block_ids[:num_blocks],
            state_id=held.state_id,
            num_tokens=held.num_tokens,
            first_token=held.first_token,
            expiry_ts=expiry,
        ).to_dict()

    def take_handoffs(self) -> dict[str, dict[str, Any]]:
        handoffs, self._handoffs = self._handoffs, {}
        return handoffs

    def poll(self) -> None:
        """Free KV the peer finished reading, or whose lease expired."""
        released: list[str] = []
        if self.transfer is not None:
            released = self.transfer.poll()[2]
        now = time.time()
        expired = [hid for hid, (_, exp) in self._held.items() if exp < now]
        for hid in expired:
            logger.warning("KV lease of %s expired before a decode peer read it", hid)
        for hid in released:
            if hid not in self._held:
                logger.warning(
                    "Release of unknown or expired hold %s; if a decode peer read "
                    "it after the lease, that request's KV may be corrupt",
                    hid,
                )
        for hid in [*released, *expired]:
            entry = self._held.pop(hid, None)
            if entry is not None:
                self.cache.release_hold(entry[0])
        if self._trace and now >= self._next_hold_log:
            self._next_hold_log = now + 10.0
            oldest = min((exp for _, exp in self._held.values()), default=None)
            logger.info(
                "prefill holds: %d, oldest held %.1f s, released %d this poll",
                len(self._held),
                0.0 if oldest is None else now - (oldest - self.lease_s),
                len(released),
            )

    def close(self) -> None:
        for held, _ in self._held.values():
            self.cache.release_hold(held)
        self._held.clear()


class DecodeFlow:
    """Decode role: allocate, pull the prefill KV, then inject the seq.

    Every handoff ends in exactly one ``KVTransfer.release_remote`` call (a
    no-op after a completed read, which already notified the peer), so the
    prefill side frees its copy promptly on success, failure and abort.
    """

    def __init__(self, engine: Any, transfer: KVTransfer, *, layout_hash: str) -> None:
        self.engine = engine
        self.transfer = transfer
        self.layout_hash = layout_hash
        self.block_size = int(engine.cache_config.block_size)
        self.vocab_size = int(engine.model_config.get_vocab_size())
        pads = engine.get_nextn_pad_token_ids()
        self._pad_drafts = [
            int(t) for t in (pads.tolist() if hasattr(pads, "tolist") else pads)
        ]
        self._waiting: deque[tuple[Any, DeepInferHandoff]] = deque()
        self._reading: dict[str, tuple[Any, DeepInferHandoff]] = {}
        # Injected seqs still in the scheduler's `prompted` queue, which
        # abort_sequence does not scan.
        self._injected: dict[str, Any] = {}
        self._cancelled: set[str] = set()
        # Outputs produced outside poll()'s transfer handling (errors, early finishes).
        self._outputs: list[Output] = []
        # DYN_DEEPINFER_TRACE_HANDOFF=1 logs each handoff's queue and read time.
        self._trace: Optional[dict[str, list[float]]] = (
            {} if os.environ.get("DYN_DEEPINFER_TRACE_HANDOFF") == "1" else None
        )
        self._next_queue_log = 0.0

    @property
    def num_pending(self) -> int:
        return len(self._waiting) + len(self._reading)

    def admit(self, seq: Any, handoff: Mapping[str, Any]) -> None:
        """Queue a seq from ``add_request(enqueue=False)`` for KV transfer."""
        h = DeepInferHandoff.from_dict(handoff)
        problem = self._check(seq, h)
        if problem is not None:
            self._release(h)
            raise ValueError(problem)
        if int(seq.sampling_params.max_tokens) <= 1:
            # The handoff's token is the whole answer. DeepInfer stops on
            # output_len == max_tokens, so an injected seq would overrun.
            self._release(h)
            self._outputs.append(
                Output(
                    request_id=seq.request_id,
                    token_ids=[h.first_token],
                    finish_reason="length",
                )
            )
            return
        if self._trace is not None:
            self._trace[seq.request_id] = [
                time.monotonic(),
                float(len(h.agent_metadata)),
            ]
        self._waiting.append((seq, h))
        self._start_reads()

    def release(self, handoff: Mapping[str, Any]) -> None:
        """Tell the prefill peer to free a handoff this rank will not read."""
        try:
            h = DeepInferHandoff.from_dict(handoff)
        except ValueError:
            logger.warning("Cannot release malformed handoff", exc_info=True)
            return
        self._release(h)

    def _check(self, seq: Any, h: DeepInferHandoff) -> Optional[str]:
        """Why a handoff cannot serve this seq; DeepInfer TORCH_CHECKs (and the
        lockstep dies) on a token count or token id that does not fit."""
        if h.layout_hash != self.layout_hash:
            return f"KV layout mismatch: prefill {h.layout_hash}, decode {self.layout_hash}"
        prompt_len = int(seq.get_prompt_len())
        if h.num_tokens != prompt_len:
            return f"handoff covers {h.num_tokens} tokens; the prompt has {prompt_len}"
        if len(h.block_ids) != -(-h.num_tokens // self.block_size):
            return f"handoff lists {len(h.block_ids)} blocks for {h.num_tokens} tokens"
        if not 0 <= h.first_token < self.vocab_size:
            return f"handoff first token {h.first_token} is outside the vocabulary"
        return None

    def _start_reads(self) -> None:
        scheduler = self.engine.scheduler
        now = time.time()
        for item in [i for i in self._waiting if now > i[1].expiry_ts - READ_MARGIN_S]:
            # The prefill side frees (and may reuse) the blocks at expiry.
            self._waiting.remove(item)
            logger.error(
                "KV lease of %s expired before decode could read it",
                item[0].request_id,
            )
            self._fail(item[0], item[1], allocated=False, release=False)
        if not self._waiting:
            return
        block_manager = self.engine.block_manager
        # Running decodes take blocks one at a time and DeepInfer preempts them
        # (stopping them with an error) once none is free: keep a reserve.
        reserve = admission_reserve(self.engine)
        while self._waiting:
            seq, h = self._waiting[0]
            rid = seq.request_id
            need = -(-(seq.get_len() + 1) // self.block_size)
            if block_manager.get_num_free_blocks() - need < reserve:
                break
            try:
                allocated = scheduler.try_allocate_seq(seq)
            except RuntimeError as e:  # can never fit (DeepInfer's TORCH_CHECK)
                self._waiting.popleft()
                logger.error("Cannot allocate KV for %s: %s", rid, e)
                self._fail(seq, h, allocated=False)
                continue
            if not allocated:
                break
            self._waiting.popleft()
            try:
                self.transfer.start_read(
                    rid, h, list(seq.block_table)[: len(h.block_ids)], int(seq.state_id)
                )
            except Exception:
                # One bad peer or handoff must not take the rank down.
                logger.exception("Cannot start the KV read of %s", rid)
                self._fail(seq, h, allocated=True)
                continue
            self._reading[rid] = (seq, h)
            if self._trace is not None and rid in self._trace:
                self._trace[rid].append(time.monotonic())

    def poll(self) -> list[Output]:
        """Inject finished transfers; returns the first-token / error outputs."""
        self._start_reads()
        if self._trace is not None and time.monotonic() >= self._next_queue_log:
            self._next_queue_log = time.monotonic() + 10.0
            engine = self.engine
            logger.info(
                "decode queues: waiting %d, reading %d, injected %d; free blocks %d, running %d",
                len(self._waiting),
                len(self._reading),
                len(self._injected),
                engine.block_manager.get_num_free_blocks(),
                len(engine.scheduler.running),
            )
        outputs, self._outputs = self._outputs, []
        self._injected = {
            rid: seq for rid, seq in self._injected.items() if _status(seq) == "WAITING"
        }
        if not self._reading:
            return outputs
        scheduler = self.engine.scheduler
        done, failed, _ = self.transfer.poll()
        now = time.time()
        for rid in done:
            seq, h = self._reading.pop(rid)
            self._release(h)
            if rid in self._cancelled:
                self._cancelled.discard(rid)
                scheduler.free_seq(seq)
                continue
            if now > h.expiry_ts - EXPIRY_SKEW_S:
                logger.error("KV read of %s finished after its lease expired", rid)
                self._fail(seq, h, allocated=True, release=False)
                continue
            # As DeepInfer's process_outputs does for a sampled token; decode
            # feeds last_output_token into the next step.
            seq.last_output_token = h.first_token
            # Every decode step verifies exactly nextn drafts (prepare_sample.cpp).
            # A prompt-only step leaves the pad drafts (_process_model_outputs);
            # the prefill engine runs prompt-only steps, so decode starts there too.
            seq.last_next_n_token = self._pad_drafts
            seq.append_token_id(h.first_token)
            seq.append_compressed_token_id(scheduler.map_engram(h.first_token))
            seq.num_cached_tokens = h.num_tokens
            scheduler.add_prompted(seq)
            if self._trace is not None:
                t = self._trace.pop(rid, None)
                if t is not None and len(t) == 3:
                    logger.info(
                        "handoff %s: %d tokens, %d blocks, agent metadata %d B, queued %.1f ms, read %.1f ms",
                        rid,
                        h.num_tokens,
                        len(h.block_ids),
                        int(t[1]),
                        (t[2] - t[0]) * 1e3,
                        (time.monotonic() - t[2]) * 1e3,
                    )
            self._injected[rid] = seq
            outputs.append(Output(request_id=rid, token_ids=[h.first_token]))
        for rid in failed:
            seq, h = self._reading.pop(rid)
            if rid in self._cancelled:
                self._cancelled.discard(rid)
                scheduler.free_seq(seq)
                self._release(h)
            else:
                self._fail(seq, h, allocated=True)
        outputs.extend(self._outputs)
        self._outputs = []
        return outputs

    def abort(self, request_id: str) -> bool:
        """True when the request is not (yet) visible to ``abort_request``."""
        if request_id in self._reading:
            # Destination blocks are being written; free them once the read ends.
            self._cancelled.add(request_id)
            return True
        for item in self._waiting:
            if item[0].request_id == request_id:
                self._waiting.remove(item)
                self._release(item[1])
                return True
        seq = self._injected.pop(request_id, None)
        if seq is not None and _status(seq) == "WAITING":
            # Admission from `prompted` turns an aborted seq into a finished one.
            seq.status = type(seq.status).FINISHED_ABORTED
            return True
        return False

    def _fail(
        self, seq: Any, h: DeepInferHandoff, *, allocated: bool, release: bool = True
    ) -> None:
        if allocated:
            self.engine.scheduler.free_seq(seq)
        if release:
            self._release(h)
        self._outputs.append(Output(request_id=seq.request_id, finish_reason="error"))

    def _release(self, h: DeepInferHandoff) -> None:
        try:
            self.transfer.release_remote(h)
        except Exception:
            logger.warning(
                "Could not ask %s to free %s; its lease will",
                h.agent_name,
                h.request_id,
                exc_info=True,
            )
