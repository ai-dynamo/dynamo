# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One DP rank: builds and owns a DeepInfer engine and applies ``Cmd``s to it.

Used in-thread for an in-process rank 0 and inside ``rank_worker`` for the
others. Each ``apply`` is one lockstep iteration: aborts, adds, then the
iteration's global flags (``Sync``: from the controller's ``Cmd``, else one
MAX all-reduce of this rank's own values), then ``close()``, or ``step()`` if
any rank has unfinished requests, or ``flush_inflight_reqs()``. The first iteration also checks that every rank
runs the same options and lockstep settings.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from dynamo.deepinfer import di_engine
from dynamo.deepinfer.protocol import (
    AddReq,
    Cmd,
    Op,
    Output,
    RankStats,
    Reply,
    Sync,
    trace_lockstep,
)
from dynamo.deepinfer.request import collect_outputs, prompt_array

logger = logging.getLogger(__name__)


@dataclass
class RankSpec:
    """Everything a rank needs to build its engine; JSON for child processes."""

    rank: int
    dp_size: int
    # deepinfer EngineArgs kwargs, without engine_type.
    engine_args: dict[str, Any]
    engine_type: str = "NORMAL"
    backend: str = "deepinfer"  # or "fake" (testing/fake_deepinfer.py)
    engine_id: str = ""
    # kv_cache.base: the per-rank KV cache manager and its options.
    cache_manager: str = "none"
    cache_options: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, data: str) -> RankSpec:
        return cls(**json.loads(data))

    def digest(self) -> str:
        """What every node of an engine must agree on for this rank: the spec
        without the (informational) engine id."""
        return _sha256(self._shared())

    def engine_digest(self) -> str:
        """What every rank of the engine must agree on: :meth:`digest`
        without the rank."""
        data = self._shared()
        del data["rank"]
        data["engine_args"].pop("engine_rank", None)
        return _sha256(data)

    def _shared(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        del data["engine_id"]
        return data


def _sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class RankRunner:
    def __init__(self, spec: RankSpec) -> None:
        self.spec = spec
        self.api = di_engine.load_api(spec.backend)
        self.engine = di_engine.build_engine(self.api, spec)
        self.cache = self.engine.dynamo_cache
        self.meta = di_engine.engine_meta(self.engine)
        self._dp_max = di_engine.dp_max(self.api, self.engine)
        self._startup_max = di_engine.dp_max(self.api, self.engine, cpu=True)
        # In-flight request id -> top_logprobs; outputs of other ids are dropped.
        self._live: dict[str, int] = {}
        # Added but not admitted yet (the cache manager has no room): (seq, prompt).
        self._later: deque[tuple[Any, np.ndarray]] = deque()
        self._thread: Optional[int] = None
        self._busy = False
        self._checked = False
        self._closed = False
        self._trace = trace_lockstep()

    def apply(self, cmd: Cmd) -> Reply:
        received = time.time() if self._trace else 0.0
        if self._closed:
            if cmd.op == Op.CLOSE:
                return Reply(seq=cmd.seq, rank=self.spec.rank, closed=True)
            raise RuntimeError(f"iteration {cmd.seq} after the engine closed")
        if self._thread != threading.get_ident():
            di_engine.bind_thread(self.api, self.engine)
            self._thread = threading.get_ident()
        if not self._checked:
            self._check_lockstep()
        engine = self.engine
        outputs: dict[str, Output] = {}
        for rid in cmd.aborts:
            self._abort(rid)
        for add in cmd.adds:
            self._add(add, outputs)
        admitted = self._admit_later()
        closing = cmd.op == Op.CLOSE
        unfinished = di_engine.has_local_work(engine)
        pending = self._num_pending() > 0
        flags = cmd.sync
        if flags is None:
            flags = self._all_reduce(
                Sync(
                    unfinished=unfinished,
                    close=closing,
                    new_prompts=admitted > 0,
                    pending=pending,
                )
            )
        synced = time.time() if self._trace else 0.0
        if closing and not flags.close:
            raise RuntimeError(f"CLOSE in iteration {cmd.seq} without the close flag")
        if flags.close:
            # Every rank of the engine closes in this iteration: DeepInfer's
            # close() is collective.
            self.close()
            reply = Reply(
                seq=cmd.seq,
                rank=self.spec.rank,
                u=flags.unfinished,
                outputs=list(outputs.values()),
                closed=True,
                unfinished=unfinished,
                pending=pending,
            )
            self._log_trace(cmd.seq, received, synced, reply)
            return reply

        if flags.new_prompts and self.spec.engine_type == "NORMAL":
            # Every rank, in the same iteration: the ANY interval makes the
            # next step an ANY prefill vote point, so ranks with new prompts
            # prefill even when another rank cannot (e.g. its KV is above
            # 90%). 0 (an ALL point) would let such a rank veto prefill
            # everywhere for as long as prompts keep coming.
            engine.scheduler.decode_step = self.meta.prefill_any_attempt_interval
        # Work that showed up here after the controller computed the flags (a
        # prompt admitted from _later) is flushed now and steps next iteration
        # (Reply.unfinished).
        u = flags.unfinished
        ptrs = engine.step() if u else engine.flush_inflight_reqs()
        collect_outputs(ptrs, self.api.RequestOutput, self._live, outputs)
        finished: list[str] = []
        for rid, out in outputs.items():
            if out.finish_reason is not None:
                self._live.pop(rid, None)
                finished.append(rid)
        if self._busy and not u:
            # DeepInfer disables the cyclic GC and collects only every N steps.
            gc.collect()
        self._busy = u
        stats = self._stats()
        reply = Reply(
            seq=cmd.seq,
            rank=self.spec.rank,
            u=u,
            outputs=list(outputs.values()),
            stats=stats,
            unfinished=di_engine.has_local_work(engine),
            pending=stats.num_pending > 0,
            finished=finished,
        )
        self._log_trace(cmd.seq, received, synced, reply)
        return reply

    def _all_reduce(self, local: Sync) -> Sync:
        unfinished, close, new_prompts, pending = self._dp_max(
            "sync",
            [
                int(local.unfinished),
                int(local.close),
                int(local.new_prompts),
                int(local.pending),
            ],
        )
        return Sync(
            unfinished=unfinished > 0,
            close=close > 0,
            new_prompts=new_prompts > 0,
            pending=pending > 0,
        )

    def _check_lockstep(self) -> None:
        """First iteration: every rank must run the same engine options and
        lockstep settings, since each derives its collectives from them. One
        MAX all-reduce of a hash and its negation (MAX and MIN), on the gloo
        group whatever the zero-overhead settings."""
        self._checked = True
        settings = di_engine.lockstep_settings(self.api, self.spec.engine_type)
        digest = self.spec.engine_digest()
        key = json.dumps([digest, settings], sort_keys=True).encode()
        mine = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") >> 1
        high, neg_low = self._startup_max("lockstep", [mine, -mine])
        if high != -neg_low:
            raise RuntimeError(
                "the DeepInfer ranks of this engine run different options or "
                "lockstep settings (LOCKSTEP_ENV); give every node the same "
                f"arguments and DEEPINFER_* environment. Rank {self.spec.rank}: "
                f"options digest {digest}, {settings}"
            )

    def _log_trace(
        self, seq: int, received: float, synced: float, reply: Reply
    ) -> None:
        if self._trace:
            logger.info(
                "lockstep rank=%d seq=%d cmd=%.6f sync=%.6f step=%.6f u=%d closed=%d",
                self.spec.rank,
                seq,
                received,
                synced,
                time.time(),
                reply.u,
                reply.closed,
            )

    def _add(self, add: AddReq, outputs: dict[str, Output]) -> None:
        rid = add.request_id
        try:
            prompt = prompt_array(add.token_ids)
            seq = self.engine.add_request(
                user_id="dynamo",
                request_id=rid,
                sampling_params=self.api.SamplingParams(**add.sampling),
                prompt_token_ids=prompt,
                multi_modal_data=None,
                enqueue=False,
            )
            if seq is None:
                raise ValueError("prompt has a token id outside the vocabulary")
            self._later.append((seq, prompt))
        except (ValueError, TypeError) as e:
            logger.warning("Rejected request %s on rank %d: %s", rid, self.spec.rank, e)
            outputs[rid] = Output(request_id=rid, finish_reason="error")
            return
        self._live[rid] = int(add.sampling.get("top_logprobs") or 0)

    def _admit_later(self) -> int:
        """Hand added prompts to the scheduler; returns how many it took."""
        # FIFO: a request the cache turns away holds back the ones behind it.
        admitted = 0
        while self._later:
            seq, prompt = self._later[0]
            if not self.cache.admit(seq, prompt).admitted:
                break
            self._later.popleft()
            admitted += 1
        return admitted

    def _abort(self, rid: str) -> None:
        self._live.pop(rid, None)
        for item in self._later:
            if item[0].request_id == rid:
                self._later.remove(item)
                return
        scheduler = self.engine.scheduler
        # An aborted waiting seq is reported through the scheduler's ignored
        # path, which never frees it. Partly prefilled prompts hold blocks
        # while waiting, so release it here.
        waiting = [
            s
            for s in scheduler.waiting
            if s.request_id == rid and not s.status.name.startswith("FINISHED")
        ]
        self.engine.abort_request(rid)
        for seq in waiting:
            self.cache.on_finished(seq)

    def _num_pending(self) -> int:
        """Requests outside the scheduler: awaiting KV space."""
        return len(self._later)

    def _stats(self) -> RankStats:
        cs = self.cache.stats()
        scheduler = self.engine.scheduler
        return RankStats(
            num_running=len(scheduler.running),
            num_waiting=len(scheduler.waiting),
            num_pending=self._num_pending(),
            kv_used_blocks=cs.used_blocks,
            kv_total_blocks=cs.total_blocks,
        )

    def close(self) -> None:
        """DeepInfer's ``close()`` is collective: call it in the CLOSE iteration only."""
        if self._closed:
            return
        self._closed = True
        self.engine.close()
