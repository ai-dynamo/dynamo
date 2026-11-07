# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fake of the DeepInfer surface that ``dynamo.deepinfer`` drives.

It mirrors the names (``LLMEngine``, ``EngineArgs``, ``EngineType``,
``SamplingParams``, ``RequestOutput``, ``SequenceStatus``) and the behaviour
the Dynamo integration depends on, as read from DeepInfer 774ce24. Engine
internals its own scheduler needs (block tables, token appends) are Python
attributes here:

- ``add_request(enqueue=...)``, ``abort_request``; ``step()`` /
  ``flush_inflight_reqs()`` return uint64 output handles that die at the next
  call. Aborting a running request emits nothing; aborting a waiting one emits
  an ``"abort"`` output that repeats the last existing token.
- ``has_unfinished_requests()``, the NORMAL prefill vote and Dynamo's own
  flag all-reduce (``all_reduce_max``) are DP all-reduces; a collective whose
  tag differs across ranks (e.g. diverged ``decode_step``) raises, standing in
  for a hang. ``collectives`` counts them by tag.
- NORMAL engines prefill only at ALL/ANY vote points derived from
  ``scheduler.decode_step`` and only while block usage is at most 90%; a
  decode step preempts (stops) the longest-output seqs whenever no block is
  free; zero-overhead scheduling delivers decode tokens one step late;
  chunked prefill honours ``chunk_checkpoints``.
- Blocks and state slots are refcounted (block 0 and slot 0 reserved);
  ``free_seq`` does not reset ``state_id``, so a double free raises.
- The P/D hooks are called where DeepInfer calls them.
- A deterministic "model": KV block rows store the token ids, the first 32
  bytes of the state slot hold a running digest of the computed tokens, and
  every sampled token hashes both. Wrong KV or state changes the output.

Environment knobs: ``DEEPINFER_FAKE_NUM_GPU_BLOCKS`` (256),
``DEEPINFER_FAKE_DP_ENDPOINT`` (ZMQ rendezvous for DP collectives; default
MASTER_ADDR:MASTER_PORT-49),
``DEEPINFER_FAKE_DP_TIMEOUT_S`` (30), ``DEEPINFER_FAKE_EOS_AFTER`` (emit EOS as
the N-th output token), ``DEEPINFER_FAKE_STEP_DELAY_S`` (0: each ``step()``
sleeps this long), ``DEEPINFER_FAKE_STEP_OVERLAP`` (1: with zero-overhead
scheduling the delay is the forward of a step, which runs while the CPU loop
goes on: a step sleeps only for what remains of the previous one, so the step
period is ``max(delay, loop)`` as on the GPU), plus DeepInfer's
``DEEPINFER_PREFILL_[ANY_]ATTEMPT_INTERVAL`` and
``DEEPINFER_ZERO_OVERHEAD_SCHEDULING``.
"""

from __future__ import annotations

import enum
import hashlib
import itertools
import json
import os
import random
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import zmq

VOCAB_SIZE = 1024
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
BLOCK_SIZE = 128
STATE_BYTES = 64
STATE_STORED_BYTES = 32  # the "stored" region a prefix snapshot keeps
KVSTORE_GRAIN = 128
# NORMAL engines prefill only at or below this block usage (scheduler.hpp:80,313).
GPU_CACHE_PREFILL_WATERMARK = 0.9


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


class _Envs:
    """DeepInfer's KV layout settings, resolved with its defaults like
    ``deepinfer/environ.py`` ``envs`` (read at each access here)."""

    @property
    def DEEPINFER_USE_NEXTN_DECODE(self) -> int:
        return _env_int("DEEPINFER_USE_NEXTN_DECODE", 5)

    @property
    def DEEPINFER_MAX_BACKTRACKING_TOKENS(self) -> int:
        return max(
            _env_int("DEEPINFER_MAX_BACKTRACKING_TOKENS", 0),
            self.DEEPINFER_USE_NEXTN_DECODE + 1,
        )

    @property
    def DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE(self) -> str:
        return os.environ.get("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE") or "off"

    @property
    def DEEPINFER_YOCO_PREFILL_KEEP_TOKENS(self) -> int:
        return _env_int("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", 0)

    @property
    def DEEPINFER_PREFILL_STORE_EXTRA_STATE(self) -> bool:
        value = os.environ.get("DEEPINFER_PREFILL_STORE_EXTRA_STATE", "0")
        return value not in ("0", "", "false", "False")


envs = _Envs()


class EngineType(enum.Enum):
    NORMAL = 0
    PROMPT = 1
    COMPLETION = 2

    @property
    def default_zero_overhead_scheduling(self) -> bool:
        return self in (EngineType.NORMAL, EngineType.COMPLETION)


class SequenceStatus(enum.Enum):
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6
    FINISHED_PREEMPTED = 7
    FINISHED_SO_FAILED = 8
    FINISHED_VE_FAILED = 9


_FINISH_REASONS = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_IGNORED: "prompt_length_exceeded",
    SequenceStatus.FINISHED_PREEMPTED: "insufficient_system_resource",
    SequenceStatus.FINISHED_SO_FAILED: "xgrammar runtime errors or timeout",
    SequenceStatus.FINISHED_VE_FAILED: "ve_error",
}


class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: bool = False,
        prompt_logprobs: bool = False,
        top_logprobs: int = 0,
        first_n_token_no_eos: int = 0,
        eos_id: int = 0,
        so_args: Optional[dict] = None,
        golden_token_ids: Optional[list[int]] = None,
        seq_seed: Optional[int] = None,
    ) -> None:
        if temperature < 0 or not 0 < top_p <= 1 or (top_k != -1 and top_k < 1):
            raise ValueError(f"invalid sampling {temperature=} {top_p=} {top_k=}")
        if not 0 <= top_logprobs <= 1000:
            raise ValueError(f"invalid {top_logprobs=}")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.ignore_eos = ignore_eos
        self.max_tokens = max(1, max_tokens)
        self.prompt_logprobs = prompt_logprobs
        self.top_logprobs = top_logprobs
        self.first_n_token_no_eos = first_n_token_no_eos
        self.eos_id = eos_id
        self.so_args = so_args
        self.golden_token_ids = golden_token_ids
        self.seq_seed = (
            random.getrandbits(64) if seq_seed is None else seq_seed & (2**64 - 1)
        )


@dataclass
class EngineArgs:
    """Same fields as ``deepinfer.engine.arg_utils.EngineArgs``."""

    model: str
    served_model_name: str = "deepinfer-model"
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    load_format: str = "auto"
    seed: int = 0
    max_model_len: Optional[int] = None
    data_parallel_size: int = 0
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    engram_parallel_size: int = 0
    engine_rank: int = 0
    engine_group_rank: int = 0
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    stop_preempted_request: bool = True
    engine_type: EngineType = EngineType.NORMAL
    chunked_prefill: bool = False
    ignore_len: int = 128 << 20
    tokens_per_block: int = -1
    lookahead_tokens: int = 0
    offline_inference: bool = False


@dataclass
class CacheConfig:
    block_size: int
    num_gpu_blocks: int
    bytes_per_block: int
    num_state_seqs: int
    bytes_of_state_per_seqs: int
    lookahead_tokens: int


@dataclass
class SchedulerConfig:
    max_num_seqs: int
    max_model_len: int
    max_num_batched_tokens: int
    prefill_attempt_interval: int
    prefill_any_attempt_interval: int
    chunked_prefill: bool
    stop_preempted_request: bool


@dataclass
class ModelConfig:
    model: str
    max_model_len: int

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE


@dataclass
class Tokenizer:
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID


class BlockManager:
    """Refcounted FIFO free list; id 0 is reserved."""

    def __init__(self, num_blocks: int) -> None:
        self._ref = [0] * num_blocks
        self._ref[0] = 1
        self._free: deque[int] = deque(range(1, num_blocks))

    def get_num_free_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n

    def allocate_nolock(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(f"cannot allocate {n} blocks, {len(self._free)} free")
        ids = [self._free.popleft() for _ in range(n)]
        for b in ids:
            self._ref[b] = 1
        return ids

    def pin_blocks(self, block_ids: Iterable[int]) -> None:
        for b in block_ids:
            if self._ref[b] <= 0:
                raise RuntimeError(f"pin of free block {b}")
            self._ref[b] += 1

    def unpin_blocks(self, block_ids: Iterable[int]) -> None:
        for b in block_ids:
            if b == 0 or self._ref[b] <= 0:
                raise RuntimeError(f"double free of block {b}")
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def ref_count(self, block_id: int) -> int:
        return self._ref[block_id]

    def lock(self) -> None:
        pass

    def unlock(self) -> None:
        pass


class Sequence:
    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: np.ndarray,
        request_id: str,
        user_id: str,
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.seq_id = seq_id
        self.request_id = request_id
        self.user_id = user_id
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.data: list[int] = prompt_token_ids.tolist()
        self.compressed_data: list[int] = list(self.data)
        self._prompt_len = len(self.data)
        self.status = SequenceStatus.WAITING
        self._block_table: list[int] = []
        self.state_id = -1
        self.num_cached_tokens = 0
        self.chunk_checkpoints: list[int] = []
        self.cached_hit_tokens = 0
        self.swa_valid_start = 0
        self.last_output_token = -1
        # MtpState::draft_tokens: the drafts the next decode step verifies.
        self.last_next_n_token: list[int] = []
        self.schedulable = True
        self.branch_id = ""

    # pybind returns a copy of the C++ vector; assignment replaces it.
    @property
    def block_table(self) -> list[int]:
        return list(self._block_table)

    @block_table.setter
    def block_table(self, value: Iterable[int]) -> None:
        self._block_table = list(value)

    def get_len(self) -> int:
        return len(self.data)

    def get_prompt_len(self) -> int:
        return self._prompt_len

    def get_output_len(self) -> int:
        return len(self.data) - self._prompt_len

    def append_token_id(self, token_id: int) -> None:
        self.data.append(int(token_id))

    def append_compressed_token_id(self, token_id: int) -> None:
        self.compressed_data.append(int(token_id))

    def is_finished(self) -> bool:
        return self.status.name.startswith("FINISHED")


@dataclass
class SchedulerOutputs:
    scheduled_seqs: list[Sequence]
    prompt_run: bool
    finished_seqs: list[Sequence] = field(default_factory=list)
    ignored_seqs: list[Sequence] = field(default_factory=list)
    # Prefill token ranges [start, end) per scheduled seq.
    chunks: list[tuple[int, int]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.scheduled_seqs


class _DPGroup:
    """Tagged all-reduce over ZMQ (rank 0 gathers and broadcasts) of an int or
    an int vector (elementwise)."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank = rank
        self.size = size
        # Collectives this rank entered, by tag without its "@step" suffix.
        self.calls: Counter[str] = Counter()
        self._timeout_ms = int(
            float(os.environ.get("DEEPINFER_FAKE_DP_TIMEOUT_S") or 30) * 1000
        )
        self._sock: Optional[zmq.Socket] = None
        if size > 1:
            # Like DeepInfer's store: rank 0 listens at MASTER_ADDR, next to
            # MASTER_PORT - 50 (deepinfer/utils.py: get_distributed_init_method).
            endpoint = os.environ.get("DEEPINFER_FAKE_DP_ENDPOINT") or (
                f"tcp://{os.environ.get('MASTER_ADDR') or '127.0.0.1'}:"
                f"{_env_int('MASTER_PORT', 23333) - 49}"
            )
            ctx = zmq.Context.instance()
            if rank == 0:
                self._sock = ctx.socket(zmq.ROUTER)
                self._sock.bind(endpoint)
            else:
                self._sock = ctx.socket(zmq.DEALER)
                self._sock.connect(endpoint)
            self._sock.setsockopt(zmq.LINGER, 0)

    def _poll(self, tag: str) -> None:
        assert self._sock is not None
        if not self._sock.poll(self._timeout_ms):
            raise TimeoutError(
                f"fake DP collective {tag!r} timed out on rank {self.rank}"
            )

    def all_reduce(
        self, tag: str, value: Any, op: Callable[[list[int]], int] = max
    ) -> Any:
        self.calls[tag.split("@")[0]] += 1
        if self._sock is None:
            return value
        if self.rank == 0:
            peers: list[bytes] = []
            got: dict[int, tuple[str, Any]] = {0: (tag, value)}
            while len(peers) < self.size - 1:
                self._poll(tag)
                ident, payload = self._sock.recv_multipart()
                peer_tag, peer_rank, peer_value = json.loads(payload)
                peers.append(ident)
                got[peer_rank] = (peer_tag, peer_value)
            # Another tag, or a vector of another length: diverged.
            keys = {
                (t, len(v) if isinstance(v, list) else None) for t, v in got.values()
            }
            values = [v for _, v in got.values()]
            result = None
            if len(keys) == 1:
                result = (
                    [op(list(col)) for col in zip(*values)]
                    if isinstance(value, list)
                    else op(values)
                )
            for ident in peers:
                self._sock.send_multipart([ident, json.dumps(result).encode()])
        else:
            assert self._sock is not None
            self._sock.send(json.dumps([tag, self.rank, value]).encode())
            self._poll(tag)
            result = json.loads(self._sock.recv())
        if result is None:
            raise RuntimeError(f"DP ranks diverged at collective {tag!r}")
        return result

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


class Scheduler:
    def __init__(
        self,
        engine_type: EngineType,
        block_manager: BlockManager,
        state_manager: BlockManager,
        config: SchedulerConfig,
        dp_group: _DPGroup,
        lookahead_tokens: int,
    ) -> None:
        self.engine_type = engine_type
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.config = config
        self.dp = dp_group
        self.lookahead_tokens = lookahead_tokens
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self.prompted: deque[Sequence] = deque()
        self.decode_step = 0

    @property
    def waiting(self) -> list[Sequence]:
        return list(self._waiting)

    @property
    def running(self) -> list[Sequence]:
        return list(self._running)

    @running.setter
    def running(self, seqs: Iterable[Sequence]) -> None:
        self._running = list(seqs)

    def num_blocks(self, seq: Sequence) -> int:
        # Room for the KV of every token so far plus the next one.
        return -(-(seq.get_len() + 1) // BLOCK_SIZE)

    def add_sequence(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    def add_prompted(self, seq: Sequence) -> None:
        if seq.status != SequenceStatus.WAITING or not seq.block_table:
            raise RuntimeError("add_prompted needs a WAITING seq with KV filled")
        self.prompted.append(seq)

    def try_allocate_seq(self, seq: Sequence) -> bool:
        return self.try_allocate_seq_with_prefix(seq, [])

    def try_allocate_seq_with_prefix(
        self, seq: Sequence, prefix_blocks: list[int]
    ) -> bool:
        if (
            seq.status != SequenceStatus.WAITING
            or seq.block_table
            or seq.state_id != -1
        ):
            raise RuntimeError(f"{seq.request_id}: already allocated or not waiting")
        num_new = self.num_blocks(seq) - len(prefix_blocks)
        if num_new < 0:
            raise RuntimeError("prefix longer than the sequence")
        if num_new > len(self.block_manager._ref) - 1:
            raise RuntimeError(f"can never allocate {num_new} blocks")
        if not self.block_manager.can_allocate(
            num_new
        ) or not self.state_manager.can_allocate(1):
            return False
        self.block_manager.pin_blocks(prefix_blocks)
        seq.block_table = list(prefix_blocks) + self.block_manager.allocate_nolock(
            num_new
        )
        seq.state_id = self.state_manager.allocate_nolock(1)[0]
        return True

    def free_seq(self, seq: Sequence) -> None:
        # Like DeepInfer: the state id is not reset, so a second free raises.
        if seq.state_id != -1:
            self.state_manager.unpin_blocks([seq.state_id])
        self.block_manager.unpin_blocks(seq.block_table)
        seq.block_table = []

    def abort_sequence(self, request_ids: list[str]) -> list[Sequence]:
        aborted = []
        for queue in (self._running, self._waiting):
            for seq in queue:
                if seq.request_id in request_ids and not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    aborted.append(seq)
        return aborted

    def get_num_unfinished_seqs(self) -> int:
        return len(self._waiting) + len(self._running) + len(self.prompted)

    def has_unfinished_seqs(self) -> bool:
        return self.get_num_unfinished_seqs() > 0

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_manager.get_num_free_blocks()

    def get_gpu_cache_usage(self) -> float:
        # As DeepInfer: the reserved block 0 counts as used.
        total = len(self.block_manager._ref)
        return (total - self.block_manager.get_num_free_blocks()) / total

    def map_engram(self, token_id: int) -> int:
        return token_id

    def schedule(self) -> SchedulerOutputs:
        if self.engine_type == EngineType.PROMPT:
            outputs = self._schedule_prefill() or SchedulerOutputs([], True)
        elif self.engine_type == EngineType.COMPLETION:
            outputs = self._schedule_decode()
        else:
            outputs = self._schedule_normal()
        if outputs.prompt_run:
            self.decode_step = 0
        else:
            self.decode_step += 1
        return outputs

    def _schedule_normal(self) -> SchedulerOutputs:
        step = self.decode_step
        is_any = step != 0 and step % self.config.prefill_any_attempt_interval == 0
        is_all = step % self.config.prefill_attempt_interval == 0
        if not is_any and not is_all:
            return self._schedule_decode()
        want = (
            self.get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK
            and bool(self._waiting)
            and self._waiting[0].schedulable
        )
        if is_any:
            do_prefill = self.dp.all_reduce(f"vote_any@{step}", int(want), max) == 1
        else:
            do_prefill = self.dp.all_reduce(f"vote_all@{step}", int(want), min) == 1
        if not do_prefill:
            return self._schedule_decode()
        outputs = self._schedule_prefill() if want else None
        if (
            self.dp.all_reduce(f"vote_scheduled@{step}", int(outputs is not None), max)
            == 0
        ):
            return self._schedule_decode()
        return outputs or SchedulerOutputs([], True)

    def _schedule_prefill(self) -> Optional[SchedulerOutputs]:
        scheduled: list[Sequence] = []
        chunks: list[tuple[int, int]] = []
        ignored: list[Sequence] = []
        budget = self.config.max_num_batched_tokens
        while self._waiting:
            seq = self._waiting[0]
            if seq.is_finished():
                ignored.append(self._waiting.popleft())
                continue
            if len(self._running) + len(scheduled) >= self.config.max_num_seqs:
                break
            if not seq.block_table:
                if (
                    seq.get_len() > self.config.max_model_len
                    or self.num_blocks(seq) > len(self.block_manager._ref) - 1
                ):
                    # Too long, or can never fit (AllocStatus::NEVER).
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored.append(self._waiting.popleft())
                    continue
                if not self.try_allocate_seq(seq):
                    break
            start, end = seq.num_cached_tokens, seq.get_len()
            for ck in sorted(seq.chunk_checkpoints):
                if start < ck < end:
                    end = ck
                    break
            if not self.config.chunked_prefill and end - start > budget:
                seq.status = SequenceStatus.FINISHED_IGNORED
                self.free_seq(seq)
                ignored.append(self._waiting.popleft())
                continue
            end = min(end, start + budget)
            if end <= start:
                break
            scheduled.append(self._waiting.popleft())
            chunks.append((start, end))
            budget -= end - start
            if end < seq.get_len() or budget <= 0:
                break
        if self.engine_type != EngineType.NORMAL or scheduled or ignored:
            return SchedulerOutputs(scheduled, True, [], ignored, chunks)
        return None

    def _schedule_decode(self) -> SchedulerOutputs:
        finished = [s for s in self._running if s.is_finished()]
        self._running = [s for s in self._running if not s.is_finished()]
        ignored = [s for s in self._waiting if s.is_finished()]
        self._waiting = deque(s for s in self._waiting if not s.is_finished())
        while self.prompted and len(self._running) < self.config.max_num_seqs:
            seq = self.prompted.popleft()
            if seq.status == SequenceStatus.WAITING:
                seq.status = SequenceStatus.RUNNING
                self._running.append(seq)
            else:
                finished.append(seq)
        # DeepInfer's LCFS loop: whenever no block is free it preempts the
        # longest-output seqs, whether or not the seq at hand needs a block.
        running = sorted(self._running, key=lambda s: s.get_output_len(), reverse=True)
        scheduled: list[Sequence] = []
        lo, hi = 0, len(running)
        while lo < hi and len(scheduled) < self.config.max_num_seqs:
            seq = running[hi - 1]
            while not self.block_manager.can_allocate(1) and hi - lo > 1:
                self._preempt(running[lo], ignored)
                lo += 1
            if self.block_manager.can_allocate(1):
                if len(seq.block_table) < self.num_blocks(seq):
                    seq.block_table = (
                        seq.block_table + self.block_manager.allocate_nolock(1)
                    )
                scheduled.append(seq)
            else:
                self._preempt(seq, ignored)
            hi -= 1
        self._running = scheduled
        return SchedulerOutputs(list(scheduled), False, finished, ignored)

    def _preempt(self, seq: Sequence, ignored: list[Sequence]) -> None:
        # stop_preempted_request: the victim stops with an error.
        seq.status = SequenceStatus.FINISHED_PREEMPTED
        self.free_seq(seq)
        ignored.append(seq)


@dataclass
class _Out:
    request_id: str
    token_id: int
    finished: bool
    finish_reason: str
    token_logprobs: float
    top_token_ids: list[int]
    top_logprobs: list[float]
    num_output_tokens: int
    prompt_len: int


_POOL: dict[int, _Out] = {}
_HANDLES = itertools.count(1)


class RequestOutput:
    """Static getters over output handles, as in ``deepinfer._C.engine``."""

    @staticmethod
    def get_request_id(ptr: int) -> str:
        return _POOL[ptr].request_id

    @staticmethod
    def get_token_id(ptr: int) -> int:
        return _POOL[ptr].token_id

    @staticmethod
    def get_finished(ptr: int) -> bool:
        return _POOL[ptr].finished

    @staticmethod
    def get_finish_reason(ptr: int) -> str:
        return _POOL[ptr].finish_reason

    @staticmethod
    def get_token_logprobs(ptr: int) -> float:
        return _POOL[ptr].token_logprobs

    @staticmethod
    def get_top_token_ids(ptr: int) -> list[int]:
        return list(_POOL[ptr].top_token_ids)

    @staticmethod
    def get_top_logprobs(ptr: int) -> list[float]:
        return list(_POOL[ptr].top_logprobs)

    @staticmethod
    def get_num_output_tokens(ptr: int) -> int:
        return _POOL[ptr].num_output_tokens

    @staticmethod
    def get_prompt_len(ptr: int) -> int:
        return _POOL[ptr].prompt_len


class CacheEngine:
    """``gpu_blocks``/``gpu_states`` as host byte arrays, plus the state store helpers."""

    state_size_to_disk = STATE_STORED_BYTES
    total_bytes_inference = STATE_BYTES

    def __init__(self, num_blocks: int, num_slots: int) -> None:
        self.gpu_blocks = np.zeros((num_blocks, BLOCK_SIZE * 4), dtype=np.uint8)
        self.gpu_states = np.zeros((num_slots, STATE_BYTES), dtype=np.uint8)
        self.block_tokens = self.gpu_blocks.view(np.int32)

    def select_state_slice_to_store(
        self,
        state_id: int,
        last_hit_position: int,
        state_cache: np.ndarray,
        state_tensor: Optional[np.ndarray] = None,
        TO_CE: bool = False,
    ) -> np.ndarray:
        if not TO_CE and last_hit_position % KVSTORE_GRAIN != 0:
            raise AssertionError(
                f"{last_hit_position=} not a multiple of {KVSTORE_GRAIN}"
            )
        size = self.total_bytes_inference if TO_CE else self.state_size_to_disk
        src = state_cache[state_id, :size]
        if state_tensor is None:
            return src
        state_tensor[...] = src
        return state_tensor

    def load_state_from_store_to_prefill_cache(
        self,
        state_id: int,
        last_hit_position: int,
        state_tensor: np.ndarray,
        state_cache: np.ndarray,
    ) -> None:
        state_cache[state_id, : self.state_size_to_disk] = state_tensor


@dataclass
class DriverWorker:
    cache_engine: CacheEngine
    device: Any = None


class LLMEngine:
    def __init__(self, args: EngineArgs) -> None:
        if (
            args.engine_type == EngineType.COMPLETION
            and not args.stop_preempted_request
        ):
            raise ValueError("COMPLETION engines require stop_preempted_request=True")
        self.engine_type = args.engine_type
        max_model_len = args.max_model_len or 4096
        num_blocks = _env_int("DEEPINFER_FAKE_NUM_GPU_BLOCKS", 256)
        num_slots = int(args.max_num_seqs * 1.2) + 1
        self.cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=num_blocks,
            bytes_per_block=BLOCK_SIZE * 4,
            num_state_seqs=num_slots,
            bytes_of_state_per_seqs=STATE_BYTES,
            lookahead_tokens=args.lookahead_tokens,
        )
        self.scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens
            or max(max_model_len, 4096),
            prefill_attempt_interval=_env_int("DEEPINFER_PREFILL_ATTEMPT_INTERVAL", 70),
            prefill_any_attempt_interval=_env_int(
                "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL", 100
            ),
            chunked_prefill=args.chunked_prefill,
            stop_preempted_request=args.stop_preempted_request,
        )
        self.model_config = ModelConfig(args.model, max_model_len)
        self.tokenizer = Tokenizer()
        self.block_manager = BlockManager(num_blocks)
        self.state_manager = BlockManager(num_slots)
        self.driver_worker = DriverWorker(CacheEngine(num_blocks, num_slots))
        dp_size = args.data_parallel_size or max(1, args.expert_parallel_size)
        self.dp_rank = args.engine_rank
        self._dp = _DPGroup(args.engine_rank, dp_size)
        self.scheduler = Scheduler(
            self.engine_type,
            self.block_manager,
            self.state_manager,
            self.scheduler_config,
            self._dp,
            args.lookahead_tokens,
        )
        zos = os.environ.get("DEEPINFER_ZERO_OVERHEAD_SCHEDULING")
        self.zero_overhead = (
            self.engine_type.default_zero_overhead_scheduling
            if zos is None
            else zos == "1"
        )
        self.eos_after = _env_int("DEEPINFER_FAKE_EOS_AFTER", 0) or None
        self.step_delay_s = float(os.environ.get("DEEPINFER_FAKE_STEP_DELAY_S") or 0)
        self.step_overlap = self.zero_overhead and bool(
            _env_int("DEEPINFER_FAKE_STEP_OVERLAP", 0)
        )
        # Overlapped steps: when the last launched "forward" completes.
        self._forward_done = 0.0
        self.seq_counter = itertools.count()
        self.closed = False
        self.num_steps = 0
        self._inflight: list[tuple[Sequence, int]] = []
        self._handles: list[int] = []
        if self.engine_type != EngineType.NORMAL:
            self.init_pd_split()

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine:
        return cls(engine_args)

    # -- DeepInfer's virtual hooks ---------------------------------------------

    def init_pd_split(self) -> None:
        pass

    def finalize_seq(self, seq: Sequence) -> None:
        self.scheduler.free_seq(seq)

    def post_schedule(self, scheduler_outputs: SchedulerOutputs) -> None:
        pass

    def do_save_kvcache(
        self, scheduled: list[Sequence], finished: list[Sequence]
    ) -> None:
        pass

    def pe_prepare_save_kvcache(
        self, scheduled: list[Sequence], finished: list[Sequence]
    ) -> None:
        pass

    def pe_do_save_kvcache(
        self, scheduled: list[Sequence], finished: list[Sequence]
    ) -> None:
        pass

    def send_processed_prompts(self) -> None:
        pass

    # -- public API -----------------------------------------------------------

    def add_request(
        self,
        user_id: str,
        request_id: str,
        sampling_params: Any,
        prompt_token_ids: np.ndarray,
        multi_modal_data: Optional[dict] = None,
        arrival_time: Optional[float] = None,
        extra: Optional[dict] = None,
        enqueue: bool = True,
        chunk_checkpoints: Optional[list[int]] = None,
    ) -> Optional[Sequence]:
        if sampling_params is None:
            sampling_params = {}
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(
                **{k: v for k, v in sampling_params.items() if v is not None}
            )
        sampling_params.eos_id = EOS_TOKEN_ID
        if (
            not isinstance(prompt_token_ids, np.ndarray)
            or prompt_token_ids.dtype != np.int32
        ):
            raise TypeError("prompt_token_ids must be an np.int32 array")
        if prompt_token_ids.size == 0:
            raise ValueError("prompt must not be empty")
        if prompt_token_ids.min() < 0 or prompt_token_ids.max() >= VOCAB_SIZE:
            return None
        seq = Sequence(
            next(self.seq_counter),
            prompt_token_ids,
            request_id,
            user_id,
            sampling_params,
            arrival_time if arrival_time is not None else time.perf_counter(),
        )
        if chunk_checkpoints is not None:
            seq.chunk_checkpoints = chunk_checkpoints
        if enqueue:
            self.scheduler.add_sequence(seq)
        return seq

    def abort_request(self, request_id: Any) -> tuple[int, int]:
        ids = [request_id] if isinstance(request_id, str) else list(request_id)
        seqs = self.scheduler.abort_sequence(ids)
        return len(seqs), sum(s.get_len() for s in seqs)

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_num_unfinished_seqs()

    def so_builder_has_pending(self) -> bool:
        return False  # no structured output

    def has_unfinished_requests(self) -> bool:
        local = int(self.scheduler.has_unfinished_seqs())
        return self._dp.all_reduce("has_unfinished", local, max) > 0

    def all_reduce_max(self, tag: str, values: list[int]) -> list[int]:
        """Stands in for Dynamo's int64 MAX ``torch.distributed.all_reduce``
        over the DP group (``di_engine.dp_max``)."""
        return self._dp.all_reduce(tag, list(values), max)

    @property
    def collectives(self) -> Counter[str]:
        """DP collectives this rank entered, by tag."""
        return self._dp.calls

    def step(self) -> np.ndarray:
        if self.closed:
            raise RuntimeError("engine is closed")
        self._release()
        self.num_steps += 1
        outs = self._process_decode(self._inflight)
        self._inflight = []
        so = self.scheduler.schedule()
        self.post_schedule(so)
        # pre_sampler_sync
        if so.prompt_run:
            self.pe_prepare_save_kvcache(so.scheduled_seqs, so.finished_seqs)
        else:
            self.do_save_kvcache(so.scheduled_seqs, so.finished_seqs)
        if self.engine_type != EngineType.PROMPT:
            for seq in so.finished_seqs:
                self.finalize_seq(seq)
        outs.extend(self._ignored_output(seq) for seq in so.ignored_seqs)
        if so.prompt_run:
            # post_process of a prompt step
            self.pe_do_save_kvcache(so.scheduled_seqs, so.finished_seqs)
            outs.extend(self._run_prefill(so))
            if self.engine_type == EngineType.PROMPT:
                self.send_processed_prompts()
        else:
            results = self._run_decode(so)
            if self.zero_overhead:
                self._inflight = results
            else:
                outs.extend(self._process_decode(results))
        if self.step_delay_s:
            if self.step_overlap:
                # Launch this step's forward behind the previous one, then
                # wait for the previous one (its post-processing needs it).
                now = time.monotonic()
                launched = max(now, self._forward_done)
                self._await_forward(now)
                self._forward_done = launched + self.step_delay_s
            else:
                time.sleep(self.step_delay_s)
        return self._publish(outs)

    def _await_forward(self, now: float) -> None:
        if self._forward_done > now:
            time.sleep(self._forward_done - now)

    def flush_inflight_reqs(self) -> np.ndarray:
        self._release()
        if self.step_overlap:
            self._await_forward(time.monotonic())
        outs = self._process_decode(self._inflight)
        self._inflight = []
        return self._publish(outs)

    def close(self) -> None:
        if self.closed:
            return
        self._dp.all_reduce("close", 0, max)  # DeepInfer's stop() barrier
        self._dp.close()
        self._release()
        self.closed = True

    # -- the fake model -------------------------------------------------------

    def _compute(self, seq: Sequence, start: int, end: int) -> None:
        """Write KV for positions [start, end) and advance the state digest."""
        ce = self.driver_worker.cache_engine
        table = seq.block_table
        digest = ce.gpu_states[seq.state_id, :32].tobytes()
        for pos in range(start, end):
            token = seq.data[pos]
            ce.block_tokens[table[pos // BLOCK_SIZE], pos % BLOCK_SIZE] = token
            digest = hashlib.sha256(digest + token.to_bytes(4, "little")).digest()
        ce.gpu_states[seq.state_id, :32] = np.frombuffer(digest, dtype=np.uint8)
        # The "dropped" half (consumer-layer SWA) is rebuilt from the suffix.
        ce.gpu_states[seq.state_id, 32:] = np.frombuffer(
            hashlib.sha256(
                np.asarray(seq.data[max(0, end - 8) : end]).tobytes()
            ).digest(),
            dtype=np.uint8,
        )

    def _sample(self, seq: Sequence) -> int:
        ce = self.driver_worker.cache_engine
        n = seq.num_cached_tokens
        rows = ce.block_tokens[seq.block_table[: -(-n // BLOCK_SIZE)]].reshape(-1)[:n]
        checksum = int(
            (rows.astype(np.int64) * np.arange(1, n + 1, dtype=np.int64)).sum()
            % (1 << 61)
        )
        digest = ce.gpu_states[seq.state_id, :32].tobytes()
        h = hashlib.sha256(digest + checksum.to_bytes(8, "little")).digest()
        token = 2 + int.from_bytes(h[:4], "little") % (VOCAB_SIZE - 2)
        sp = seq.sampling_params
        if (
            self.eos_after
            and seq.get_output_len() + 1 >= self.eos_after
            and not sp.ignore_eos
        ):
            token = EOS_TOKEN_ID
        return token

    def get_nextn_pad_token_ids(self) -> list[int]:
        # mtp_info_pool.get_nextn_pad_token_ids (the real engine returns a tensor).
        return [VOCAB_SIZE - 1] * envs.DEEPINFER_USE_NEXTN_DECODE

    def _run_prefill(self, so: SchedulerOutputs) -> list[_Out]:
        outs: list[_Out] = []
        for seq, (start, end) in zip(so.scheduled_seqs, so.chunks):
            self._compute(seq, start, end)
            seq.num_cached_tokens = end
            if end < seq.get_len():
                self.scheduler._waiting.appendleft(
                    seq
                )  # chunked: continue next prompt step
                continue
            seq.status = SequenceStatus.RUNNING
            outs.append(self._accept(seq, self._sample(seq)))
            self.scheduler._running.append(seq)
        return outs

    def _run_decode(self, so: SchedulerOutputs) -> list[tuple[Sequence, int]]:
        results = []
        for seq in so.scheduled_seqs:
            pos = seq.get_len() - 1
            if seq.last_output_token != seq.data[-1] or seq.num_cached_tokens != pos:
                raise RuntimeError(
                    f"{seq.request_id}: decode input mismatch "
                    f"(last_output_token={seq.last_output_token}, last={seq.data[-1]}, "
                    f"cached={seq.num_cached_tokens}, len={seq.get_len()})"
                )
            # prepare_sample.cpp: every decode step verifies a fixed-width draft.
            nextn = envs.DEEPINFER_USE_NEXTN_DECODE
            if len(seq.last_next_n_token) != nextn:
                raise RuntimeError(
                    f"{seq.request_id}: Check failed: draft_tokens.size() == nextn_token_cnt "
                    f"({len(seq.last_next_n_token)} vs. {nextn})"
                )
            self._compute(seq, pos, pos + 1)
            seq.num_cached_tokens = pos + 1
            results.append((seq, self._sample(seq)))
        return results

    def _process_decode(self, results: list[tuple[Sequence, int]]) -> list[_Out]:
        # Sequences aborted while their token was in flight emit nothing.
        return [self._accept(seq, tok) for seq, tok in results if not seq.is_finished()]

    def _accept(self, seq: Sequence, token: int) -> _Out:
        seq.last_output_token = token
        # A prompt-only step pads the drafts (llm_engine.py _process_model_outputs).
        seq.last_next_n_token = self.get_nextn_pad_token_ids()
        seq.append_token_id(token)
        seq.append_compressed_token_id(self.scheduler.map_engram(token))
        sp = seq.sampling_params
        if not sp.ignore_eos and token == EOS_TOKEN_ID:
            seq.status = SequenceStatus.FINISHED_STOPPED
        elif seq.get_len() > self.scheduler_config.max_model_len or (
            seq.get_output_len() == sp.max_tokens
        ):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
        logprob = -0.1 * (token % 10)
        k = sp.top_logprobs
        return _Out(
            request_id=seq.request_id,
            token_id=token,
            finished=seq.is_finished(),
            finish_reason=_FINISH_REASONS.get(seq.status, ""),
            token_logprobs=logprob,
            top_token_ids=[(token + i) % VOCAB_SIZE for i in range(k)],
            top_logprobs=[logprob - 0.1 * i for i in range(k)],
            num_output_tokens=seq.get_output_len(),
            prompt_len=seq.get_prompt_len(),
        )

    def _ignored_output(self, seq: Sequence) -> _Out:
        return _Out(
            request_id=seq.request_id,
            token_id=seq.data[-1],
            finished=True,
            finish_reason=_FINISH_REASONS.get(seq.status, ""),
            token_logprobs=1.0,
            top_token_ids=[],
            top_logprobs=[],
            num_output_tokens=seq.get_output_len(),
            prompt_len=seq.get_prompt_len(),
        )

    def _publish(self, outs: list[_Out]) -> np.ndarray:
        handles = [next(_HANDLES) for _ in outs]
        for handle, out in zip(handles, outs):
            _POOL[handle] = out
        self._handles = handles
        return np.asarray(handles, dtype=np.uint64)

    def _release(self) -> None:
        for handle in self._handles:
            _POOL.pop(handle, None)
        self._handles = []
