# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The opaque engine -- a separate OS process, exactly as in the real worker.

Why a process and not a thread
------------------------------
The diagram treats the engine as opaque ("identical C++/CUDA core") and the
whole argument is about the *app* interpreter's GIL. TRT-LLM runs the executor
in a separate process (``executor/worker.py`` under mpi/Rpc), so its Python
never contends for the app's GIL. Simulating it in a thread would put the
engine's bookkeeping on the same GIL as the event loop and quietly invent
contention that does not exist on the real worker, which would flatter the push
path. It forks.

What it models
--------------
One decode iteration, repeated:

1. admit newly submitted requests up to the configured batch (:class:`BatchConfig`),
2. burn one iteration of wall clock (:class:`ConstantIteration` or
   :class:`BatchDependentIteration`) -- the GPU work, deliberately *not*
   modelled in detail,
3. emit exactly one token for every in-flight request and ship the whole
   iteration as ONE IPC message.

Step 3 is ``_AwaitResponseHelper.handle_for_ipc_batched``
(``executor/base_worker.py:1252``), which accumulates every response of the
iteration into ``rsp_batch`` and hands the list to ``FusedIpcQueue.put``. That
single message is what makes the proxy issue exactly one ``notify_many``, and
therefore what makes a whole batch of responses share one event-loop
ready-deque entry.

No context phase is modelled. The capture is decode rank 0 of a disaggregated
deployment (``point_offline_disagg_dynamo``), so the request arrives with its
KV already transferred and goes straight into decode.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Protocol, runtime_checkable

from egress_experiments.fake_trtllm.ipc import Link
from egress_experiments.fake_trtllm.result import Response, ResultPayload

_perf = time.perf_counter_ns


# ---------------------------------------------------------------------------
# Batch geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchConfig:
    """How many responses come back from the engine per iteration.

    Two equivalent ways to say it:

    * ``BatchConfig(total=1024)`` -- the aggregate decode batch,
    * ``BatchConfig(per_rank=64, ranks=16)`` -- attention-DP geometry.

    They differ in one thing that matters: :class:`BatchDependentIteration`
    scales with the **per-rank** batch, because each rank does attention over
    its own sequences. Same total batch spread over more ranks is a faster
    iteration and therefore a higher response rate.

    The IPC structure does not change with ``ranks``. Rank 0 owns the executor
    loop that talks to the proxy, so the whole iteration still arrives as one
    message. (Modelling assumption -- per-rank result lanes are not simulated.)

    This is the engine's **decode batch** -- requests in flight -- not the
    number of responses per iteration. Those differ by
    :attr:`EngineConfig.stream_interval`::

        responses per iteration = total_batch / stream_interval

    On the 355778 capture that is ``986 x 8 ranks / si 40 = 197.2`` against a
    measured 200.1, and it closes the rest of the geometry to 1.5 %: demand
    3,785/s vs 3,841 measured, loop load 32.3 % vs 32.8 %.

    ``986`` is therefore the **per-rank** batch, which is also what
    :class:`BatchDependentIteration` was fitted on -- and evaluating that fit at
    986 gives 51.69 ms against the measured 52.10 ms iteration.
    """

    total: Optional[int] = None
    per_rank: Optional[int] = None
    ranks: int = 1

    def __post_init__(self) -> None:
        if self.total is not None and self.per_rank is not None:
            raise ValueError("give either total= or per_rank=, not both")
        if self.ranks < 1:
            raise ValueError("ranks must be >= 1")
        if self.per_rank is not None and self.per_rank < 1:
            raise ValueError("per_rank must be >= 1")
        if self.total is not None and self.total < 1:
            raise ValueError("total must be >= 1")

    @property
    def total_batch(self) -> int:
        if self.total is not None:
            return self.total
        if self.per_rank is not None:
            return self.per_rank * self.ranks
        return 132

    @property
    def per_rank_batch(self) -> int:
        if self.per_rank is not None:
            return self.per_rank
        return math.ceil(self.total_batch / self.ranks)

    def per_rank_inflight(self, active: int) -> int:
        """Sequences on one rank when ``active`` are in flight overall."""
        return math.ceil(active / self.ranks)

    def describe(self) -> str:
        if self.ranks > 1:
            return (
                f"{self.total_batch} total "
                f"({self.per_rank_batch}/rank x {self.ranks} ranks)"
            )
        return f"{self.total_batch} total"


# ---------------------------------------------------------------------------
# Iteration-time models
# ---------------------------------------------------------------------------


@runtime_checkable
class IterationModel(Protocol):
    """How long one decode iteration takes."""

    def duration_ms(self, active: int, batch: BatchConfig) -> float:
        ...

    def describe(self) -> str:
        ...


@dataclass(frozen=True)
class ConstantIteration:
    """Fixed iteration time, whatever the batch.

    The default 34 ms is chosen so that the default batch of 132 produces
    ``132 / 0.034 = 3,882`` responses/s, against the capture's measured 3,871.
    It is deliberately *below* the observed 46.4 ms TPOT: the diagram's claim is
    that the difference is loop queueing, so it has to emerge rather than be
    configured in.
    """

    iteration_ms: float = 34.0

    def duration_ms(self, active: int, batch: BatchConfig) -> float:
        return self.iteration_ms

    def describe(self) -> str:
        return f"constant {self.iteration_ms:g} ms"


@dataclass(frozen=True)
class BatchDependentIteration:
    """Iteration time linear in the **per-rank** batch.

    Both coefficients are fitted to the capture, not invented. Two
    ``(batch, iteration)`` points are recoverable from ``ASYNCIO_GIL_PATH.md``'s
    header table plus its ``us per req per iteration`` row::

        dynamo         5.169 s / 100 iters = 51.69 ms   at batch 986
        trtllm-serve   4.853 s / 100 iters = 48.53 ms   at batch 880

    Those reproduce the document's own 52.42 and 55.15 us/req/iter exactly, so
    the window and batch figures are self-consistent. The line through them::

        iteration_ms = 22.296 + 0.029811 * batch

    which has the right shape for decode: a fixed cost that does not scale with
    the batch (weight streaming, collectives, launch overhead) plus a per-
    sequence cost (KV reads).

    Two assumptions, both load-bearing:

    * The points come from **different systems**. Treating them as two points
      on one curve relies on the document's own claim that the GPU work is
      comparable ("identical C++/CUDA core · GPU work comparable"). If that is
      wrong, so is the slope.
    * The fit is anchored at batch 880-986 and both points are single-rank
      geometry. Applying it per-rank under attention DP is an extrapolation,
      and so is evaluating it at small batches -- at batch 132 it predicts
      26.2 ms against :class:`ConstantIteration`'s calibrated 34 ms.
    """

    #: Fixed per-iteration cost, ms.
    base_ms: float = 22.296
    #: Marginal cost per sequence on a rank, microseconds.
    per_request_us: float = 29.811

    def duration_ms(self, active: int, batch: BatchConfig) -> float:
        per_rank = batch.per_rank_inflight(active)
        return self.base_ms + self.per_request_us * per_rank / 1000.0

    def describe(self) -> str:
        return (
            f"batch-dependent {self.base_ms:g} ms + "
            f"{self.per_request_us:g} us/req/rank"
        )


@dataclass
class EngineConfig:
    """Engine geometry."""

    batch: BatchConfig = field(default_factory=BatchConfig)
    iteration: IterationModel = field(default_factory=ConstantIteration)
    #: Tokens generated per request before ``is_final``. TOKENS, not responses:
    #: at ``stream_interval=S`` a request yields ``max_tokens / S`` responses.
    max_tokens: int = 64
    #: TRT-LLM's ``stream_interval``: create a response only every S iterations,
    #: carrying the S tokens generated since the last one. Its own docs give the
    #: reason -- "set this to a larger value when the resource bottleneck is on
    #: the CPU side" -- which is precisely the bottleneck this whole simulation
    #: is about.
    #:
    #: It is the single most powerful knob on loop load, because it divides the
    #: response rate without touching the token rate::
    #:
    #:     responses/s = total_batch / (stream_interval * iteration_s)
    #:
    #: The 355778 capture ran ``stream_interval: 40``
    #: (``server-gen-si40.yaml:62``). At the same batch, si=1 would offer
    #: 151,401 responses/s against a loop capacity of 11,718/s.
    stream_interval: int = 1

    def __post_init__(self) -> None:
        if self.stream_interval < 1:
            raise ValueError("stream_interval must be >= 1")

    @property
    def responses_per_iteration(self) -> float:
        """What the loop actually has to absorb, per engine iteration."""
        return self.batch.total_batch / self.stream_interval

    def describe(self) -> str:
        return (
            f"batch {self.batch.describe()} · stream_interval "
            f"{self.stream_interval} · iteration {self.iteration.describe()}"
        )


# ---------------------------------------------------------------------------
# The engine process
# ---------------------------------------------------------------------------


def _reader(endpoint, inbox: Deque, stop: threading.Event) -> None:
    """Drain the request lane. Mirrors the worker's request thread."""
    while not stop.is_set():
        batch = endpoint.get()
        if batch is None:
            stop.set()
            return
        for item in batch:
            if item is None:  # shutdown sentinel: worker.py:285 result_queue.put(None)
                stop.set()
                return
            inbox.append(item)


def engine_main(request_link: Link, result_link: Link, cfg: EngineConfig) -> None:
    """Child-process entry point."""
    requests = request_link.open_child()
    results = result_link.open_child()

    inbox: Deque[dict] = collections.deque()
    stop = threading.Event()
    reader = threading.Thread(
        target=_reader, args=(requests, inbox, stop), name="engine_request_reader"
    )
    reader.daemon = True
    reader.start()

    # client_id -> [tokens left, tokens buffered since last stream]
    active: Dict[int, List[int]] = {}
    active_generations: Dict[int, Optional[int]] = {}
    waiting: Deque[dict] = collections.deque()
    max_batch = cfg.batch.total_batch
    stream_interval = cfg.stream_interval

    try:
        while not stop.is_set():
            while inbox:
                waiting.append(inbox.popleft())
            while waiting and len(active) < max_batch:
                req = waiting.popleft()
                active[req["client_id"]] = [
                    int(req.get("max_tokens", cfg.max_tokens)),
                    0,
                ]
                active_generations[req["client_id"]] = req.get("generation")

            if not active:
                # Idle: nothing in flight, so no iteration to run.
                time.sleep(0.0005)
                continue

            # The iteration cost is decided by what is in flight NOW, so a
            # batch-dependent engine speeds up as the batch drains.
            period_ns = int(cfg.iteration.duration_ms(len(active), cfg.batch) * 1e6)
            deadline = _perf() + period_ns
            while _perf() < deadline and not stop.is_set():
                time.sleep(0.0005)

            # handle_for_ipc_batched: one rsp_batch for the whole iteration.
            now = _perf()
            rsp_batch = []
            finished = []
            for client_id, state in active.items():
                # Every in-flight request generates a token every iteration --
                # stream_interval changes only how often that becomes a
                # RESPONSE, never the token rate.
                state[0] -= 1
                state[1] += 1
                is_final = state[0] <= 0
                if state[1] < stream_interval and not is_final:
                    continue  # buffered on the engine side; the loop sees nothing

                # One response carrying every token since the last one.
                buffered = state[1]
                state[1] = 0
                rsp_batch.append(
                    Response(
                        client_id=client_id,
                        generation=active_generations[client_id],
                        result=ResultPayload(
                            new_token_ids=[
                                [
                                    _token_for(client_id, state[0] + offset)
                                    for offset in range(buffered)
                                ]
                            ],
                            is_final=is_final,
                            finish_reasons=["length"] if is_final else None,
                        ),
                        emitted_ns=now,
                    )
                )
                if is_final:
                    finished.append(client_id)
            for client_id in finished:
                active.pop(client_id, None)
                active_generations.pop(client_id, None)

            if not rsp_batch:
                # Nothing crossed this iteration -- at stream_interval > 1 that
                # is normal and must NOT become an empty IPC message, or the
                # observed responses-per-message would be wrong.
                continue

            results.put(rsp_batch)  # ONE message, len(rsp_batch) responses
    finally:
        stop.set()
        try:
            results.put(None)  # unblock the proxy dispatch thread
        except Exception:
            pass
        results.close()
        requests.close()


def _token_for(client_id: int, remaining: int) -> int:
    """A deterministic, meaningless token id. The engine is opaque."""
    return 1000 + ((client_id * 31 + remaining) % 5000)


def spawn_engine(
    cfg: Optional[EngineConfig] = None,
) -> "EngineHandle":
    """Fork the engine. ``fork`` so socketpair fds are inherited."""
    import multiprocessing

    cfg = cfg or EngineConfig()
    request_link = Link("request")
    result_link = Link("result")

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=engine_main,
        args=(request_link, result_link, cfg),
        name="trtllm_executor_worker",
        daemon=True,
    )
    proc.start()
    # The parent must drop its copies of the child fds or EOF never arrives.
    request_link.close_child_in_parent()
    result_link.close_child_in_parent()
    return EngineHandle(proc, request_link, result_link, cfg)


@dataclass
class EngineHandle:
    proc: "object"
    request_link: Link
    result_link: Link
    cfg: EngineConfig

    def shutdown(self, timeout: float = 5.0) -> None:
        try:
            self.request_link.parent.put(None)
        except Exception:
            pass
        try:
            self.proc.join(timeout)  # type: ignore[attr-defined]
        except Exception:
            pass
        if getattr(self.proc, "is_alive", lambda: False)():
            self.proc.terminate()  # type: ignore[attr-defined]
            self.proc.join(1.0)  # type: ignore[attr-defined]
        self.request_link.close()
        self.result_link.close()
