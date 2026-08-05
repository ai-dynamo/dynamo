# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wires the two halves together and runs a load.

Process/thread topology, matching the dynamo column of the diagram:

    ┌─ engine process ─────────────┐        ┌─ app process ───────────────────┐
    │ trtllm_executor_worker       │        │ MainThread  = the ONE asyncio   │
    │  · iterate, emit one token   │  IPC   │               loop (worker)     │
    │    per in-flight request     │◀──────▶│ proxy_dispatch_result_thread    │
    │  · ONE message per iteration │        │ tokio-runtime-worker (ingress + │
    │    (handle_for_ipc_batched)  │        │               egress drivers)   │
    └──────────────────────────────┘        └─────────────────────────────────┘

Two independent sets of knobs:

**Ingress** -- requests arriving off the wire. :attr:`SimConfig.arrival` picks
the process (``constant`` / ``poisson`` / ``closed``) and :attr:`SimConfig.qps`
the rate. Default is the steady-state QPS, i.e. exactly the rate that keeps the
engine's batch full; feeding faster than that only queues inside the engine and
measures the engine rather than the loop.

**Engine** -- :attr:`SimConfig.batch` (total, or per-rank × ranks) sets how many
responses come back per iteration, and :attr:`SimConfig.iteration` picks
constant or batch-dependent iteration time. See
``fake_trtllm/engine.py``.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from egress_experiments import architectures, loop_meter
from egress_experiments.costs import (
    SERVE_LOOP_US_PER_RESPONSE,
    Costs,
    reset_spin_ledger,
    spin_ledger,
)
from egress_experiments.dynamo_sim.gil_noise import GilNoise, GilNoiseConfig
from egress_experiments.dynamo_sim.probes import LoopProbe, RequestRecord, summarize
from egress_experiments.dynamo_sim.rust_bridge import TokioRuntime
from egress_experiments.fake_trtllm.engine import (
    BatchConfig,
    ConstantIteration,
    EngineConfig,
    IterationModel,
)

_perf = time.perf_counter_ns

#: Arrival processes for the ingress.
ARRIVALS = ("constant", "poisson", "closed")


@dataclass
class SimConfig:
    #: "pull" (default Rust behaviour) or "push" (DYN_TRTLLM_PUSH_EGRESS=1).
    #: Ignored when :attr:`architecture` is set to something else -- the
    #: architecture owns the driver shape.
    egress: str = "push"
    #: Response-path architecture, from ``egress_experiments.architectures``.
    #: ``None`` means "baseline-{egress}".
    architecture: Optional[str] = None
    #: Total requests to run.
    requests: int = 500
    #: Output tokens per request; each one is a response.
    max_tokens: int = 16
    #: Input length, only used for completion_usage bookkeeping.
    isl: int = 1024

    # -- ingress: requests coming in off the wire ------------------------
    #: ``constant`` -- evenly spaced arrivals at :attr:`qps`.
    #: ``poisson``  -- exponential inter-arrivals with mean 1/qps.
    #: ``closed``   -- no schedule; hold :attr:`concurrency` in flight.
    #:
    #: Open by default: a closed loop admits its first `concurrency` requests
    #: simultaneously, and that thundering herd dominates admission-latency
    #: percentiles instead of the loop contention under study.
    arrival: str = "constant"
    #: Offered load. ``None`` means :attr:`steady_state_qps`.
    qps: Optional[float] = None
    #: Seeded, so a poisson run is reproducible.
    arrival_seed: int = 1234
    #: Closed-loop only. ``None`` means the engine's total batch.
    concurrency: Optional[int] = None

    # -- engine ----------------------------------------------------------
    #: Responses per iteration: total, or per-rank x ranks.
    batch: BatchConfig = field(default_factory=BatchConfig)
    #: Constant or batch-dependent iteration time.
    iteration: IterationModel = field(default_factory=ConstantIteration)
    #: TRT-LLM stream_interval: one response per S iterations, carrying S
    #: tokens. Divides the response rate without touching the token rate, so
    #: it is the dominant control on loop load. The capture ran 40.
    stream_interval: int = 1
    #: Threads in the spawn_blocking pool. tokio's own default is far larger,
    #: but what matters here is that there is more than one and that none of
    #: them is the event loop. The capture's app interpreter had 50
    #: GIL-capable threads in total.
    blocking_threads: int = 8
    #: Hard wall-clock bound on the run, seconds. Without it a saturation
    #: benchmark that does NOT saturate never ends: closed-loop requests never
    #: complete, arrivals never stop, and the backlog -- the only other stop
    #: condition -- stays flat precisely because the loop is keeping up.
    duration_s: Optional[float] = None
    #: Abort once the loop is this far behind the engine. An overloaded loop
    #: (si=1 at a real batch) otherwise queues responses until the process
    #: dies, which is a worse demonstration than a clean stop.
    max_backlog: Optional[int] = None

    # -- everything else --------------------------------------------------
    costs: Costs = field(default_factory=Costs)
    lag_ms: float = 5.0
    #: Extra GIL-capable threads (the decode worker had ~45 total). See
    #: dynamo_sim/gil_noise.py -- required to reproduce the pull/push
    #: latency difference, which is a contention effect.
    gil_noise: GilNoiseConfig = field(default_factory=GilNoiseConfig)

    @property
    def architecture_name(self) -> str:
        return self.architecture or f"baseline-{self.egress}"

    def __post_init__(self) -> None:
        if self.arrival not in ARRIVALS:
            raise ValueError(f"arrival must be one of {ARRIVALS}, got {self.arrival!r}")
        if self.egress not in ("pull", "push"):
            raise ValueError(f"egress must be pull or push, got {self.egress!r}")

    # -- derived ----------------------------------------------------------

    @property
    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            batch=self.batch,
            iteration=self.iteration,
            max_tokens=self.max_tokens,
            stream_interval=self.stream_interval,
        )

    @property
    def full_batch_iteration_ms(self) -> float:
        """Iteration time with the batch full -- the steady-state cost."""
        return self.iteration.duration_ms(self.batch.total_batch, self.batch)

    @property
    def responses_per_iteration(self) -> float:
        """What the loop has to absorb per iteration: batch / stream_interval."""
        return self.batch.total_batch / self.stream_interval

    @property
    def offered_response_rate(self) -> float:
        """Responses/s the engine will offer once the batch is full."""
        return self.responses_per_iteration / (self.full_batch_iteration_ms / 1000.0)

    @property
    def residency_s(self) -> float:
        """How long a request occupies a batch slot: max_tokens ITERATIONS.

        Unaffected by stream_interval -- a request still generates one token
        per iteration, it is only reported less often.
        """
        return self.max_tokens * self.full_batch_iteration_ms / 1000.0

    @property
    def steady_state_qps(self) -> float:
        """The arrival rate that holds exactly ``batch.total_batch`` in flight.

        ``total_batch / residency``. Offering more than this just fills the
        engine's waiting queue.
        """
        return self.batch.total_batch / self.residency_s

    @property
    def offered_qps(self) -> float:
        return self.qps or self.steady_state_qps

    @property
    def closed_loop_concurrency(self) -> int:
        return self.concurrency or self.batch.total_batch

    def describe_ingress(self) -> str:
        if self.arrival == "closed":
            return f"closed loop, {self.closed_loop_concurrency} in flight"
        return f"{self.arrival} @ {self.offered_qps:.0f} qps"


@dataclass
class SimResult:
    config: SimConfig
    wall_s: float
    responses: int
    requests_completed: int
    #: FakeLLM counters.
    ipc_messages: int
    responses_dispatched: int
    notify_many_calls: int
    responses_per_deque_entry: float
    #: Driver counters.
    loop_handoffs: int
    blocking_gil_acquisitions: int
    delivered: int
    fallback_yields: int
    #: Probe output.
    probe: Dict[str, Any]
    #: Derived per-request timings, in ms.
    queue_wait: Dict[str, float]
    ttft: Dict[str, float]
    tpot: Dict[str, float]
    threads: Dict[str, Optional[str]]
    errors: List[str]
    #: Extra GIL contenders that were running, so a report can never imply a
    #: contention regime it did not actually apply.
    gil_noise_threads: int = 0

    # -- steady-state window ----------------------------------------------
    #
    # The capture this models is a 5.169 s window at max batch, so a whole-run
    # average is not the comparable quantity: ramp-up and drain both run at
    # roughly half batch and drag it down. These fields cover the window
    # between "the batch has filled" and "arrivals stopped".
    window_s: float = 0.0
    window_responses: int = 0
    window_ipc_messages: int = 0
    window_requests: int = 0
    #: False when the run was too short to contain a usable window, in which
    #: case the window fields fall back to the whole run.
    window_valid: bool = False

    # -- backlog: how far the loop fell behind the engine -------------------
    #
    # Responses the dispatch thread has put_nowait-ed into per-request queues
    # minus those the loop has actually delivered. Flat means the loop keeps
    # up; a rising line IS the asyncio queue growing, which is what an
    # overloaded loop looks like from the outside.
    backlog_samples: List[int] = field(default_factory=list)
    backlog_final: int = 0
    backlog_max: int = 0
    #: Least-squares slope over the samples, responses/s. ~0 = keeping up.
    backlog_growth_per_s: float = 0.0
    #: True when --max-backlog tripped and the run was stopped early.
    backlog_aborted: bool = False
    #: True when the run hit :attr:`SimConfig.duration_s` instead. Distinct
    #: from backlog_aborted: a timed-out run was NOT necessarily saturated.
    timed_out: bool = False
    #: Responses per IPC message as EMITTED by the engine, in the window.
    window_emitted_per_message: float = 0.0
    #: perf_counter_ns at which the run began, and one timestamp per delivered
    #: response. bench.py windows these itself -- a saturation run never
    #: reaches the arrival-driven steady window this module computes.
    started_ns: int = 0
    response_times: List[int] = field(default_factory=list)
    #: One timestamp per item AS IT LEAVES THE LOOP -- the benchmark's score.
    #: Distinct from response_times, which is measured after the tokio-side
    #: consumer and therefore reflects that consumer when it is the laggard.
    loop_item_times: List[int] = field(default_factory=list)
    #: Which thread ticked the meter. Should be the loop and nothing else.
    loop_meter_threads: Dict[str, int] = field(default_factory=dict)
    #: Modelled work by thread, from costs.spin_ledger().
    spin_us_by_thread: Dict[str, float] = field(default_factory=dict)
    #: Whatever the architecture wants to report.
    arch_report: Dict[str, Any] = field(default_factory=dict)

    # -- observed engine behaviour ----------------------------------------

    @property
    def mean_engine_batch(self) -> float:
        """Responses per iteration the engine EMITTED, in the window.

        Compare with ``config.responses_per_iteration``: a shortfall means the
        offered QPS never filled the batch, so any conclusion about loop load
        is understated.
        """
        return self.window_emitted_per_message

    @property
    def batch_fill(self) -> float:
        target = self.config.responses_per_iteration
        return self.mean_engine_batch / target if target else 0.0

    @property
    def mean_delivered_per_message(self) -> float:
        """Responses per message the loop actually got through.

        Below :attr:`mean_engine_batch` exactly when the loop is falling
        behind, so the gap between the two is the backlog accruing.
        """
        if not self.window_ipc_messages:
            return 0.0
        return self.window_responses / self.window_ipc_messages

    @property
    def mean_iteration_ms(self) -> float:
        if not self.window_ipc_messages:
            return 0.0
        return self.window_s * 1000.0 / self.window_ipc_messages

    @property
    def achieved_qps(self) -> float:
        if not self.window_s:
            return 0.0
        return self.window_requests / self.window_s

    @property
    def whole_run_engine_batch(self) -> float:
        if not self.ipc_messages:
            return 0.0
        return self.responses_dispatched / self.ipc_messages

    # -- the diagram's "supporting aggregates" -----------------------------

    @property
    def loop_us_per_response(self) -> float:
        c = self.config.costs
        return (
            c.loop_us_per_response_push
            if self.config.egress == "push"
            else c.loop_us_per_response_pull
        )

    @property
    def loop_capacity_per_s(self) -> float:
        return 1e6 / self.loop_us_per_response

    @property
    def response_demand_per_s(self) -> float:
        """Responses per second in the steady-state window."""
        return self.window_responses / self.window_s if self.window_s else 0.0

    @property
    def loop_load(self) -> float:
        return self.response_demand_per_s / self.loop_capacity_per_s

    @property
    def offered_load(self) -> float:
        """Offered responses/s over loop capacity. > 1 means it cannot keep up.

        Unlike :attr:`loop_load`, which divides ACHIEVED demand by capacity and
        therefore saturates just under 1 no matter how overloaded the run is,
        this uses what the engine *offers*. It is the honest overload number.
        """
        return self.config.offered_response_rate / self.loop_capacity_per_s

    #: Backlog growth counts as "not keeping up" once it exceeds this fraction
    #: of what the loop can DRAIN. Measuring against the offered rate instead
    #: would be self-defeating: the more overloaded a run is, the larger the
    #: threshold it has to clear, and a badly swamped loop whose engine is also
    #: throttled can grow steadily while still looking small next to the offer.
    BACKLOG_GROWTH_TOLERANCE = 0.05

    @property
    def measured_items_per_s(self) -> float:
        """Throughput actually observed at the loop's exit, whole run."""
        times = self.loop_item_times
        if len(times) < 2:
            return 0.0
        span_s = (times[-1] - times[0]) / 1e9
        return len(times) / span_s if span_s > 0 else 0.0

    @property
    def backlog_growing(self) -> bool:
        if self.backlog_aborted:
            return True
        if self.timed_out and self.backlog_max < 1000:
            # Ran the clock out with the loop keeping up: not saturated.
            return False
        # Relative to what this architecture MEASURABLY does, not to
        # loop_capacity_per_s -- that is derived from `Costs`, i.e. from the
        # BASELINE cost model, and is the same 11,718/s for every
        # architecture. Judging a 30,000/s architecture against a threshold
        # sized for a 9,500/s one makes ordinary drift look like saturation,
        # which stops the benchmark's ladder early and reports an
        # engine-limited run as a loop-limited one. Bug found by the
        # offloaded-postproc experiment.
        reference = self.measured_items_per_s or self.loop_capacity_per_s
        return self.backlog_growth_per_s > self.BACKLOG_GROWTH_TOLERANCE * reference

    @property
    def overloaded(self) -> bool:
        return self.offered_load > 1.0 or self.backlog_growing

    @property
    def serve_ratio(self) -> float:
        """How many times dearer than trtllm-serve's 1.94 us bookkeeping."""
        return self.loop_us_per_response / SERVE_LOOP_US_PER_RESPONSE


async def _run_async(cfg: SimConfig) -> SimResult:
    py_loop = asyncio.get_running_loop()

    probe = LoopProbe(lag_ms=cfg.lag_ms)
    probe.install(py_loop)

    arch = architectures.get(cfg.architecture_name)

    reset_spin_ledger()
    loop_meter.reset()
    llm = arch.build_llm(cfg.engine_config, cfg.costs)
    llm.start(py_loop)

    records: Dict[str, RequestRecord] = {}
    handler = arch.build_handler(llm, cfg.costs, records)

    tokio = TokioRuntime(blocking_threads=cfg.blocking_threads)
    tokio.start()

    driver = arch.build_driver(handler, py_loop, tokio, cfg.costs)
    arch.on_started(llm, driver)

    done = asyncio.Event()
    started_ns = _perf()
    #: When the last request was issued -- the moment the drain begins.
    arrivals_done_ns = [0]

    async def orchestrate() -> None:
        """Runs on the tokio loop: this is the ingress side."""

        async def issue(index: int) -> None:
            request_id = f"req-{index}"
            record = RequestRecord(request_id=request_id)
            records[request_id] = record
            request = {
                "id": request_id,
                "token_ids": list(range(cfg.isl)),
                "max_tokens": cfg.max_tokens,
            }
            await driver.run(request, record)

        if cfg.arrival == "closed":
            semaphore = asyncio.Semaphore(cfg.closed_loop_concurrency)

            async def issue_limited(index: int) -> None:
                async with semaphore:
                    await issue(index)

            gathered = asyncio.gather(*(issue_limited(i) for i in range(cfg.requests)))
            # A closed loop issues continuously, so the drain only begins when
            # the last request has actually been admitted; approximate with
            # the end of the run.
            await gathered
            arrivals_done_ns[0] = _perf()
            return

        # Open loop: arrivals on a schedule.
        loop = asyncio.get_running_loop()
        qps = cfg.offered_qps
        rng = random.Random(cfg.arrival_seed)
        start = loop.time()
        elapsed = 0.0
        inflight = []
        for index in range(cfg.requests):
            if cfg.arrival == "poisson":
                # Exponential inter-arrivals: the memoryless process a real
                # front end sees, not evenly spaced traffic.
                elapsed += rng.expovariate(qps)
            else:
                elapsed = index / qps
            delay = (start + elapsed) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            inflight.append(asyncio.ensure_future(issue(index)))
        arrivals_done_ns[0] = _perf()
        await asyncio.gather(*inflight)

    noise = GilNoise(cfg.gil_noise)
    noise.start()
    orchestrator = tokio.submit(orchestrate())

    # Backlog sampler. A plain thread: the loop is the thing under
    # observation, so sampling from it would be measuring the instrument.
    backlog_samples: List[int] = []
    backlog_stop = threading.Event()
    backlog_tripped = [False]
    timed_out = [False]

    def sample_backlog() -> None:
        deadline = started_ns + int(cfg.duration_s * 1e9) if cfg.duration_s else None
        while not backlog_stop.wait(0.1):
            backlog_samples.append(llm.responses_dispatched - driver.delivered)
            over_backlog = (
                cfg.max_backlog is not None and backlog_samples[-1] > cfg.max_backlog
            )
            # A saturation benchmark that does NOT saturate has no other stop
            # condition: closed-loop requests never finish, arrivals never end,
            # and the backlog stays flat precisely because the loop is coping.
            expired = deadline is not None and _perf() > deadline
            if over_backlog or expired:
                backlog_tripped[0] = True
                timed_out[0] = expired and not over_backlog
                # Cancel the orchestrator rather than stopping the engine:
                # killing the engine would strand every in-flight request
                # waiting for a final response that never arrives, and the
                # gather below would never return.
                orchestrator.cancel()
                return

    sampler = threading.Thread(
        target=sample_backlog, name="backlog-sampler", daemon=True
    )
    sampler.start()

    def _finish(_fut) -> None:
        py_loop.call_soon_threadsafe(done.set)

    orchestrator.add_done_callback(_finish)
    await done.wait()
    # Ask the future whether it was cancelled rather than catching
    # CancelledError: on this interpreter concurrent.futures._base.CancelledError
    # is a DIFFERENT class from asyncio.CancelledError -- not an alias, not a
    # subclass -- so `except asyncio.CancelledError` silently fails to match and
    # the abort escapes as an unhandled error.
    if orchestrator.cancelled():
        if not backlog_tripped[0]:
            raise RuntimeError("orchestration was cancelled unexpectedly")
    else:
        orchestrator.result()  # surface orchestration failures

    wall_s = (_perf() - started_ns) / 1e9

    backlog_stop.set()
    sampler.join(timeout=2.0)

    arch.on_finished(llm, driver)
    report = probe.report()
    probe.uninstall()
    noise.stop()
    llm.shutdown()
    tokio.stop()

    # ---- steady-state window -------------------------------------------
    # Opens once the batch has had time to fill (one full request residency)
    # and closes when arrivals stop. Outside it the batch is ramping or
    # draining at roughly half strength, which is not what the capture's
    # 5.169 s window at max batch measured.
    residency_ns = int(cfg.max_tokens * cfg.full_batch_iteration_ms * 1e6)
    win_start = started_ns + residency_ns
    win_end = arrivals_done_ns[0] or _perf()
    window_valid = (win_end - win_start) > 0.5e9
    if not window_valid:
        win_start, win_end = started_ns, _perf()

    def _in_window(times: List[int]) -> int:
        return sum(1 for t in times if win_start <= t <= win_end)

    window_s = (win_end - win_start) / 1e9
    window_responses = _in_window(driver.response_times)
    window_ipc = _in_window(llm.ipc_times)
    # Emitted per message, from the boundary rather than from what was
    # delivered: under overload the loop lags and delivered/message would
    # under-report the engine's real batch.
    emitted = [
        size
        for t, size in zip(llm.ipc_times, llm.ipc_batch_sizes)
        if win_start <= t <= win_end
    ]
    window_emitted_per_message = (sum(emitted) / len(emitted)) if emitted else 0.0

    # Least-squares slope of the backlog, in responses/s. Sampled at 10 Hz, so
    # x is the sample index over 10.
    growth = 0.0
    if len(backlog_samples) >= 4:
        n = len(backlog_samples)
        mean_x = (n - 1) / 2.0
        mean_y = sum(backlog_samples) / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(backlog_samples))
        den = sum((i - mean_x) ** 2 for i in range(n))
        growth = (num / den) * 10.0 if den else 0.0

    finished = [r for r in records.values() if r.responses]
    # Latency percentiles over requests admitted inside the window, so the
    # opening burst and the tail are excluded from those too.
    windowed = [r for r in finished if win_start <= r.accepted_ns <= win_end]
    if not windowed:
        windowed = finished

    return SimResult(
        config=cfg,
        wall_s=wall_s,
        responses=driver.delivered,
        requests_completed=len(finished),
        ipc_messages=llm.ipc_messages,
        responses_dispatched=llm.responses_dispatched,
        notify_many_calls=llm.notify_many_calls,
        responses_per_deque_entry=llm.responses_per_deque_entry,
        loop_handoffs=driver.loop_handoffs,
        blocking_gil_acquisitions=driver.blocking_gil_acquisitions,
        delivered=driver.delivered,
        fallback_yields=getattr(driver, "fallback_yields", 0),
        probe=report,
        queue_wait=summarize([r.queue_wait_ms for r in windowed]),
        ttft=summarize([r.ttft_ms for r in windowed]),
        tpot=summarize([r.tpot_ms for r in windowed if r.responses > 1]),
        threads={
            "loop": llm.loop_thread_name,
            "dispatch": llm.dispatch_thread_name,
        },
        errors=list(driver.errors),
        gil_noise_threads=cfg.gil_noise.threads,
        window_s=window_s,
        window_responses=window_responses,
        window_ipc_messages=window_ipc,
        window_requests=len(windowed),
        window_valid=window_valid,
        window_emitted_per_message=window_emitted_per_message,
        backlog_samples=backlog_samples,
        backlog_final=backlog_samples[-1] if backlog_samples else 0,
        backlog_max=max(backlog_samples) if backlog_samples else 0,
        backlog_growth_per_s=growth,
        backlog_aborted=backlog_tripped[0] and not timed_out[0],
        timed_out=timed_out[0],
        started_ns=started_ns,
        response_times=list(driver.response_times),
        loop_item_times=loop_meter.timestamps(),
        loop_meter_threads=loop_meter.report(),
        spin_us_by_thread=spin_ledger(),
        arch_report=arch.extra_report(),
    )


def run_simulation(cfg: Optional[SimConfig] = None) -> SimResult:
    """Blocking entry point. Runs the app process's event loop on this thread."""
    return asyncio.run(_run_async(cfg or SimConfig()))
