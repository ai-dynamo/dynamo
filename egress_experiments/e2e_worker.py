# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python and Rust response-path mock workers served by the real Dynamo runtime.

The synthetic TRT-LLM engine remains a separate process. Its dispatch thread
delivers one response batch per engine iteration into per-request queues, then
wakes the worker's single asyncio loop once. The loop performs the calibrated
``handle_response`` and ``build_response`` work before the real Dynamo bridge
egresses each chunk.

The Rust arm converts each raw IPC batch once on the dispatch thread, releases
the GIL, and performs response state updates, construction, and bounded runtime
egress in Rust. Python is woken only when a request completes.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import signal
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any

from egress_experiments.costs import Costs
from egress_experiments.dynamo_sim.probes import LoopProbe
from egress_experiments.dynamo_sim.worker import (
    TrtllmWorkerHandler,
    USING_REAL_PUSH_EGRESS,
    push_egress_capable,
)
from egress_experiments.fake_trtllm.engine import (
    BatchConfig,
    ConstantIteration,
    EngineConfig,
)
from egress_experiments.fake_trtllm.llm import FakeLLM

logger = logging.getLogger(__name__)
_perf = time.perf_counter_ns


def adapt_runtime_request(
    request: dict[str, Any], *, request_id: str, default_max_tokens: int = 64
) -> dict[str, Any]:
    """Map the real frontend request shape to the simulated TRT-LLM shape."""
    stop_conditions = request.get("stop_conditions") or {}
    sampling_options = request.get("sampling_options") or {}
    max_tokens = (
        stop_conditions.get("max_tokens")
        or request.get("max_tokens")
        or default_max_tokens
    )
    n = sampling_options.get("n") or request.get("n") or 1
    return {
        **request,
        "id": request_id,
        "max_tokens": int(max_tokens),
        "n": int(n),
    }


def response_path_costs(scale: float) -> Costs:
    """Scale only Python response work; keep request admission identical."""
    baseline = Costs()
    return replace(
        baseline,
        handle_response_us=baseline.handle_response_us * scale,
        build_response_us=baseline.build_response_us * scale,
    )


class RuntimeTrtllmWorkerHandler(TrtllmWorkerHandler):
    """Adapt real preprocessed requests without changing the measured path."""

    def __init__(self, llm: FakeLLM, costs: Costs | None = None) -> None:
        super().__init__(llm, costs=costs)
        self._request_ids = itertools.count(1)

    @property
    def completed_results(self):
        return self.llm.completed_results

    # This decorator must remain outermost: the branch's Rust bridge detects
    # the response_sender parameter on the registered callable.
    @push_egress_capable
    async def generate(self, request, context=None):
        adapted = adapt_runtime_request(
            request,
            request_id=f"runtime-{next(self._request_ids)}",
            default_max_tokens=self.llm.engine_config.max_tokens,
        )
        async for out in self.generate_locally(adapted, context):
            yield out


class RuntimeRustEgressWorkerHandler:
    """Keep request admission in Python and move each response batch to Rust."""

    def __init__(self, llm: FakeLLM, processor: Any, costs: Costs | None = None) -> None:
        self.llm = llm
        self.processor = processor
        self.costs = costs or Costs()
        self._request_path = TrtllmWorkerHandler(llm, costs=self.costs)
        self._request_ids = itertools.count(1)

    @property
    def completed_results(self):
        return self.llm.completed_results

    @property
    def responses_yielded(self) -> int:
        return int(self.processor.frames_sent)

    async def generate(self, request, context=None, response_sender=None):
        if response_sender is None:
            raise RuntimeError("Rust response path requires response_sender")

        adapted = adapt_runtime_request(
            request,
            request_id=f"runtime-{next(self._request_ids)}",
            default_max_tokens=self.llm.engine_config.max_tokens,
        )
        prompt_tokens = len(adapted.get("token_ids") or [])
        generation_result, _ = self._request_path._start_generation(
            adapted,
            response_processor=self.processor,
            response_sender=response_sender,
            prompt_tokens=prompt_tokens,
        )
        try:
            await generation_result.wait_native()
        except (asyncio.CancelledError, GeneratorExit):
            self.llm.cancel_native(generation_result)
            raise

        if False:  # pragma: no cover - keep the handler an async generator
            yield


@dataclass
class StatsSampler:
    """Convert cumulative worker counters into interval rates."""

    start_ns: int
    _last_ns: int | None = None
    _last_dispatched: int = 0
    _last_yielded: int = 0
    _last_notify: int = 0
    _last_batch_index: int = 0

    def sample(
        self,
        *,
        now_ns: int,
        responses_dispatched: int,
        responses_yielded: int,
        notify_many_calls: int,
        ipc_batch_sizes: list[int],
    ) -> dict[str, float | int]:
        previous_ns = self.start_ns if self._last_ns is None else self._last_ns
        elapsed_s = max((now_ns - previous_ns) / 1e9, 1e-9)
        dispatched_delta = responses_dispatched - self._last_dispatched
        yielded_delta = responses_yielded - self._last_yielded
        notify_delta = notify_many_calls - self._last_notify
        new_batches = ipc_batch_sizes[self._last_batch_index :]

        result: dict[str, float | int] = {
            "response_rate": dispatched_delta / elapsed_s,
            "yield_rate": yielded_delta / elapsed_s,
            "loop_backlog": responses_dispatched - responses_yielded,
            "responses_per_notify": (
                dispatched_delta / notify_delta if notify_delta else 0.0
            ),
            "mean_ipc_batch": statistics.fmean(new_batches) if new_batches else 0.0,
            "responses_dispatched": responses_dispatched,
            "responses_yielded_to_egress": responses_yielded,
            "notify_many_calls": notify_many_calls,
        }
        self._last_ns = now_ns
        self._last_dispatched = responses_dispatched
        self._last_yielded = responses_yielded
        self._last_notify = notify_many_calls
        self._last_batch_index = len(ipc_batch_sizes)
        return result


async def _report_stats(
    llm: FakeLLM,
    handler: Any,
    probe: LoopProbe,
    interval_s: float,
) -> None:
    sampler = StatsSampler(start_ns=_perf())
    while True:
        await asyncio.sleep(interval_s)
        sample = sampler.sample(
            now_ns=_perf(),
            responses_dispatched=llm.responses_dispatched,
            responses_yielded=handler.responses_yielded,
            notify_many_calls=llm.notify_many_calls,
            ipc_batch_sizes=list(llm.ipc_batch_sizes),
        )
        sample["loop"] = probe.report()
        logger.info("GIL_PATH_STATS %s", json.dumps(sample, sort_keys=True))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the simulated TRT-LLM GIL response path through Dynamo"
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-name", default="gil-path-mocker")
    parser.add_argument("--endpoint", default="dynamo.backend.generate")
    parser.add_argument("--discovery-backend", default="etcd")
    parser.add_argument("--request-plane", default="tcp")
    parser.add_argument(
        "--response-path",
        choices=("python", "rust"),
        default="python",
        help="run response handling on the Python loop or in native Rust",
    )
    parser.add_argument("--batch-total", type=_positive_int, default=200)
    parser.add_argument("--iteration-ms", type=_positive_float, default=52.1)
    parser.add_argument("--max-tokens", type=_positive_int, default=89)
    parser.add_argument("--stream-interval", type=_positive_int, default=1)
    parser.add_argument("--response-shards", type=_positive_int, default=4)
    parser.add_argument("--response-queue-depth", type=_positive_int, default=2)
    parser.add_argument(
        "--response-cost-scale",
        "--cost-scale",
        dest="response_cost_scale",
        type=_non_negative_float,
        default=1.0,
        help="scale handle_response and build_response only",
    )
    parser.add_argument("--stats-interval", type=_positive_float, default=1.0)
    parser.add_argument("--loop-lag-ms", type=_positive_float, default=5.0)
    return parser.parse_args(argv)


def _validate_push_runtime() -> None:
    if not USING_REAL_PUSH_EGRESS:
        raise RuntimeError(
            "the real push_egress_capable decorator was not loaded"
        )

    import dynamo._core as core

    if not hasattr(core, "ResponseSender"):
        raise RuntimeError(
            "the runtime requires bindings built from this checkout; "
            "ResponseSender is missing"
        )


def _validate_rust_runtime(core: Any) -> None:
    if not hasattr(core, "NativeResponseEgress"):
        raise RuntimeError(
            "--response-path rust requires bindings with NativeResponseEgress; "
            "rebuild lib/bindings/python from this checkout"
        )


async def serve(args: argparse.Namespace) -> None:
    _validate_push_runtime()
    import dynamo._core as core

    if args.response_path == "rust":
        _validate_rust_runtime(core)
    loop = asyncio.get_running_loop()
    costs = response_path_costs(args.response_cost_scale)
    engine_config = EngineConfig(
        batch=BatchConfig(total=args.batch_total),
        iteration=ConstantIteration(args.iteration_ms),
        max_tokens=args.max_tokens,
        stream_interval=args.stream_interval,
    )

    # Fork the synthetic engine before constructing DistributedRuntime and its
    # Tokio threads. Forking a multithreaded runtime process is unsafe.
    llm = FakeLLM(engine_config, costs=costs)
    handler = None
    probe = None
    runtime = None
    reporter = None
    probe_reset_installed = False

    try:
        llm.start(loop)
        if args.response_path == "rust":
            benchmark_response_work_us = (
                costs.handle_response_us + costs.build_response_us
            )
            handler = RuntimeRustEgressWorkerHandler(
                llm,
                processor=core.NativeResponseEgress(
                    shards=args.response_shards,
                    queue_depth=args.response_queue_depth,
                    benchmark_response_work_us=benchmark_response_work_us,
                ),
                costs=costs,
            )
        else:
            handler = RuntimeTrtllmWorkerHandler(llm, costs=costs)
        probe = LoopProbe(lag_ms=args.loop_lag_ms)
        probe.install(loop)
        if hasattr(signal, "SIGUSR1"):
            def reset_probe() -> None:
                probe.reset()
                logger.info("GIL_PATH_PROBE_RESET")

            loop.add_signal_handler(signal.SIGUSR1, reset_probe)
            probe_reset_installed = True
        reporter = loop.create_task(
            _report_stats(llm, handler, probe, args.stats_interval)
        )
        from dynamo.llm import (
            ModelInput,
            ModelType,
            WorkerType,
            register_model,
        )
        from dynamo.runtime import DistributedRuntime

        runtime = DistributedRuntime(
            loop, args.discovery_backend, args.request_plane
        )
        endpoint = runtime.endpoint(args.endpoint)
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat | ModelType.Completions,
            endpoint,
            args.model_path,
            model_name=args.model_name,
            kv_cache_block_size=64,
            worker_type=WorkerType.Aggregated,
            ignore_weights=True,
        )

        demand = engine_config.responses_per_iteration / (
            args.iteration_ms / 1000.0
        )
        loop_cost = 0.0 if args.response_path == "rust" else costs.loop_us_per_response_push
        logger.info(
            "%s response-path worker ready: %s; demand=%.1f responses/s, "
            "modelled_loop_cost=%.2f us/response, modelled_loop_load=%.1f%%",
            args.response_path,
            engine_config.describe(),
            demand,
            loop_cost,
            demand * loop_cost / 1e4,
        )
        if args.response_path == "rust":
            logger.info(
                "native response configuration: shards=%d, queue_depth=%d, "
                "response_work_target_us=%.2f",
                args.response_shards,
                args.response_queue_depth,
                benchmark_response_work_us,
            )
        await endpoint.serve_endpoint(handler.generate)
    finally:
        try:
            if reporter is not None:
                reporter.cancel()
                await asyncio.gather(reporter, return_exceptions=True)
        finally:
            try:
                if probe is not None:
                    if probe_reset_installed:
                        loop.remove_signal_handler(signal.SIGUSR1)
                    probe.uninstall()
            finally:
                try:
                    if runtime is not None:
                        runtime.shutdown()
                finally:
                    try:
                        llm.shutdown()
                    finally:
                        if handler is not None and args.response_path == "rust":
                            logger.info(
                                "RUST_EGRESS_FINAL %s",
                                json.dumps(
                                    {
                                        "active_requests": (
                                            handler.processor.active_requests
                                        ),
                                        "responses_processed": (
                                            handler.processor.responses_processed
                                        ),
                                        "responses_dropped": (
                                            handler.processor.responses_dropped
                                        ),
                                        "frames_sent": handler.processor.frames_sent,
                                    },
                                    sort_keys=True,
                                ),
                            )


def _interrupt_for_shutdown(_signum: int, _frame: Any) -> None:
    """Turn SIGTERM into normal async-runner cancellation and unwinding."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvloop

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_for_shutdown)
    try:
        try:
            uvloop.run(serve(args))
        except KeyboardInterrupt:
            pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
