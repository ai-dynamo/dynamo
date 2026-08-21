# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU end-to-end coverage for ``InstrumentedScheduler`` self-benchmark
mode (``DYN_BENCHMARK_MODE``).

Pure-Python unit tests on ``_bench_inject_fake_decode`` and the empty-frame
schedule branch can't catch the recently-fixed regressions in isolation --
both only manifest with the model running on a GPU:

1. ``kv_connector_metadata`` must be attached to every benchmark-built
   ``SchedulerOutput``. Otherwise vLLM's worker-side
   ``_get_kv_connector_output`` asserts and EngineCore dies before the
   first synthetic decode batch in any disagg config (any worker with a
   KV connector configured: NixlConnector, FlexKVConnectorV1, etc.).

2. The synthetic decode prompt must be padded to ``ctx_len + 1``.
   Otherwise the async-scheduler's ``-1`` placeholder write at
   ``token_ids_cpu[req_idx, ctx_len]`` collides with the read slot the
   next benchmark batch's request reads as its decode input -- the
   embedding lookup OOBs and EngineCore dies on the second decode point
   (first ``batch_size > 1`` sweep).

This test runs the worker through its **normal** startup -- with
``--benchmark-mode`` set, the worker performs the self-benchmark before
registering with the runtime, then proceeds to serve. So a passing
serving health check already proves the benchmark didn't break the
engine. We additionally:

* validate every worker wrote a benchmark JSON whose every point has
  at least one FPM with positive ``wall_time``,
* send a real chat-completion request and assert a non-empty response.

This means a regression in either fix (or anything else that breaks
the benchmark for a serving worker) fails this test.

Coverage matrix:

* ``test_self_benchmark_agg_serves_after_bench`` -- aggregated worker,
  ``--benchmark-mode agg`` (covers prompt-padding regression on the
  decode sweep within agg mode; ``gpu_1``).
* ``test_self_benchmark_disagg_serves_after_bench`` -- prefill worker
  (``--benchmark-mode prefill``) + decode worker
  (``--benchmark-mode decode``, NixlConnector ``kv_both``); the
  user-reported configuration. Exercises BOTH fixes simultaneously
  (``gpu_2``).

Setup mirrors ``tests/fault_tolerance/cancellation/test_vllm.py``:
same model (``FAULT_TOLERANCE_MODEL_NAME`` = ``Qwen/Qwen3-0.6B``),
same ``DynamoFrontendProcess`` + ``ManagedProcess`` worker pattern,
same ``runtime_services_dynamic_ports`` + ``predownload_models``
fixtures, same NixlConnector / kv-events / VLLM_NIXL_SIDE_CHANNEL_PORT
wiring for the disagg case.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from tests.utils.client import send_request
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port, deallocate_ports

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
]


_BENCH_WARMUP_ITERATIONS = "2"

# Match the cancellation tests' worker config. max-model-len is a bit
# tighter here so the benchmark sweep finishes quickly.
_MAX_MODEL_LEN = "512"
_GPU_MEMORY_UTILIZATION = "0.45"


def _validate_benchmark_results(output_path: Path, expected_mode: str) -> dict:
    """Parse the benchmark JSON and assert every point produced a real
    forward-pass measurement (positive ``wall_time``)."""
    assert (
        output_path.exists()
    ), f"benchmark JSON missing at {output_path} -- worker never wrote it"
    data = json.loads(output_path.read_text())
    assert data.get("schema_version") == 2

    actual_mode = data.get("config", {}).get("mode")
    assert actual_mode == expected_mode, (
        f"benchmark mode in JSON ({actual_mode}) does not match "
        f"expected ({expected_mode})"
    )
    assert data.get("valid") is True, (
        f"benchmark reported incomplete coverage: coverage={data.get('coverage')} "
        f"skipped_points={data.get('skipped_points')} "
        f"missing_phases={data.get('missing_phases')}"
    )

    results = data.get("results") or []
    assert (
        len(results) > 0
    ), f"benchmark JSON has no result points (mode={expected_mode})"

    assert [r["point"]["benchmark_id"] for r in results] == list(
        range(1, len(results) + 1)
    )

    for r in results:
        point = r["point"]
        fpms = r.get("fpms") or []
        assert len(fpms) > 0, (
            f"point {point} produced no FPMs -- the model didn't actually "
            f"execute that batch (regression in _bench_inject_fake_decode "
            f"or the empty-frame schedule branch)"
        )
        for fpm_index, fpm in enumerate(fpms):
            assert fpm.get("counter_id") == point["benchmark_id"]
            wall_time = fpm.get("wall_time", 0.0)
            assert wall_time > 0, (
                f"point {point} FPM has non-positive wall_time={wall_time}; "
                f"the benchmark recorded a heartbeat instead of a real "
                f"forward-pass measurement"
            )

            if point["point_type"] == "prefill":
                scheduled = fpm.get("scheduled_requests") or {}
                assert scheduled.get("num_prefill_requests", 0) > 0, (
                    f"prefill point {point} captured a non-prefill FPM: "
                    f"scheduled_requests={scheduled}"
                )
                if fpm_index == 0:
                    batch_size = point.get("batch_size", 1)
                    assert scheduled.get("num_prefill_requests") == batch_size, (
                        f"point {point} measured the wrong prefill batch size: "
                        f"scheduled_requests={scheduled}"
                    )
                    assert scheduled.get("sum_prefill_tokens") == point.get(
                        "total_prefill_tokens"
                    ), (
                        f"point {point} measured the wrong prefill token total: "
                        f"scheduled_requests={scheduled}"
                    )
                    assert scheduled.get("sum_prefill_kv_tokens") == point.get(
                        "total_kv_read_tokens", 0
                    ), (
                        f"point {point} measured the wrong initial KV reads: "
                        f"scheduled_requests={scheduled}"
                    )
            else:
                scheduled = fpm.get("scheduled_requests") or {}
                assert scheduled.get("num_decode_requests", 0) > 0, (
                    f"decode point {point} captured a non-decode FPM: "
                    f"scheduled_requests={scheduled}"
                )
                if fpm_index == 0:
                    batch_size = point.get("batch_size", 1)
                    assert scheduled.get("num_decode_requests") == batch_size, (
                        f"point {point} measured the wrong decode batch size: "
                        f"scheduled_requests={scheduled}"
                    )
                    assert scheduled.get("sum_decode_kv_tokens") == point.get(
                        "total_kv_read_tokens"
                    ), (
                        f"point {point} measured the wrong decode context: "
                        f"scheduled_requests={scheduled}"
                    )

    merged_path = output_path.with_name(
        f"{output_path.stem}_merged{output_path.suffix}"
    )
    assert merged_path.exists(), f"merged benchmark JSON missing at {merged_path}"
    merged = json.loads(merged_path.read_text())
    assert merged.get("run_id") == data.get("run_id")
    assert len(merged.get("iteration_groups", [])) == len(results)
    assert merged.get("dp", {}).get("ranks") == [0]
    return data


class _DynamoBenchmarkWorker(ManagedProcess):
    """Process manager for a vLLM worker started with ``--benchmark-mode``.

    Modeled on ``tests/fault_tolerance/cancellation/test_vllm.py``'s
    ``DynamoWorkerProcess`` so the disagg worker pair (prefill + decode)
    wires NixlConnector / kv-events / NIXL side channel exactly the
    same way -- this keeps CI ports and process layout consistent with
    other vLLM e2e tests.
    """

    def __init__(
        self,
        request,
        frontend_port: int,
        bench_output_path: Path,
        bench_mode: str,
        is_prefill: bool | None,
    ):
        self.bench_output_path = bench_output_path
        self.bench_mode = bench_mode
        self.is_prefill = is_prefill
        self.frontend_port = frontend_port
        allocated_ports: list[int] = []
        request.addfinalizer(lambda ports=allocated_ports: deallocate_ports(ports))

        # Allocate a per-worker system port like the cancellation test does.
        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        allocated_ports.append(self.system_port)
        # Allocate a per-worker forward-pass-metrics ZMQ publisher port.
        # ``InstrumentedScheduler`` (auto-injected by --benchmark-mode) binds
        # ``tcp://*:DYN_FORWARDPASS_METRIC_PORT + dp_rank`` for the FPM
        # publisher. Two workers on the same host with the default port
        # collide on the second bind (``Address already in use (addr='tcp://*:20380')``).
        # The operator sets this per-engine via DynamoFPMBasePort; the
        # cancellation test never hits this because it doesn't run with
        # --benchmark-mode and so doesn't auto-inject InstrumentedScheduler.
        self.fpm_port = allocate_port(DynamoPortRange.FPM.value)
        allocated_ports.append(self.fpm_port)

        env = os.environ.copy()
        if "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES" not in env:
            kv_mark = request.node.get_closest_marker("requested_vllm_kv_cache_bytes")
            if kv_mark:
                env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(int(kv_mark.args[0]))

        gpu_mem_args = build_gpu_mem_args("build_vllm_gpu_mem_args", env=env)
        if not gpu_mem_args:
            gpu_mem_args = ["--gpu-memory-utilization", _GPU_MEMORY_UTILIZATION]

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--compilation-config",
            '{"cudagraph_capture_sizes":[1,2,4,8]}',
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "128",
            *gpu_mem_args,
            "--max-model-len",
            _MAX_MODEL_LEN,
            # Benchmark flags
            "--benchmark-mode",
            bench_mode,
            "--benchmark-warmup-iterations",
            _BENCH_WARMUP_ITERATIONS,
            "--benchmark-output-path",
            str(bench_output_path),
            # Bound how long the worker waits internally for the
            # benchmark file before failing startup. The outer
            # ``timeout=`` below is a separate safety net.
            "--benchmark-timeout",
            "240",
        ]

        if is_prefill is True:
            command.extend(["--disaggregation-mode", "prefill"])
            command.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                ]
            )
            health_check_urls = [
                (f"http://localhost:{self.system_port}/health", self._is_ready),
            ]
        elif is_prefill is False:
            command.extend(["--disaggregation-mode", "decode"])
            command.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                ]
            )
            health_check_urls = [
                (f"http://localhost:{self.system_port}/health", self._is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ]
        else:
            health_check_urls = [
                (f"http://localhost:{self.system_port}/health", self._is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ]

        env["DYN_LOG"] = "info"
        # Match cancellation tests' config to avoid CI flake from
        # canary health checks during startup.
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)
        # Required so InstrumentedScheduler's FPM publisher doesn't fight
        # the other worker's publisher for tcp://*:20380.
        env["DYN_FORWARDPASS_METRIC_PORT"] = str(self.fpm_port)

        # Prefill worker publishes KV events on its own ZMQ port and uses
        # a distinct NIXL side-channel port. Same constants as
        # ``tests/fault_tolerance/cancellation/test_vllm.py``.
        if is_prefill is True:
            command.extend(
                [
                    "--kv-events-config",
                    json.dumps(
                        {
                            "publisher": "zmq",
                            "topic": "kv-events",
                            "endpoint": "tcp://*:20082",
                            "enable_kv_cache_events": True,
                        }
                    ),
                ]
            )
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5601"

        if is_prefill is True:
            worker_type = "prefill_worker"
        elif is_prefill is False:
            worker_type = "decode_worker"
        else:
            worker_type = "worker"
        log_dir = f"{request.node.name}_{worker_type}"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            # Generous: cold model load + benchmark sweep + frontend
            # registration all need to complete within this window.
            timeout=420,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            deallocate_port(self.system_port)
        except Exception as e:
            logger.warning(f"Failed to release worker system port: {e}")
        try:
            deallocate_port(self.fpm_port)
        except Exception as e:
            logger.warning(f"Failed to release worker FPM port: {e}")
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _is_ready(self, response) -> bool:
        try:
            data = response.json()
            if data.get("status") == "ready":
                kind = (
                    "Prefill"
                    if self.is_prefill is True
                    else "Decode"
                    if self.is_prefill is False
                    else "Aggregated"
                )
                logger.info(
                    f"{kind} worker ready (bench_mode={self.bench_mode}, "
                    f"system_port={self.system_port})"
                )
                return True
            logger.warning(f"Worker status not ready yet: {data.get('status')!r}")
        except ValueError:
            logger.warning("Worker /health response was not valid JSON")
        return False


def _send_chat_completion(frontend_port: int) -> str:
    """Send a real chat completion via the frontend and return the
    assistant's content. Asserts the response is non-empty so a worker
    that registered but can't actually generate fails the test."""
    url = f"http://localhost:{frontend_port}/v1/chat/completions"
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    response = send_request(url=url, payload=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    assert len(choices) > 0, f"chat completion returned no choices: {body}"
    content = (choices[0].get("message") or {}).get("content") or ""
    assert len(content.strip()) > 0, f"chat completion returned empty content: {body}"
    logger.info(f"chat completion ok: content={content!r}")
    return content


@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(3.8)
@pytest.mark.requested_vllm_kv_cache_bytes(1_119_388_000)
@pytest.mark.timeout(600)
def test_self_benchmark_agg_serves_after_bench(
    request, runtime_services_dynamic_ports, predownload_models, tmp_path
):
    """Aggregated worker runs ``--benchmark-mode agg`` during startup,
    then serves a normal chat completion.

    Catches the prompt-padding regression on the decode sweep within
    agg mode (``_bench_inject_fake_decode`` is also called from the
    agg path).
    """
    bench_output = tmp_path / "bench_agg.json"

    with DynamoFrontendProcess(request, frontend_port=0) as frontend:
        logger.info(f"Frontend up on port {frontend.http_port}")

        with _DynamoBenchmarkWorker(
            request,
            frontend_port=frontend.http_port,
            bench_output_path=bench_output,
            bench_mode="agg",
            is_prefill=None,
        ):
            # Health checks already passed by the time we get here:
            # worker startup includes the benchmark sweep, then frontend
            # registration. So the engine survived the sweep.
            data = _validate_benchmark_results(bench_output, "agg")

            # Sanity: agg sweep produces both prefill and decode points,
            # and the decode sweep includes batch>1 (the regression case).
            point_types = {r["point"]["point_type"] for r in data["results"]}
            assert (
                "prefill" in point_types and "decode" in point_types
            ), f"agg benchmark missing point types: got {point_types}"
            prefill_kv_reads = sorted(
                r["point"].get("total_kv_read_tokens", 0)
                for r in data["results"]
                if r["point"]["point_type"] == "prefill"
            )
            assert any(kv_reads > 0 for kv_reads in prefill_kv_reads), (
                "agg prefill sweep did not exercise a prefix-cache hit: "
                f"kv_read_tokens={prefill_kv_reads}"
            )
            prefill_hit_batches = sorted(
                r["point"]["batch_size"]
                for r in data["results"]
                if r["point"]["point_type"] == "prefill"
                and r["point"].get("total_kv_read_tokens", 0) > 0
            )
            assert any(b > 1 for b in prefill_hit_batches), (
                "agg prefill sweep did not exercise batch>1 after KV reads: "
                f"batch_sizes={prefill_hit_batches}"
            )
            decode_batches = sorted(
                r["point"]["batch_size"]
                for r in data["results"]
                if r["point"]["point_type"] == "decode"
            )
            assert any(b > 1 for b in decode_batches), (
                f"agg decode sweep only ran batch=1, prompt-padding "
                f"regression would slip through. batch_sizes={decode_batches}"
            )
            assert {8, 9}.issubset(decode_batches)
            assert max(decode_batches) == data["limits"]["feasible_max_batch_size"]
            decode_boundary = {
                r["point"]["batch_size"]: r["point"]
                for r in data["results"]
                if r["point"]["point_type"] == "decode"
                and r["point"]["batch_size"] in {8, 9}
            }
            assert decode_boundary[8]["expected_cudagraph_mode"] == "FULL"
            assert decode_boundary[8]["expected_capture_size"] == 8
            assert decode_boundary[9]["expected_cudagraph_mode"] == "NONE"

            prefill_tokens = {
                r["point"]["total_prefill_tokens"]
                for r in data["results"]
                if r["point"]["point_type"] == "prefill"
            }
            assert {8, 9}.issubset(prefill_tokens)
            assert any(tokens > 8 for tokens in prefill_tokens)
            prefill_boundary = {
                r["point"]["total_prefill_tokens"]: r["point"]
                for r in data["results"]
                if r["point"]["point_type"] == "prefill"
                and r["point"]["total_prefill_tokens"] in {8, 9}
            }
            assert prefill_boundary[8]["expected_cudagraph_mode"] == "PIECEWISE"
            assert prefill_boundary[8]["expected_capture_size"] == 8
            assert prefill_boundary[9]["expected_cudagraph_mode"] == "NONE"

            # Now exercise normal serving end-to-end.
            _send_chat_completion(frontend.http_port)


@pytest.mark.gpu_2
@pytest.mark.timeout(900)
def test_self_benchmark_disagg_serves_after_bench(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    tmp_path,
):
    """Disagg prefill + decode workers each run their own
    ``--benchmark-mode``, then together serve a normal chat completion.

    This is the user-reported configuration. Exercises:

    * connector-metadata fix on the decode worker (NixlConnector
      kv_both -- worker would die on the first synthetic decode batch
      if the metadata isn't attached);
    * prompt-padding fix on both workers' decode sweeps (the prefill
      worker also runs decode warmup steps via ``_bench_step_warmup``).
    """
    bench_output_p = tmp_path / "bench_prefill.json"
    bench_output_d = tmp_path / "bench_decode.json"

    with DynamoFrontendProcess(request, frontend_port=0) as frontend:
        logger.info(f"Frontend up on port {frontend.http_port}")

        with _DynamoBenchmarkWorker(
            request,
            frontend_port=frontend.http_port,
            bench_output_path=bench_output_p,
            bench_mode="prefill",
            is_prefill=True,
        ) as prefill:
            logger.info(f"Prefill worker ready (PID {prefill.proc.pid})")

            with _DynamoBenchmarkWorker(
                request,
                frontend_port=frontend.http_port,
                bench_output_path=bench_output_d,
                bench_mode="decode",
                is_prefill=False,
            ) as decode:
                logger.info(f"Decode worker ready (PID {decode.proc.pid})")

                # Both workers passed serving health checks -- both
                # survived their benchmark sweeps and registered.
                p_data = _validate_benchmark_results(bench_output_p, "prefill")
                d_data = _validate_benchmark_results(bench_output_d, "decode")

                # Decode sweep on the decode worker must include batch>1.
                decode_batches = sorted(
                    r["point"]["batch_size"] for r in d_data["results"]
                )
                assert any(b > 1 for b in decode_batches), (
                    f"disagg decode sweep only ran batch=1, prompt-padding "
                    f"regression would slip through. "
                    f"batch_sizes={decode_batches}"
                )
                # Prefill sweep on the prefill worker must produce
                # prefill points.
                p_types = {r["point"]["point_type"] for r in p_data["results"]}
                assert (
                    "prefill" in p_types
                ), f"prefill benchmark missing prefill points: {p_types}"
                prefill_kv_reads = sorted(
                    r["point"].get("total_kv_read_tokens", 0) for r in p_data["results"]
                )
                assert any(kv_reads > 0 for kv_reads in prefill_kv_reads), (
                    "disaggregated prefill sweep did not exercise a prefix-cache "
                    f"hit: kv_read_tokens={prefill_kv_reads}"
                )
                prefill_hit_batches = sorted(
                    r["point"]["batch_size"]
                    for r in p_data["results"]
                    if r["point"].get("total_kv_read_tokens", 0) > 0
                )
                assert any(b > 1 for b in prefill_hit_batches), (
                    "disaggregated prefill sweep did not exercise batch>1 after "
                    f"KV reads: batch_sizes={prefill_hit_batches}"
                )

                # End-to-end: a real chat completion that traverses
                # both workers (prefill on one, decode on the other).
                _send_chat_completion(frontend.http_port)
