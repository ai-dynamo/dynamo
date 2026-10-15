# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-model native async EAGLE3 shape, CUDA timing and cleanup oracle."""

import json
import os
import sys

import msgspec
import pytest
import requests

from dynamo.vllm.benchmark_points import BenchmarkPoints
from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler, _NativeFPMEmitter
from dynamo.vllm.worker_factory import _merge_benchmark_rank_results
from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_ports

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.model("Nicolassuez/Qwen3-0.6B-eagle3"),
    pytest.mark.timeout(600),
    pytest.mark.requested_vllm_kv_cache_bytes(1073741824),
    # NVML peak 3625 MiB on RTX 6000 Ada; round up the scheduling bound.
    pytest.mark.profiled_vram_gib(3.6),
]


class NativeObserverCheckingScheduler(InstrumentedScheduler):
    """Observe the real producer and cleanup without replacing execution/timing."""

    def _bench_install_native_fpm_observer(self):
        self._original_emitter_factory = _NativeFPMEmitter.__dict__["from_vllm_config"]
        super()._bench_install_native_fpm_observer()

    def _bench_attach_native_fpm(self, emitter):
        self._observed_emitter = emitter
        self._original_native_begin = emitter.begin_iteration
        self._original_native_emit = emitter._emit_if_ready
        super()._bench_attach_native_fpm(emitter)

    def _bench_receive_native_fpm(self, iteration_id, pending):
        before = len(self._bench_current_fpms)
        super()._bench_receive_native_fpm(iteration_id, pending)
        if len(self._bench_current_fpms) > before:
            actual = self._bench_current_fpms[-1]
            assert actual["benchmark_sample"]["native_iteration_id"] == iteration_id
            assert actual["wall_time"] == pending.duration_seconds
            assert actual["scheduled_requests"] == msgspec.to_builtins(
                pending.scheduled
            )
            assert actual["queued_requests"] == msgspec.to_builtins(pending.queued)

    def add_request(self, request):
        if self._bench_active and request.request_id.startswith("__bench_"):
            assert request.sampling_params.temperature == 0.7
            assert request.sampling_params.top_p == 0.9
        return super().add_request(request)

    def _assert_native_observer_restored(self):
        assert self._bench_sd_restore_native is None
        assert self._bench_sd_restore_factory is None
        assert (
            _NativeFPMEmitter.__dict__["from_vllm_config"]
            is self._original_emitter_factory
        )
        assert self._observed_emitter.begin_iteration == self._original_native_begin
        assert self._observed_emitter._emit_if_ready == self._original_native_emit

    def _bench_deactivate(self, *, resume_publisher=True):
        super()._bench_deactivate(resume_publisher=resume_publisher)
        if hasattr(self, "_observed_emitter"):
            self._assert_native_observer_restored()

    def schedule(self, throttle_prefills=False):
        if not self._bench_active and hasattr(self, "_observed_emitter"):
            self._assert_native_observer_restored()
        return super().schedule(throttle_prefills)


@pytest.mark.parametrize(
    "mode,graph_mode,acceptance_length",
    [
        ("decode", "NONE", 1.0),
        ("decode", "NONE", 2.78),
        ("decode", "NONE", 4.0),
        ("decode", "FULL_DECODE_ONLY", 1.0),
        ("decode", "FULL_DECODE_ONLY", 2.78),
        ("decode", "FULL_DECODE_ONLY", 4.0),
        ("prefill", "NONE", 2.78),
        ("agg", "NONE", 2.78),
    ],
)
def test_eagle3_self_benchmark_native_verification(
    tmp_path, request, predownload_models, mode, acceptance_length, graph_mode
):
    ports = []
    request.addfinalizer(lambda: deallocate_ports(ports))
    api_port = allocate_port(DynamoPortRange.SERVE.value)
    ports.append(api_port)
    fpm_port = allocate_port(DynamoPortRange.FPM.value)
    ports.append(fpm_port)
    native_fpm_port = allocate_port(DynamoPortRange.FPM.value)
    ports.append(native_fpm_port)
    output = tmp_path / "benchmark.json"
    benchmark = {
        "mode": mode,
        "warmup_iterations": 2,
        "timeout": 180,
        "output_path": str(output),
        "points": {
            "schema_version": 1,
            # Qwen's cache blocks are 16 tokens; each cached prefill has
            # more than one block of new tokens for EAGLE's prefix lookup.
            "prefill": [
                {
                    "batch_size": 1,
                    "total_prefill_tokens": 32,
                    "total_kv_read_tokens": 0,
                },
                {
                    "batch_size": 1,
                    "total_prefill_tokens": 32,
                    "total_kv_read_tokens": 64,
                },
                {
                    "batch_size": 2,
                    "total_prefill_tokens": 64,
                    "total_kv_read_tokens": 128,
                },
            ]
            if mode != "decode"
            else [],
            "decode": [
                {"batch_size": 1, "total_kv_read_tokens": 32},
                {"batch_size": 2, "total_kv_read_tokens": 256},
                {"batch_size": 4, "total_kv_read_tokens": 128},
            ]
            if mode != "prefill"
            else [],
        },
    }
    BenchmarkPoints.model_validate(benchmark["points"])
    env = dict(os.environ)
    env.update(
        DYN_FORWARDPASS_METRIC_PORT=str(fpm_port),
        VLLM_PLUGINS="",
        VLLM_USE_V2_MODEL_RUNNER="0",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        DYN_BENCH_PREFILL_REAL_SEED="1",
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "Qwen/Qwen3-0.6B",
        "--revision",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "--host",
        "127.0.0.1",
        "--port",
        str(api_port),
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--kv-cache-memory-bytes",
        "1073741824",
        "--scheduler-cls",
        "tests.vllm_self_benchmark.test_vllm_speculative_self_benchmark_gpu.NativeObserverCheckingScheduler",
        "--forward-pass-metrics-port",
        str(native_fpm_port),
        "--async-scheduling",
        "--override-generation-config",
        json.dumps({"temperature": 0.7, "top_p": 0.9}),
        "--cudagraph-metrics",
        "--additional-config",
        json.dumps({"benchmark": benchmark}),
        "--speculative-config",
        json.dumps(
            {
                "method": "eagle3",
                "model": "Nicolassuez/Qwen3-0.6B-eagle3",
                "revision": "00b7c41f32655e90ee65ac5b90bb381905ce1700",
                "num_speculative_tokens": 3,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": acceptance_length,
            }
        ),
    ]
    if graph_mode == "NONE":
        command += ["--enforce-eager"]
    else:
        command += [
            "--compilation-config",
            json.dumps({"mode": 0, "cudagraph_mode": graph_mode}),
            "--max-cudagraph-capture-size",
            "16",
        ]
    with ManagedProcess(
        command=command,
        env=env,
        log_dir=str(tmp_path),
        display_name="eagle3-selfbench",
        health_check_urls=[f"http://127.0.0.1:{api_port}/health"],
        health_check_funcs=[lambda remaining: output.exists()],
        timeout=480,
        terminate_all_matching_process_names=False,
    ):
        data = json.loads(output.read_text())
        assert data["valid"], data.get("error")
        assert data["schema_version"] == 3
        assert data["measurement_protocol"]["timing_metric"] == "model_step_cuda"
        assert (
            data["measurement_protocol"]["execution_evidence"]["timing_clock"]
            == "cuda_event_elapsed"
        )
        assert data["measurement_policy"]["decode"] == "continuous_native_async_decode"
        assert len(data["results"]) == (6 if mode == "agg" else 3)
        shape_warmups = [
            record
            for record in data["warmup_evidence"]["records"]
            if record["kind"] == "speculative_shape"
            and record["requested_shape"].get("point_type") == "decode"
        ]
        assert len(shape_warmups) == (0 if mode == "prefill" else 3)
        assert all(record["status"] == "completed" for record in shape_warmups)
        assert all(
            record["validation"]["status"] == "passed" for record in shape_warmups
        )
        if mode != "decode":
            for point in benchmark["points"]["prefill"]:
                expected = {
                    "num_prefill_requests": point["batch_size"],
                    "sum_prefill_tokens": point["total_prefill_tokens"],
                    "sum_prefill_kv_tokens": point["total_kv_read_tokens"],
                    "num_decode_requests": 0,
                }
                assert any(
                    record["status"] == "completed"
                    and record["kind"]
                    in {"eager_shape", "speculative_shape", "real_prefix_shape"}
                    and record["observed_forward_count"] == 1
                    and all(
                        record["first_scheduled_requests"].get(key) == value
                        and record["last_scheduled_requests"].get(key) == value
                        for key, value in expected.items()
                    )
                    for record in data["warmup_evidence"]["records"]
                ), point
        spec = data["speculative_benchmark"]
        assert spec["verify_tokens_per_request"] == 4
        assert spec["timing_scope"] == "model_step_cuda"
        assert spec["window_warmup_iterations"] == 5
        assert spec["window_measured_iterations"] == 20
        assert spec["window_tail_iterations"] == 2
        assert spec["max_observed_steps_in_flight"] >= 1
        if mode != "prefill":
            assert spec["max_observed_steps_in_flight"] > 1
        assert spec["sampling"]["temperature"] == 0.7
        assert spec["sampling"]["top_p"] == 0.9
        identity = spec["execution_identity"]
        assert identity["status"] == "resolved", identity
        assert identity["unresolved_fields"] == []
        assert identity["binding"] == {
            "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "execution_profile": {
                "kind": "speculative_round",
                "method": "eagle3",
                "num_draft_tokens": 3,
                "draft_model": "Nicolassuez/Qwen3-0.6B-eagle3",
                "draft_revision": "00b7c41f32655e90ee65ac5b90bb381905ce1700",
                "timing_scope": "model_step_cuda",
            },
        }
        assert spec["kv_capacity"]["usable_blocks"] > 0
        assert spec["kv_capacity"]["context_capacity_tokens"] > 0
        assert spec["kv_capacity"]["allocated_kv_tensor_bytes"] > 0
        assert (
            sum(
                tensor["size_bytes"]
                for tensor in spec["kv_capacity"]["tensor_allocations"]
            )
            == spec["kv_capacity"]["allocated_kv_tensor_bytes"]
        )
        assert (
            spec["free_blocks_after_benchmark"]
            == spec["kv_capacity"]["free_blocks_before_benchmark"]
        )
        native_ids_seen = set()
        for row in data["results"]:
            point = row["point"]
            fpm = row["fpms"][0]
            measurement = fpm["benchmark_measurement"]
            raw_fpms = measurement["raw_fpms"]
            for raw in raw_fpms:
                sample = raw["benchmark_sample"]
                native_id = sample["native_iteration_id"]
                assert type(native_id) is int and native_id >= 0
                assert native_id not in native_ids_seen
                native_ids_seen.add(native_id)
                assert raw["timing_scope"] == "model_step_cuda"
                assert sample["timing"] == {
                    "basis": "model_step_cuda",
                    "start_monotonic": None,
                    "end_monotonic": None,
                    "duration_seconds": raw["wall_time"],
                }
                assert raw["wall_time"] > 0
            if point["point_type"] == "prefill":
                assert len(raw_fpms) == measurement["expected_internal_samples"] == 1
                assert measurement["estimate"]["method"] == "single_step"
                assert measurement["estimate"]["raw_sample_indices"] == [0]
                raw = raw_fpms[0]
                for key in (
                    "wall_time",
                    "scheduled_requests",
                    "queued_requests",
                    "timing_scope",
                ):
                    assert fpm[key] == raw[key]
                shape = raw["scheduled_requests"]
                assert shape["num_prefill_requests"] == point["batch_size"]
                assert shape["sum_prefill_tokens"] == point["total_prefill_tokens"]
                assert shape["sum_prefill_kv_tokens"] == point["total_kv_read_tokens"]
                assert (
                    shape["num_decode_requests"] == shape["sum_decode_kv_tokens"] == 0
                )
                graph = raw["benchmark_sample"]["cudagraph"]
                assert graph["status"] == "observed"
                assert graph["num_unpadded_tokens"] == point["total_prefill_tokens"]
                assert graph["runtime_mode"] == "NONE"
                if point["total_kv_read_tokens"]:
                    assert row["kv_seed_regime"] == "real_prefix"
                continue
            assert row["kv_seed_regime"] == "real_kv"
            assert (
                fpm["scheduled_requests"]["num_decode_requests"] == point["batch_size"]
            )
            assert (
                fpm["scheduled_requests"]["sum_decode_kv_tokens"]
                == point["total_kv_read_tokens"]
            )
            assert len(raw_fpms) == 20
            samples = [raw["benchmark_sample"] for raw in raw_fpms]
            assert [sample["window_index"] for sample in samples] == list(range(5, 25))
            assert all(sample["window_phase"] == "measured" for sample in samples)
            native_ids = [sample["native_iteration_id"] for sample in samples]
            assert native_ids == list(range(native_ids[0], native_ids[0] + 20))
            initial_point = measurement["initial_point"]
            batch_size = point["batch_size"]
            kv_lengths = [
                raw["scheduled_requests"]["sum_decode_kv_tokens"] for raw in raw_fpms
            ]
            expected_outputs = (
                {1}
                if acceptance_length == 1.0
                else {4}
                if acceptance_length == 4.0
                else {2, 3}
            )
            assert (
                initial_point["total_kv_read_tokens"]
                + 5 * batch_size * min(expected_outputs)
                <= kv_lengths[0]
                <= initial_point["total_kv_read_tokens"]
                + 5 * batch_size * max(expected_outputs)
            )
            for index, (raw, sample) in enumerate(zip(raw_fpms, samples, strict=True)):
                assert raw["scheduled_requests"]["num_decode_requests"] == batch_size
                assert sample["cudagraph"]["status"] == "observed"
                assert sample["cudagraph"]["num_unpadded_tokens"] == 4 * batch_size
                assert sample["cudagraph"]["runtime_mode"] == (
                    "NONE" if graph_mode == "NONE" else "FULL"
                )
                assert set(sample["speculation"]["scheduled_tokens"].values()) == {4}
                assert set(
                    sample["speculation"]["scheduled_draft_tokens"].values()
                ) == {3}
                actual_outputs = sample["speculation"]["output_tokens_per_request"]
                assert len(actual_outputs) == batch_size
                assert set(actual_outputs) <= expected_outputs
                timing = sample["timing"]
                assert timing["basis"] == "model_step_cuda"
                assert timing["duration_seconds"] == raw["wall_time"] > 0
                if index + 1 < len(raw_fpms):
                    # Corrected next-iteration context grows by accepted outputs,
                    # not optimistically by the four scheduled query tokens.
                    assert kv_lengths[index + 1] - kv_lengths[index] == sum(
                        actual_outputs
                    )
            assert fpm["wall_time"] > 0
        merged = _merge_benchmark_rank_results(
            [(0, output, data)], tmp_path / "merged.json"
        )
        assert merged["schema_version"] == 3
        assert merged["speculative_benchmark"] == spec
        assert merged["measurement_policy"] == data["measurement_policy"]
        artifact_before_serving = output.read_bytes()
        response = requests.post(
            f"http://127.0.0.1:{api_port}/v1/completions",
            json={"model": "Qwen/Qwen3-0.6B", "prompt": "The sky is", "max_tokens": 16},
            timeout=30,
        )
        response.raise_for_status()
        assert response.json()["usage"]["completion_tokens"] > 0
        assert output.read_bytes() == artifact_before_serving
