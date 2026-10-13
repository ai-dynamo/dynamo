# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-model native EAGLE3 preparation, verify-shape and cleanup oracle."""

import json
import os
import sys

import pytest
import requests

from dynamo.vllm.benchmark_points import BenchmarkPoints
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
    # NVML peak on RTX 6000 Ada, eager/full graphs and acceptance endpoints.
    pytest.mark.profiled_vram_gib(3.5),
]


@pytest.mark.parametrize("acceptance_length", [1.0, 2.78, 4.0])
@pytest.mark.parametrize("graph_mode", ["NONE", "FULL_DECODE_ONLY"])
def test_eagle3_self_benchmark_native_verification(
    tmp_path, request, predownload_models, acceptance_length, graph_mode
):
    ports = []
    request.addfinalizer(lambda: deallocate_ports(ports))
    api_port = allocate_port(DynamoPortRange.SERVE.value)
    ports.append(api_port)
    fpm_port = allocate_port(DynamoPortRange.FPM.value)
    ports.append(fpm_port)
    output = tmp_path / "benchmark.json"
    benchmark = {
        "mode": "decode",
        "warmup_iterations": 2,
        "timeout": 180,
        "output_path": str(output),
        "points": {
            "schema_version": 1,
            "prefill": [],
            "decode": [
                {"batch_size": 1, "total_kv_read_tokens": 32},
                {"batch_size": 2, "total_kv_read_tokens": 256},
                {"batch_size": 4, "total_kv_read_tokens": 128},
            ],
        },
    }
    BenchmarkPoints.model_validate(benchmark["points"])
    env = dict(os.environ)
    env.update(
        DYN_FORWARDPASS_METRIC_PORT=str(fpm_port),
        VLLM_PLUGINS="",
        VLLM_USE_V2_MODEL_RUNNER="0",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
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
        "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
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
        assert (
            data["measurement_policy"]["decode"]
            == "first_native_verification_after_prefill"
        )
        assert len(data["results"]) == 3
        spec = data["speculative_benchmark"]
        assert spec["verify_tokens_per_request"] == 4
        assert spec["kv_capacity"]["usable_blocks"] > 0
        assert spec["kv_capacity"]["context_capacity_tokens"] > 0
        assert spec["kv_capacity"]["allocated_kv_tensor_bytes"] > 0
        assert all(group["specs_by_layer"] for group in spec["kv_capacity"]["groups"])
        assert (
            spec["free_blocks_after_benchmark"]
            == spec["kv_capacity"]["free_blocks_before_benchmark"]
        )
        for row in data["results"]:
            assert row["kv_seed_regime"] == "real_kv"
            point = row["point"]
            fpm = row["fpms"][0]
            assert (
                fpm["scheduled_requests"]["num_decode_requests"] == point["batch_size"]
            )
            assert (
                fpm["scheduled_requests"]["sum_decode_kv_tokens"]
                == point["total_kv_read_tokens"]
            )
            measurement = fpm["benchmark_measurement"]
            sample = measurement["raw_fpms"][0]["benchmark_sample"]
            assert sample["cudagraph"]["status"] == "observed"
            assert sample["cudagraph"]["num_unpadded_tokens"] == 4 * point["batch_size"]
            assert sample["cudagraph"]["runtime_mode"] == (
                "NONE" if graph_mode == "NONE" else "FULL"
            )
            assert set(sample["speculation"]["scheduled_tokens"].values()) == {4}
            assert set(sample["speculation"]["scheduled_draft_tokens"].values()) == {3}
            assert (
                len(sample["speculation"]["output_tokens_per_request"])
                == point["batch_size"]
            )
            actual_outputs = sample["speculation"]["output_tokens_per_request"]
            expected_outputs = (
                {1}
                if acceptance_length == 1.0
                else {4}
                if acceptance_length == 4.0
                else {2, 3}
            )
            assert set(actual_outputs) <= expected_outputs
            assert fpm["wall_time"] > 0
        merged = _merge_benchmark_rank_results(
            [(0, output, data)], tmp_path / "merged.json"
        )
        assert merged["schema_version"] == 3
        assert merged["speculative_benchmark"] == spec
        assert merged["measurement_policy"] == data["measurement_policy"]
        response = requests.post(
            f"http://127.0.0.1:{api_port}/v1/completions",
            json={"model": "Qwen/Qwen3-0.6B", "prompt": "The sky is", "max_tokens": 16},
            timeout=30,
        )
        response.raise_for_status()
        assert response.json()["usage"]["completion_tokens"] > 0
