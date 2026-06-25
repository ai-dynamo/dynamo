# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.self_benchmark import (
    TrtllmSelfBenchmarkConfig,
    apply_self_benchmark_engine_args,
    benchmark_options_without_mode,
    normalize_self_benchmark_payload,
    trtllm_self_benchmark_cli_args,
    wait_for_self_benchmark_output,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
]


def _payload(*, timed_out: bool = False, include_skipped: bool = False) -> dict:
    results = [
        {
            "point": {
                "point_type": "prefill",
                "index": 5,
                "isl": 1024,
                "kv_read_tokens": 512,
                "context_length": 0,
                "batch_size": 1,
            },
            "iteration_stats": [
                {
                    "hostStepTimeMS": 12.3,
                    "prevDeviceStepTimeMS": 10.8,
                    "schedulerMode": "overlap",
                    "inflightBatchingStats": {
                        "numContextRequests": 1,
                        "numCtxTokens": 512,
                        "numCtxKvTokens": 512,
                        "numGenRequests": 0,
                        "numGenKvTokens": 0,
                    },
                }
            ],
            "skipped_reason": None,
            "observed_kv_read_tokens": 512,
            "cache_hit_validated": True,
        },
        {
            "point": {
                "point_type": "decode",
                "index": 6,
                "isl": 0,
                "context_length": 2048,
                "batch_size": 4,
            },
            "iteration_stats": [
                {
                    "hostStepTimeMS": 8.0,
                    "schedulerMode": "overlap",
                    "inflightBatchingStats": {
                        "numContextRequests": 0,
                        "numCtxTokens": 0,
                        "numCtxKvTokens": 0,
                        "numGenRequests": 4,
                        "numGenKvTokens": 8192,
                    },
                }
            ],
            "skipped_reason": None,
        },
    ]
    if include_skipped:
        results.append(
            {
                "point": {
                    "point_type": "decode",
                    "index": 7,
                    "isl": 0,
                    "context_length": 4096,
                    "batch_size": 8,
                },
                "iteration_stats": [],
                "skipped_reason": "kv cache exhausted",
            }
        )
    return {
        "config": {"mode": "agg", "output_path": "/tmp/trt.json"},
        "limits": {"max_num_scheduled_tokens": 8192},
        "timed_out": timed_out,
        "results": results,
    }


def test_cli_args_and_engine_args_use_trtllm_native_surface():
    cfg = TrtllmSelfBenchmarkConfig(
        mode="agg",
        prefill_isl_granularity=2,
        prefill_kv_read_granularity=5,
        decode_context_granularity=3,
        decode_batch_granularity=4,
        warmup_iterations=1,
        output_path="/tmp/out.json",
        timeout_s=9,
    )

    assert trtllm_self_benchmark_cli_args(cfg) == [
        "--self_benchmark_mode",
        "agg",
        "--self_benchmark_prefill_granularity",
        "2",
        "--self_benchmark_prefill_kv_read_granularity",
        "5",
        "--self_benchmark_decode_length_granularity",
        "3",
        "--self_benchmark_decode_batch_granularity",
        "4",
        "--self_benchmark_warmup_iterations",
        "1",
        "--self_benchmark_output_path",
        "/tmp/out.json",
        "--self_benchmark_timeout",
        "9",
    ]

    engine_args = {}
    apply_self_benchmark_engine_args(engine_args, cfg)
    assert engine_args == {
        "self_benchmark_mode": "agg",
        "self_benchmark_prefill_granularity": 2,
        "self_benchmark_prefill_kv_read_granularity": 5,
        "self_benchmark_decode_length_granularity": 3,
        "self_benchmark_decode_batch_granularity": 4,
        "self_benchmark_warmup_iterations": 1,
        "self_benchmark_output_path": "/tmp/out.json",
        "self_benchmark_timeout": 9,
    }


@pytest.mark.asyncio
async def test_wait_for_output_parses_and_normalizes_fpms(tmp_path):
    output = tmp_path / "trt.json"
    output.write_text(json.dumps(_payload()))
    cfg = TrtllmSelfBenchmarkConfig(mode="agg", output_path=str(output), timeout_s=1)

    data = await wait_for_self_benchmark_output(cfg, worker_id="worker-1")

    assert data["status"] == "ok"
    assert data["backend"] == "trtllm"
    assert "1024,512" in data["profile"]["prefill"]
    assert "2048,4" in data["profile"]["decode"]
    assert data["results"][0]["normalized_key"] == [1024, 512]
    assert data["profile"]["prefill"]["1024,512"]["cache_hit_validated"] is True
    assert (
        data["profile"]["prefill"]["1024,512"]["observed_kv_read_tokens"] == 512
    )
    prefill_fpm = data["results"][0]["fpms"][0]
    assert prefill_fpm["worker_id"] == "worker-1"
    assert prefill_fpm["wall_time"] == pytest.approx(0.0123)
    assert prefill_fpm["scheduled_requests"]["sum_prefill_tokens"] == 512
    assert prefill_fpm["scheduled_requests"]["sum_prefill_kv_tokens"] == 512
    assert data["metadata"]["raw_trtllm_self_benchmark"]["limits"][
        "max_num_scheduled_tokens"
    ] == 8192


def test_prefill_kv_read_fallback_uses_point_schema():
    data = normalize_self_benchmark_payload(
        {
            "timed_out": False,
            "results": [
                {
                    "point": {
                        "point_type": "prefill",
                        "index": 1,
                        "isl": 1024,
                        "kv_read_tokens": 256,
                        "batch_size": 2,
                    },
                    "iteration_stats": [{"hostStepTimeMS": 1.0}],
                    "skipped_reason": None,
                }
            ],
        }
    )

    fpm = data["results"][0]["fpms"][0]
    assert data["results"][0]["normalized_key"] == [1024, 256]
    assert fpm["scheduled_requests"]["num_prefill_requests"] == 2
    assert fpm["scheduled_requests"]["sum_prefill_tokens"] == 1536
    assert fpm["scheduled_requests"]["sum_prefill_kv_tokens"] == 512


@pytest.mark.asyncio
async def test_wait_for_output_missing_file_times_out(tmp_path):
    cfg = TrtllmSelfBenchmarkConfig(
        mode="agg",
        output_path=str(tmp_path / "missing.json"),
        timeout_s=0,
    )

    with pytest.raises(TimeoutError, match="did not write"):
        await wait_for_self_benchmark_output(cfg)


@pytest.mark.asyncio
async def test_invalid_json_fails_startup(tmp_path):
    output = tmp_path / "bad.json"
    output.write_text("{not-json")
    cfg = TrtllmSelfBenchmarkConfig(mode="agg", output_path=str(output), timeout_s=1)

    with pytest.raises(ValueError, match="Invalid TRT-LLM self-benchmark JSON"):
        await wait_for_self_benchmark_output(cfg)


def test_timed_out_payload_warns_and_publishes_partial_data():
    data = normalize_self_benchmark_payload(_payload(timed_out=True))

    assert data["metadata"]["timed_out"] is True
    assert data["metadata"]["warnings"] == [
        "TRT-LLM self-benchmark timed out; publishing partial data"
    ]
    assert data["profile"]["prefill"]


def test_skipped_points_are_kept_but_excluded_from_profile_tables():
    data = normalize_self_benchmark_payload(_payload(include_skipped=True))

    skipped_result = data["results"][-1]
    assert skipped_result["skipped_reason"] == "kv cache exhausted"
    assert skipped_result["fpms"] == []
    assert "4096,8" not in data["profile"]["decode"]
    assert data["metadata"]["skipped_points"][-1]["normalized_key"] == [4096, 8]


def test_benchmark_options_require_mode():
    config = SimpleNamespace(
        benchmark_mode=None,
        benchmark_prefill_isl_granularity=None,
        benchmark_prefill_kv_read_granularity=None,
        benchmark_decode_context_granularity=None,
        benchmark_decode_batch_granularity=None,
        benchmark_warmup_iterations=None,
        benchmark_output_path="/tmp/out.json",
        benchmark_timeout_s=None,
    )

    assert benchmark_options_without_mode(config) == ["--benchmark-output-path"]


def test_native_trt_self_benchmark_aliases_parse():
    try:
        from dynamo.trtllm.backend_args import DynamoTrtllmArgGroup
    except ImportError as exc:
        pytest.skip(f"TRT-LLM package is unavailable: {exc}")

    parser = argparse.ArgumentParser()
    DynamoTrtllmArgGroup().add_arguments(parser)
    config = parser.parse_args(
        [
            "--self_benchmark_mode",
            "agg",
            "--self_benchmark_prefill_granularity",
            "2",
            "--self_benchmark_prefill_kv_read_granularity",
            "5",
            "--self_benchmark_decode_length_granularity",
            "3",
            "--self_benchmark_decode_batch_granularity",
            "4",
            "--self_benchmark_warmup_iterations",
            "1",
            "--self_benchmark_output_path",
            "/tmp/trt.json",
            "--self_benchmark_timeout",
            "9",
        ]
    )

    assert config.benchmark_mode == "agg"
    assert config.benchmark_prefill_isl_granularity == 2
    assert config.benchmark_prefill_kv_read_granularity == 5
    assert config.benchmark_decode_context_granularity == 3
    assert config.benchmark_decode_batch_granularity == 4
    assert config.benchmark_warmup_iterations == 1
    assert config.benchmark_output_path == "/tmp/trt.json"
    assert config.benchmark_timeout_s == 9


@pytest.mark.asyncio
async def test_init_worker_waits_for_benchmark_before_registering(monkeypatch, tmp_path):
    try:
        from dynamo.trtllm.workers import llm_worker as worker_mod
    except ImportError as exc:
        pytest.skip(f"TRT-LLM worker runtime binding is unavailable: {exc}")

    events: list[str] = []
    output_path = tmp_path / "bench.json"

    class FakeEndpoint:
        def __init__(self, name: str):
            self.name = name

        def connection_id(self):
            return 42

        async def serve_endpoint(self, *args, **kwargs):
            events.append(f"serve:{self.name}")

    class FakeRuntime:
        def endpoint(self, name: str):
            return FakeEndpoint(name)

    class FakeEngine:
        def start_health_monitor(self, *args, **kwargs):
            events.append("health")

        def get_attention_dp_size(self):
            return 1

    @asynccontextmanager
    async def fake_get_llm_engine(*args, **kwargs):
        events.append("engine")
        yield FakeEngine()

    async def fake_wait_for_output(*args, **kwargs):
        events.append("benchmark")
        return {"status": "ok", "results": [], "metadata": {}}

    async def fake_register_model(*args, **kwargs):
        events.append("register")

    class FakeHandler:
        _benchmark_results = None

        async def generate(self, *args, **kwargs):
            yield {}

        async def get_perf_metrics(self, *args, **kwargs):
            yield self._benchmark_results

    class FakeFactory:
        def get_request_handler(self, config):
            return FakeHandler()

    monkeypatch.setattr(worker_mod, "get_llm_engine", fake_get_llm_engine)
    monkeypatch.setattr(
        worker_mod, "wait_for_self_benchmark_output", fake_wait_for_output
    )
    monkeypatch.setattr(worker_mod, "register_model", fake_register_model)
    monkeypatch.setattr(worker_mod, "RequestHandlerFactory", lambda: FakeFactory())
    monkeypatch.setattr(worker_mod, "tokenizer_factory", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_mod, "dump_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker_mod, "register_engine_metrics_callback", lambda *a, **k: None
    )
    monkeypatch.setattr(
        worker_mod,
        "LLMBackendMetrics",
        lambda *args, **kwargs: SimpleNamespace(set_model_load_time=lambda *a: None),
    )

    config = SimpleNamespace(
        allowed_local_media_path="",
        benchmark_decode_batch_granularity=None,
        benchmark_decode_context_granularity=None,
        benchmark_mode="agg",
        benchmark_output_path=str(output_path),
        benchmark_prefill_isl_granularity=None,
        benchmark_prefill_kv_read_granularity=None,
        benchmark_timeout_s=1,
        benchmark_warmup_iterations=None,
        component="backend",
        connector=["none"],
        custom_jinja_template=None,
        default_guidance_scale=5.0,
        default_height=1024,
        default_num_frames=81,
        default_num_images_per_prompt=1,
        default_num_inference_steps=50,
        default_width=1024,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        dit_cfg_size=1,
        dit_ring_size=1,
        dit_ulysses_size=1,
        dump_config_to=None,
        dyn_enable_structural_tag=False,
        dyn_reasoning_parser=None,
        dyn_structural_tag_schema="auto",
        dyn_structural_tag_scope="auto",
        dyn_tool_call_parser=None,
        enable_attention_dp=False,
        enable_cuda_graph=False,
        enable_fullgraph=False,
        enable_layerwise_nvtx_marker=False,
        enable_local_indexer=True,
        enable_teacache=False,
        encode_endpoint="",
        endpoint="generate",
        endpoint_types="chat,completions",
        exclude_tools_when_tool_choice_none=True,
        expert_parallel_size=None,
        extra_engine_args="",
        free_gpu_memory_fraction=0.9,
        frontend_decoding=False,
        gpus_per_node=1,
        guided_decoding_backend=None,
        kv_block_size=32,
        load_format="auto",
        max_batch_size=8,
        max_beam_width=1,
        max_file_size_mb=50,
        max_num_tokens=8192,
        max_seq_len=32768,
        modality=Modality.TEXT,
        model="Qwen/Qwen3-0.6B",
        model_loader_extra_config="",
        multimodal_embedding_cache_capacity_gb=0,
        namespace="dynamo",
        override_engine_args="",
        pipeline_parallel_size=1,
        publish_events_and_metrics=False,
        quant_algo=None,
        quant_dynamic=True,
        revision=None,
        served_model_name=None,
        skip_warmup=False,
        tensor_parallel_size=1,
        torch_dtype="bfloat16",
        disable_torch_compile=False,
        teacache_thresh=0.2,
        teacache_use_ret_steps=True,
    )
    config.has_connector = lambda name: name in config.connector

    await worker_mod.init_llm_worker(
        FakeRuntime(),
        config,
        shutdown_event=AsyncMock(),
        shutdown_endpoints=[],
    )

    assert events.index("benchmark") < events.index("register")
    assert events.index("register") < events.index("serve:dynamo.backend.generate")
    assert "serve:dynamo.backend.get_perf_metrics" in events
