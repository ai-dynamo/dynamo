#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket

from dynamo.llm import MockEngineArgs, ModelRuntimeConfig, ReasoningConfig, SglangArgs

_DEFAULT_NUM_GPU_BLOCKS = 16384
_DEFAULT_MAX_NUM_SEQS = 256
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


def _parse_reasoning_config(reasoning_json: str | None) -> ReasoningConfig | None:
    if not reasoning_json:
        return None

    reasoning = json.loads(reasoning_json)
    return ReasoningConfig(
        start_thinking_token_id=reasoning["start_thinking_token_id"],
        end_thinking_token_id=reasoning["end_thinking_token_id"],
        thinking_ratio=reasoning["thinking_ratio"],
    )


def _build_sglang_args(args: argparse.Namespace) -> SglangArgs | None:
    sglang_args = {
        "schedule_policy": getattr(args, "sglang_schedule_policy", None),
        "page_size": getattr(args, "sglang_page_size", None),
        "max_prefill_tokens": getattr(args, "sglang_max_prefill_tokens", None),
        "chunked_prefill_size": getattr(args, "sglang_chunked_prefill_size", None),
        "clip_max_new_tokens": getattr(args, "sglang_clip_max_new_tokens", None),
        "schedule_conservativeness": getattr(
            args, "sglang_schedule_conservativeness", None
        ),
    }
    if not any(value is not None for value in sglang_args.values()):
        return None
    return SglangArgs(**sglang_args)


def build_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    worker_type = (
        "prefill"
        if getattr(args, "is_prefill_worker", False)
        else "decode"
        if getattr(args, "is_decode_worker", False)
        else "aggregated"
    )
    aic_backend = None
    aic_system = None
    aic_backend_version = None
    aic_tp_size = None
    aic_model_path = None
    aic_moe_tp_size = None
    aic_moe_ep_size = None
    aic_attention_dp_size = None
    if getattr(args, "aic_perf_model", False):
        aic_backend = (
            getattr(args, "aic_backend", None)
            or getattr(args, "engine_type", None)
            or "vllm"
        )
        aic_system = getattr(args, "aic_system", None)
        aic_backend_version = getattr(args, "aic_backend_version", None)
        aic_tp_size = getattr(args, "aic_tp_size", None)
        aic_model_path = getattr(args, "model_path", None)
        aic_moe_tp_size = getattr(args, "aic_moe_tp_size", None)
        aic_moe_ep_size = getattr(args, "aic_moe_ep_size", None)
        aic_attention_dp_size = getattr(args, "aic_attention_dp_size", None)
    return MockEngineArgs(
        engine_type=getattr(args, "engine_type", None) or "vllm",
        num_gpu_blocks=getattr(args, "num_gpu_blocks", _DEFAULT_NUM_GPU_BLOCKS),
        block_size=getattr(args, "block_size", 0) or 0,
        max_num_seqs=getattr(args, "max_num_seqs", _DEFAULT_MAX_NUM_SEQS),
        max_num_batched_tokens=getattr(
            args, "max_num_batched_tokens", _DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
        enable_chunked_prefill=getattr(args, "enable_chunked_prefill", True),
        speedup_ratio=getattr(args, "speedup_ratio", 1.0),
        decode_speedup_ratio=getattr(args, "decode_speedup_ratio", 1.0),
        dp_size=getattr(args, "dp_size", 1),
        startup_time=getattr(args, "startup_time", None),
        worker_type=worker_type,
        planner_profile_data=getattr(args, "planner_profile_data", None),
        aic_backend=aic_backend,
        aic_system=aic_system,
        aic_backend_version=aic_backend_version,
        aic_tp_size=aic_tp_size,
        aic_model_path=aic_model_path,
        aic_moe_tp_size=aic_moe_tp_size,
        aic_moe_ep_size=aic_moe_ep_size,
        aic_attention_dp_size=aic_attention_dp_size,
        enable_local_indexer=not getattr(args, "durable_kv_events", False),
        kv_transfer_bandwidth=getattr(args, "kv_transfer_bandwidth", None),
        reasoning=_parse_reasoning_config(getattr(args, "reasoning", None)),
        sglang=_build_sglang_args(args),
        preemption_mode=getattr(args, "preemption_mode", "lifo"),
    )


def load_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    if args.extra_engine_args:
        return MockEngineArgs.from_json(args.extra_engine_args.read_text())
    return build_mocker_engine_args(args)


def apply_worker_engine_args_overrides(
    engine_args: MockEngineArgs,
    *,
    kv_bytes_per_token: int | None = None,
    bootstrap_port: int | None = None,
    zmq_kv_events_port: int | None = None,
    zmq_replay_port: int | None = None,
) -> MockEngineArgs:
    return engine_args.with_overrides(
        bootstrap_port=bootstrap_port,
        zmq_kv_events_port=zmq_kv_events_port,
        zmq_replay_port=zmq_replay_port,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def build_runtime_config(
    engine_args: MockEngineArgs,
) -> tuple[int, ModelRuntimeConfig]:
    rc = ModelRuntimeConfig()
    rc.total_kv_blocks = engine_args.num_gpu_blocks
    rc.max_num_seqs = engine_args.max_num_seqs
    if rc.max_num_seqs is None:
        rc.max_num_seqs = _DEFAULT_MAX_NUM_SEQS
    rc.max_num_batched_tokens = engine_args.max_num_batched_tokens
    if rc.max_num_batched_tokens is None:
        rc.max_num_batched_tokens = _DEFAULT_MAX_NUM_BATCHED_TOKENS
    rc.enable_local_indexer = (
        engine_args.enable_local_indexer and not engine_args.is_decode()
    )
    rc.data_parallel_size = engine_args.dp_size

    bootstrap_port = engine_args.bootstrap_port
    if engine_args.is_prefill() and bootstrap_port is not None:
        host = os.environ.get(
            "DYN_HTTP_RPC_HOST", socket.gethostbyname(socket.gethostname())
        )
        rc.set_disaggregated_endpoint(
            bootstrap_host=host, bootstrap_port=bootstrap_port
        )

    return engine_args.block_size, rc
