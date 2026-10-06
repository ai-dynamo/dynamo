#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket

from dynamo._internal.ais import estimate_canonical_num_gpu_blocks
from dynamo.common.configuration.groups.ais_perf_args import parse_ais_perf_config
from dynamo.common.utils.topology import apply_topology_config
from dynamo.llm import ModelRuntimeConfig
from dynamo.mocker import MockEngineArgs, ReasoningConfig, SglangArgs, TrtllmArgs

_DEFAULT_NUM_GPU_BLOCKS = 16384
_DEFAULT_MAX_NUM_SEQS = 256
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_DEFAULT_AIS_SYSTEM = "h200_sxm"
_DEFAULT_VLLM_BLOCK_SIZE = 64
_DEFAULT_SGLANG_BLOCK_SIZE = 1
# Recent TRT-LLM PyTorch backend default tokens_per_block (older builds use 64).
_DEFAULT_TRTLLM_BLOCK_SIZE = 32


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


def _build_trtllm_args(args: argparse.Namespace) -> TrtllmArgs | None:
    trtllm_args = {
        "capacity_scheduler_policy": getattr(
            args, "trtllm_capacity_scheduler_policy", None
        ),
    }
    if not any(value is not None for value in trtllm_args.values()):
        return None
    return TrtllmArgs(**trtllm_args)


def _resolve_block_size_for_capacity(
    engine_type: str,
    block_size: int | None,
    sglang_page_size: int | None,
) -> int:
    if block_size is not None:
        return block_size
    if engine_type == "sglang":
        if sglang_page_size is not None:
            return sglang_page_size
        return _DEFAULT_SGLANG_BLOCK_SIZE
    if engine_type == "trtllm":
        return _DEFAULT_TRTLLM_BLOCK_SIZE
    return _DEFAULT_VLLM_BLOCK_SIZE


def _resolve_raw_engine_args(raw: dict) -> dict:
    canonical = raw.get("ais_perf_config")
    if canonical is not None and raw.get("num_gpu_blocks") is None:
        sglang = raw.get("sglang") or {}
        raw["num_gpu_blocks"] = estimate_canonical_num_gpu_blocks(
            canonical,
            block_size=_resolve_block_size_for_capacity(
                raw.get("engine_type", "vllm"),
                raw.get("block_size"),
                sglang.get("page_size"),
            ),
            **{
                key: raw[key]
                for key in (
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "gpu_memory_utilization",
                    "mem_fraction_static",
                    "free_gpu_memory_fraction",
                )
                if raw.get(key) is not None
            },
        )
    return raw


def build_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    worker_type = (
        "prefill"
        if getattr(args, "is_prefill_worker", False)
        else "decode"
        if getattr(args, "is_decode_worker", False)
        else "aggregated"
    )
    engine_type = args.engine_type or "vllm"
    canonical = getattr(args, "ais_perf_config", None)
    flat = {
        key: getattr(args, "ais_" + name, None)
        for name, key in (
            ("backend", "backend"),
            ("system", "system"),
            ("backend_version", "backend_version"),
            ("tp_size", "tp"),
            ("moe_tp_size", "moe_tp_size"),
            ("moe_ep_size", "moe_ep_size"),
            ("attention_dp_size", "attention_dp"),
            ("nextn", "nextn"),
        )
        if getattr(args, "ais_" + name, None) is not None
    }
    if canonical is not None:
        if args.ais_perf_model or flat:
            raise ValueError(
                "--ais-perf-config cannot be combined with flat AIS/AIC identity flags"
            )
        canonical = parse_ais_perf_config(canonical)
        if canonical.get("worker_type", worker_type) != worker_type:
            raise ValueError("AIS worker_type must match the mocker worker role")
        canonical["worker_type"] = worker_type
    elif args.ais_perf_model:
        canonical = {
            "model": args.model_path,
            "backend": engine_type,
            "system": _DEFAULT_AIS_SYSTEM,
            "worker_type": worker_type,
            **flat,
        }
        if not canonical["model"]:
            raise ValueError("--ais-perf-model requires --model-path")
    if canonical is not None:
        from aisimulate_core.sdk import ForwardPassPerfModelConfig

        canonical = ForwardPassPerfModelConfig(**canonical).to_dict()
    raw = _resolve_raw_engine_args(
        {
            "ais_perf_config": canonical,
            "num_gpu_blocks": args.num_gpu_blocks,
            "engine_type": engine_type,
            "block_size": args.block_size,
            "sglang": {"page_size": args.sglang_page_size},
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "mem_fraction_static": args.mem_fraction_static,
            "free_gpu_memory_fraction": args.free_gpu_memory_fraction,
        }
    )
    num_gpu_blocks = raw["num_gpu_blocks"]
    if num_gpu_blocks is None:
        num_gpu_blocks = _DEFAULT_NUM_GPU_BLOCKS
    return MockEngineArgs(
        ais_perf_config=canonical,
        engine_type=engine_type,
        num_gpu_blocks=num_gpu_blocks,
        block_size=getattr(args, "block_size", 0) or 0,
        max_model_len=args.max_model_len,
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
        ais_nextn=None if canonical is not None else args.ais_nextn,
        ais_nextn_accept_rates=args.ais_nextn_accept_rates,
        ais_mtp_seed=args.ais_mtp_seed,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        mem_fraction_static=getattr(args, "mem_fraction_static", None),
        free_gpu_memory_fraction=getattr(args, "free_gpu_memory_fraction", None),
        enable_local_indexer=True,
        kv_bytes_per_token=getattr(args, "kv_bytes_per_token", None),
        kv_transfer_bandwidth=getattr(args, "kv_transfer_bandwidth", None),
        kv_transfer_timing_mode=getattr(args, "kv_transfer_timing_mode", "full_prompt"),
        reasoning=_parse_reasoning_config(getattr(args, "reasoning", None)),
        response_replay_trace_path=args.response_replay_trace_path,
        sglang=_build_sglang_args(args),
        trtllm=_build_trtllm_args(args),
        preemption_mode=getattr(args, "preemption_mode", "lifo"),
    )


def load_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    if args.extra_engine_args:
        raw = json.loads(args.extra_engine_args.read_text())
        if not isinstance(raw, dict):
            raise ValueError("extra engine args must be a JSON object")
        raw = _resolve_raw_engine_args(raw)
        return MockEngineArgs.from_json(json.dumps(raw))
    return build_mocker_engine_args(args)


def apply_worker_engine_args_overrides(
    engine_args: MockEngineArgs,
    *,
    kv_bytes_per_token: int | None = None,
    bootstrap_port: int | None = None,
    zmq_kv_events_port: int | None = None,
    zmq_replay_port: int | None = None,
    ais_mtp_seed: int | None = None,
) -> MockEngineArgs:
    return engine_args.with_overrides(
        bootstrap_port=bootstrap_port,
        zmq_kv_events_port=zmq_kv_events_port,
        zmq_replay_port=zmq_replay_port,
        kv_bytes_per_token=kv_bytes_per_token,
        ais_mtp_seed=ais_mtp_seed,
    )


def build_runtime_config(
    engine_args: MockEngineArgs,
) -> tuple[int, ModelRuntimeConfig]:
    rc = ModelRuntimeConfig()
    rc.context_length = engine_args.max_model_len or 0
    # MockEngineArgs defines num_gpu_blocks per independently simulated DP rank.
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
    rc.kv_event_publishing_enabled = (
        engine_args.enable_prefix_caching and not engine_args.is_decode()
    )
    rc.data_parallel_size = engine_args.dp_size
    rc.set_engine_specific("output_replay_consumer", "true")

    bootstrap_port = engine_args.bootstrap_port
    if engine_args.is_prefill() and bootstrap_port is not None:
        host = os.environ.get("DYN_HTTP_RPC_HOST") or socket.gethostbyname(
            socket.gethostname()
        )
        rc.set_disaggregated_endpoint(
            bootstrap_host=host, bootstrap_port=bootstrap_port
        )

    apply_topology_config(rc)

    return engine_args.block_size, rc
