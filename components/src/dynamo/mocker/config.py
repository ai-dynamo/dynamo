#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import socket

from dynamo._internal.aic import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MEM_FRACTION_STATIC,
    AicMemoryEstimatorUnavailableError,
    estimate_num_gpu_blocks,
)
from dynamo.common.utils.topology import apply_topology_config
from dynamo.llm import ModelRuntimeConfig
from dynamo.mocker import MockEngineArgs, ReasoningConfig, SglangArgs, TrtllmArgs

logger = logging.getLogger(__name__)

_DEFAULT_NUM_GPU_BLOCKS = 16384
_DEFAULT_MAX_NUM_SEQS = 256
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_DEFAULT_AIC_SYSTEM = "h200_sxm"
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


def _estimate_aic_num_gpu_blocks(
    *,
    engine_type: str,
    block_size: int | None,
    max_num_batched_tokens: int | None,
    aic_backend: str,
    aic_system: str | None,
    aic_backend_version: str | None,
    aic_tp_size: int | None,
    aic_model_path: str | None,
    aic_moe_tp_size: int | None,
    aic_moe_ep_size: int | None,
    aic_attention_dp_size: int | None,
    gpu_memory_utilization: float | None,
    mem_fraction_static: float | None,
    free_gpu_memory_fraction: float | None,
    sglang_page_size: int | None,
) -> int:
    if not aic_model_path:
        raise ValueError(
            "AIC KV cache capacity estimation requires a model path; "
            "set --model-path or aic_model_path"
        )
    resolved_block_size = _resolve_block_size_for_capacity(
        engine_type, block_size, sglang_page_size
    )
    try:
        return estimate_num_gpu_blocks(
            backend_name=aic_backend,
            system=aic_system or _DEFAULT_AIC_SYSTEM,
            model_path=aic_model_path,
            tp_size=aic_tp_size if aic_tp_size is not None else 1,
            block_size=resolved_block_size,
            max_num_batched_tokens=(
                max_num_batched_tokens
                if max_num_batched_tokens is not None
                else _DEFAULT_MAX_NUM_BATCHED_TOKENS
            ),
            gpu_memory_utilization=(
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else DEFAULT_GPU_MEMORY_UTILIZATION
            ),
            mem_fraction_static=(
                mem_fraction_static
                if mem_fraction_static is not None
                else DEFAULT_MEM_FRACTION_STATIC
            ),
            # None -> aic.py applies the TRT-LLM default (0.9).
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            backend_version=aic_backend_version,
            moe_tp_size=aic_moe_tp_size,
            moe_ep_size=aic_moe_ep_size,
            attention_dp_size=aic_attention_dp_size,
        )
    except AicMemoryEstimatorUnavailableError as exc:
        logger.warning(
            "AIC KV-cache capacity estimation is unavailable: %s. Falling back "
            "to default num_gpu_blocks=%d; upgrade aiconfigurator or set "
            "--num-gpu-blocks-override explicitly.",
            exc,
            _DEFAULT_NUM_GPU_BLOCKS,
        )
        return _DEFAULT_NUM_GPU_BLOCKS


def _resolve_num_gpu_blocks(
    *,
    explicit_num_gpu_blocks: int | None,
    engine_type: str,
    block_size: int | None,
    max_num_batched_tokens: int | None,
    aic_backend: str | None,
    aic_system: str | None,
    aic_backend_version: str | None,
    aic_tp_size: int | None,
    aic_model_path: str | None,
    aic_moe_tp_size: int | None,
    aic_moe_ep_size: int | None,
    aic_attention_dp_size: int | None,
    gpu_memory_utilization: float | None,
    mem_fraction_static: float | None,
    free_gpu_memory_fraction: float | None,
    sglang_page_size: int | None,
) -> int:
    if explicit_num_gpu_blocks is not None:
        return explicit_num_gpu_blocks
    if aic_backend is None:
        return _DEFAULT_NUM_GPU_BLOCKS
    return _estimate_aic_num_gpu_blocks(
        engine_type=engine_type,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        aic_backend=aic_backend,
        aic_system=aic_system,
        aic_backend_version=aic_backend_version,
        aic_tp_size=aic_tp_size,
        aic_model_path=aic_model_path,
        aic_moe_tp_size=aic_moe_tp_size,
        aic_moe_ep_size=aic_moe_ep_size,
        aic_attention_dp_size=aic_attention_dp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        mem_fraction_static=mem_fraction_static,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        sglang_page_size=sglang_page_size,
    )


def _resolve_raw_engine_args(
    raw: dict,
    *,
    fallback_model_path: str | None = None,
) -> dict:
    if raw.get("num_gpu_blocks") is not None:
        return raw

    aic_backend = raw.get("aic_backend")
    if aic_backend is None:
        return raw

    engine_type = raw.get("engine_type") or "vllm"
    sglang = raw.get("sglang")
    sglang_page_size = sglang.get("page_size") if isinstance(sglang, dict) else None
    raw["num_gpu_blocks"] = _estimate_aic_num_gpu_blocks(
        engine_type=engine_type,
        block_size=raw.get("block_size"),
        max_num_batched_tokens=raw.get("max_num_batched_tokens"),
        aic_backend=aic_backend,
        aic_system=raw.get("aic_system"),
        aic_backend_version=raw.get("aic_backend_version"),
        aic_tp_size=raw.get("aic_tp_size"),
        aic_model_path=raw.get("aic_model_path") or fallback_model_path,
        aic_moe_tp_size=raw.get("aic_moe_tp_size"),
        aic_moe_ep_size=raw.get("aic_moe_ep_size"),
        aic_attention_dp_size=raw.get("aic_attention_dp_size"),
        gpu_memory_utilization=raw.get("gpu_memory_utilization"),
        mem_fraction_static=raw.get("mem_fraction_static"),
        free_gpu_memory_fraction=raw.get("free_gpu_memory_fraction"),
        sglang_page_size=sglang_page_size,
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
    engine_type = getattr(args, "engine_type", None) or "vllm"
    max_model_len = getattr(args, "max_model_len", None)
    num_gpu_blocks = _resolve_num_gpu_blocks(
        explicit_num_gpu_blocks=getattr(args, "num_gpu_blocks", None),
        engine_type=engine_type,
        block_size=getattr(args, "block_size", None),
        max_num_batched_tokens=getattr(
            args, "max_num_batched_tokens", _DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        aic_backend=aic_backend,
        aic_system=aic_system,
        aic_backend_version=aic_backend_version,
        aic_tp_size=aic_tp_size,
        aic_model_path=aic_model_path,
        aic_moe_tp_size=aic_moe_tp_size,
        aic_moe_ep_size=aic_moe_ep_size,
        aic_attention_dp_size=aic_attention_dp_size,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        mem_fraction_static=getattr(args, "mem_fraction_static", None),
        free_gpu_memory_fraction=getattr(args, "free_gpu_memory_fraction", None),
        sglang_page_size=getattr(args, "sglang_page_size", None),
    )
    return MockEngineArgs(
        engine_type=engine_type,
        num_gpu_blocks=num_gpu_blocks,
        block_size=getattr(args, "block_size", 0) or 0,
        max_model_len=max_model_len,
        max_num_seqs=getattr(args, "max_num_seqs", _DEFAULT_MAX_NUM_SEQS),
        max_num_batched_tokens=getattr(
            args, "max_num_batched_tokens", _DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
        g1_backend=getattr(args, "g1_backend", "kvbm"),
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
        aic_nextn=getattr(args, "aic_nextn", None),
        aic_nextn_accept_rates=getattr(args, "aic_nextn_accept_rates", None),
        aic_mtp_seed=getattr(args, "aic_mtp_seed", 42),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        mem_fraction_static=getattr(args, "mem_fraction_static", None),
        free_gpu_memory_fraction=getattr(args, "free_gpu_memory_fraction", None),
        enable_local_indexer=True,
        kv_bytes_per_token=getattr(args, "kv_bytes_per_token", None),
        kv_transfer_bandwidth=getattr(args, "kv_transfer_bandwidth", None),
        kv_transfer_timing_mode=getattr(args, "kv_transfer_timing_mode", "full_prompt"),
        num_g2_blocks=getattr(args, "num_g2_blocks", None),
        num_g3_blocks=getattr(args, "num_g3_blocks", None),
        enable_g4_storage=getattr(args, "enable_g4_storage", False),
        offload_batch_size=getattr(args, "offload_batch_size", None),
        bandwidth_g1_to_g2_gbps=getattr(args, "bandwidth_g1_to_g2_gbps", None),
        bandwidth_g2_to_g1_gbps=getattr(args, "bandwidth_g2_to_g1_gbps", None),
        bandwidth_g2_to_g3_gbps=getattr(args, "bandwidth_g2_to_g3_gbps", None),
        bandwidth_g3_to_g2_gbps=getattr(args, "bandwidth_g3_to_g2_gbps", None),
        bandwidth_g2_to_g4_gbps=getattr(args, "bandwidth_g2_to_g4_gbps", None),
        bandwidth_g4_to_g2_gbps=getattr(args, "bandwidth_g4_to_g2_gbps", None),
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
        raw = _resolve_raw_engine_args(
            raw, fallback_model_path=getattr(args, "model_path", None)
        )
        return MockEngineArgs.from_json(json.dumps(raw))
    return build_mocker_engine_args(args)


def apply_worker_engine_args_overrides(
    engine_args: MockEngineArgs,
    *,
    kv_bytes_per_token: int | None = None,
    bootstrap_port: int | None = None,
    zmq_kv_events_port: int | None = None,
    zmq_replay_port: int | None = None,
    aic_mtp_seed: int | None = None,
) -> MockEngineArgs:
    return engine_args.with_overrides(
        bootstrap_port=bootstrap_port,
        zmq_kv_events_port=zmq_kv_events_port,
        zmq_replay_port=zmq_replay_port,
        kv_bytes_per_token=kv_bytes_per_token,
        aic_mtp_seed=aic_mtp_seed,
    )


def build_runtime_config(
    engine_args: MockEngineArgs,
) -> tuple[int, ModelRuntimeConfig]:
    rc = ModelRuntimeConfig()
    rc.context_length = engine_args.max_model_len or 0
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
    rc.set_engine_specific("output_replay_consumer", "true")

    bootstrap_port = engine_args.bootstrap_port
    if engine_args.is_prefill() and bootstrap_port is not None:
        host = os.environ.get(
            "DYN_HTTP_RPC_HOST", socket.gethostbyname(socket.gethostname())
        )
        rc.set_disaggregated_endpoint(
            bootstrap_host=host, bootstrap_port=bootstrap_port
        )

    apply_topology_config(rc)

    return engine_args.block_size, rc
