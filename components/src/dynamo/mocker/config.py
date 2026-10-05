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

_DEFAULT_AIS_SYSTEM = "h200_sxm"


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


def _resolve_capacity(engine_args: MockEngineArgs, *, explicit: bool) -> MockEngineArgs:
    canonical = engine_args.ais_perf_config
    if canonical is not None and not explicit:
        defaults = MockEngineArgs()
        engine_args.num_gpu_blocks = estimate_canonical_num_gpu_blocks(
            canonical,
            block_size=engine_args.block_size,
            max_num_batched_tokens=(
                engine_args.max_num_batched_tokens or defaults.max_num_batched_tokens
            ),
            max_num_seqs=engine_args.max_num_seqs or defaults.max_num_seqs,
            **{
                key: getattr(engine_args, key)
                for key in (
                    "gpu_memory_utilization",
                    "mem_fraction_static",
                    "free_gpu_memory_fraction",
                )
                if getattr(engine_args, key) is not None
            },
        )
    return engine_args


def build_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    worker_type = (
        "prefill"
        if getattr(args, "is_prefill_worker", False)
        else "decode"
        if getattr(args, "is_decode_worker", False)
        else "aggregated"
    )
    engine_type = args.engine_type or MockEngineArgs().engine_type
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
        canonical.setdefault("worker_type", worker_type)
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
    # Forward supplied CLI values. AISimulate materializes engine defaults.
    raw = {
        key: getattr(args, key)
        for key in (
            "num_gpu_blocks",
            "block_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "speedup_ratio",
            "decode_speedup_ratio",
            "dp_size",
            "startup_time",
            "planner_profile_data",
            "ais_nextn",
            "ais_nextn_accept_rates",
            "ais_mtp_seed",
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "kv_bytes_per_token",
            "kv_transfer_bandwidth",
            "kv_transfer_timing_mode",
            "response_replay_trace_path",
            "preemption_mode",
        )
        if getattr(args, key, None) is not None
    }
    raw.update(
        ais_perf_config=canonical,
        engine_type=engine_type,
        worker_type=worker_type,
        enable_local_indexer=True,
        reasoning=_parse_reasoning_config(getattr(args, "reasoning", None)),
        sglang=_build_sglang_args(args),
        trtllm=_build_trtllm_args(args),
    )
    return _resolve_capacity(
        MockEngineArgs(**raw), explicit=args.num_gpu_blocks is not None
    )


def load_mocker_engine_args(args: argparse.Namespace) -> MockEngineArgs:
    if args.extra_engine_args:
        raw = json.loads(args.extra_engine_args.read_text())
        if not isinstance(raw, dict):
            raise ValueError("extra engine args must be a JSON object")
        explicit = raw.get("num_gpu_blocks") is not None or (
            isinstance(raw.get("engine"), dict)
            and raw["engine"].get("num_gpu_blocks") is not None
        )
        return _resolve_capacity(
            MockEngineArgs.from_json(json.dumps(raw)), explicit=explicit
        )
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
        rc.max_num_seqs = MockEngineArgs().max_num_seqs
    rc.max_num_batched_tokens = engine_args.max_num_batched_tokens
    if rc.max_num_batched_tokens is None:
        rc.max_num_batched_tokens = MockEngineArgs().max_num_batched_tokens
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
