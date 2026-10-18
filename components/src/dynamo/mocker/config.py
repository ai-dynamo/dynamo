# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo CLI/runtime adapters for AISimulate's canonical engine configuration."""

import argparse
import copy
import dataclasses
import json
import os
import socket
from collections.abc import Mapping
from typing import Any

from dynamo.common.configuration.groups.ais_perf_args import parse_ais_perf_config
from dynamo.common.kv_state_agent import worker_raw_topic
from dynamo.common.utils.topology import TopologyConfig, apply_topology_config
from dynamo.llm import ModelRuntimeConfig

# Fixed indexer domain and pool for mocker CacheOwners, in the canonical
# `domain/pool/owner` format (see examples/backends/vllm/deploy/kvcr/agg.yaml).
_KV_STATE_AGENT_INDEXER_DOMAIN = (
    "01010101010101010101010101010101:02020202020202020202020202020202"
)
_KV_STATE_AGENT_POOL = "0303030303030303"


def normalize_mocker_config(config: Mapping[str, Any] | str | None = None) -> dict:
    from dynamo._core import _normalize_mocker_config

    return _normalize_mocker_config({} if config is None else config)


def performance_config(config: Mapping[str, Any]) -> dict | None:
    timing = config.get("engine", {}).get("timing_model", {})
    if timing.get("type") == "external" and timing.get("provider") in {"ais", "aic"}:
        return timing["config"]
    return None


def engine_limit(config: Mapping[str, Any], name: str) -> int | None:
    return config["engine"][name]


def _build_native_host_offload(args: argparse.Namespace) -> dict | None:
    num_host_blocks = getattr(args, "num_host_blocks", None)
    controls = {
        key: value
        for key, value in (
            (
                "d2h_bandwidth_gbps",
                getattr(args, "host_offload_d2h_bandwidth_gbps", None),
            ),
            (
                "h2d_bandwidth_gbps",
                getattr(args, "host_offload_h2d_bandwidth_gbps", None),
            ),
        )
        if value is not None
    }
    if num_host_blocks is None:
        if controls:
            raise ValueError("--host-offload-* flags require --num-host-blocks")
        return None
    return {"num_host_blocks": num_host_blocks, **controls}


def build_mocker_engine_args(args: argparse.Namespace) -> dict:
    worker_type = (
        "prefill"
        if args.is_prefill_worker
        else "decode"
        if args.is_decode_worker
        else "aggregated"
    )
    rank = {
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
            "kv_transfer_bandwidth",
            "kv_transfer_timing_mode",
            "preemption_mode",
        )
        if getattr(args, key, None) is not None
    }
    rank.update(backend=args.engine_type, worker_type=worker_type)
    for flag, field in (
        ("ais_nextn", "aic_nextn"),
        ("ais_nextn_accept_rates", "aic_nextn_accept_rates"),
        ("ais_mtp_seed", "aic_mtp_seed"),
        ("kv_bytes_per_token", "kv_transfer_bytes_per_token"),
    ):
        if getattr(args, flag, None) is not None:
            rank[field] = getattr(args, flag)
    for backend, fields in (
        (
            "sglang",
            (
                "schedule_policy",
                "max_prefill_tokens",
                "chunked_prefill_size",
                "clip_max_new_tokens",
                "schedule_conservativeness",
            ),
        ),
        ("trtllm", ("capacity_scheduler_policy",)),
    ):
        overrides = {
            key: getattr(args, f"{backend}_{key}")
            for key in fields
            if getattr(args, f"{backend}_{key}", None) is not None
        }
        if overrides:
            rank[backend] = overrides

    canonical = getattr(args, "ais_perf_config", None)
    flat = {
        target: getattr(args, f"ais_{source}")
        for source, target in (
            ("backend", "backend"),
            ("system", "system"),
            ("backend_version", "backend_version"),
            ("tp_size", "tp"),
            ("moe_tp_size", "moe_tp_size"),
            ("moe_ep_size", "moe_ep_size"),
            ("attention_dp_size", "attention_dp"),
            ("nextn", "nextn"),
        )
        if getattr(args, f"ais_{source}", None) is not None
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
            "backend": args.engine_type,
            "system": "h200_sxm",
            "worker_type": worker_type,
            **flat,
        }
    profile = getattr(args, "planner_profile_data", None)
    if canonical is not None:
        if profile is not None:
            raise ValueError("choose one timing provider: AIS or planner_profile_data")
        rank["timing_model"] = {
            "type": "external",
            "provider": "ais",
            "config": canonical,
        }
    elif profile is not None:
        rank["timing_model"] = {
            "type": "external",
            "provider": "dynamo_profile",
            "config": {"path": str(profile)},
        }

    host_offload = _build_native_host_offload(args)
    if host_offload is not None:
        rank["native_host_offload"] = host_offload
        rank["kv_cache_bytes_per_token"] = getattr(args, "kv_bytes_per_token", None)

    config = {
        key: getattr(args, key)
        for key in (
            "dp_size",
            "startup_time",
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
        )
        if getattr(args, key, None) is not None
    }
    runtime: dict[str, Any] = {"enable_local_indexer": True}
    if getattr(args, "reasoning", None):
        runtime["reasoning"] = json.loads(args.reasoning)
    if getattr(args, "response_replay_trace_path", None):
        runtime["response_replay_trace_path"] = str(args.response_replay_trace_path)
    return normalize_mocker_config({**config, "engine": rank, "dynamo": runtime})


def load_mocker_engine_args(args: argparse.Namespace) -> dict:
    if args.extra_engine_args:
        return normalize_mocker_config(args.extra_engine_args.read_text())
    return build_mocker_engine_args(args)


def apply_worker_engine_args_overrides(
    engine_args: Mapping[str, Any],
    *,
    kv_bytes_per_token: int | None = None,
    bootstrap_port: int | None = None,
    zmq_kv_events_port: int | None = None,
    zmq_kv_events_topic: str | None = None,
    zmq_replay_port: int | None = None,
    kv_state_agent: bool | None = None,
    ais_mtp_seed: int | None = None,
) -> dict:
    config = copy.deepcopy(dict(engine_args))
    for key, value in (
        ("bootstrap_port", bootstrap_port),
        ("zmq_kv_events_port", zmq_kv_events_port),
        ("zmq_kv_events_topic", zmq_kv_events_topic),
        ("zmq_replay_port", zmq_replay_port),
        ("kv_state_agent", kv_state_agent),
    ):
        if value is not None:
            config.setdefault("dynamo", {})[key] = value
    for key, value in (
        ("kv_transfer_bytes_per_token", kv_bytes_per_token),
        ("aic_mtp_seed", ais_mtp_seed),
    ):
        if value is not None:
            config["engine"][key] = value
    return normalize_mocker_config(config)


def build_runtime_config(
    engine_args: Mapping[str, Any],
    topology_config: TopologyConfig | None = None,
) -> tuple[int, ModelRuntimeConfig]:
    rank = engine_args["engine"]
    runtime = engine_args["dynamo"]
    rc = ModelRuntimeConfig()
    rc.context_length = rank["max_model_len"] or 0
    rc.total_kv_blocks = rank["num_gpu_blocks"]
    rc.max_num_seqs = engine_limit(engine_args, "max_num_seqs")
    rc.max_num_batched_tokens = engine_limit(engine_args, "max_num_batched_tokens")
    is_decode = rank["worker_type"] == "decode"
    rc.enable_local_indexer = runtime["enable_local_indexer"] and not is_decode
    rc.kv_event_publishing_enabled = rank["enable_prefix_caching"] and not is_decode
    rc.data_parallel_size = engine_args["dp_size"]
    rc.set_engine_specific("output_replay_consumer", "true")
    port = runtime["bootstrap_port"]
    if rank["worker_type"] == "prefill" and port is not None:
        host = os.environ.get("DYN_HTTP_RPC_HOST") or socket.gethostbyname(
            socket.gethostname()
        )
        rc.set_disaggregated_endpoint(bootstrap_host=host, bootstrap_port=port)
    apply_topology_config(rc, topology_config)
    return rank["block_size"], rc


def worker_topology_config(
    topology_config: TopologyConfig, worker_id: int, island_size: int | None
) -> TopologyConfig:
    """Topology of mocker worker ``worker_id`` in this process.

    One mocker process stands for many workers. With ``island_size``, workers
    are split into KV-transfer islands of that size: each worker's value for the
    KV transfer domain gets the suffix ``-{worker_id // island_size}``.
    """
    if not island_size or not topology_config.enabled:
        return topology_config
    domain = topology_config.kv_transfer_domain
    domains = dict(topology_config.topology_domains)
    domains[domain] = f"{domains[domain]}-{worker_id // island_size}"
    return dataclasses.replace(topology_config, topology_domains=domains)


def kv_state_agent_cache_owner_ids(dp_size: int, first_slot: int) -> list[str]:
    """CacheOwner IDs of one mocker worker, one per DP rank.

    Operator-assigned durable slots in the canonical `domain/pool/slot` format
    of the vLLM KVCR manifests, so a restarted worker reattaches to its slots.
    """
    return [
        f"{_KV_STATE_AGENT_INDEXER_DOMAIN}/{_KV_STATE_AGENT_POOL}/{first_slot + rank:032x}"
        for rank in range(dp_size)
    ]


def build_kv_state_agent_descriptors(
    engine_args: Mapping[str, Any],
    *,
    cache_owner_ids: list[str],
    kv_state_endpoint: str,
    raw_advertise_host: str,
    zmq_kv_events_port: int,
    router_hint_worker_type: str | None = None,
) -> list[dict]:
    """KV state-agent attachment descriptors for one mocker worker, one per DP rank,
    built as a vLLM state-agent worker builds them.

    Every rank publishes the worker's raw topic. With ``router_hint_worker_type``,
    each owner advertises a placeholder source control endpoint: the router builds
    hints for it, and the mocker ignores them.
    """
    raw_topic = worker_raw_topic(cache_owner_ids)
    descriptors = []
    for rank, cache_owner_id in enumerate(cache_owner_ids):
        slot = cache_owner_id.rsplit("/", 1)[1]
        descriptors.append(
            {
                "cache_owner_id": cache_owner_id,
                "global_dp_rank": rank,
                "kv_state_endpoint": kv_state_endpoint,
                "indexer_domain_id": cache_owner_id.split("/", 1)[0],
                "kv_block_size": engine_args["engine"]["block_size"],
                "raw_zmq_endpoint": (
                    f"tcp://{raw_advertise_host}:{zmq_kv_events_port + rank}"
                ),
                "raw_topic": raw_topic,
                "router_hint_source": (
                    {
                        "source_control_endpoint": f"tcp://mocker-{slot}.invalid:1",
                        "worker_type": router_hint_worker_type,
                    }
                    if router_hint_worker_type is not None
                    else None
                ),
            }
        )
    return descriptors
