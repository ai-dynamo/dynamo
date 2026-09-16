# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional, pinned Mooncake Store metadata, never an inference dependency."""

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_KEY = "vllm_mooncake_store"
VLLM_REVISION = "1085b64425a9e6f5ca52876ad32e55fda5665f4e"
WORKER_EXTENSION = "dynamo.vllm.mooncake_store_runtime.MooncakeStoreWorkerExtension"
FPM_WORKER_EXTENSION = (
    "dynamo.vllm.mooncake_store_worker.MooncakeStoreFpmWorkerExtension"
)
RPC_METHOD = "dynamo_mooncake_store_descriptor"
RPC_TIMEOUT_SECONDS = 10.0
MAX_DESCRIPTOR_BYTES = 256 * 1024

_STORE_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"
_SPEC_MODULE = "vllm.v1.kv_cache_interface"
_MANAGER_MODULE = "vllm.v1.core.single_type_kv_cache_manager"

# Fingerprints are of the unmodified files at VLLM_REVISION. Version strings alone
# do not detect a patched wheel with different private cache/hash semantics.
_PINNED_FILES = (
    (
        "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
        "bddabd71b6394c6d4eddc07238dbb5204c1c984aadb96db586346a0da59db099",
    ),
    (
        "distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py",
        "2f8c7ae9bb6b9b43119a86421892d77c591533159752489a79e6e46e6255a2c2",
    ),
    (
        "distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py",
        "1ecdc99faa10435bbb988b8ea083e48942a0cc7d51b69e273861a34638b5fea4",
    ),
    (
        "distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py",
        "386033a398ea38495de5d951e990fa9b0139f0134a45157767ae1795ae1cd13b",
    ),
    (
        "v1/kv_cache_interface.py",
        "718ee51c8b845352f89c5eae8a449984713bc269829f4315e27c0fedf5305ef5",
    ),
    (
        "v1/core/single_type_kv_cache_manager.py",
        "15360abf039513fab8f0fe99d7bc36283e63251d83e1c13f99368ae89bf13135",
    ),
    (
        "v1/core/kv_cache_utils.py",
        "2b3d29bffd425ad65c0ce4e6f3fce6c6562642f0f8f18880adaa14c4cb3c0862",
    ),
    (
        "utils/hashing.py",
        "2e8fcc81dd675ed0fba49de8fd8658b4dab9ae26a917edaf5164bf266c56b4df",
    ),
    (
        "v1/core/block_pool.py",
        "35b8427b6144b8cf7662ad5a5a14c0c3051e3542c4c196a920dc99d093c4499d",
    ),
)


class UnsupportedContract(ValueError):
    """The optional hint cannot be justified by this worker's runtime."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise UnsupportedContract(reason)


def _positive(value: Any) -> int:
    _require(type(value) is int and 0 < value <= 2**32 - 1, "invalid token span")
    return value


def _verify_pinned_sources() -> None:
    distribution = importlib.metadata.distribution("vllm")
    direct_url = distribution.read_text("direct_url.json")
    commit = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    version = distribution.version
    scm = re.search(r"(?:\+|\.)g([0-9a-f]{8,40})(?:\.|$)", version)
    _require(
        commit == VLLM_REVISION
        or (
            scm is not None
            and VLLM_REVISION.startswith(scm.group(1))
            and re.search(r"\.d\d{8}", version) is None
        ),
        "vLLM build does not identify the pinned revision",
    )
    vllm = importlib.import_module("vllm")
    root = Path(vllm.__file__).parent
    for relative, expected in _PINNED_FILES:
        _require(
            hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected,
            "vLLM private cache/hash sources differ from the pinned adapter",
        )


def _validate_input_config(config: Any) -> None:
    model = config.model_config
    _require(
        model.is_multimodal_model is False
        and model.enable_prompt_embeds is False
        and model.is_encoder_decoder is False
        and model.runner_type == "generate",
        "only text generation without prompt embeddings is supported",
    )
    _require(config.speculative_config is None, "speculative decoding is unsupported")
    _require(
        config.cache_config.enable_prefix_caching is True
        and config.cache_config.prefix_caching_hash_algo == "sha256"
        and os.environ.get("PYTHONHASHSEED") == "0",
        "sha256 prefix caching and PYTHONHASHSEED=0 are required",
    )
    envs = importlib.import_module("vllm.envs")
    _require(
        envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES is True,
        "integer GPU event hashes are required",
    )
    _require(
        config.kv_events_config is not None
        and config.kv_events_config.enable_kv_cache_events is True,
        "GPU KV event publication is required",
    )
    scheduler = config.scheduler_config.scheduler_cls
    _require(
        scheduler is None
        or scheduler
        in (
            "vllm.v1.core.sched.scheduler.Scheduler",
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
            "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
        ),
        "custom scheduler input/hash semantics are unsupported",
    )
    parallel = config.parallel_config
    _require(
        parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1,
        "context-parallel cache geometry is unsupported",
    )


def _find_store_connector(connector: Any) -> Any:
    # The adapter is opt-in and older supported engines need not have these
    # modules. Resolve optional private imports only inside the guarded RPC.
    store_cls = importlib.import_module(
        f"{_STORE_MODULE}.connector"
    ).MooncakeStoreConnector
    multi_cls = importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ).MultiConnector
    pending = [connector]
    stores = []
    seen = set()
    while pending:
        current = pending.pop()
        _require(id(current) not in seen and len(seen) < 64, "invalid connector graph")
        seen.add(id(current))
        if type(current) is store_cls:
            stores.append(current)
        elif type(current) is multi_cls:
            pending.extend(current.sub_connectors)
    _require(len(stores) == 1, "exactly one Mooncake Store connector is required")
    return stores[0]


def _spec_semantics(spec: Any) -> tuple[str, str, str, int | None, str | None]:
    specs = importlib.import_module(_SPEC_MODULE)
    if type(spec) is specs.UniformTypeKVCacheSpecs:
        members = list(spec.kv_cache_specs.values())
        _require(bool(members), "empty uniform cache group")
        semantics = [_spec_semantics(member) for member in members]
        _require(
            all(item == semantics[0] for item in semantics)
            and all(member.block_size == spec.block_size for member in members),
            "mixed uniform cache semantics",
        )
        return semantics[0]
    if type(spec) is specs.FullAttentionSpec:
        _require(
            spec.sliding_window is None
            and spec.attention_chunk_size is None
            and spec.non_causal is False,
            "nonstandard full-attention semantics",
        )
        return "full_attention", "FullAttentionSpec", "FullAttentionManager", None, None
    if type(spec) is specs.SlidingWindowSpec:
        _require(spec.extra_retained_tokens == 0, "speculative window retention")
        window = _positive(spec.sliding_window)
        _require(window > 1, "sliding window must exceed one token")
        return (
            "sliding_window",
            "SlidingWindowSpec",
            "SlidingWindowManager",
            window,
            None,
        )
    if type(spec) is specs.MambaSpec:
        _require(
            spec.mamba_cache_mode == "align" and spec.num_speculative_blocks == 0,
            "only non-speculative Mamba align is supported",
        )
        return "mamba", "MambaSpec", "MambaManager", None, "align"
    raise UnsupportedContract("unsupported cache specification")


def _export_descriptor(connector: Any, config: Any) -> dict[str, Any]:
    _validate_input_config(config)
    store_connector = _find_store_connector(connector)
    _require(
        store_connector._kv_transfer_config.kv_connector_extra_config.get(
            "enable_lookup", True
        )
        is True,
        "store lookup is disabled",
    )
    worker = store_connector.connector_worker
    worker_cls = importlib.import_module(f"{_STORE_MODULE}.worker").MooncakeStoreWorker
    _require(type(worker) is worker_cls, "missing initialized Mooncake Store worker")
    _require(not worker._capacity_only, "capacity-only connector")
    _require(worker.pcp_size == worker.dcp_size == 1, "context parallelism unsupported")
    coord = worker.coord
    coord_cls = importlib.import_module(
        f"{_STORE_MODULE}.coordinator"
    ).MooncakeStoreCoordinator
    _require(type(coord) is coord_cls, "unsupported store coordinator")
    _require(not coord.use_eagle and not coord.eagle_group_ids, "Eagle block drop")
    original = worker._kv_cache_config
    group_ids = original.prefix_cacheable_group_ids
    _require(
        0 < len(original.kv_cache_groups) <= 64
        and 0 < len(group_ids) <= 64
        and len(group_ids) == len(worker.token_dbs) == len(worker._lookup_key_prefixes)
        and len(group_ids)
        == len(worker._kv_cache_groups)
        == len(coord.kv_cache_groups),
        "incomplete store-group projection",
    )
    projection = [None] * len(original.kv_cache_groups)
    groups = []
    prefixes_seen: set[str] = set()
    managers = importlib.import_module(_MANAGER_MODULE)
    data = importlib.import_module(f"{_STORE_MODULE}.data")
    hash_span = _positive(coord.hash_block_size)
    alignment = _positive(coord.lcm_block_size)
    _require(
        worker.hash_block_size == hash_span and worker.block_size == alignment,
        "inconsistent coordinator geometry",
    )
    namespace = None
    for group_id, gpu_group_id in enumerate(group_ids):
        _require(
            type(gpu_group_id) is int
            and 0 <= gpu_group_id < len(projection)
            and projection[gpu_group_id] is None,
            "invalid GPU group projection",
        )
        projection[gpu_group_id] = group_id
        group = worker._kv_cache_groups[group_id]
        spec = group.kv_cache_spec
        semantics = _spec_semantics(spec)
        kind, spec_name, manager_name, window, mamba_mode = semantics
        _require(
            not group.is_eagle_group
            and spec == original.kv_cache_groups[gpu_group_id].kv_cache_spec
            and spec == coord.kv_cache_groups[group_id].kv_cache_spec,
            "inconsistent resolved group specification",
        )
        matches = [
            entry for entry in coord.attention_groups if group_id in entry.group_ids
        ]
        _require(len(matches) == 1, "incomplete coordinator manager coverage")
        entry = matches[0]
        _require(
            entry.manager_cls is getattr(managers, manager_name)
            and _spec_semantics(entry.spec) == semantics
            and not entry.use_eagle,
            "unsupported coordinator manager",
        )
        db = worker.token_dbs[group_id]
        _require(type(db) is data.ChunkedTokenDatabase, "unsupported token database")
        span = _positive(db.block_size)
        _require(
            span == spec.block_size
            and db.hash_block_size == hash_span
            and span % hash_span == 0
            and alignment % span == 0
            and db.metadata.group_id == group_id,
            "invalid group hash/physical spans",
        )
        cache_prefix = db.metadata.cache_prefix
        _require(
            type(cache_prefix) is str and bool(cache_prefix.strip()),
            "cache_prefix is required",
        )
        if namespace is None:
            namespace = cache_prefix
        _require(cache_prefix == namespace, "inconsistent cache namespaces")
        prefixes = worker._lookup_key_prefixes[group_id]
        _require(
            type(prefixes) in (list, tuple) and len(prefixes) > 0,
            "missing object prefixes",
        )
        for prefix in prefixes:
            _require(
                type(prefix) is str
                and 0 < len(prefix.encode()) <= 4096
                and prefix.startswith(cache_prefix + "@")
                and not prefix.endswith("@")
                and prefix not in prefixes_seen
                and len(prefixes_seen) < 1024,
                "invalid or duplicate object prefix",
            )
            prefixes_seen.add(prefix)
        groups.append(
            {
                "group_id": group_id,
                "kind": kind,
                "spec": spec_name,
                "manager": manager_name,
                "block_size": span,
                "hash_block_size": hash_span,
                "key_prefixes": sorted(prefixes),
                "sliding_window": window,
                "mamba_cache_mode": mamba_mode,
            }
        )
    main = next((g for g in groups if g["kind"] == "full_attention"), None)
    _require(main is not None, "main full-attention event group is required")
    event_span = main["block_size"]
    for group in groups:
        span = group["block_size"]
        if group["kind"] == "full_attention":
            _require(span % event_span == 0, "unobservable full-attention hashes")
        elif group["kind"] == "sliding_window":
            _require(
                span % event_span == 0
                or (group["sliding_window"] - 2) // span + 1 == 1,
                "unobservable interior sliding-window hashes",
            )
        alignment = math.lcm(alignment, span, event_span)
        _positive(alignment)
    _require(type(coord.enable_partial_hash_hits) is bool, "invalid partial-hash mode")
    descriptor = {
        "schema_version": 1,
        "adapter": "vllm-1085b644",
        "vllm_revision": VLLM_REVISION,
        "hash": {
            "algorithm": "sha256",
            "digest_encoding": "hex",
            "gpu_event_hash": "low64",
            "seed_policy": "pythonhashseed-0",
            "key_separator": "@",
        },
        "input": {"text_only": True, "normalized_namespaces": True, "lora": True},
        "main_event_group": group_ids[main["group_id"]],
        "main_event_block_size": event_span,
        "gpu_to_store_group": projection,
        "coordinator": {
            "lcm_block_size": coord.lcm_block_size,
            "speculative": False,
            "drop_blocks": False,
            "partial_hash_hits": coord.enable_partial_hash_hits,
        },
        "groups": groups,
    }
    _require(
        len(_canonical(descriptor)) <= MAX_DESCRIPTOR_BYTES, "descriptor size limit"
    )
    return descriptor


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class MooncakeStoreWorkerExtension:
    """Named, read-only RPC mixed into vLLM workers by explicit configuration."""

    def dynamo_mooncake_store_descriptor(self) -> dict[str, Any]:
        try:
            _verify_pinned_sources()
            config = self.vllm_config
            transfer = importlib.import_module("vllm.distributed.kv_transfer")
            descriptor = _export_descriptor(transfer.get_kv_transfer_group(), config)
            return {
                "rank": config.parallel_config.rank,
                "dp_rank": config.parallel_config.data_parallel_index,
                "descriptor": descriptor,
            }
        except Exception as exc:  # noqa: BLE001 - optional hints must not stop serving
            # An optional inspection must never turn an unsupported private API
            # into a failed collective that can prevent normal model serving.
            reason = (
                str(exc) if isinstance(exc, UnsupportedContract) else type(exc).__name__
            )
            return {"unsupported": reason[:200]}


def _agree_descriptors(
    replies: Any, *, world_size: int, dp_rank: int, event_span: int
) -> dict[str, Any]:
    _require(
        type(replies) is list and len(replies) == world_size,
        "missing worker RPC replies",
    )
    expected_keys = {
        "schema_version",
        "adapter",
        "vllm_revision",
        "hash",
        "input",
        "main_event_group",
        "main_event_block_size",
        "gpu_to_store_group",
        "coordinator",
        "groups",
    }
    seen = set()
    agreed = None
    result = None
    for reply in replies:
        _require(type(reply) is dict, "invalid worker RPC reply")
        if "unsupported" in reply:
            raise UnsupportedContract("a worker did not validate the pinned contract")
        _require(
            set(reply) == {"rank", "dp_rank", "descriptor"},
            "invalid worker RPC envelope",
        )
        rank = reply["rank"]
        _require(
            type(rank) is int
            and 0 <= rank < world_size
            and rank not in seen
            and type(reply["dp_rank"]) is int
            and reply["dp_rank"] == dp_rank,
            "duplicate or unexpected worker rank",
        )
        seen.add(rank)
        descriptor = reply["descriptor"]
        _require(
            type(descriptor) is dict
            and set(descriptor) == expected_keys
            and descriptor["schema_version"] == 1
            and descriptor["adapter"] == "vllm-1085b644"
            and descriptor["vllm_revision"] == VLLM_REVISION
            and descriptor["main_event_block_size"] == event_span,
            "worker metadata disagrees with registered main-event span or schema",
        )
        canonical = _canonical(descriptor)
        _require(len(canonical) <= MAX_DESCRIPTOR_BYTES, "descriptor size limit")
        if agreed is None:
            agreed, result = canonical, descriptor
        _require(canonical == agreed, "workers disagree on the resolved store contract")
    _require(result is not None, "empty worker RPC result")
    return result


async def publish_mooncake_store_runtime(
    runtime_config: Any,
    engine: Any,
    vllm_config: Any,
    *,
    dp_range: tuple[int, int],
    event_span: int,
) -> bool:
    """Publish only an explicit, supported, all-rank-agreed connector contract."""
    if getattr(vllm_config.parallel_config, "worker_extension_cls", "") not in (
        WORKER_EXTENSION,
        FPM_WORKER_EXTENSION,
    ):
        return False
    try:
        _require(
            dp_range[1] == 1, "collective_rpc does not return all internal DP engines"
        )
        _verify_pinned_sources()
        _validate_input_config(vllm_config)
        world_size = _positive(vllm_config.parallel_config.world_size)
        _require(world_size <= 1024, "worker RPC rank limit")
        replies = await asyncio.wait_for(
            engine.collective_rpc(RPC_METHOD, timeout=RPC_TIMEOUT_SECONDS),
            timeout=RPC_TIMEOUT_SECONDS,
        )
        descriptor = _agree_descriptors(
            replies,
            world_size=world_size,
            dp_rank=dp_range[0],
            event_span=event_span,
        )
        runtime_config.set_engine_specific(RUNTIME_KEY, _canonical(descriptor))
        return True
    except Exception as exc:  # noqa: BLE001 - optional hints must not stop serving
        # Preserve serving on RPC, optional dependency and private-shape failures.
        reason = (
            str(exc) if isinstance(exc, UnsupportedContract) else type(exc).__name__
        )
        logger.warning("Mooncake Store routing hints disabled: %s", reason[:200])
        return False
