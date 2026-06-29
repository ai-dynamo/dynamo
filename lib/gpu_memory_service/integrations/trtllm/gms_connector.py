# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM GMS KV connector.

This module mirrors the vLLM/SGLang GMS connector contracts while keeping
TensorRT-LLM imports lazy. The scheduler side owns prefix lookup and metadata
production; the worker side owns GMS block movement through ``BlockGdsConnector``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from gms_kv_ring.common.prefix_hashes import prefix_block_hashes
from gms_kv_ring.common.prefix_index import PrefixIndex
from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
from gms_kv_ring.engines.trtllm.install_kv_ring import install_for_trtllm

logger = logging.getLogger(__name__)
_META_VERSION = 6


@dataclass
class GMSTrtllmConnectorMeta:
    evict_queue: list[tuple[str, list[int], list[int], list[bytes]]] = field(
        default_factory=list
    )
    restore_queue: list[tuple[str, list[tuple[int, int, int]]]] = field(
        default_factory=list
    )
    staging_restore_queue: list[tuple[str, list[tuple[bytes, int, int]]]] = field(
        default_factory=list
    )


def _encode_meta(meta: GMSTrtllmConnectorMeta) -> bytes:
    payload: dict[str, Any] = {
        "v": _META_VERSION,
        "evict": [
            [str(rid), [int(x) for x in bids], [int(g) for g in gens]]
            for rid, bids, gens, _hashes in meta.evict_queue
        ],
        "restore": [
            [str(rid), [[int(a), int(b), int(c)] for a, b, c in triples]]
            for rid, triples in meta.restore_queue
        ],
        "restore_staging": [
            [
                str(rid),
                [[bytes(h).hex(), int(dst), int(gen)] for h, dst, gen in triples],
            ]
            for rid, triples in meta.staging_restore_queue
        ],
    }
    hashes_per_evict = [
        [bytes(h).hex() for h in hashes]
        for _rid, _bids, _gens, hashes in meta.evict_queue
    ]
    if any(hashes_per_evict):
        payload["hashes_per_evict"] = hashes_per_evict
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _decode_meta(
    payload: bytes | bytearray | memoryview | None,
) -> GMSTrtllmConnectorMeta:
    if not payload:
        return GMSTrtllmConnectorMeta()
    obj = json.loads(bytes(payload).decode("utf-8"))
    if int(obj.get("v", 0)) != _META_VERSION:
        raise ValueError(f"unsupported TRT-LLM GMS metadata version: {obj.get('v')!r}")
    hashes_per_evict = obj.get("hashes_per_evict") or []
    evict_queue: list[tuple[str, list[int], list[int], list[bytes]]] = []
    for idx, raw in enumerate(obj.get("evict") or []):
        rid, bids, gens = raw
        raw_hashes = hashes_per_evict[idx] if idx < len(hashes_per_evict) else []
        evict_queue.append(
            (
                str(rid),
                [int(x) for x in bids],
                [int(g) for g in gens],
                [bytes.fromhex(h) for h in raw_hashes],
            )
        )
    restore_queue = [
        (
            str(rid),
            [(int(src), int(dst), int(gen)) for src, dst, gen in triples],
        )
        for rid, triples in (obj.get("restore") or [])
    ]
    staging_restore_queue = [
        (
            str(rid),
            [(bytes.fromhex(h), int(dst), int(gen)) for h, dst, gen in triples],
        )
        for rid, triples in (obj.get("restore_staging") or [])
    ]
    return GMSTrtllmConnectorMeta(evict_queue, restore_queue, staging_restore_queue)


def _build_block_layout_trtllm(kv_cache_tensor: Any):
    if not hasattr(kv_cache_tensor, "shape") or not hasattr(
        kv_cache_tensor, "data_ptr"
    ):
        raise TypeError("expected a torch.Tensor-like TRT-LLM KV cache tensor")
    shape = tuple(int(x) for x in kv_cache_tensor.shape)
    if len(shape) < 4:
        raise ValueError("TRT-LLM KV cache tensor must be at least 4D")
    num_blocks, num_layers, kv_factor = shape[:3]
    elems_per_kv = 1
    for dim in shape[3:]:
        elems_per_kv *= int(dim)
    elem_size = int(kv_cache_tensor.element_size())
    bytes_per_kv = elems_per_kv * elem_size
    flat_layers = num_layers * kv_factor
    block_bytes_per_block = flat_layers * bytes_per_kv
    base = int(kv_cache_tensor.data_ptr())
    layers = [
        {
            "layer_idx": flat,
            "va": base + flat * bytes_per_kv,
            "size": int(num_blocks) * block_bytes_per_block,
            "stride": block_bytes_per_block,
        }
        for flat in range(flat_layers)
    ]

    def layout(block_id: int) -> list[tuple[int, int, int]]:
        block_base = int(block_id) * block_bytes_per_block
        return [
            (flat, block_base + flat * bytes_per_kv, bytes_per_kv)
            for flat in range(flat_layers)
        ]

    return layers, layout, flat_layers, block_bytes_per_block


def _request_tokens(req: Any) -> Optional[list[int]]:
    try:
        return [int(x) for x in req.get_tokens(0)]
    except Exception:
        tokens = getattr(req, "prompt_token_ids", None)
        if tokens is None:
            return None
        try:
            return [int(x) for x in tokens]
        except Exception:
            return None


def _request_salt(req: Any) -> Optional[str]:
    salt = getattr(req, "cache_salt_id", None)
    if salt is None:
        salt = getattr(req, "cache_salt", None)
    return None if salt is None else str(salt)


class GMSTrtllmKvCacheScheduler:
    def __init__(self, llm_args: Any) -> None:
        self.llm_args = llm_args
        self.block_size = int(llm_args.kv_cache_config.tokens_per_block)
        self.engine_id = os.environ.get("GMS_TRTLLM_ENGINE_ID", "trtllm")
        self.source_engine_id = os.environ.get(
            "GMS_TRTLLM_SOURCE_ENGINE_ID", self.engine_id
        )
        snapshot_path = os.environ.get("GMS_TRTLLM_PREFIX_INDEX_SNAPSHOT")
        self._prefix_index = PrefixIndex(snapshot_path=snapshot_path)
        self._sched_evict_queue: list[
            tuple[str, list[int], list[int], list[bytes]]
        ] = []
        self._sched_restore_queue: list[tuple[str, list[tuple[int, int, int]]]] = []
        self._sched_staging_restore_queue: list[
            tuple[str, list[tuple[bytes, int, int]]]
        ] = []
        self._pending_restore: dict[str, list[tuple[int, int]]] = {}
        self._pending_staging: dict[str, list[tuple[bytes, int]]] = {}
        self._cross_node_xfer_enabled = os.environ.get("GMS_KVR_CROSS_NODE") == "1"

    def _staging_scan(self, _hashes: list[bytes]) -> dict[bytes, dict]:
        return {}

    def get_num_new_matched_tokens(self, req: Any, num_computed_tokens: int):
        tokens = _request_tokens(req)
        if not tokens:
            return 0, False
        req_id = str(getattr(req, "request_id", id(req)))
        salt = _request_salt(req)
        local = self._prefix_index.lookup(
            tokens, self.block_size, salt, self.source_engine_id
        )
        if local:
            self._pending_restore[req_id] = local
            return len(local) * self.block_size, False
        if self._cross_node_xfer_enabled:
            hashes = prefix_block_hashes(tokens, self.block_size, salt)
            hits = self._staging_scan(hashes)
            staging: list[tuple[bytes, int]] = []
            for h in hashes:
                hit = hits.get(h)
                if hit is None:
                    break
                staging.append((h, int(hit.get("generation", 0))))
            if staging:
                self._pending_staging[req_id] = staging
                return len(staging) * self.block_size, False
        return 0, False

    def update_state_after_alloc(self, req: Any, block_ids: list[int]) -> None:
        req_id = str(getattr(req, "request_id", id(req)))
        if req_id in self._pending_staging:
            entries = self._pending_staging.pop(req_id)
            triples = [
                (h, int(dst), int(gen)) for (h, gen), dst in zip(entries, block_ids)
            ]
            if triples:
                self._sched_staging_restore_queue.append((req_id, triples))
            return
        entries = self._pending_restore.pop(req_id, [])
        triples = [
            (int(src), int(dst), int(gen))
            for (src, gen), dst in zip(entries, block_ids)
        ]
        if triples:
            self._sched_restore_queue.append((req_id, triples))

    def request_finished(self, req: Any, block_ids: list[int]) -> bool:
        tokens = _request_tokens(req)
        if not tokens or not block_ids:
            return False
        req_id = str(getattr(req, "request_id", id(req)))
        salt = _request_salt(req)
        bids = [int(x) for x in block_ids]
        generations = self._prefix_index.record(
            tokens, bids, self.block_size, salt, self.engine_id
        )
        hashes = []
        if self._cross_node_xfer_enabled:
            hashes = prefix_block_hashes(tokens, self.block_size, salt)[: len(bids)]
        self._sched_evict_queue.append((req_id, bids, generations, hashes))
        return True

    def shutdown(self) -> None:
        self._prefix_index.snapshot()

    def build_connector_meta(self, _scheduler_output: Any) -> GMSTrtllmConnectorMeta:
        meta = GMSTrtllmConnectorMeta(
            evict_queue=list(self._sched_evict_queue),
            restore_queue=list(self._sched_restore_queue),
            staging_restore_queue=list(self._sched_staging_restore_queue),
        )
        self._sched_evict_queue.clear()
        self._sched_restore_queue.clear()
        self._sched_staging_restore_queue.clear()
        return meta

    def reclaim_cached_kv(self, target_free_blocks: int) -> int:
        from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

        return int(install_kv_leases_v2.reclaim_cached_kv_v2(int(target_free_blocks)))


class GMSTrtllmKvCacheWorker:
    def __init__(self, llm_args: Any) -> None:
        self.llm_args = llm_args
        self.block_size = int(llm_args.kv_cache_config.tokens_per_block)
        self.engine_id = os.environ.get("GMS_TRTLLM_ENGINE_ID", "trtllm")
        self.source_engine_id = os.environ.get(
            "GMS_TRTLLM_SOURCE_ENGINE_ID", self.engine_id
        )
        self.daemon_socket = os.environ.get("GMS_TRTLLM_DAEMON_SOCKET", "")
        self._cross_node_xfer_enabled = os.environ.get("GMS_KVR_CROSS_NODE") == "1"
        self._host_tier_fallback = os.environ.get("GMS_KVR_HOST_TIER_FALLBACK") == "1"
        self._meta = GMSTrtllmConnectorMeta()
        self._gds_conn: Optional[BlockGdsConnector] = None
        self._handle = None
        self._block_layout = None
        self._n_layers = 0
        self._block_bytes = 0
        self._finished_saves: set[str] = set()
        self._finished_loads: set[str] = set()

    def register_kv_caches(self, kv_cache_tensor: Any) -> None:
        if isinstance(kv_cache_tensor, dict):
            if not kv_cache_tensor:
                raise ValueError("register_kv_caches: empty kv_caches dict")
            kv_cache_tensor = next(iter(kv_cache_tensor.values()))
        layers, layout, n_layers, block_bytes = _build_block_layout_trtllm(
            kv_cache_tensor
        )
        self._block_layout = layout
        self._n_layers = n_layers
        self._block_bytes = block_bytes
        if self.daemon_socket:
            self._handle = install_for_trtllm(
                engine_id=self.engine_id,
                daemon_socket=self.daemon_socket,
                layers=layers,
            )
            self._gds_conn = BlockGdsConnector(self._handle, layout)

    def bind_connector_meta(self, meta: GMSTrtllmConnectorMeta | bytes) -> None:
        self._meta = (
            _decode_meta(meta) if isinstance(meta, (bytes, bytearray)) else meta
        )

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        payload = getattr(connector_metadata, "payload", connector_metadata)
        self.bind_connector_meta(payload)

    def clear_connector_metadata(self) -> None:
        self._meta = GMSTrtllmConnectorMeta()

    def _register_cross_node(self, block_ids: list[int], hashes: list[bytes]) -> None:
        if not self._cross_node_xfer_enabled or self._handle is None or not hashes:
            return
        if self._block_layout is None:
            return
        items = []
        for bid, h in zip(block_ids, hashes):
            items.append({"content_hash": h, "ranges": self._block_layout(int(bid))})
        try:
            _total, skipped = self._handle._client.register_content_addresses_batch(
                items
            )
        except Exception:  # noqa: BLE001
            logger.warning("TRT-LLM GMS cross-node registration failed", exc_info=True)
            self._cross_node_xfer_enabled = False
            return
        if skipped:
            self._cross_node_xfer_enabled = False

    def wait_for_save(self, _stream: Any) -> None:
        for req_id, block_ids, generations, hashes in self._meta.evict_queue:
            gens = {int(b): int(g) for b, g in zip(block_ids, generations)}
            if self._gds_conn is None:
                self._finished_saves.add(str(req_id))
                continue
            if self._gds_conn.is_available():
                self._gds_conn.evict_blocks_to_storage(block_ids, generations=gens)
                self._register_cross_node(block_ids, hashes)
            elif self._host_tier_fallback:
                self._gds_conn.evict_blocks_to_host(block_ids, generations=gens)
            self._finished_saves.add(str(req_id))

    def start_load_kv(self, _stream: Any) -> None:
        for req_id, triples in self._meta.restore_queue:
            if self._gds_conn is not None:
                if self._gds_conn.is_available():
                    self._gds_conn.restore_blocks_remap(triples)
                elif self._host_tier_fallback:
                    self._gds_conn.restore_blocks_remap_from_host(
                        self.source_engine_id, triples
                    )
            self._finished_loads.add(str(req_id))
        for req_id, triples in self._meta.staging_restore_queue:
            ok = False
            if self._handle is not None:
                ok = bool(self._handle.restore_staging_blocks_sync(triples))
            if ok:
                self._finished_loads.add(str(req_id))

    def get_finished(
        self, finished_save_req_ids: list[str], finished_load_req_ids: list[str]
    ):
        saves = [
            rid for rid in finished_save_req_ids if str(rid) in self._finished_saves
        ]
        loads = [
            rid for rid in finished_load_req_ids if str(rid) in self._finished_loads
        ]
        for rid in saves:
            self._finished_saves.discard(str(rid))
        for rid in loads:
            self._finished_loads.discard(str(rid))
        return saves, loads

    def allocate_kv_caches(self, shape, dtype, device):
        """Persistent VMM allocation hook used by the TRT-LLM engine patch."""
        import torch

        socket = os.environ.get("GMS_TRTLLM_VMM_IPC_SOCKET", "")
        engine_id = os.environ.get(
            "GMS_TRTLLM_VMM_IPC_ENGINE_ID",
            os.environ.get("GMS_TRTLLM_ENGINE_ID", self.engine_id),
        )
        if not socket:
            raise RuntimeError("persistent KV allocation failed: missing GMS socket")
        try:
            dev_idx = 0
            if hasattr(device, "index") and device.index is not None:
                dev_idx = int(device.index)
            import gpu_memory_service.client.torch.allocator as alloc_mod

            alloc_mod.get_or_create_persistent_allocator(
                socket, dev_idx, engine_id, tag="kv_pool", shared=False
            )
            with alloc_mod.gms_use_persistent_pool("kv_pool", dev_idx):
                out = torch.empty(shape, dtype=dtype, device=device)
            self._vmm_ipc_enabled = True
            return out
        except Exception as exc:  # noqa: BLE001
            self._vmm_ipc_enabled = False
            raise RuntimeError("persistent KV allocation failed") from exc
