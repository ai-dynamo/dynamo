# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in retained-GPU cache with isolated CUDA IPC or MNNVL delivery.

The primary cache uses vLLM's existing ref-counted cache policy and scheduling
contract, but its storage is a CUDA allocation, never a host mmap. Local stores
and loads are D2D copies; the remote leg is a GPU-only KVCR instance. This raw
layout requires an explicit identical model/layout compatibility domain.
"""

import msgspec
from kvcr.types import BlockKey
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_offload.base import OffloadingEvent, OffloadKey, get_offload_block_hash
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.metrics.cache_hit_source import CacheHitSource


class _VllmKeyAdapter:
    """Keep complete framework keys separate from lossy router hashes."""

    def decode_from_framework(self, framework_key: object) -> BlockKey:
        if not isinstance(framework_key, bytes):
            raise TypeError("vLLM offload keys must be bytes")
        return BlockKey(framework_key)

    def encode_to_framework(self, key: BlockKey) -> OffloadKey:
        return OffloadKey(self.decode_from_framework(key))

    def encode_to_router(self, key: BlockKey) -> int | bytes:
        block_hash = BlockHash(get_offload_block_hash(self.encode_to_framework(key)))
        return maybe_convert_block_hash(block_hash)

    def decode_from_router(self, router_key: int | bytes) -> BlockKey:
        raise NotImplementedError("A router hash cannot recover the KV group identity")

    # Pinned vLLM lookup and older KVCR versions use the two legacy names.
    encode = decode_from_framework
    decode = encode_to_router


def _gpu_metric(name):
    return name.replace("kv_offload_cpu_", "kv_gpu_")


class GPURetainedLoadStoreSpec(CPULoadStoreSpec):
    @staticmethod
    def medium():
        return "GPU_CACHE"


class GPUPrimaryManager(CPUOffloadingManager):
    """Reuse cache-policy metadata; no CPU payload storage is constructed."""

    def __init__(self, spec, region):
        super().__init__(
            num_chunks=spec.num_chunks,
            cache_policy=spec.extra_config.get("eviction_policy", "lru"),
            enable_events=spec.kv_events_config.enable_kv_cache_events,
        )
        self.region = region
        self._published_keys = set()
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

    def get_kv_memoryview(self):
        # The parent tiering manager reads row geometry only. This object has
        # no host buffer protocol; the GPU adapter uses its CUDA tensor.
        return self.region

    def _get_load_store_spec(self, keys, chunks):
        return GPURetainedLoadStoreSpec([chunk.chunk_id for chunk in chunks])

    def get_load_source(self, key, req_context):
        return CacheHitSource.DEVICE

    def prepare_read(self, keys, req_context):
        return self._prepare_load(keys, req_context, record_access=False)

    def take_events(self):
        for event in super().take_events():
            if event.removed:
                self._published_keys.difference_update(event.keys)
            else:
                self._published_keys.update(event.keys)
            yield event

    def reset_cache(self):
        super().reset_cache()
        if self.events is not None:
            # Native Device clears have an independent lifetime. Revoke this
            # retained tier explicitly, and never publish a pending pre-reset
            # store after its physical slots have been invalidated.
            self.events.clear()
            if self._published_keys:
                self.events.append(
                    OffloadingEvent(
                        keys=sorted(self._published_keys),
                        medium=self.medium,
                        removed=True,
                    )
                )
            # Keep the set until take_events consumes the removal, so a
            # repeated reset cannot discard a still-unpublished revocation.

    def get_stats(self):
        stats = super().get_stats()
        for name in list(stats._types):
            new_name = _gpu_metric(name)
            if new_name != name:
                stats._types[new_name] = stats._types.pop(name)
                stats._values[new_name] = stats._values.pop(name)
        return stats

    def shutdown(self):
        self.region.cleanup()


class GPUOffloadingConnector(OffloadingConnector):
    """Publish retained-cache residency as GPU memory, never CPU inventory."""

    def reset_cache(self):
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        tracker = scheduler._events_tracker
        published = scheduler.manager.primary_tier._published_keys
        removal_metadata = {
            key: tracker._pending_event_metadata[key]
            for key in published
            if key in tracker._pending_event_metadata
        }
        result = super().reset_cache()
        # The pinned scheduler clears its tracker after manager.reset_cache.
        # Preserve only metadata needed for queued published-key removals, so
        # multi-block chunks revoke every constituent hash, without reviving
        # pending stores or any physical cache state.
        tracker._pending_event_metadata.update(removal_metadata)
        return result

    def take_events(self):
        for event in super().take_events():
            yield (
                msgspec.structs.replace(event, medium="gpu_direct")
                if isinstance(event, (BlockStored, BlockRemoved))
                else event
            )
