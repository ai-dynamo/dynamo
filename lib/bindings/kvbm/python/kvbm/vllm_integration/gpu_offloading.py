# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in retained-GPU cache with isolated CUDA IPC or MNNVL delivery.

The primary cache uses vLLM's existing ref-counted cache policy and scheduling
contract, but its storage is a CUDA allocation, never a host mmap. Local stores
and loads are D2D copies; the remote leg is a GPU-only KVCR instance. This raw
layout requires an explicit identical model/layout compatibility domain.
"""

import logging
import socket
import time
import uuid

import msgspec
from kvcr import KVCR, KVCRBindings, ZmqPeerControlChannel
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWGPUOptions
from kvcr.types import BlockKey, MemoryRef, RegionDescriptor, TransferError
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_offload.base import (
    OffloadingEvent,
    OffloadKey,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.kvcr.manager import (
    KVCRSecondaryTierManager,
    _FrameworkPinAdapter,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

logger = logging.getLogger(__name__)


_RETAINED_FAILED_GPU_MANAGERS = []


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


class _GPUFrameworkPinAdapter(_FrameworkPinAdapter):
    """Pin one logical slot once, then expose every tensor-parallel part."""

    def _resolve_pin(self, parent, keys):
        result = super()._resolve_pin(parent, keys)
        if result is None:
            return None
        pin_handle, descriptors = result
        try:
            expanded = {}
            for key, refs in descriptors.items():
                if refs is None:
                    expanded[key] = None
                    continue
                if len(refs) != 1:
                    raise ValueError("Expected one logical retained-GPU slot")
                expanded[key] = self._tier._make_descriptors(refs[0].element_index)
            return pin_handle, expanded
        except BaseException:
            # The inherited adapter has already pinned the primary slot. Its
            # completion must be returned even when multipart validation fails.
            self.release_pin(pin_handle)
            raise


class GPUKVCRSecondaryTierManager(KVCRSecondaryTierManager):
    """Transport-only secondary over the retained CUDA primary cache."""

    medium = None
    cache_hit_source = CacheHitSource.P2P

    def __init__(self, spec, region, config):
        SecondaryTierManager.__init__(self, spec, region, "gpu_kvcr")
        transport = config.get("gpu_transport", "cuda_ipc")
        if transport != spec.gpu_transport:
            raise ValueError("GPU tier and retained bank transport profiles differ")
        domain = config.get("compatibility_domain")
        if domain != spec.extra_config.get("router_hint_transfer_domain"):
            raise ValueError(
                "GPU KVCR domain must match the certified connector domain"
            )
        ports = config.get("control_ports")
        if not isinstance(ports, list) or len(ports) != 1 or type(ports[0]) is not int:
            raise ValueError(
                "GPU KVCR requires exactly one integer control port per engine"
            )
        if not 1 <= ports[0] <= 65535:
            raise ValueError("GPU KVCR control port is out of range")
        if config.get("router_capabilities") != ["router_hint_gpu_direct"]:
            raise ValueError(
                "GPU KVCR requires its explicit GPU-only router capability"
            )
        if any(
            config.get(name)
            for name in (
                "secondary_g2_slots",
                "g3",
                "kvcr_service_socket_path",
                "local_dram",
            )
        ):
            raise ValueError("GPU KVCR forbids CPU, disk and Guard payload storage")
        remote_options = {
            "compatibility_domain": domain,
            "ucx_tls": tuple(config.get("ucx_tls", ("cuda_ipc", "posix", "self"))),
        }
        if transport == "mnnvl":
            if tuple(config.get("ucx_tls", ())) != ("cuda_ipc", "tcp", "self"):
                raise ValueError(
                    "MNNVL requires explicit cuda_ipc,tcp,self context TLS"
                )
            remote_options.update(
                gpu_transport="mnnvl", fabric_domain=spec.fabric_domain
            )
        self._key_adapter = _VllmKeyAdapter()
        self._framework_pin_adapter = _GPUFrameworkPinAdapter(self)
        self._inventory_events = []
        self._finished_jobs = []
        self._jobs_by_op = {}
        self._local_dram_mmap = None
        self._hazard_ops = set()
        self._region = region
        banks = getattr(region, "banks", (region,))
        self._part_labels = (
            tuple(f":rank{rank}" for rank in range(len(banks)))
            if len(banks) > 1
            else ("",)
        )
        if len(banks) != spec.config.parallel.world_size:
            raise ValueError("Retained GPU bank count differs from tensor parallelism")
        if any(bank.shape != (spec.num_chunks, spec.row_bytes) for bank in banks):
            raise ValueError("Retained GPU rank banks have different row geometry")
        control = ZmqPeerControlChannel(
            config.get("control_host", "0.0.0.0"),
            ports[0],
            config.get("control_advertise_host") or socket.gethostname(),
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            native_port = listener.getsockname()[1]
        try:
            self._kvcr = KVCR(
                KVCRConfig(
                    pool_layouts=[("", spec.row_bytes)],
                    enable_telemetry=True,
                    operation_timeout_ms=config.get("operation_timeout_ms", 3000),
                    abandon_timeout_ms=config.get("abandon_timeout_ms", 5000),
                    nixl_agent_name=f"GPU-KVCR-{uuid.uuid4()}",
                    nixl_listen_port=native_port,
                ),
                KVCRBindings(
                    request_pin=self._framework_pin_adapter.request_pin,
                    poll_pin_results=self._framework_pin_adapter.poll_pin_results,
                    release_pin=self._framework_pin_adapter.release_pin,
                    cancel_pin_request=self._framework_pin_adapter.cancel_pin_request,
                    framework_control=control,
                    key_adapter=self._key_adapter,
                    stats_factory=OffloadingConnectorStats,
                    on_resilience_event=self._resilience_event,
                ),
                KVCRBackendConfigs(
                    framework_regions=[
                        RegionDescriptor(
                            mem_type="VRAM",
                            device_Id=bank.tensor.device.index,
                            addr=bank.tensor.data_ptr(),
                            count=bank.shape[0],
                            size=bank.shape[1],
                            label=label,
                        )
                        for label, bank in zip(self._part_labels, banks, strict=True)
                    ],
                    remote_fw_gpu=RemoteFWGPUOptions(**remote_options),
                ),
            )
        except BaseException:
            # Startup may have left native work alive; do not free CUDA storage
            # beneath it. Process exit is the final ownership boundary.
            _RETAINED_FAILED_GPU_MANAGERS.append(self)
            raise

    def drain_jobs(self):
        deadline = time.monotonic() + 30
        while self._jobs_by_op or self._framework_pin_adapter.has_active_pins():
            self._finished_jobs.extend(self._poll_finished_jobs())
            if time.monotonic() >= deadline:
                _RETAINED_FAILED_GPU_MANAGERS.append(self)
                raise TimeoutError(
                    "GPU DMA did not quiesce; retaining its CUDA allocation"
                )
            time.sleep(0.001)

    def _resilience_event(self, event):
        if isinstance(event, TransferError):
            key = (event.op_handle, event.destination_owner is not None)
            if event.state == "uncertain":
                self._hazard_ops.add(key)
            elif event.state == "quiesced":
                self._hazard_ops.discard(key)
        logger.error("GPU KVCR lifecycle event: %s", event)

    def submit_store(self, job_metadata):
        # The primary already owns the completed GPU copy. Do not deposit to
        # any KVCR-managed host/disk pool or make an additional payload copy.
        self._finished_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def on_new_request(self, req_context):
        if req_context.kv_hints is not None:
            self._kvcr.submit_hint(
                request_id=req_context.req_id, hints=req_context.kv_hints
            )
        return RequestOffloadingContext()

    def _make_descriptors(self, chunk_id):
        if type(chunk_id) is not int or not 0 <= chunk_id < self._region.shape[0]:
            raise ValueError("Retained GPU slot is outside the registered banks")
        return [
            MemoryRef(
                end_point_name=self._kvcr.config.nixl_agent_name,
                element_index=chunk_id,
                label=label,
            )
            for label in self._part_labels
        ]

    def submit_load(self, job_metadata):
        # KVCR owns one operation for all rank parts of each key. A partial
        # transfer cannot publish a logical block or release its primary slot.
        blocks = {
            self._key_adapter.decode_from_framework(key): self._make_descriptors(
                int(chunk_id)
            )
            for key, chunk_id in zip(
                job_metadata.keys, job_metadata.chunk_ids, strict=True
            )
        }
        if not blocks:
            self._finished_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=True)
            )
            return
        op_handle = self._kvcr.deliver(
            blocks, request_id=job_metadata.req_context.req_id
        )
        self._jobs_by_op[op_handle] = (job_metadata.job_id, len(blocks), True, set())

    def shutdown(self):
        try:
            # GPU-only KVCR withholds failed destination results until native
            # quiescence, so inherited job completion cannot recycle live DMA.
            super().shutdown()
        except BaseException:
            _RETAINED_FAILED_GPU_MANAGERS.append(self)
            raise


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
