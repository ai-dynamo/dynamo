# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP1 retained-GPU KV cache with CUDA-IPC-only remote delivery.

The primary cache uses vLLM's existing ref-counted cache policy and scheduling
contract, but its storage is a CUDA allocation, never a host mmap. Local stores
and loads are D2D copies; the remote leg is a GPU-only KVCR instance. This raw
layout requires an explicit identical model/layout compatibility domain.
"""

import functools
import logging
import socket
import threading
import time
import types
import uuid
from dataclasses import replace

import msgspec
from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWGPUOptions
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.types import RegionDescriptor, TransferError
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.kvcr.manager import (
    KVCRSecondaryTierManager,
    _FrameworkPinAdapter,
    _kvcr_metric_definitions,
    _VllmKeyAdapter,
)
from vllm.v1.kv_offload.tiering.manager import (
    TieringOffloadingManager,
    _SecondaryTierFacingParent,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from .gpu_memory import GPUCacheRegion, GPUCacheWorker

logger = logging.getLogger(__name__)
_RETAINED_FAILED_GPU_MANAGERS = []


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


class GPUKVCRSecondaryTierManager(KVCRSecondaryTierManager):
    """Transport-only secondary over the retained CUDA primary cache."""

    medium = None
    cache_hit_source = CacheHitSource.P2P

    def __init__(self, spec, region, config):
        SecondaryTierManager.__init__(self, spec, region, "gpu_kvcr")
        domain = config.get("compatibility_domain")
        if domain != spec.extra_config.get("router_hint_transfer_domain"):
            raise ValueError(
                "GPU KVCR domain must match the certified connector domain"
            )
        ports = config.get("control_ports")
        if not isinstance(ports, list) or len(ports) != 1 or type(ports[0]) is not int:
            raise ValueError("GPU KVCR TP1 requires exactly one integer control port")
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
        self._key_adapter = _VllmKeyAdapter()
        self._framework_pin_adapter = _FrameworkPinAdapter(self)
        self._inventory_events = []
        self._finished_jobs = []
        self._jobs_by_op = {}
        self._local_dram_mmap = None
        self._hazard_ops = set()
        self._region = region
        control = ZmqPeerControlChannel(
            config.get("control_host", "0.0.0.0"),
            ports[0],
            config.get("control_advertise_host") or socket.gethostname(),
        )
        with socket.socket() as listener:
            listener.bind(("", 0))
            native_port = listener.getsockname()[1]
        try:
            self._kvcr = KVCR(
                KVCRConfig(
                    pool_layouts=[("", region.shape[1])],
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
                            device_Id=region.tensor.device.index,
                            addr=region.tensor.data_ptr(),
                            count=region.shape[0],
                            size=region.shape[1],
                        )
                    ],
                    remote_fw_gpu=RemoteFWGPUOptions(
                        compatibility_domain=domain,
                        ucx_tls=tuple(
                            config.get("ucx_tls", ("cuda_ipc", "posix", "self"))
                        ),
                    ),
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

    def shutdown(self):
        try:
            # GPU-only KVCR withholds failed destination results until native
            # quiescence, so inherited job completion cannot recycle live DMA.
            super().shutdown()
        except BaseException:
            _RETAINED_FAILED_GPU_MANAGERS.append(self)
            raise


def _serialized(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        with self._gpu_lock:
            result = method(self, *args, **kwargs)
            return tuple(result) if isinstance(result, types.GeneratorType) else result

    return call


class GPUTieringManager(TieringOffloadingManager):
    """Serialize policy access while servicing peers between model steps."""

    def __init__(self, primary, tier):
        self._gpu_lock = threading.RLock()
        self._gpu_stop = threading.Event()
        super().__init__(primary_tier=primary, secondary_tiers=[tier])
        self._gpu_thread = threading.Thread(target=self._poll_gpu, daemon=True)
        self._gpu_thread.start()

    def _poll_gpu(self):
        while not self._gpu_stop.wait(0.001):
            if not self._gpu_lock.acquire(timeout=0.1):
                continue
            try:
                self._processed_jobs_this_step = False
                self._maybe_process_finished_jobs()
                self.secondary_tiers[0].serve_external_requests(
                    _SecondaryTierFacingParent(self, 0)
                )
            except Exception:
                logger.exception(
                    "GPU KV peer progress failed; retaining outstanding leases"
                )
                self._gpu_stop.set()
            finally:
                self._gpu_lock.release()

    lookup = _serialized(TieringOffloadingManager.lookup)
    get_load_source = _serialized(TieringOffloadingManager.get_load_source)
    prepare_load = _serialized(TieringOffloadingManager.prepare_load)
    complete_load = _serialized(TieringOffloadingManager.complete_load)
    prepare_store = _serialized(TieringOffloadingManager.prepare_store)
    complete_store = _serialized(TieringOffloadingManager.complete_store)
    touch = _serialized(TieringOffloadingManager.touch)
    create_store_job = _serialized(TieringOffloadingManager.create_store_job)
    on_new_request = _serialized(TieringOffloadingManager.on_new_request)
    on_request_finished = _serialized(TieringOffloadingManager.on_request_finished)
    on_schedule_end = _serialized(TieringOffloadingManager.on_schedule_end)
    take_events = _serialized(TieringOffloadingManager.take_events)
    get_stats = _serialized(TieringOffloadingManager.get_stats)
    has_pending_work = _serialized(TieringOffloadingManager.has_pending_work)

    @_serialized
    def reset_cache(self):
        primary = self.primary_tier
        active = (
            primary._num_allocated_chunks
            - len(primary._free_list)
            - primary._num_evictable_cache_chunks
        )
        if self._jobs or active or self.secondary_tiers[0]._hazard_ops:
            raise RuntimeError("Cannot reset GPU cache while transfers own its slots")
        return super().reset_cache()

    def shutdown(self):
        self._gpu_stop.set()
        self._gpu_thread.join(timeout=5)
        if self._gpu_thread.is_alive():
            _RETAINED_FAILED_GPU_MANAGERS.append(self)
            raise RuntimeError("GPU peer progress did not stop; retaining GPU buffers")
        with self._gpu_lock:
            super().shutdown()


class GPUOffloadingSpec(OffloadingSpec):
    """Out-of-tree offloading spec; no CPU KV storage or disk tier exists."""

    @classmethod
    def build_metric_definitions(cls, extra_config):
        definitions = {
            _gpu_metric(name): replace(
                value, documentation=value.documentation.replace("CPU", "GPU cache")
            )
            for name, value in TieringOffloadingSpec.build_metric_definitions(
                extra_config
            ).items()
        }
        definitions.update(_kvcr_metric_definitions())
        return definitions

    def __init__(self, config):
        super().__init__(config)
        if (
            config.parallel.world_size != 1
            or config.canonical_layout
            or config.replicated_layout
        ):
            raise ValueError("Direct GPU KV cache currently requires exact-layout TP1")
        if (
            not self.kv_events_config.enable_kv_cache_events
            or not self.kv_events_config.self_describing_kv_events
        ):
            raise ValueError(
                "Direct GPU KV reuse requires self-describing ready inventory"
            )
        capacity = self.extra_config.get("gpu_bytes_to_use")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("gpu_bytes_to_use must be a positive integer")
        if self.extra_config.get("cpu_bytes_to_use") or self.extra_config.get(
            "canonical_layout"
        ):
            raise ValueError(
                "GPU KV cache forbids CPU staging and canonical CPU layouts"
            )
        domain = self.extra_config.get("router_hint_transfer_domain")
        if not isinstance(domain, str) or not domain or domain != domain.strip():
            raise ValueError(
                "GPU KV cache requires an explicit certified layout domain"
            )
        self.row_bytes = config.worker_kv_bytes_per_block * self.blocks_per_chunk
        if self.row_bytes <= 0:
            raise ValueError("GPU KV row geometry must be known at startup")
        self.num_chunks = capacity // self.row_bytes
        if self.num_chunks <= 0:
            raise ValueError("GPU cache budget cannot hold one full KV row")
        self.tiers = self.extra_config.get("secondary_tiers")
        if not isinstance(self.tiers, list) or len(self.tiers) != 1:
            raise ValueError("GPU cache requires exactly one GPU-only KVCR tier")
        if self.tiers[0].get("type") != "GPUKVCRSecondaryTierManager":
            raise ValueError("GPU cache permits only the GPU-only KVCR tier")
        self._manager = self._worker = None

    def get_manager(self):
        if self._manager is None:
            region = GPUCacheRegion(
                self.config.engine_id, self.num_chunks, self.row_bytes
            )
            primary = GPUPrimaryManager(self, region)
            tier = GPUKVCRSecondaryTierManager(self, region, self.tiers[0])
            self._manager = GPUTieringManager(primary, tier)
        return self._manager

    def get_worker(self, kv_caches):
        if self._worker is None:
            self._worker = GPUCacheWorker(self, kv_caches)
        return self._worker


class GPUOffloadingConnector(OffloadingConnector):
    """Publish retained-cache residency as GPU memory, never CPU inventory."""

    def take_events(self):
        for event in super().take_events():
            yield (
                msgspec.structs.replace(event, medium="gpu_direct")
                if isinstance(event, (BlockStored, BlockRemoved))
                else event
            )
