# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-owned retained KV storage and local device-to-device cache copies.

Only IPC handles and copy descriptors cross host memory. KV payloads stay on
CUDA. The scheduler owns the allocation; a model worker imports its selected
CUDA IPC or FABRIC handle lazily after scheduler construction.
"""

import base64
import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from vllm import _custom_ops as ops
from vllm.v1.kv_offload.base import OffloadingWorker
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

from . import gpu_mnnvl


def region_path(engine_id: str) -> Path:
    digest = hashlib.sha256(engine_id.encode()).hexdigest()
    return Path(tempfile.gettempdir()) / f"dynamo-gpu-kv-{os.getuid()}-{digest}"


def export_ipc(tensor: torch.Tensor) -> dict:
    if not tensor.is_cuda or tensor.dtype != torch.int8 or tensor.ndim != 2:
        raise ValueError("GPU KV storage must be a two-dimensional CUDA byte tensor")
    rebuild, args = reduce_tensor(tensor)
    if rebuild is not rebuild_cuda_tensor or len(args) != 15:
        raise RuntimeError("Unsupported PyTorch CUDA IPC reduction contract")
    # Use a closed JSON schema, rather than deserializing executable pickles.
    if args[0] is not torch.Tensor or args[4] is not torch.storage.TypedStorage:
        raise RuntimeError("Unsupported CUDA IPC tensor/storage class")
    return {
        "schema": 1,
        "pid": os.getpid(),
        "gpu_uuid": str(torch.cuda.get_device_properties(tensor.device).uuid),
        "shape": list(args[1]),
        "stride": list(args[2]),
        "offset": args[3],
        "handle": base64.b64encode(args[7]).decode(),
        "storage_size_bytes": args[8],
        "storage_offset_bytes": args[9],
        "ref_handle": base64.b64encode(args[11]).decode(),
        "ref_offset": args[12],
        "event_handle": base64.b64encode(args[13]).decode(),
        "event_sync": args[14],
    }


def import_ipc(data: dict, shape: tuple[int, int]) -> torch.Tensor:
    if data.get("schema") != 1 or data.get("shape") != list(shape):
        raise ValueError("GPU cache IPC layout differs from the local configuration")
    device = torch.cuda.current_device()
    if data.get("gpu_uuid") != str(torch.cuda.get_device_properties(device).uuid):
        raise ValueError("GPU cache owner and worker must use the same physical GPU")
    decode = lambda name: base64.b64decode(data[name], validate=True)
    result = rebuild_cuda_tensor(
        torch.Tensor,
        tuple(data["shape"]),
        tuple(data["stride"]),
        data["offset"],
        torch.storage.TypedStorage,
        torch.int8,
        device,
        decode("handle"),
        data["storage_size_bytes"],
        data["storage_offset_bytes"],
        False,
        decode("ref_handle"),
        data["ref_offset"],
        decode("event_handle"),
        data["event_sync"],
    )
    if not result.is_cuda or result.dtype != torch.int8 or tuple(result.shape) != shape:
        raise ValueError("CUDA IPC did not produce the required GPU cache")
    return result


class GPUCacheRegion:
    """One TP1 CUDA allocation, owned for the complete KVCR lifetime."""

    gpu_transport = "cuda_ipc"

    def __init__(self, engine_id: str, chunks: int, row_bytes: int):
        if chunks <= 0 or row_bytes <= 0:
            raise ValueError("GPU cache capacity must contain a whole positive row")
        self.tensor = torch.zeros((chunks, row_bytes), dtype=torch.int8, device="cuda")
        self.tensor.record_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.shape = (chunks, row_bytes)
        self.strides = (row_bytes, 1)
        self.path = region_path(engine_id)
        self.path.mkdir(mode=0o700)
        self.metadata = export_ipc(self.tensor)
        self._local_export_consumed = False
        self._local_export_lock = threading.Lock()
        self._closed = False
        fd = os.open(
            self.path / "ipc.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        with os.fdopen(fd, "w") as output:
            json.dump(self.metadata, output)
        _OWNED_GPU_REGIONS[self.path] = self

    def __len__(self):
        return self.shape[0]

    def consume_local_export(self):
        with self._local_export_lock:
            self._consume_local_export()

    def _consume_local_export(self):
        if self._local_export_consumed:
            return
        # reduce_tensor creates one producer reference for a future importer.
        # UniProcExecutor already owns the storage, so balance that unused
        # reference just as PyTorch's cached-storage rebuild branch does.
        torch.storage.TypedStorage._release_ipc_counter(
            base64.b64decode(self.metadata["ref_handle"], validate=True),
            self.metadata["ref_offset"],
            device=self.tensor.device.index,
        )
        self._local_export_consumed = True
        # This export has been consumed locally and cannot also serve an IPC
        # worker. The TP1 worker keeps using the original allocation.
        (self.path / "ipc.json").unlink()

    def cleanup(self):
        # The caller has drained local/native work. Local views own their
        # storage; PyTorch's IPC references retain it for external consumers.
        # Keeping the producer Tensor until static allocator teardown would
        # instead leave its IPC bookkeeping alive past the owning singleton.
        if self._closed:
            return
        self._closed = True
        (self.path / "ipc.json").unlink(missing_ok=True)
        self.path.rmdir()
        self.tensor = None


def _process_start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])


def _fabric_tensor(owner):
    tensor = torch.as_tensor(
        owner, dtype=torch.int8, device=torch.device("cuda", owner.device)
    )
    if (
        not tensor.is_cuda
        or tensor.dtype != torch.int8
        or tensor.device.index != owner.device
        or tuple(tensor.shape) != owner.shape
        or tuple(tensor.stride()) != (owner.shape[1], 1)
        or tensor.data_ptr() != owner.pointer
    ):
        raise RuntimeError(
            "PyTorch must alias the exact FABRIC GPU bank without a copy"
        )
    return tensor


class GPUFabricCacheRegion:
    """An isolated FABRIC VMM bank; never changes the model's CUDA allocator."""

    gpu_transport = "mnnvl"

    def __init__(self, engine_id, chunks, row_bytes, base_domain):
        self.shape = (chunks, row_bytes)
        self.strides = (row_bytes, 1)
        self._owner = gpu_mnnvl.FabricAllocation(
            self.shape, device=torch.cuda.current_device()
        )
        self.fabric_domain = self._owner.identity.domain
        self.compatibility_domain = gpu_mnnvl.effective_domain(
            base_domain, self.fabric_domain
        )
        self.tensor = _fabric_tensor(self._owner)
        self._closed = False
        self.path = region_path(engine_id)
        self.path.mkdir(mode=0o700)
        self.metadata = {
            "schema": "dynamo-gpu-fabric-v1",
            "gpu_transport": "mnnvl",
            "engine_id": engine_id,
            "owner_pid": os.getpid(),
            "owner_start_ticks": _process_start_ticks(os.getpid()),
            "gpu_uuid": self._owner.identity.gpu_uuid,
            "fabric_domain": self.fabric_domain,
            "base_domain": base_domain,
            "compatibility_domain": self.compatibility_domain,
            "shape": list(self.shape),
            "stride": list(self.strides),
            "logical_bytes": self._owner.logical_bytes,
            "allocation_bytes": self._owner.allocation_bytes,
            "fabric_handle": base64.b64encode(self._owner.fabric_handle).decode(),
        }
        temporary = self.path / "ipc.json.pending"
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(self.metadata, output)
            output.flush()
            os.fsync(output.fileno())
        os.rename(temporary, self.path / "ipc.json")
        _OWNED_GPU_REGIONS[self.path] = self

    def __len__(self):
        return self.shape[0]

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        (self.path / "ipc.json").unlink(missing_ok=True)
        self.path.rmdir()
        # Native owners must have quiesced first. Tensor storage retains the
        # mapping owner for any remaining views; the owner synchronizes local
        # CUDA work before it unmaps at the final reference boundary.
        self.tensor = None
        self._owner = None


def import_fabric(
    data, engine_id, shape, base_domain, compatibility_domain, fabric_domain
):
    expected = {
        "schema",
        "gpu_transport",
        "engine_id",
        "owner_pid",
        "owner_start_ticks",
        "gpu_uuid",
        "fabric_domain",
        "base_domain",
        "compatibility_domain",
        "shape",
        "stride",
        "logical_bytes",
        "allocation_bytes",
        "fabric_handle",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Closed local FABRIC descriptor schema required")
    identity = gpu_mnnvl.observe_fabric(torch.cuda.current_device())
    if (
        data["schema"] != "dynamo-gpu-fabric-v1"
        or data["gpu_transport"] != "mnnvl"
        or data["engine_id"] != engine_id
        or data["shape"] != list(shape)
        or not isinstance(data["shape"], list)
        or any(type(value) is not int for value in data["shape"])
        or data["stride"] != [shape[1], 1]
        or not isinstance(data["stride"], list)
        or any(type(value) is not int for value in data["stride"])
        or data["gpu_uuid"] != identity.gpu_uuid
        or data["fabric_domain"] != fabric_domain
        or fabric_domain != identity.domain
        or data["base_domain"] != base_domain
        or data["compatibility_domain"] != compatibility_domain
        or compatibility_domain
        != gpu_mnnvl.effective_domain(base_domain, identity.domain)
        or type(data["logical_bytes"]) is not int
        or data["logical_bytes"] != shape[0] * shape[1]
        or type(data["allocation_bytes"]) is not int
        or not data["allocation_bytes"] >= data["logical_bytes"] > 0
        or type(data["owner_pid"]) is not int
        or data["owner_pid"] <= 0
        or type(data["owner_start_ticks"]) is not int
        or data["owner_start_ticks"] != _process_start_ticks(data["owner_pid"])
    ):
        raise ValueError(
            "Local FABRIC bank GPU, owner, layout or transport domain differs"
        )
    if not isinstance(data["fabric_handle"], str) or len(data["fabric_handle"]) > 128:
        raise ValueError("Bounded base64 FABRIC metadata required")
    handle = base64.b64decode(data["fabric_handle"], validate=True)
    if len(handle) != 64:
        raise ValueError("Local FABRIC handle must contain exactly64 bytes")
    owner = gpu_mnnvl.FabricAllocation(
        shape,
        device=torch.cuda.current_device(),
        imported_handle=handle,
        allocation_bytes=data["allocation_bytes"],
    )
    if owner.identity != identity:
        raise RuntimeError("Assigned GPU/fabric changed during local FABRIC import")
    return _fabric_tensor(owner)


_OWNED_GPU_REGIONS: dict[Path, GPUCacheRegion | GPUFabricCacheRegion] = {}


def open_region(
    engine_id: str,
    shape: tuple[int, int],
    timeout: float = 30,
    *,
    gpu_transport="cuda_ipc",
    base_domain=None,
    compatibility_domain=None,
    fabric_domain=None,
):
    if gpu_transport not in ("cuda_ipc", "mnnvl"):
        raise ValueError("Unknown retained GPU transport")
    directory = region_path(engine_id)
    owned = _OWNED_GPU_REGIONS.get(directory)
    if owned is not None:
        if owned.gpu_transport != gpu_transport:
            raise ValueError("Local GPU bank uses a different transport profile")
        if owned.shape != shape or owned._closed:
            raise ValueError("Local GPU cache owner has a different or closed layout")
        if gpu_transport == "mnnvl":
            if (
                owned.fabric_domain != fabric_domain
                or owned.metadata["base_domain"] != base_domain
                or owned.compatibility_domain != compatibility_domain
            ):
                raise ValueError("Local MNNVL bank domain differs")
            return owned.tensor
        # UniProcExecutor runs the model worker and scheduler in one process.
        # CUDA forbids importing an IPC handle exported by that same process.
        owned.consume_local_export()
        return owned.tensor
    path = directory / "ipc.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "GPU cache scheduler did not publish its CUDA IPC handle"
            )
        time.sleep(0.005)
    if gpu_transport == "mnnvl":
        directory_info = directory.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_uid != os.getuid()
            or directory_info.st_mode & 0o077
        ):
            raise ValueError(
                "FABRIC descriptor directory must be private and owner-controlled"
            )
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size > 65536
            ):
                raise ValueError("FABRIC metadata must be a bounded private owner file")
            data = json.load(source)
        return import_fabric(
            data, engine_id, shape, base_domain, compatibility_domain, fabric_domain
        )
    info = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError(
            "GPU cache IPC metadata must be a private owner-controlled file"
        )
    return import_ipc(json.loads(path.read_text()), shape)


def _copy_gpu_payload(source, destination, sizes, **kwargs):
    # Both payload endpoints are CUDA. Host tensors contain addresses/sizes
    # only. Preserve source stream ordering in both cache-copy directions.
    return ops.swap_blocks_batch(
        source, destination, sizes, is_src_access_order_any=False
    )


class GPUCacheCopyHandler(SingleDirectionOffloadingHandler):
    """Reuse vLLM's block geometry and CUDA event fences with CUDA endpoints.

    The pinned vLLM handler's constructor enforces CPU destinations. This
    initializer supplies its documented raw-page state with GPU cache views;
    its block/chunk cursor, stream ordering and completion code are unchanged.
    Canonical cross-topology layouts are deliberately unsupported here.
    """

    def __init__(self, gpu_tensors, cache_tensors, blocks_per_chunk, groups, store):
        if not gpu_tensors or len(gpu_tensors) != len(cache_tensors):
            raise ValueError("GPU cache tensor set differs from the model allocation")
        for original, cached in zip(gpu_tensors, cache_tensors, strict=True):
            if (
                not original.is_cuda
                or not cached.is_cuda
                or original.device != cached.device
            ):
                raise ValueError(
                    "Local KV cache copies require CUDA buffers on one GPU"
                )
            if original.dtype != torch.int8 or cached.dtype != torch.int8:
                raise ValueError("GPU KV copies require byte views")
            if cached.shape[1] != original.shape[1] * blocks_per_chunk:
                raise ValueError("GPU cache page size differs from the model layout")
        self.src_tensors = gpu_tensors if store else cache_tensors
        self.dst_tensors = cache_tensors if store else gpu_tensors
        self.gpu_to_cpu = store  # Inherited geometry direction, not a memory type.
        self.layer_refs_per_group = groups
        self._swap_blocks_batch = _copy_gpu_payload
        self.src_blocks_per_chunk = 1 if store else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if store else 1
        self._canonical_copy_plans = None
        self._fill_group_ops = self._fill_direct_ops
        self._scratch_bases_src = np.empty(0, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(0, dtype=np.uint64)
        self._transfer_events = {}
        self._transfers = deque()
        self._stream_pool = []
        self._event_pool = []
        self._buffer_pool = []

    def shutdown(self):
        super().shutdown()
        # The parent releases CUDA views after successful event synchronization.
        # Break the remaining bound-method cycle so this handler also retires
        # deterministically, without a garbage-collection pass.
        self._fill_group_ops = None


class GPUCacheWorker(OffloadingWorker):
    def __init__(self, spec, caches):
        self.spec, self.caches = spec, caches
        self.handlers = None
        self.region = None

    def _initialize(self):
        if self.handlers is not None:
            return
        options = {}
        if self.spec.gpu_transport == "mnnvl":
            options = {
                "gpu_transport": "mnnvl",
                "base_domain": self.spec.base_domain,
                "compatibility_domain": self.spec.extra_config[
                    "router_hint_transfer_domain"
                ],
                "fabric_domain": self.spec.fabric_domain,
            }
        self.region = open_region(
            self.spec.config.engine_id,
            (self.spec.num_chunks, self.spec.row_bytes),
            **options,
        )
        originals, cached, offset = [], [], 0
        for cache in self.caches.tensors:
            page_bytes = cache.page_size_bytes
            width = page_bytes * self.spec.blocks_per_chunk
            if offset + width > self.spec.row_bytes:
                raise ValueError("Actual GPU banks exceed the configured cache row")
            originals.append(cache.tensor.view(torch.int8).view(-1, page_bytes))
            cached.append(
                self.region.as_strided(
                    (self.spec.num_chunks, width), (self.spec.row_bytes, 1), offset
                )
            )
            offset += width
        if offset != self.spec.row_bytes:
            raise ValueError(
                "Actual GPU bank set does not cover the configured cache row"
            )
        self.handlers = tuple(
            GPUCacheCopyHandler(
                originals,
                cached,
                self.spec.blocks_per_chunk,
                self.caches.group_data_refs,
                store,
            )
            for store in (True, False)
        )

    def submit_store(self, job_id, src_spec, dst_spec):
        self._initialize()
        return self.handlers[0].transfer_async(job_id, src_spec, dst_spec)

    def submit_load(self, job_id, src_spec, dst_spec):
        self._initialize()
        return self.handlers[1].transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self):
        return (
            []
            if self.handlers is None
            else [
                result for handler in self.handlers for result in handler.get_finished()
            ]
        )

    def wait(self, job_ids):
        for handler in self.handlers or ():
            handler.wait(job_ids)

    def shutdown(self):
        for handler in self.handlers or ():
            handler.shutdown()
        # A failed handler shutdown retains its views above. Once all CUDA
        # events have drained, release imported/local views before IPC teardown.
        self.handlers = None
        self.region = None
