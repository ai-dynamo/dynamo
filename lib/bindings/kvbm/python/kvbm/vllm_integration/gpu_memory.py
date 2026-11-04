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
from pathlib import Path

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor


def region_path(engine_id: str, *, rank: int | None = None) -> Path:
    digest = hashlib.sha256(engine_id.encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"dynamo-gpu-kv-{os.getuid()}-{digest}"
    if rank is None:
        return directory
    if type(rank) is not int or rank < 0:
        raise ValueError("GPU bank rank must be a nonnegative integer")
    return directory / f"rank-{rank}"


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

    def decode(name):
        return base64.b64decode(data[name], validate=True)

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

    def __init__(
        self, engine_id: str, chunks: int, row_bytes: int, *, _path: Path | None = None
    ):
        if chunks <= 0 or row_bytes <= 0:
            raise ValueError("GPU cache capacity must contain a whole positive row")
        self.tensor = torch.zeros((chunks, row_bytes), dtype=torch.int8, device="cuda")
        self.tensor.record_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.shape = (chunks, row_bytes)
        self.strides = (row_bytes, 1)
        self.path = region_path(engine_id) if _path is None else _path
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


_OWNED_GPU_REGIONS: dict[Path, GPUCacheRegion] = {}


def open_region(engine_id: str, shape: tuple[int, int], timeout: float = 30):
    directory = region_path(engine_id)
    owned = _OWNED_GPU_REGIONS.get(directory)
    if owned is not None:
        if owned.shape != shape or owned._closed:
            raise ValueError("Local GPU cache owner has a different or closed layout")
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
