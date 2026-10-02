# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""One-writer mmap metadata, safe across process crashes on a local filesystem.

flock serializes metadata transactions; it does not fence imported CUDA handles.
Aligned lock-free atomic words commit the header and each fixed block record.
Power-loss durability and recovery after GMS/GPU loss are outside this contract.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import hashlib
import json
import mmap
import os
import struct
import threading
import zlib
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.types import (
    KVBlockRecord,
    KVRecoveryManifest,
    RecoveryResult,
)

_HEADER = 4096
_STRIDE = 128
_MAGIC = b"DYNKV001"
_EPOCH = 16
_COMMIT = 24
_BINDING = 32
_OWNER = 64
_PAYLOAD = struct.Struct("<IQ68s")
_CRC = struct.Struct("<I")
_SEQ_CST = 5


class _AtomicWords:
    def __init__(self):
        library = ctypes.util.find_library("atomic")
        if library is None:
            raise RuntimeError("KV recovery requires libatomic")
        self.library = ctypes.CDLL(library)
        self.load = getattr(self.library, "__atomic_load_8")
        self.load.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.load.restype = ctypes.c_uint64
        self.store = getattr(self.library, "__atomic_store_8")
        self.store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        self.store.restype = None
        lock_free = getattr(self.library, "__atomic_is_lock_free")
        lock_free.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        lock_free.restype = ctypes.c_bool
        if not lock_free(8, None):
            raise RuntimeError("KV recovery requires lock-free 64-bit atomics")


class MmapKVRecoveryManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._atomics = _AtomicWords()

    def recover(
        self,
        allocations: Sequence[PersistentAllocation],
        manifest: KVRecoveryManifest,
    ) -> MmapRecoverySession:
        if not allocations or not manifest.tensors or manifest.num_blocks <= 1:
            raise ValueError("recovery requires complete allocations and KV layout")
        ids = {a.allocation_id for a in allocations}
        if len(ids) != len(allocations) or any(
            tensor.allocation_id not in ids for tensor in manifest.tensors
        ):
            raise ValueError("KV layout refers to missing or duplicate allocations")
        binding = hashlib.sha256(
            json.dumps(
                {
                    "allocations": sorted(
                        (asdict(a) for a in allocations),
                        key=lambda a: a["allocation_id"],
                    ),
                    "manifest": asdict(manifest),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).digest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        mapping = None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = _HEADER + manifest.num_blocks * _STRIDE
            same_size = os.fstat(fd).st_size == size
            if not same_size:
                os.ftruncate(fd, size)
            mapping = mmap.mmap(fd, size)
            session = MmapRecoverySession(
                fd, mapping, self._atomics, manifest.num_blocks
            )
            valid_header = same_size and mapping[:8] == _MAGIC
            previous_epoch = session._load(_EPOCH) if valid_header else 0
            recovered = (
                valid_header
                and previous_epoch != 0
                and session._load(_COMMIT) == previous_epoch
                and mapping[_BINDING : _BINDING + 32] == binding
            )
            records = session._read_blocks() if recovered else ()
            if records is None:
                recovered = False
                records = ()
            # A crash after epoch advancement cannot expose the old committed
            # header as a completed takeover. The next owner starts fresh.
            session.epoch = previous_epoch + 1
            session._store(_EPOCH, session.epoch)
            session._store(_COMMIT, 0)
            session.owner = uuid4().bytes
            mapping[:8] = _MAGIC
            mapping[_BINDING : _BINDING + 32] = binding
            mapping[_OWNER : _OWNER + 16] = session.owner
            if not recovered:
                session._clear_records()
            session._store(_COMMIT, session.epoch)
            session.result = (
                RecoveryResult.RECOVERED if recovered else RecoveryResult.FRESH
            )
            session.reason = (
                "binding matched" if recovered else "new or invalid binding"
            )
            session._blocks = tuple(records)
            return session
        except BaseException:
            if mapping is not None:
                mapping.close()
                mapping = None
            os.close(fd)
            raise
        finally:
            if mapping is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)


class MmapRecoverySession:
    def __init__(self, fd, mapping, atomics, capacity):
        self._fd = fd
        self._mapping = mapping
        self._atomics = atomics
        self._capacity = capacity
        self._lock = threading.RLock()
        self._closed = False

    def blocks(self) -> tuple[KVBlockRecord, ...]:
        with self._transaction():
            return self._blocks

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("KV recovery session is closed")
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                if (
                    self._load(_EPOCH) != self.epoch
                    or self._load(_COMMIT) != self.epoch
                    or self._mapping[_OWNER : _OWNER + 16] != self.owner
                ):
                    raise RuntimeError("stale KV recovery writer session")
                yield
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _address(self, offset):
        return ctypes.addressof(ctypes.c_char.from_buffer(self._mapping, offset))

    def _load(self, offset):
        return self._atomics.load(self._address(offset), _SEQ_CST)

    def _store(self, offset, value):
        self._atomics.store(self._address(offset), value, _SEQ_CST)

    def _offset(self, block_id):
        if not 0 < block_id < self._capacity:
            raise ValueError("KV recovery block ID is outside the native pool")
        return _HEADER + block_id * _STRIDE

    def invalidate_blocks(self, block_ids: Sequence[int]) -> None:
        offsets = [self._offset(block_id) for block_id in block_ids]
        with self._transaction():
            for offset in offsets:
                self._store(offset, 0)

    def publish_blocks(self, blocks: Sequence[KVBlockRecord]) -> None:
        prepared = []
        for block in blocks:
            if not 4 < len(block.block_hash) <= 68 or block.num_tokens <= 0:
                raise ValueError("invalid native KV block record")
            payload = _PAYLOAD.pack(
                len(block.block_hash), block.num_tokens, block.block_hash
            )
            prepared.append((self._offset(block.block_id), payload))
        with self._transaction():
            for offset, payload in prepared:
                self._store(offset, 0)
                self._mapping[offset + 8 : offset + 8 + _PAYLOAD.size] = payload
                _CRC.pack_into(
                    self._mapping, offset + 8 + _PAYLOAD.size, zlib.crc32(payload)
                )
                self._store(offset, 1)

    def _read_blocks(self):
        blocks = []
        for block_id in range(1, self._capacity):
            offset = self._offset(block_id)
            valid = self._load(offset)
            if valid == 0:
                continue
            if valid != 1:
                return None
            payload = self._mapping[offset + 8 : offset + 8 + _PAYLOAD.size]
            checksum = _CRC.unpack_from(self._mapping, offset + 8 + _PAYLOAD.size)[0]
            if checksum != zlib.crc32(payload):
                return None
            length, num_tokens, key = _PAYLOAD.unpack(payload)
            if not 4 < length <= 68 or not num_tokens:
                return None
            blocks.append(KVBlockRecord(block_id, key[:length], num_tokens))
        return tuple(blocks)

    def _clear_records(self):
        for block_id in range(1, self._capacity):
            self._store(self._offset(block_id), 0)

    def clear(self) -> None:
        with self._transaction():
            self._clear_records()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._mapping.close()
                os.close(self._fd)
                self._closed = True
