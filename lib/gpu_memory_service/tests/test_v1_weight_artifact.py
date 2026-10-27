# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

import msgspec
from _fake_vmm import FakeVMM
from blake3 import blake3
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.snapshot import disk as snapshot_disk
from gpu_memory_service.snapshot.transfer import (
    TransferBackendKind,
    verify_file_source_digests,
)
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import AllocationRecord
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.snapshot import weight_artifact

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


class _ByteVMM(FakeVMM):
    def __init__(self) -> None:
        super().__init__(granularity=64)
        self._physical: dict[int, bytearray] = {}
        self._exported: dict[int, int] = {}
        self._imported: dict[int, int] = {}

    def create_tolerate_oom(self, size, device):
        allocated, handle = super().create_tolerate_oom(size, device)
        self._physical[handle] = bytearray(size)
        return allocated, handle

    def release(self, handle):
        if handle in self.server_handles:
            self._physical.pop(handle)
        self._imported.pop(handle, None)
        super().release(handle)

    def export_to_shareable_handle(self, handle):
        fd = super().export_to_shareable_handle(handle)
        self._exported[os.fstat(fd).st_ino] = handle
        return fd

    def import_shareable_handle_close_fd(self, fd):
        physical_handle = self._exported[os.fstat(fd).st_ino]
        imported_handle = super().import_shareable_handle_close_fd(fd)
        self._imported[imported_handle] = physical_handle
        return imported_handle

    def _allocation(self, va, size):
        mapped_size, imported_handle = self.mapped[va]
        assert size <= mapped_size
        return self._physical[self._imported[imported_handle]]

    def read(self, va, size):
        return bytes(self._allocation(va, size)[:size])

    def write(self, va, data):
        self._allocation(va, len(data))[: len(data)] = data


class _Writer:
    vmm: _ByteVMM

    def __init__(self, path, *, device, hash_name=None, hash_threads=1):
        assert device == 0
        assert hash_threads > 0
        self._path = Path(path)
        self._data = bytearray()
        assert hash_name in (None, "blake3")
        self._hasher = blake3() if hash_name == "blake3" else None

    def __enter__(self):
        return self

    def write_device(self, src_ptr, byte_count):
        data = self.vmm.read(src_ptr, byte_count)
        self._data.extend(data)
        if self._hasher is not None:
            self._hasher.update(data)

    def __exit__(self, *_args):
        self._path.write_bytes(self._data)

    def hexdigest(self):
        assert self._hasher is not None
        return self._hasher.hexdigest()


class _Transfer:
    def __init__(self, vmm, sources, integrity_by_path, *, fail=False):
        self._vmm = vmm
        self._sources = sources
        self._integrity_by_path = integrity_by_path
        self._fail = fail

    def restore(self, targets):
        if self._fail:
            raise RuntimeError("restore failed")
        verify_file_source_digests(
            self._sources, self._integrity_by_path, max_workers=1
        )
        for source in self._sources:
            target = targets[source.allocation_id]
            shard = Path(source.file_path).read_bytes()
            data = shard[source.file_offset : source.file_offset + source.byte_count]
            assert len(data) == target.byte_count
            self._vmm.write(target.va, data)

    def close(self):
        pass


class _Backend:
    def __init__(self, vmm, *, fail=False):
        self._vmm = vmm
        self._fail = fail
        self.closed = False
        self.restores = 0

    def start_restore(self, sources, integrity_by_path=None):
        self.restores += 1
        return _Transfer(self._vmm, sources, integrity_by_path or {}, fail=self._fail)

    def close(self):
        self.closed = True


@contextmanager
def _server(path: str, vmm: FakeVMM):
    manager = GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield manager
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


def _configure_device(monkeypatch, vmm: _ByteVMM) -> None:
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "invalidate_device_uuid_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "get_device_uuid",
        lambda _device: "GPU-0",
    )


def _map(session, record, vmm, access):
    return weight_artifact._map_export(
        session,
        record,
        vmm,
        0,
        64,
        access,
    )


def test_device_writer_blake3_hashes_the_exact_multi_chunk_file(tmp_path, monkeypatch):
    source = bytes(range(11))
    base = 1000

    class Slot:
        def __init__(self):
            self._buffer = bytearray(8)
            self.view = memoryview(self._buffer)

        def wait(self):
            pass

        def copy_from_device_async(self, src_ptr, size):
            offset = src_ptr - base
            self._buffer[:size] = source[offset : offset + size]

    class VMM:
        def runtime_set_device(self, device):
            assert device == 0

    closed = []
    monkeypatch.setattr(snapshot_disk, "get_vmm", lambda: VMM())
    monkeypatch.setattr(
        snapshot_disk,
        "make_pinned_copy_slots",
        lambda _vmm, buffers: [Slot() for _ in range(buffers)],
    )
    monkeypatch.setattr(
        snapshot_disk,
        "close_pinned_copy_slots",
        lambda slots, *_args: closed.extend(slots),
    )
    shard = tmp_path / "shard.bin"
    writer = snapshot_disk.DeviceToFileWriter(
        str(shard),
        device=0,
        buffers=2,
        chunk_size=3,
        hash_name="blake3",
    )
    with writer:
        writer.write_device(base, 5)
        writer.write_device(base + 5, len(source) - 5)
        with pytest.raises(RuntimeError, match="before close"):
            writer.hexdigest()

    assert shard.read_bytes() == source
    assert writer.hexdigest() == blake3(source).hexdigest()
    assert len(closed) == 2


def test_legacy_shard_writer_does_not_require_hash_support(tmp_path, monkeypatch):
    calls = []

    class LegacyWriter:
        def __init__(self, path, *, device):
            calls.append((path, device))

        def __enter__(self):
            return self

        def write_device(self, src_ptr, byte_count):
            calls.append((src_ptr, byte_count))

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(snapshot_disk, "DeviceToFileWriter", LegacyWriter)
    placements, digests = snapshot_disk.write_device_shards(
        [(100, 64), (200, 64)],
        [str(tmp_path)],
        device=0,
        shard_size_bytes=64,
        max_workers=1,
        relative_to=str(tmp_path),
    )

    assert placements == [("shard_0000.bin", 0), ("shard_0001.bin", 0)]
    assert digests == {}
    assert (100, 64) in calls
    assert (200, 64) in calls


@pytest.mark.timeout(10)
@pytest.mark.parametrize("schema_version", [1, 2])
def test_weight_artifact_round_trip_preserves_exact_ids_and_bytes(
    tmp_path,
    monkeypatch,
    schema_version,
) -> None:
    transfer_backend = TransferBackendKind.NIXL_GDS
    warnings: list[str] = []
    monkeypatch.setattr(
        weight_artifact.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    records = (
        AllocationRecord("weight-0", 64),
        AllocationRecord("weight-1", 128),
    )
    expected = {
        "weight-0": bytes(range(64)),
        "weight-1": bytes((255 - index) % 256 for index in range(128)),
    }

    source_vmm = _ByteVMM()
    _Writer.vmm = source_vmm
    _configure_device(monkeypatch, source_vmm)
    monkeypatch.setattr(snapshot_disk, "DeviceToFileWriter", _Writer)
    with _server(socket_path, source_vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        mappings = []
        for record in records:
            writer.allocate(record.allocation_id, record.aligned_size)
            mapping = _map(writer, record, source_vmm, GrantedLockType.RW)
            source_vmm.write(mapping[0].base, expected[record.allocation_id])
            mappings.append(mapping)
        writer.commit()
        manifest = weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            shard_size_bytes=64,
        )
        for mapping in reversed(mappings):
            weight_artifact._release_mapping(source_vmm, mapping)
        writer.close()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(record.allocation_id, record.aligned_size) for record in records]
    assert manifest.version == 2
    assert len(manifest.shards) == 2
    for shard in manifest.shards:
        shard_path = artifact / shard.path
        assert shard.byte_count == shard_path.stat().st_size
        assert shard.blake3 == blake3(shard_path.read_bytes()).hexdigest()

    if schema_version == 1:
        legacy_manifest = {
            "version": 1,
            "allocations": msgspec.to_builtins(manifest.allocations),
        }
        (artifact / "manifest.json").write_bytes(msgspec.json.encode(legacy_manifest))

    target_vmm = _ByteVMM()
    backends = []
    backend_calls = []
    _configure_device(monkeypatch, target_vmm)

    def create_backend(name, config):
        backend_calls.append((name, config))
        backends.append(_Backend(target_vmm))
        return backends[-1]

    monkeypatch.setattr(weight_artifact, "create_transfer_backend", create_backend)
    with _server(socket_path, target_vmm):
        weight_artifact.load_weights(
            str(artifact), socket_path, 0, transfer_backend=transfer_backend.value
        )
        reader = _GMSClientSession(socket_path, RequestedLockType.RO)
        # A restarted loader must not wait for RW behind the reader's lease.
        weight_artifact.load_weights(
            str(artifact), socket_path, 0, transfer_backend=transfer_backend.value
        )
        assert weight_artifact._list_allocations(reader) == records
        restored = [
            _map(reader, record, target_vmm, GrantedLockType.RO) for record in records
        ]
        assert {
            mapping.allocation_id: target_vmm.read(mapping.base, mapping.aligned_size)
            for mapping, _handle in restored
        } == expected
        for mapping in reversed(restored):
            weight_artifact._release_mapping(target_vmm, mapping)
        reader.close()
    # The second load finds the weights committed: its backend restores nothing.
    assert [backend.restores for backend in backends] == [1, 0]
    assert all(backend.closed for backend in backends)
    assert {name for name, _config in backend_calls} == {transfer_backend.value}
    assert {config.device for _name, config in backend_calls} == {0}
    assert any(
        "content-digest integrity verification is unavailable" in warning
        for warning in warnings
    ) == (schema_version == 1)


@pytest.mark.timeout(10)
def test_failed_weight_load_does_not_publish_partial_epoch(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    allocation = weight_artifact.WeightArtifactAllocation(
        "weight",
        64,
        "shard.bin",
        0,
    )
    shard_bytes = bytes(64)
    (artifact / "manifest.json").write_bytes(
        msgspec.json.encode(
            weight_artifact.WeightArtifactManifest(
                2,
                (allocation,),
                (
                    weight_artifact.WeightArtifactShard(
                        "shard.bin",
                        len(shard_bytes),
                        blake3(shard_bytes).hexdigest(),
                    ),
                ),
            )
        )
    )
    (artifact / "shard.bin").write_bytes(shard_bytes)

    vmm = _ByteVMM()
    backend = _Backend(vmm, fail=True)
    _configure_device(monkeypatch, vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )
    with _server(socket_path, vmm) as manager:
        with pytest.raises(RuntimeError, match="restore failed"):
            weight_artifact.load_weights(str(artifact), socket_path, 0)
        assert not manager._sessions._committed
        assert not vmm.server_handles
        replacement = _GMSClientSession(socket_path, RequestedLockType.RW)
        replacement.close()
    assert backend.closed


def _write_test_artifact(
    artifact: Path,
    *,
    shard_bytes: bytes = bytes(64),
    digest: str | None = None,
    declared_byte_count: int | None = None,
    version: int = 2,
    include_shard: bool = True,
) -> None:
    artifact.mkdir()
    allocation = weight_artifact.WeightArtifactAllocation(
        "weight",
        64,
        "shard.bin",
        0,
    )
    manifest = {
        "version": version,
        "allocations": [msgspec.to_builtins(allocation)],
    }
    if include_shard:
        expected_digest = blake3(shard_bytes).hexdigest()
        manifest["shards"] = [
            {
                "path": "shard.bin",
                "byte_count": len(shard_bytes)
                if declared_byte_count is None
                else declared_byte_count,
                "blake3": digest or expected_digest,
            }
        ]
    (artifact / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    (artifact / "shard.bin").write_bytes(shard_bytes)


def test_weight_artifact_rejects_invalid_shard_digest(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, digest="0" * 63)

    with pytest.raises(RuntimeError, match="invalid BLAKE3"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_same_length_shard_mutation_without_publish(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    (artifact / "shard.bin").write_bytes(bytes([1]) + bytes(63))
    vmm = _ByteVMM()
    _configure_device(monkeypatch, vmm)
    backend = _Backend(vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )

    socket_path = str(tmp_path / "weights.sock")
    with _server(socket_path, vmm) as manager:
        with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
            weight_artifact.load_weights(str(artifact), socket_path, 0)
        assert not manager._sessions._committed
        assert not vmm.server_handles
    assert backend.closed


def test_v2_blake3_artifact_detects_same_length_mutation(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, version=2)
    _manifest, sources, integrity = weight_artifact._load_manifest(str(artifact), 64)
    assert integrity[sources[0].file_path].blake3 == blake3(bytes(64)).hexdigest()
    (artifact / "shard.bin").write_bytes(bytes([1]) + bytes(63))

    with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
        verify_file_source_digests(sources, integrity, max_workers=1)


def test_v2_sha256_manifest_is_rejected_instead_of_ignoring_its_digest(
    tmp_path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    manifest_path = artifact / "manifest.json"
    manifest = msgspec.json.decode(manifest_path.read_bytes())
    manifest["shards"][0]["sha256"] = manifest["shards"][0].pop("blake3")
    manifest_path.write_bytes(msgspec.json.encode(manifest))

    with pytest.raises(msgspec.ValidationError):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_validates_external_sharded_ssd_digest(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    external_root = tmp_path / "external"
    external_root.mkdir()
    shard_path = external_root / "shard.bin"
    shard_bytes = bytes(64)
    shard_path.write_bytes(shard_bytes)
    artifact.mkdir()
    manifest = weight_artifact.WeightArtifactManifest(
        2,
        (
            weight_artifact.WeightArtifactAllocation(
                "weight",
                64,
                str(shard_path),
                0,
            ),
        ),
        (
            weight_artifact.WeightArtifactShard(
                str(shard_path),
                len(shard_bytes),
                blake3(shard_bytes).hexdigest(),
            ),
        ),
    )
    (artifact / "manifest.json").write_bytes(msgspec.json.encode(manifest))

    _manifest, sources, integrity = weight_artifact._load_manifest(
        str(artifact), 64, (str(external_root),)
    )
    assert integrity[sources[0].file_path].byte_count == len(shard_bytes)
    assert integrity[sources[0].file_path].blake3 == blake3(shard_bytes).hexdigest()
    assert verify_file_source_digests(sources, integrity, max_workers=1) == (1, 64)
    shard_path.write_bytes(bytes([1]) + bytes(63))
    with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
        verify_file_source_digests(sources, integrity, max_workers=1)


def test_weight_artifact_requires_integrity_record_for_every_shard(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, include_shard=False)

    with pytest.raises(msgspec.ValidationError, match="shards"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_duplicate_shard_integrity_records(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    manifest = msgspec.json.decode((artifact / "manifest.json").read_bytes())
    manifest["shards"].append(manifest["shards"][0])
    (artifact / "manifest.json").write_bytes(msgspec.json.encode(manifest))

    with pytest.raises(RuntimeError, match="duplicate GMS V1 shard record"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_unreferenced_shard_integrity_record(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    manifest = msgspec.json.decode((artifact / "manifest.json").read_bytes())
    manifest["shards"].append(
        {
            "path": "extra.bin",
            "byte_count": 64,
            "blake3": blake3(bytes(64)).hexdigest(),
        }
    )
    (artifact / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    (artifact / "extra.bin").write_bytes(bytes(64))

    with pytest.raises(RuntimeError, match="records do not match allocation shards"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_declared_shard_length_mismatch(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, declared_byte_count=128)

    with pytest.raises(RuntimeError, match="length does not match record"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_truncated_shard(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    (artifact / "shard.bin").write_bytes(bytes(32))

    with pytest.raises(RuntimeError, match="length does not match extents"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_weight_artifact_rejects_missing_shard(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    (artifact / "shard.bin").unlink()

    with pytest.raises(RuntimeError, match="shard does not exist.*shard.bin"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_legacy_artifact_warns_and_cannot_detect_same_length_corruption(
    tmp_path, monkeypatch
):
    warnings: list[str] = []
    monkeypatch.setattr(
        weight_artifact.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, version=1, include_shard=False)

    before, sources, integrity = weight_artifact._load_manifest(str(artifact), 64)
    (artifact / "shard.bin").write_bytes(bytes([1]) + bytes(63))
    after, changed_sources, changed_integrity = weight_artifact._load_manifest(
        str(artifact), 64
    )
    assert before == after
    assert sources == changed_sources
    assert integrity == changed_integrity == {}
    assert before.version == 1
    assert len(warnings) == 2
    assert all(
        "content-digest integrity verification is unavailable" in warning
        for warning in warnings
    )
    assert all(
        "same-length weight file corruption cannot be detected" in warning
        for warning in warnings
    )
    assert all(str(artifact) in warning for warning in warnings)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "damage", ["missing", "truncated", "escape", "duplicate", "offset"]
)
def test_public_loader_rejects_structural_damage_before_backend_or_session(
    tmp_path,
    monkeypatch,
    version,
    damage,
) -> None:
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, version=version, include_shard=version != 1)
    manifest_path = artifact / "manifest.json"
    manifest = msgspec.json.decode(manifest_path.read_bytes())
    if damage == "missing":
        (artifact / "shard.bin").unlink()
    elif damage == "truncated":
        (artifact / "shard.bin").write_bytes(bytes(32))
    elif damage == "escape":
        (tmp_path / "outside.bin").write_bytes(bytes(64))
        manifest["allocations"][0]["shard"] = "../outside.bin"
    elif damage == "duplicate":
        manifest["allocations"].append(manifest["allocations"][0])
    else:
        manifest["allocations"][0]["offset"] = 64
    manifest_path.write_bytes(msgspec.json.encode(manifest))
    vmm = _ByteVMM()
    _configure_device(monkeypatch, vmm)

    def unexpected_backend(*_args, **_kwargs):
        pytest.fail("structural validation must precede backend creation")

    def unexpected_session(*_args, **_kwargs):
        pytest.fail("structural validation must precede GMS session creation")

    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        unexpected_backend,
    )
    monkeypatch.setattr(weight_artifact, "_GMSClientSession", unexpected_session)

    with pytest.raises(RuntimeError):
        weight_artifact.load_weights(
            str(artifact),
            str(tmp_path / "unused.sock"),
            0,
        )


@pytest.mark.parametrize("version", [0, 3])
def test_unknown_artifact_version_is_rejected(tmp_path, version):
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, version=version)
    with pytest.raises(RuntimeError, match="unsupported.*version"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_v2_empty_shards_never_falls_back_to_legacy(tmp_path):
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact)
    manifest_path = artifact / "manifest.json"
    manifest = msgspec.json.decode(manifest_path.read_bytes())
    manifest["shards"] = []
    manifest_path.write_bytes(msgspec.json.encode(manifest))
    with pytest.raises(RuntimeError, match="no shard integrity records"):
        weight_artifact._load_manifest(str(artifact), 64)


def test_changing_v2_version_alone_does_not_bypass_verification(tmp_path):
    artifact = tmp_path / "artifact"
    _write_test_artifact(artifact, version=1, include_shard=True)
    with pytest.raises(msgspec.ValidationError, match="unknown field.*shards"):
        weight_artifact._load_manifest(str(artifact), 64)
