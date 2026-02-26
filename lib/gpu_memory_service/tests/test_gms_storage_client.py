# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client.gms_storage_client (GMSStorageClient).

Test classes:
- TestAllocationEntry        – dataclass creation and asdict roundtrip (no GPU)
- TestSaveManifest           – to_dict/from_dict/JSON roundtrip (no GPU)
- TestShardWriter            – _ShardWriter write behaviour (no GPU)
- TestTorchSerializationRoundtrip – torch.save/.load roundtrip (no GPU + optional CUDA)
- TestLoadFromPrebuiltSave        – load_tensors() against hand-crafted save dir (no GPU)
- TestGMSStorageClientInit        – __init__ attribute checks (no GPU)
- TestGMSStorageClientSaveMock    – mocked GMSClientMemoryManager (no GPU)
- TestGMSStorageIntegration       – real GMS server in background thread (GPU required)
- TestGMSStorageClientLoadMock    – mocked load_to_gms (no GPU)
- TestNixlGdsWriter               – POSIX roundtrip (NIXL required); GDS (GPU + GDS HW)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import threading
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import uvloop  # noqa: F401

    _UVLOOP_AVAILABLE = True
except ImportError:
    _UVLOOP_AVAILABLE = False

_CUDA_AVAILABLE = _TORCH_AVAILABLE and torch.cuda.is_available()

try:
    from nixl._api import nixl_agent  # noqa: F401

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from gpu_memory_service.client.gms_storage_client import (  # noqa: E402
    _CURRENT_VERSION,
    AllocationEntry,
    GMSStorageClient,
    SaveManifest,
    _NixlGdsWriter,
    _read_shard_sequential,
    _ShardWriter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    alloc_id: str = "alloc-1",
    size: int = 1024,
    aligned_size: int = 2097152,
    tag: str = "default",
    tensor_file: str = "tensors/alloc-1.pt",
    tensor_offset: int = 0,
) -> AllocationEntry:
    return AllocationEntry(
        allocation_id=alloc_id,
        size=size,
        aligned_size=aligned_size,
        tag=tag,
        tensor_file=tensor_file,
        tensor_offset=tensor_offset,
    )


def _make_manifest(allocations=None) -> SaveManifest:
    return SaveManifest(
        version=_CURRENT_VERSION,
        timestamp=1_700_000_000.0,
        layout_hash="deadbeef",
        device=0,
        use_gds=False,
        gds_available=False,
        allocations=allocations or [],
    )


# ===========================================================================
# TestAllocationEntry
# ===========================================================================


class TestAllocationEntry:
    def test_creation(self):
        e = _make_entry()
        assert e.allocation_id == "alloc-1"
        assert e.size == 1024
        assert e.aligned_size == 2097152
        assert e.tag == "default"
        assert e.tensor_file == "tensors/alloc-1.pt"
        assert e.tensor_offset == 0

    def test_creation_with_offset(self):
        e = _make_entry(tensor_file="shards/shard_0000.bin", tensor_offset=4096)
        assert e.tensor_file == "shards/shard_0000.bin"
        assert e.tensor_offset == 4096

    def test_frozen(self):
        e = _make_entry()
        with pytest.raises((AttributeError, TypeError)):
            e.size = 9999  # type: ignore[misc]

    def test_asdict_roundtrip(self):
        e = _make_entry()
        d = asdict(e)
        assert d == {
            "allocation_id": "alloc-1",
            "size": 1024,
            "aligned_size": 2097152,
            "tag": "default",
            "tensor_file": "tensors/alloc-1.pt",
            "tensor_offset": 0,
        }
        e2 = AllocationEntry(**d)
        assert e2 == e

    def test_different_ids_not_equal(self):
        e1 = _make_entry("a-1")
        e2 = _make_entry("a-2")
        assert e1 != e2


# ===========================================================================
# TestSaveManifest
# ===========================================================================


class TestSaveManifest:
    def test_to_dict(self):
        m = _make_manifest()
        d = m.to_dict()
        assert d["version"] == _CURRENT_VERSION
        assert d["timestamp"] == 1_700_000_000.0
        assert d["layout_hash"] == "deadbeef"
        assert d["device"] == 0
        assert d["use_gds"] is False
        assert d["gds_available"] is False
        assert d["allocations"] == []

    def test_from_dict_roundtrip(self):
        entry = _make_entry(tensor_file="shards/shard_0000.bin", tensor_offset=128)
        m = _make_manifest([entry])
        d = m.to_dict()
        m2 = SaveManifest.from_dict(d)
        assert m2.version == m.version
        assert m2.timestamp == m.timestamp
        assert m2.layout_hash == m.layout_hash
        assert m2.device == m.device
        assert m2.use_gds == m.use_gds
        assert m2.gds_available == m.gds_available
        assert len(m2.allocations) == 1
        a = m2.allocations[0]
        assert a.allocation_id == entry.allocation_id
        assert a.size == entry.size
        assert a.aligned_size == entry.aligned_size
        assert a.tag == entry.tag
        assert a.tensor_file == entry.tensor_file
        assert a.tensor_offset == 128

    def test_from_dict_backward_compat_no_tensor_offset(self):
        """Old manifests without tensor_offset load with tensor_offset=0."""
        entry = _make_entry()
        m = _make_manifest([entry])
        d = m.to_dict()
        # Simulate an old manifest that doesn't have tensor_offset
        for a in d["allocations"]:
            a.pop("tensor_offset", None)
        m2 = SaveManifest.from_dict(d)
        assert m2.allocations[0].tensor_offset == 0

    def test_json_file_roundtrip(self):
        entry = _make_entry()
        m = _make_manifest([entry])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.json")
            with open(path, "w") as f:
                json.dump(m.to_dict(), f)
            with open(path) as f:
                m2 = SaveManifest.from_dict(json.load(f))
        assert m2.layout_hash == m.layout_hash
        assert len(m2.allocations) == 1

    def test_empty_allocations_roundtrip(self):
        m = _make_manifest()
        m2 = SaveManifest.from_dict(m.to_dict())
        assert m2.allocations == []

    def test_multiple_allocations(self):
        entries = [_make_entry(f"a-{i}", size=i * 100) for i in range(5)]
        m = _make_manifest(entries)
        m2 = SaveManifest.from_dict(m.to_dict())
        assert len(m2.allocations) == 5
        for orig, restored in zip(entries, m2.allocations):
            assert orig.allocation_id == restored.allocation_id
            assert orig.size == restored.size


# ===========================================================================
# TestShardWriter  (no GPU required)
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestShardWriter:
    """Tests for _ShardWriter: sequential write, shard rolling, file contents."""

    def test_single_shard_sequential_offsets(self):
        """Two allocations fit in one shard; offsets are contiguous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            with _ShardWriter(shards_dir, shard_size_bytes=1024 * 1024) as writer:
                t1 = torch.full((64,), 0xAA, dtype=torch.uint8)
                t2 = torch.full((128,), 0xBB, dtype=torch.uint8)
                path1, off1 = writer.write(t1)
                path2, off2 = writer.write(t2)

            # Assertions must be inside the tempdir context (dir is deleted on exit)
            assert path1 == path2, "Both allocations should land in the same shard"
            assert off1 == 0
            assert off2 == 64  # t1 is 64 bytes

            shard_path = os.path.join(tmpdir, path1)
            with open(shard_path, "rb") as f:
                data = f.read()
            assert len(data) == 64 + 128
            assert all(b == 0xAA for b in data[:64])
            assert all(b == 0xBB for b in data[64:])

    def test_shard_rolling(self):
        """When a shard is full, writer starts a new shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            # shard_size=100 bytes; two 64-byte writes → two shards
            # (after first write: 64 bytes; 64+64=128 > 100 → roll)
            with _ShardWriter(shards_dir, shard_size_bytes=100) as writer:
                t1 = torch.full((64,), 1, dtype=torch.uint8)
                t2 = torch.full((64,), 2, dtype=torch.uint8)
                path1, off1 = writer.write(t1)
                path2, off2 = writer.write(t2)

            # File existence assertions must be inside the tempdir context
            assert path1 != path2, "Overflow must produce a new shard"
            assert off1 == 0
            assert off2 == 0  # second shard starts at offset 0
            assert path1 == os.path.join("shards", "shard_0000.bin")
            assert path2 == os.path.join("shards", "shard_0001.bin")
            assert os.path.exists(os.path.join(tmpdir, path1))
            assert os.path.exists(os.path.join(tmpdir, path2))

    def test_oversized_allocation_gets_own_shard(self):
        """An allocation larger than shard_size still writes (sole entry in shard)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            with _ShardWriter(shards_dir, shard_size_bytes=16) as writer:
                big = torch.full((256,), 0xCC, dtype=torch.uint8)
                path, off = writer.write(big)

            # File access must be inside the tempdir context
            assert off == 0
            shard_path = os.path.join(tmpdir, path)
            assert os.path.getsize(shard_path) == 256

    def test_four_allocs_two_shards(self):
        """Four 64-byte allocations with 100-byte shard → 2 shards, 2 per shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            paths_offsets = []
            with _ShardWriter(shards_dir, shard_size_bytes=130) as writer:
                for i in range(4):
                    t = torch.full((64,), i, dtype=torch.uint8)
                    paths_offsets.append(writer.write(t))

        shard_paths = {p for p, _ in paths_offsets}
        assert len(shard_paths) == 2, f"Expected 2 shards, got {shard_paths}"

        # First two land in shard 0 (0 bytes, 64 bytes); second two in shard 1
        assert paths_offsets[0] == (os.path.join("shards", "shard_0000.bin"), 0)
        assert paths_offsets[1] == (os.path.join("shards", "shard_0000.bin"), 64)
        assert paths_offsets[2] == (os.path.join("shards", "shard_0001.bin"), 0)
        assert paths_offsets[3] == (os.path.join("shards", "shard_0001.bin"), 64)

    def test_shard_rel_path_starts_with_shards(self):
        """All returned relative paths start with 'shards/'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            with _ShardWriter(shards_dir) as writer:
                path, _ = writer.write(torch.zeros(32, dtype=torch.uint8))
        assert path.startswith("shards/")

    @pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")
    def test_sequential_read_matches_write(self):
        """Data written to a shard can be read back in order via _read_shard_sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "shards")
            entries: List[AllocationEntry] = []
            originals: Dict[str, torch.Tensor] = {}

            with _ShardWriter(shards_dir, shard_size_bytes=512) as writer:
                for i in range(3):
                    data = torch.full((64,), i + 1, dtype=torch.uint8)
                    originals[f"a{i}"] = data
                    rel_path, offset = writer.write(data)
                    entries.append(
                        AllocationEntry(
                            allocation_id=f"a{i}",
                            size=64,
                            aligned_size=64,
                            tag="t",
                            tensor_file=rel_path,
                            tensor_offset=offset,
                        )
                    )

            # All three in the same shard (3 × 64 = 192 < 512)
            assert len({e.tensor_file for e in entries}) == 1

            entries.sort(key=lambda e: e.tensor_offset)
            abs_path = os.path.join(tmpdir, entries[0].tensor_file)
            result = _read_shard_sequential(abs_path, entries, device=0)

            assert set(result.keys()) == {"a0", "a1", "a2"}
            for aid, expected in originals.items():
                assert torch.equal(result[aid].cpu(), expected), f"Mismatch for {aid}"


# ===========================================================================
# TestTorchSerializationRoundtrip
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestTorchSerializationRoundtrip:
    def test_uint8_tensor_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "t.pt")
            data = torch.arange(256, dtype=torch.uint8)
            torch.save(data, path)
            loaded = torch.load(path, weights_only=True)
        assert torch.equal(data, loaded)

    def test_float_as_bytes(self):
        """Simulate how float tensors are stored: as a uint8 view."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "t.pt")
            src = torch.randn(32, dtype=torch.float32)
            # View as uint8 (how dump works)
            raw = src.view(torch.uint8)
            torch.save(raw, path)
            loaded_raw = torch.load(path, weights_only=True)
            restored = loaded_raw.view(torch.float32)
        assert torch.equal(src, restored)

    def test_large_tensor(self):
        """Round-trip a 16 MiB tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "t.pt")
            data = torch.randint(0, 256, (16 * 1024 * 1024,), dtype=torch.uint8)
            torch.save(data, path)
            loaded = torch.load(path, weights_only=True)
        assert torch.equal(data, loaded)

    @pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_tensor_roundtrip(self):
        """GPU tensor → CPU save → GPU restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "t.pt")
            gpu_tensor = torch.randint(
                0, 256, (1024,), dtype=torch.uint8, device="cuda"
            )
            torch.save(gpu_tensor.cpu(), path)
            loaded = torch.load(path, weights_only=True, map_location="cuda")
        assert torch.equal(gpu_tensor, loaded)


# ===========================================================================
# TestRestoreFromPrebuiltDump
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestLoadFromPrebuiltSave:
    """Build a dump directory by hand and verify load_tensors() reads it correctly."""

    @staticmethod
    def _build_dump_dir(tmpdir: str) -> Dict[str, Any]:
        """Create a minimal dump directory with two allocations and metadata.

        Uses the legacy .pt format (one file per allocation) to verify
        backward compatibility with the new sequential-read code path.
        """
        tensors_dir = os.path.join(tmpdir, "tensors")
        os.makedirs(tensors_dir)

        # Allocation A: 16 bytes
        data_a = torch.arange(16, dtype=torch.uint8)
        path_a = os.path.join(tensors_dir, "alloc-a.pt")
        torch.save(data_a, path_a)

        # Allocation B: 32 bytes
        data_b = torch.arange(32, dtype=torch.uint8)
        path_b = os.path.join(tensors_dir, "alloc-b.pt")
        torch.save(data_b, path_b)

        # Metadata
        meta_value_a = b"tensor_meta_a"
        meta_value_b = b"tensor_meta_b"
        metadata = {
            "key_a": {
                "allocation_id": "alloc-a",
                "offset_bytes": 0,
                "value": base64.b64encode(meta_value_a).decode("ascii"),
            },
            "key_b": {
                "allocation_id": "alloc-b",
                "offset_bytes": 8,
                "value": base64.b64encode(meta_value_b).decode("ascii"),
            },
        }
        with open(os.path.join(tmpdir, "gms_metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Manifest — legacy format: no tensor_offset field
        entries = [
            AllocationEntry("alloc-a", 16, 16, "default", "tensors/alloc-a.pt"),
            AllocationEntry("alloc-b", 32, 32, "default", "tensors/alloc-b.pt"),
        ]
        manifest = SaveManifest(
            version=_CURRENT_VERSION,
            timestamp=time.time(),
            layout_hash="cafebabe",
            device=0,
            use_gds=False,
            gds_available=False,
            allocations=entries,
        )
        with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
            json.dump(manifest.to_dict(), f)

        return {
            "data_a": data_a,
            "data_b": data_b,
            "meta_a": meta_value_a,
            "meta_b": meta_value_b,
        }

    def test_loads_all_allocations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir)
            tensors, _ = GMSStorageClient.load_tensors(tmpdir, max_workers=2)
        assert "alloc-a" in tensors
        assert "alloc-b" in tensors

    def test_tensor_values_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._build_dump_dir(tmpdir)
            tensors, _ = GMSStorageClient.load_tensors(tmpdir, max_workers=2)
        assert torch.equal(tensors["alloc-a"].cpu(), expected["data_a"])
        assert torch.equal(tensors["alloc-b"].cpu(), expected["data_b"])

    def test_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._build_dump_dir(tmpdir)
            _, metadata = GMSStorageClient.load_tensors(tmpdir, max_workers=2)
        assert metadata["key_a"]["allocation_id"] == "alloc-a"
        assert metadata["key_a"]["offset_bytes"] == 0
        assert metadata["key_a"]["value"] == expected["meta_a"]
        assert metadata["key_b"]["allocation_id"] == "alloc-b"
        assert metadata["key_b"]["offset_bytes"] == 8
        assert metadata["key_b"]["value"] == expected["meta_b"]

    def test_parallel_restore(self):
        """load_tensors with max_workers > number of allocations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir)
            tensors, _ = GMSStorageClient.load_tensors(tmpdir, max_workers=8)
        assert len(tensors) == 2

    def test_missing_metadata_file_ok(self):
        """load_tensors still works when gms_metadata.json is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir)
            os.unlink(os.path.join(tmpdir, "gms_metadata.json"))
            tensors, metadata = GMSStorageClient.load_tensors(tmpdir, max_workers=1)
        assert len(metadata) == 0
        assert len(tensors) == 2


# ===========================================================================
# TestGMSStorageClientInit
# ===========================================================================


class TestGMSStorageClientInit:
    def test_no_gds_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = GMSStorageClient(
                tmpdir, socket_path="/tmp/fake.sock", device=0, use_gds=False
            )
        assert client.output_dir == tmpdir
        assert client.device == 0
        assert client._nixl_writer is None

    def test_gds_without_nixl_warns(self):
        """use_gds=True without nixl installed should warn and not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "gpu_memory_service.client.gms_storage_client._NIXL_AVAILABLE", False
            ):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    client = GMSStorageClient(
                        tmpdir, socket_path="/tmp/fake.sock", device=0, use_gds=True
                    )
        assert client._nixl_writer is None
        assert any("nixl" in str(warning.message).lower() for warning in w)

    def test_output_dir_attr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "my_dump")
            client = GMSStorageClient(
                subdir, socket_path="/tmp/fake.sock", device=2, use_gds=False
            )
        assert client.output_dir == subdir
        assert client.device == 2

    def test_restore_only_no_output_dir(self):
        """GMSStorageClient can be created without output_dir for restore-only use."""
        client = GMSStorageClient(socket_path="/tmp/fake.sock", device=0)
        assert client.output_dir is None

    def test_dump_without_output_dir_raises(self):
        """dump() raises ValueError when output_dir is None."""
        client = GMSStorageClient(socket_path="/tmp/fake.sock", device=0)
        with patch(
            "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE", True
        ):
            with pytest.raises(ValueError, match="output_dir"):
                client.save()

    def test_custom_shard_size(self):
        """shard_size_bytes parameter is stored."""
        client = GMSStorageClient(
            socket_path="/tmp/fake.sock", shard_size_bytes=128 * 1024 * 1024
        )
        assert client._shard_size == 128 * 1024 * 1024


# ===========================================================================
# TestGMSStorageClientDumpMock
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestGMSStorageClientSaveMock:
    """Primary correctness tests: full serialisation pipeline without GPU.

    Mocks GMSClientMemoryManager and _tensor_from_pointer so tests run
    entirely in CPU memory.
    """

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _make_cpu_tensor(size: int, fill: int = 42) -> "torch.Tensor":
        return torch.full((size,), fill, dtype=torch.uint8)

    @staticmethod
    def _build_mock_mm(allocations: List[Dict], metadata: Dict[str, Any]):
        """Return a MagicMock that mimics GMSClientMemoryManager API."""
        mm = MagicMock()
        mm.__enter__ = MagicMock(return_value=mm)
        mm.__exit__ = MagicMock(return_value=False)
        mm._client_rpc.committed = True
        mm._client_rpc.get_memory_layout_hash.return_value = "hash-abc"
        mm.list_allocations.return_value = allocations
        mm.import_allocation.side_effect = lambda aid: 0xDEAD_BEEF  # fake VA

        def _metadata_list():
            return list(metadata.keys())

        def _metadata_get(key):
            entry = metadata.get(key)
            if entry is None:
                return None
            return (entry["allocation_id"], entry["offset_bytes"], entry["value"])

        mm.metadata_list.side_effect = lambda: _metadata_list()
        mm.metadata_get.side_effect = _metadata_get
        return mm

    def _run_dump(
        self,
        tmpdir: str,
        allocations: List[Dict],
        metadata: Dict[str, Any],
        cpu_tensors: Dict[str, torch.Tensor],
        shard_size_bytes: int = 4 * 1024**3,
    ) -> SaveManifest:
        """Patch GMSClientMemoryManager and _tensor_from_pointer, then call dump()."""
        mock_mm = self._build_mock_mm(allocations, metadata)

        # Track call order to map import_allocation call → allocation id
        call_counter = [0]

        def fake_tensor_from_pointer(va, shape, stride, dtype, device):
            idx = call_counter[0]
            call_counter[0] += 1
            if idx >= len(allocations):
                raise IndexError(
                    f"Call index {idx} out of range for {len(allocations)} allocations"
                )
            alloc_id = allocations[idx]["allocation_id"]
            return cpu_tensors[alloc_id]

        client = GMSStorageClient(
            tmpdir,
            socket_path="/tmp/fake.sock",
            device=0,
            shard_size_bytes=shard_size_bytes,
        )

        with (
            patch(
                "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                return_value=mock_mm,
            ),
            patch(
                "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                side_effect=fake_tensor_from_pointer,
            ),
            patch(
                "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                True,
            ),
        ):
            manifest = client.save()

        return manifest

    # -- tests -------------------------------------------------------------

    def test_creates_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocations = [
                {"allocation_id": "a1", "size": 16, "aligned_size": 16, "tag": "t1"},
            ]
            cpu_tensors = {"a1": self._make_cpu_tensor(16, fill=1)}
            manifest = self._run_dump(tmpdir, allocations, {}, cpu_tensors)

            assert os.path.exists(os.path.join(tmpdir, "manifest.json"))
            assert os.path.exists(os.path.join(tmpdir, "gms_metadata.json"))
            tensor_file = os.path.join(tmpdir, manifest.allocations[0].tensor_file)
            assert os.path.exists(tensor_file)

    def test_uses_shard_format(self):
        """dump() packs allocations into shards/shard_*.bin files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allocations = [
                {"allocation_id": "a1", "size": 16, "aligned_size": 16, "tag": "t1"},
                {"allocation_id": "a2", "size": 32, "aligned_size": 32, "tag": "t2"},
            ]
            cpu_tensors = {
                "a1": self._make_cpu_tensor(16, fill=1),
                "a2": self._make_cpu_tensor(32, fill=2),
            }
            manifest = self._run_dump(tmpdir, allocations, {}, cpu_tensors)

        # Both allocations in shard files
        for entry in manifest.allocations:
            assert entry.tensor_file.startswith(
                "shards/"
            ), f"Expected shards/ prefix, got {entry.tensor_file!r}"

        # With default 4 GiB shard size, both fit in one shard
        shard_files = {a.tensor_file for a in manifest.allocations}
        assert len(shard_files) == 1

        # Offsets are sequential: first is 0, second is 16 (size of first)
        sorted_entries = sorted(manifest.allocations, key=lambda e: e.tensor_offset)
        assert sorted_entries[0].tensor_offset == 0
        assert sorted_entries[1].tensor_offset == 16

    def test_shard_rolling_with_small_shard_size(self):
        """Small shard_size causes allocations to roll into multiple shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allocations = [
                {
                    "allocation_id": f"a{i}",
                    "size": 64,
                    "aligned_size": 64,
                    "tag": f"t{i}",
                }
                for i in range(4)
            ]
            cpu_tensors = {f"a{i}": self._make_cpu_tensor(64, fill=i) for i in range(4)}
            # shard_size=100: first alloc writes 64 bytes; second alloc would
            # make 64+64=128 > 100, so it rolls to a new shard.  Each alloc
            # of 64 bytes triggers a roll → 4 shards for 4 allocations.
            manifest = self._run_dump(
                tmpdir, allocations, {}, cpu_tensors, shard_size_bytes=100
            )

        shard_files = {a.tensor_file for a in manifest.allocations}
        assert (
            len(shard_files) == 4
        ), f"Expected 4 shards (one per 64-byte alloc with 100-byte limit), got {shard_files}"

    def test_manifest_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocations = [
                {
                    "allocation_id": "a1",
                    "size": 16,
                    "aligned_size": 32,
                    "tag": "weights",
                },
                {
                    "allocation_id": "a2",
                    "size": 8,
                    "aligned_size": 16,
                    "tag": "kvcache",
                },
            ]
            cpu_tensors = {
                "a1": self._make_cpu_tensor(32, fill=1),
                "a2": self._make_cpu_tensor(16, fill=2),
            }
            manifest = self._run_dump(tmpdir, allocations, {}, cpu_tensors)

        assert manifest.version == _CURRENT_VERSION
        assert manifest.layout_hash == "hash-abc"
        assert manifest.device == 0
        assert manifest.use_gds is False
        assert len(manifest.allocations) == 2
        ids = {a.allocation_id for a in manifest.allocations}
        assert ids == {"a1", "a2"}

    def test_metadata_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allocations = [
                {"allocation_id": "a1", "size": 16, "aligned_size": 16, "tag": "t"},
            ]
            cpu_tensors = {"a1": self._make_cpu_tensor(16)}
            raw_meta = {
                "key1": {
                    "allocation_id": "a1",
                    "offset_bytes": 0,
                    "value": b"hello_meta",
                }
            }
            self._run_dump(tmpdir, allocations, raw_meta, cpu_tensors)

            with open(os.path.join(tmpdir, "gms_metadata.json")) as f:
                saved = json.load(f)

        assert "key1" in saved
        assert saved["key1"]["allocation_id"] == "a1"
        assert saved["key1"]["offset_bytes"] == 0
        decoded = base64.b64decode(saved["key1"]["value"])
        assert decoded == b"hello_meta"

    def test_full_dump_load_tensors_roundtrip(self):
        """Dump then load_tensors: tensor data must be bit-exact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_a = torch.arange(64, dtype=torch.uint8)
            data_b = torch.full((32,), 0xFF, dtype=torch.uint8)
            allocations = [
                {"allocation_id": "a1", "size": 64, "aligned_size": 64, "tag": "w"},
                {"allocation_id": "a2", "size": 32, "aligned_size": 32, "tag": "k"},
            ]
            cpu_tensors = {"a1": data_a, "a2": data_b}
            raw_meta = {
                "m1": {"allocation_id": "a1", "offset_bytes": 0, "value": b"meta1"},
            }
            self._run_dump(tmpdir, allocations, raw_meta, cpu_tensors)
            tensors, metadata = GMSStorageClient.load_tensors(tmpdir, max_workers=2)

        assert torch.equal(tensors["a1"].cpu(), data_a)
        assert torch.equal(tensors["a2"].cpu(), data_b)
        assert metadata["m1"]["value"] == b"meta1"
        assert metadata["m1"]["allocation_id"] == "a1"

    def test_empty_allocations(self):
        """Dump with no allocations produces valid (empty) manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._run_dump(tmpdir, [], {}, {})
        assert manifest.allocations == []

    def test_uncommitted_raises(self):
        """dump() raises RuntimeError when GMS has no committed weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._build_mock_mm([], {})
            mm._client_rpc.committed = False
            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock")
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
            ):
                with pytest.raises(RuntimeError, match="committed"):
                    client.save()


# ===========================================================================
# TestGMSDumpIntegration  (GPU required)
# ===========================================================================


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not _UVLOOP_AVAILABLE, reason="uvloop not available")
class TestGMSStorageIntegration:
    """Full integration test: real GMS server → dump → restore → verify."""

    @staticmethod
    def _run_server(
        socket_path: str, ready_event: threading.Event, stop_event: threading.Event
    ):
        """Start a GMS server in a background thread."""
        import uvloop
        from gpu_memory_service.server import GMSRPCServer

        async def _serve():
            server = GMSRPCServer(socket_path, device=0)
            await server.start()
            ready_event.set()
            # Poll until stop is requested
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
            await server.stop()

        uvloop.install()
        asyncio.run(_serve())

    def test_write_dump_restore_verify(self):
        import tempfile

        from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
        from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
        from gpu_memory_service.common.types import RequestedLockType

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "gms.sock")
            dump_dir = os.path.join(tmpdir, "dump")

            ready = threading.Event()
            stop = threading.Event()
            srv_thread = threading.Thread(
                target=self._run_server,
                args=(socket_path, ready, stop),
                daemon=True,
            )
            srv_thread.start()
            ready.wait(timeout=10)

            try:
                # Write two allocations
                with GMSClientMemoryManager(
                    socket_path, mode=RequestedLockType.RW, device=0
                ) as mm:
                    va1 = mm.allocate_and_map(2097152, tag="layer0")
                    t1 = _tensor_from_pointer(va1, [2097152], [1], torch.uint8, 0)
                    t1.fill_(0xAB)

                    va2 = mm.allocate_and_map(4194304, tag="layer1")
                    t2 = _tensor_from_pointer(va2, [4194304], [1], torch.uint8, 0)
                    t2.fill_(0xCD)

                    mm.metadata_put(
                        "t1_meta", list(mm._allocation_id_to_va.keys())[0], 0, b"meta1"
                    )
                    mm.commit()

                # Dump
                client = GMSStorageClient(dump_dir, socket_path=socket_path, device=0)
                manifest = client.save()

                assert len(manifest.allocations) == 2
                assert manifest.layout_hash != ""

                # Verify shard format
                for entry in manifest.allocations:
                    assert entry.tensor_file.startswith("shards/")

                # load_tensors: disk-only, returns standalone GPU tensors
                tensors, metadata = GMSStorageClient.load_tensors(dump_dir, device=0)

                assert len(tensors) == 2
                for tensor in tensors.values():
                    assert tensor.dtype == torch.uint8
                    assert tensor.is_cuda

                # Check metadata round-tripped
                assert "t1_meta" in metadata
                assert metadata["t1_meta"]["value"] == b"meta1"

            finally:
                stop.set()
                srv_thread.join(timeout=5)

    def test_dump_load_to_gms_roundtrip(self):
        """Full cycle: write → commit → dump → fresh server → load_to_gms → verify."""
        from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
        from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
        from gpu_memory_service.common.types import RequestedLockType

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "gms.sock")
            dump_dir = os.path.join(tmpdir, "dump")

            ready = threading.Event()
            stop = threading.Event()
            srv_thread = threading.Thread(
                target=self._run_server,
                args=(socket_path, ready, stop),
                daemon=True,
            )
            srv_thread.start()
            ready.wait(timeout=10)

            try:
                # Write and commit initial data
                with GMSClientMemoryManager(
                    socket_path, mode=RequestedLockType.RW, device=0
                ) as mm:
                    va1 = mm.allocate_and_map(2097152, tag="layerA")
                    t1 = _tensor_from_pointer(va1, [2097152], [1], torch.uint8, 0)
                    t1.fill_(0x11)

                    va2 = mm.allocate_and_map(2097152, tag="layerB")
                    t2 = _tensor_from_pointer(va2, [2097152], [1], torch.uint8, 0)
                    t2.fill_(0x22)

                    orig_alloc_ids = list(mm._allocation_id_to_va.keys())
                    mm.metadata_put("mA", orig_alloc_ids[0], 0, b"meta_a")
                    mm.metadata_put("mB", orig_alloc_ids[1], 4, b"meta_b")
                    mm.commit()

                # Dump
                dump_client = GMSStorageClient(
                    dump_dir, socket_path=socket_path, device=0
                )
                manifest = dump_client.save()
                assert len(manifest.allocations) == 2

                # Simulate fresh start: clear GMS by acquiring RW and clearing
                with GMSClientMemoryManager(
                    socket_path, mode=RequestedLockType.RW, device=0
                ) as mm:
                    mm.clear_all()
                    mm.commit()

                # load_to_gms: writes data back
                id_map = dump_client.load_to_gms(dump_dir, max_workers=2)

                assert len(id_map) == 2
                # All old IDs must be mapped to new IDs
                for old_id in [e.allocation_id for e in manifest.allocations]:
                    assert old_id in id_map

                # Verify data via RO reader
                with GMSClientMemoryManager(
                    socket_path, mode=RequestedLockType.RO, device=0
                ) as mm:
                    allocs = mm.list_allocations()
                    assert len(allocs) == 2

                    # Check tags preserved
                    tags = {a["allocation_id"]: a.get("tag") for a in allocs}
                    new_ids = list(id_map.values())
                    assert tags[new_ids[0]] in ("layerA", "layerB")
                    assert tags[new_ids[1]] in ("layerA", "layerB")

                    # Check tensor values
                    for alloc in allocs:
                        va = mm.import_allocation(alloc["allocation_id"])
                        t = _tensor_from_pointer(
                            va, [alloc["aligned_size"]], [1], torch.uint8, 0
                        )
                        fill = 0x11 if alloc.get("tag") == "layerA" else 0x22
                        assert (
                            t[0].item() == fill
                        ), f"Wrong fill for {alloc['allocation_id']}"

                    # Check metadata keys survived and have bytes values
                    assert set(mm.metadata_list()) == {"mA", "mB"}
                    got_a = mm.metadata_get("mA")
                    assert got_a is not None
                    assert got_a[2] == b"meta_a"
                    got_b = mm.metadata_get("mB")
                    assert got_b is not None
                    assert got_b[2] == b"meta_b"

            finally:
                stop.set()
                srv_thread.join(timeout=5)

    def test_true_restart_roundtrip(self):
        """The primary workflow test: start → write → dump → RESTART → restore → verify.

        The server thread is fully stopped between dump and restore so the
        socket file is deleted and a brand-new server process starts with
        EMPTY state (committed=False).  This is the exact production scenario.
        """
        from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
        from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
        from gpu_memory_service.common.types import RequestedLockType

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "gms.sock")
            dump_dir = os.path.join(tmpdir, "dump")

            # ---- Phase 1: start server --------------------------------
            ready1 = threading.Event()
            stop1 = threading.Event()
            srv1 = threading.Thread(
                target=self._run_server,
                args=(socket_path, ready1, stop1),
                daemon=True,
            )
            srv1.start()
            assert ready1.wait(timeout=10), "Server 1 did not start"

            # ---- Phase 2: write and commit ----------------------------
            with GMSClientMemoryManager(
                socket_path, mode=RequestedLockType.RW, device=0
            ) as mm:
                va1 = mm.allocate_and_map(2097152, tag="layer0")
                t1 = _tensor_from_pointer(va1, [2097152], [1], torch.uint8, 0)
                t1.fill_(0xAA)

                va2 = mm.allocate_and_map(2097152, tag="layer1")
                t2 = _tensor_from_pointer(va2, [2097152], [1], torch.uint8, 0)
                t2.fill_(0xBB)

                alloc_ids = list(mm._allocation_id_to_va.keys())
                mm.metadata_put("w0", alloc_ids[0], 0, b"weight_meta_0")
                mm.metadata_put("w1", alloc_ids[1], 0, b"weight_meta_1")
                mm.commit()

            # ---- Phase 3: dump ----------------------------------------
            dump_client = GMSStorageClient(dump_dir, socket_path=socket_path, device=0)
            manifest = dump_client.save()
            assert len(manifest.allocations) == 2
            # Verify shard format was used
            for entry in manifest.allocations:
                assert entry.tensor_file.startswith("shards/")

            # ---- Phase 4: RESTART server (stop old, start new) --------
            stop1.set()
            srv1.join(timeout=10)
            assert not srv1.is_alive(), "Server 1 did not stop"
            # Socket file must be gone after stop()
            assert not os.path.exists(socket_path), "Socket file was not cleaned up"

            ready2 = threading.Event()
            stop2 = threading.Event()
            srv2 = threading.Thread(
                target=self._run_server,
                args=(socket_path, ready2, stop2),
                daemon=True,
            )
            srv2.start()
            assert ready2.wait(timeout=10), "Server 2 did not start"

            try:
                # ---- Phase 5: restore into fresh server ---------------
                restore_client = GMSStorageClient(socket_path=socket_path, device=0)
                id_map = restore_client.load_to_gms(dump_dir, max_workers=2)

                assert len(id_map) == 2, f"Expected 2 remapped IDs, got {id_map}"

                # ---- Phase 6: verify via RO reader --------------------
                with GMSClientMemoryManager(
                    socket_path, mode=RequestedLockType.RO, device=0
                ) as mm:
                    allocs = mm.list_allocations()
                    assert len(allocs) == 2

                    tag_to_alloc = {a.get("tag"): a for a in allocs}
                    assert "layer0" in tag_to_alloc
                    assert "layer1" in tag_to_alloc

                    # Verify tensor values
                    for tag, fill in (("layer0", 0xAA), ("layer1", 0xBB)):
                        a = tag_to_alloc[tag]
                        va = mm.import_allocation(a["allocation_id"])
                        t = _tensor_from_pointer(
                            va, [a["aligned_size"]], [1], torch.uint8, 0
                        )
                        assert (
                            t[0].item() == fill
                        ), f"{tag}: expected 0x{fill:02X}, got 0x{t[0].item():02X}"

                    # Verify metadata
                    assert set(mm.metadata_list()) == {"w0", "w1"}
                    assert mm.metadata_get("w0")[2] == b"weight_meta_0"
                    assert mm.metadata_get("w1")[2] == b"weight_meta_1"

            finally:
                stop2.set()
                srv2.join(timeout=5)


# ===========================================================================
# TestGMSStorageClientRestoreMock  (no GPU)
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestGMSStorageClientLoadMock:
    """Tests for load_to_gms() using mocked GMSClientMemoryManager.

    No GPU required.  Verifies: allocation IDs remapped correctly, tensor data
    copied into GMS memory, metadata written with new IDs, commit called.
    """

    @staticmethod
    def _build_dump_dir(tmpdir: str, num_allocs: int = 2) -> Dict[str, Any]:
        """Create a minimal dump directory using legacy .pt format."""
        tensors_dir = os.path.join(tmpdir, "tensors")
        os.makedirs(tensors_dir)

        alloc_ids = [f"orig-{i}" for i in range(num_allocs)]
        entries = []
        cpu_data: Dict[str, torch.Tensor] = {}
        for i, aid in enumerate(alloc_ids):
            data = torch.full((64,), i + 1, dtype=torch.uint8)
            cpu_data[aid] = data
            path = os.path.join(tensors_dir, f"{aid}.pt")
            torch.save(data, path)
            entries.append(AllocationEntry(aid, 64, 64, f"tag{i}", f"tensors/{aid}.pt"))

        meta_raw = {
            "key0": {
                "allocation_id": alloc_ids[0],
                "offset_bytes": 0,
                "value": base64.b64encode(b"v0").decode(),
            },
        }
        if num_allocs > 1:
            meta_raw["key1"] = {
                "allocation_id": alloc_ids[1],
                "offset_bytes": 8,
                "value": base64.b64encode(b"v1").decode(),
            }

        with open(os.path.join(tmpdir, "gms_metadata.json"), "w") as f:
            json.dump(meta_raw, f)

        manifest = SaveManifest(
            version=_CURRENT_VERSION,
            timestamp=time.time(),
            layout_hash="abc",
            device=0,
            use_gds=False,
            gds_available=False,
            allocations=entries,
        )
        with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
            json.dump(manifest.to_dict(), f)

        return {"alloc_ids": alloc_ids, "cpu_data": cpu_data}

    @staticmethod
    def _build_mock_rw_mm(alloc_ids_iter):
        """Mock that simulates allocate_and_map, _mappings, metadata_put, commit."""

        mm = MagicMock()
        mm.__enter__ = MagicMock(return_value=mm)
        mm.__exit__ = MagicMock(return_value=False)

        # Each call to allocate_and_map returns a fake VA; new allocation IDs
        # are stored in mm._mappings keyed by that VA.
        call_count = [0]
        va_base = 0x1000_0000

        new_ids = [f"new-{i}" for i in range(10)]  # pool of new IDs

        def _alloc_and_map(size, tag="default"):
            idx = call_count[0]
            call_count[0] += 1
            va = va_base + idx * 0x100_0000

            # Build a LocalMapping-like namedtuple
            mapping = MagicMock()
            mapping.allocation_id = new_ids[idx]
            mapping.aligned_size = size
            mm._mappings[va] = mapping

            return va

        mm._mappings = {}
        mm.allocate_and_map = MagicMock(side_effect=_alloc_and_map)
        mm.clear_all = MagicMock(return_value=0)
        mm.metadata_put = MagicMock(return_value=True)
        mm.commit = MagicMock(return_value=True)

        return mm, new_ids

    def test_id_map_returned(self):
        """load_to_gms returns a dict mapping old → new allocation IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = self._build_dump_dir(tmpdir, num_allocs=2)
            mock_mm, new_ids = self._build_mock_rw_mm(None)

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=torch.zeros(64, dtype=torch.uint8),
                ),
            ):
                id_map = client.load_to_gms(tmpdir)

        assert len(id_map) == 2
        for old_id in info["alloc_ids"]:
            assert old_id in id_map
            assert id_map[old_id].startswith("new-")

    def test_metadata_remapped(self):
        """Metadata allocation IDs are translated to new IDs before metadata_put."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir, num_allocs=2)
            mock_mm, new_ids = self._build_mock_rw_mm(None)

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=torch.zeros(64, dtype=torch.uint8),
                ),
            ):
                client.load_to_gms(tmpdir)

        # metadata_put should have been called with the NEW alloc IDs
        calls = mock_mm.metadata_put.call_args_list
        put_keys = {c.args[0] for c in calls}
        assert "key0" in put_keys
        assert "key1" in put_keys

        # The allocation_id argument must be from the new ID pool, not the old
        for c in calls:
            _key, alloc_id_arg, _offset, _value = c.args
            assert alloc_id_arg.startswith(
                "new-"
            ), f"Expected remapped new ID, got {alloc_id_arg!r}"

    def test_commit_called(self):
        """commit() is called exactly once after all allocations are restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir, num_allocs=1)
            mock_mm, _ = self._build_mock_rw_mm(None)

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=torch.zeros(64, dtype=torch.uint8),
                ),
            ):
                client.load_to_gms(tmpdir)

        mock_mm.commit.assert_called_once()

    def test_clear_existing_default(self):
        """clear_all() is called by default (clear_existing=True)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir, num_allocs=1)
            mock_mm, _ = self._build_mock_rw_mm(None)

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=torch.zeros(64, dtype=torch.uint8),
                ),
            ):
                client.load_to_gms(tmpdir, clear_existing=True)

        mock_mm.clear_all.assert_called_once()

    def test_no_clear_skips_clear_all(self):
        """clear_all() is NOT called when clear_existing=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_dump_dir(tmpdir, num_allocs=1)
            mock_mm, _ = self._build_mock_rw_mm(None)

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=torch.zeros(64, dtype=torch.uint8),
                ),
            ):
                client.load_to_gms(tmpdir, clear_existing=False)

        mock_mm.clear_all.assert_not_called()

    def test_tensor_data_copied(self):
        """The tensor data loaded from disk is copied into the GMS memory view."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = self._build_dump_dir(tmpdir, num_allocs=1)
            mock_mm, _ = self._build_mock_rw_mm(None)

            # Capture what gets written into dst_tensor via copy_()
            copied = {}

            dst_mock = MagicMock(spec=torch.Tensor)
            dst_mock.copy_ = MagicMock(
                side_effect=lambda src, **_kw: copied.update({"src": src})
            )

            client = GMSStorageClient(tmpdir, socket_path="/tmp/fake.sock", device=0)
            with (
                patch(
                    "gpu_memory_service.client.gms_storage_client.GMSClientMemoryManager",
                    return_value=mock_mm,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._GMS_IMPORTS_AVAILABLE",
                    True,
                ),
                patch(
                    "gpu_memory_service.client.gms_storage_client._tensor_from_pointer",
                    return_value=dst_mock,
                ),
            ):
                client.load_to_gms(tmpdir)

        dst_mock.copy_.assert_called_once()
        src_tensor = copied["src"]
        expected = info["cpu_data"][info["alloc_ids"][0]]
        assert torch.equal(src_tensor.cpu(), expected)


# ===========================================================================
# TestNixlGdsWriter  (NIXL required)
# ===========================================================================


@pytest.mark.skipif(not _NIXL_AVAILABLE, reason="nixl not installed")
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
class TestNixlGdsWriter:
    def test_posix_backend_roundtrip(self):
        """Write a CPU-accessible tensor via NIXL and read it back."""
        writer = _NixlGdsWriter()
        if not writer.is_active:
            pytest.skip("No NIXL backend available")

        # Requires CUDA for GPU memory registration
        if not _CUDA_AVAILABLE:
            pytest.skip("CUDA required for NIXL memory registration")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bin")
            size = 4096

            src = torch.arange(size, dtype=torch.uint8, device="cuda:0")
            try:
                writer.write_tensor(src, path)
            except Exception as exc:
                pytest.skip(f"NIXL write_tensor failed (likely unsupported fs): {exc}")

            assert os.path.getsize(path) == size

            loaded = writer.read_tensor(path, size, torch.uint8, [size], 0)
            assert torch.equal(src.cpu(), loaded.cpu())

    @pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required for GDS")
    def test_gds_backend_roundtrip(self):
        """GDS path: requires GDS-capable NVMe hardware; skips otherwise."""
        writer = _NixlGdsWriter()
        if not writer.is_gds:
            pytest.skip("GDS backend not available (hardware or driver missing)")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gds_test.bin")
            size = 8192
            src = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda:0")
            try:
                writer.write_tensor(src, path)
            except Exception as exc:
                pytest.skip(f"GDS write failed (likely unsupported filesystem): {exc}")
            loaded = writer.read_tensor(path, size, torch.uint8, [size], 0)
        assert torch.equal(src.cpu(), loaded.cpu())

    def test_is_active_property(self):
        writer = _NixlGdsWriter()
        # is_active should be True if any backend was initialised
        assert isinstance(writer.is_active, bool)

    def test_is_gds_property(self):
        writer = _NixlGdsWriter()
        assert isinstance(writer.is_gds, bool)
        if writer.is_gds:
            assert writer.is_active
