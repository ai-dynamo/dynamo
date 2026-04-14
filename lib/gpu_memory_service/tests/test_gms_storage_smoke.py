# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from gpu_memory_service.cli.storage_runner import _build_parser
from gpu_memory_service.client._gms_storage_disk import load_manifest_and_metadata
from gpu_memory_service.client._gms_storage_model import AllocationEntry, SaveManifest
from gpu_memory_service.common import cuda_utils


class GMSStorageSmokeTest(unittest.TestCase):
    def test_manifest_round_trip(self) -> None:
        manifest = SaveManifest(
            version="1.0",
            timestamp=123.0,
            layout_hash="abc",
            device=2,
            allocations=[
                AllocationEntry(
                    allocation_id="alloc-1",
                    size=16,
                    aligned_size=32,
                    tag="weights",
                    tensor_file="shards/shard_0000.bin",
                    tensor_offset=64,
                )
            ],
        )

        restored = SaveManifest.from_dict(manifest.to_dict())

        self.assertEqual(restored.version, "1.0")
        self.assertEqual(restored.layout_hash, "abc")
        self.assertEqual(restored.device, 2)
        self.assertEqual(len(restored.allocations), 1)
        self.assertEqual(restored.allocations[0].tensor_offset, 64)

    def test_load_manifest_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = {
                "version": "1.0",
                "timestamp": 1.0,
                "layout_hash": "layout",
                "device": 0,
                "allocations": [
                    {
                        "allocation_id": "alloc-1",
                        "size": 4,
                        "aligned_size": 8,
                        "tag": "weights",
                        "tensor_file": "shards/shard_0000.bin",
                        "tensor_offset": 0,
                    }
                ],
            }
            metadata = {
                "tensor-key": {
                    "allocation_id": "alloc-1",
                    "offset_bytes": 0,
                    "value": base64.b64encode(b"payload").decode("ascii"),
                }
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "gms_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            loaded_manifest, loaded_metadata = load_manifest_and_metadata(tmpdir)

            self.assertEqual(loaded_manifest.layout_hash, "layout")
            self.assertEqual(loaded_manifest.allocations[0].allocation_id, "alloc-1")
            self.assertEqual(loaded_metadata["tensor-key"]["value"], b"payload")

    def test_cli_parser_builds_save_and_load_commands(self) -> None:
        parser = _build_parser()

        save_args = parser.parse_args(["save", "--output-dir", "/tmp/out"])
        load_args = parser.parse_args(["load", "--input-dir", "/tmp/in"])

        self.assertEqual(save_args.subcommand, "save")
        self.assertEqual(save_args.device, 0)
        self.assertEqual(load_args.subcommand, "load")
        self.assertEqual(load_args.workers, 4)

    def test_cuda_set_current_device_initializes_before_retain(self) -> None:
        calls: list[object] = []
        fake_cuda = types.SimpleNamespace(
            CUresult=types.SimpleNamespace(CUDA_SUCCESS=0),
            cuInit=lambda flags: calls.append("cuInit") or (0,),
            cuDevicePrimaryCtxRetain=lambda device: calls.append(
                ("retain", device)
            ) or (0, f"ctx-{device}"),
            cuCtxSetCurrent=lambda ctx: calls.append(("set", ctx)) or (0,),
            cuDevicePrimaryCtxRelease=lambda device: (0,),
        )

        with mock.patch.object(cuda_utils, "cuda", fake_cuda):
            with mock.patch.object(cuda_utils, "_primary_contexts", {}):
                with mock.patch.object(
                    cuda_utils,
                    "_primary_context_release_registered",
                    False,
                ):
                    cuda_utils.cuda_set_current_device(3)

        self.assertEqual(calls[0], "cuInit")
        self.assertEqual(calls[1], ("retain", 3))
        self.assertEqual(calls[2], ("set", "ctx-3"))

    def test_cumem_get_allocation_granularity_initializes_first(self) -> None:
        calls: list[object] = []

        class _Prop:
            def __init__(self) -> None:
                self.type = None
                self.location = types.SimpleNamespace(type=None, id=None)
                self.requestedHandleTypes = None

        fake_cuda = types.SimpleNamespace(
            CUresult=types.SimpleNamespace(CUDA_SUCCESS=0),
            CUmemAllocationProp=_Prop,
            CUmemAllocationType=types.SimpleNamespace(
                CU_MEM_ALLOCATION_TYPE_PINNED=1
            ),
            CUmemLocationType=types.SimpleNamespace(
                CU_MEM_LOCATION_TYPE_DEVICE=2
            ),
            CUmemAllocationHandleType=types.SimpleNamespace(
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=3
            ),
            CUmemAllocationGranularity_flags=types.SimpleNamespace(
                CU_MEM_ALLOC_GRANULARITY_MINIMUM=4
            ),
            cuInit=lambda flags: calls.append("cuInit") or (0,),
            cuMemGetAllocationGranularity=lambda prop, flag: calls.append(
                ("granularity", prop.location.id, flag)
            ) or (0, 2097152),
        )

        with mock.patch.object(cuda_utils, "cuda", fake_cuda):
            granularity = cuda_utils.cumem_get_allocation_granularity(5)

        self.assertEqual(granularity, 2097152)
        self.assertEqual(calls[0], "cuInit")
        self.assertEqual(calls[1], ("granularity", 5, 4))


if __name__ == "__main__":
    unittest.main()
