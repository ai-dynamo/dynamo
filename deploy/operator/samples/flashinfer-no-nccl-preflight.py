# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank FlashInfer checkpoint lifecycle and no-NCCL preflight."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_DISABLE_NCCL", "1")
os.environ.setdefault("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")

import torch
import torch.distributed as dist

EXPECTED_VLLM_SHA = "855054d3a61ad0c8597ed29d6d1979cbebdb475a"
EXPECTED_FLASHINFER_SHA = "330cc8e1a09f59c1241084459f3df3204b9b8327"
EXPECTED_FLASHINFER_VERSION = "0.6.14"
EXPECTED_WORLD_SIZE = 8


def _read_provenance() -> dict[str, str]:
    path = Path("/opt/dynamo/source-provenance.txt")
    if not path.is_file():
        raise RuntimeError(f"Missing source provenance: {path}")
    return dict(
        line.strip().split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def _module_origin(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Native module is unavailable: {name}")
    importlib.import_module(name)
    return spec.origin


def _elf_nccl_dependencies(paths: dict[str, str]) -> dict[str, list[str]]:
    dependencies: dict[str, list[str]] = {}
    for name, path in paths.items():
        result = subprocess.run(
            ["readelf", "-d", path],
            check=True,
            capture_output=True,
            text=True,
        )
        dependencies[name] = [
            line.strip()
            for line in result.stdout.splitlines()
            if "NEEDED" in line and "nccl" in line.lower()
        ]
    return dependencies


def _mapped_nccl_dsos() -> list[str]:
    maps = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    return sorted({line.split()[-1] for line in maps if "nccl" in line.lower()})


def _validate_runtime() -> dict[str, Any]:
    provenance = _read_provenance()
    expected_provenance = {
        "install_mode": "full-native-source",
        "vllm_source_sha": EXPECTED_VLLM_SHA,
        "flashinfer_source_sha": EXPECTED_FLASHINFER_SHA,
        "flashinfer_source_version": EXPECTED_FLASHINFER_VERSION,
    }
    for key, expected in expected_provenance.items():
        actual = provenance.get(key)
        if actual != expected:
            raise RuntimeError(f"Provenance {key}={actual!r}, expected {expected!r}")

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "nvidia-nccl-cu13",
            "flashinfer-python",
            "flashinfer-cubin",
        )
    }
    expected_versions = {
        "torch": "2.12.0+cu130",
        "torchvision": "0.27.0+cu130",
        "nvidia-nccl-cu13": "2.29.7",
        "flashinfer-python": EXPECTED_FLASHINFER_VERSION,
        "flashinfer-cubin": EXPECTED_FLASHINFER_VERSION,
    }
    if versions != expected_versions:
        raise RuntimeError(f"Unexpected package versions: {versions!r}")

    from flashinfer.comm import SymmetricAllGatherWorkspace
    from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory
    from vllm.plugins import load_general_plugins
    from vllm.utils.import_utils import has_deep_ep_v2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import GPUWorker

    if not callable(SymmetricAllGatherWorkspace.checkpoint_prepare):
        raise RuntimeError("Symmetric all-gather checkpoint API is missing")
    if not callable(MoeAlltoAll.checkpoint_restore):
        raise RuntimeError("One-sided MoE checkpoint API is missing")
    if has_deep_ep_v2():
        raise RuntimeError("DeepEPv2 must be disabled without probing NCCL")
    if hasattr(GPUWorker, "checkpoint_prepare") or hasattr(
        GPUWorker, "checkpoint_restore"
    ):
        raise RuntimeError("GPUWorker must not contain snapshot lifecycle overrides")
    wake_source = inspect.getsource(GPUModelRunner.init_fp8_kv_scales)
    if "isinstance(cache_entry, list)" not in wake_source:
        raise RuntimeError("Hybrid FP8 KV wake fix is missing")

    plugin_names = {
        entry_point.name
        for entry_point in importlib.metadata.entry_points(group="vllm.general_plugins")
    }
    if "dynamo_snapshot" not in plugin_names:
        raise RuntimeError(f"Dynamo snapshot plugin is missing: {plugin_names}")
    load_general_plugins()
    backend = SleepModeBackendFactory.get_backend_class("dynamo_snapshot")
    if not backend.preserves_communicators():
        raise RuntimeError("Dynamo snapshot backend does not preserve communicators")
    if not backend.preserves_graphs_with_communicators():
        raise RuntimeError("Dynamo snapshot backend does not preserve graphs")

    native_modules = {
        name: _module_origin(name)
        for name in (
            "vllm._C_stable_libtorch",
            "vllm._moe_C_stable_libtorch",
            "vllm.cumem_allocator",
            "vllm.vllm_flash_attn._vllm_fa2_C",
        )
    }
    return {
        "provenance": provenance,
        "versions": versions,
        "native_modules": native_modules,
        "elf_nccl_dependencies": _elf_nccl_dependencies(native_modules),
    }


def _assert_raises_detached(operation: Any, label: str) -> None:
    try:
        operation()
    except RuntimeError:
        return
    raise RuntimeError(f"{label} unexpectedly ran with detached workspace")


def _expected_all_gather(
    world_size: int, shape: tuple[int, ...], offset: int
) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                shape,
                rank + offset,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for rank in range(world_size)
        ],
        dim=0,
    )


def _run_collectives(rank: int, world_size: int) -> dict[str, Any]:
    import flashinfer.comm as comm
    from flashinfer.comm import MoeAlltoAll, SymmetricAllGatherWorkspace
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig, TorchDistBackend
    from vllm.distributed.device_communicators.mnnvl_compat import (
        CustomCommunicator,
    )

    control_group = dist.group.WORLD
    ar_workspace = comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=world_size,
        rank=rank,
        max_token_num=4,
        hidden_dim=4096,
        dtype=torch.bfloat16,
        comm_backend=TorchDistBackend(group=control_group),
    )
    ar_input = torch.full((4, 4096), rank + 1, dtype=torch.bfloat16, device="cuda")
    ar_output = torch.empty_like(ar_input)

    def all_reduce() -> None:
        comm.allreduce_fusion(
            input=ar_input,
            workspace=ar_workspace,
            output=ar_output,
            pattern=comm.AllReduceFusionPattern.kAllReduce,
            use_oneshot=False,
        )

    all_reduce()
    torch.cuda.synchronize()
    torch.testing.assert_close(ar_output, torch.full_like(ar_output, 36))
    ar_graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(ar_graph):
        all_reduce()

    ag_shape = (8, 64)
    ag_workspace = SymmetricAllGatherWorkspace(
        max_elems=ag_shape[0] * ag_shape[1],
        world_size=world_size,
        rank=rank,
        comm_backend=TorchDistBackend(group=control_group),
        dtype=torch.bfloat16,
    )
    ag_input = torch.full(ag_shape, rank + 1, dtype=torch.bfloat16, device="cuda")
    ag_output = torch.empty(
        (ag_shape[0] * world_size, ag_shape[1]),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ag_workspace.all_gather(ag_input, ag_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(ag_output, _expected_all_gather(world_size, ag_shape, 1))
    ag_graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(ag_graph):
        ag_workspace.all_gather(ag_input, ag_output)

    max_tokens = 4
    hidden_size = 64
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
    )
    moe_workspace_size = MoeAlltoAll.get_moe_workspace_size_per_rank(
        world_size,
        top_k=1,
        max_num_tokens=max_tokens,
        hidden_size=hidden_size,
    )
    moe_workspace = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_tokens,
        top_k=1,
        num_experts=world_size,
        workspace_size_per_rank=moe_workspace_size,
        mnnvl_config=MnnvlConfig(
            comm_backend=CustomCommunicator(control_group),
        ),
    )
    moe_input = torch.full(
        (max_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    remote_expert = (rank + 1) % world_size
    selected_experts = torch.full(
        (max_tokens, 1),
        remote_expert,
        dtype=torch.int32,
        device="cuda",
    )

    def moe_round_trip() -> torch.Tensor:
        received = moe_workspace.dispatch(
            selected_experts,
            [moe_input],
            max_tokens,
        )[0]
        combine_payload = moe_workspace.get_combine_payload_tensor_in_workspace(
            max_tokens,
            hidden_size,
            torch.bfloat16,
        )
        combine_payload.copy_(received)
        return moe_workspace.combine(
            combine_payload,
            max_tokens,
            payload_in_workspace=True,
        )

    moe_output = moe_round_trip()
    torch.cuda.synchronize()
    torch.testing.assert_close(moe_output, moe_input)
    moe_graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(moe_graph):
        moe_output = moe_round_trip()
    torch.cuda.synchronize()
    torch.testing.assert_close(moe_output, moe_input)

    dist.barrier()
    for workspace in (ar_workspace, ag_workspace, moe_workspace):
        workspace.checkpoint_prepare()
        workspace.checkpoint_prepare()
    _assert_raises_detached(all_reduce, "all-reduce")
    _assert_raises_detached(
        lambda: ag_workspace.all_gather(ag_input, ag_output),
        "all-gather",
    )
    _assert_raises_detached(moe_round_trip, "one-sided MoE")

    dist.barrier()
    ar_backend = TorchDistBackend(group=control_group)
    ag_backend = TorchDistBackend(group=control_group)
    moe_backend = CustomCommunicator(control_group)
    ar_workspace.checkpoint_restore(ar_backend)
    ar_workspace.checkpoint_restore(ar_backend)
    ag_workspace.checkpoint_restore(ag_backend)
    ag_workspace.checkpoint_restore(ag_backend)
    moe_workspace.checkpoint_restore(moe_backend)
    moe_workspace.checkpoint_restore(moe_backend)
    dist.barrier()

    ar_input.fill_(rank + 2)
    ag_input.fill_(rank + 8)
    moe_input.fill_(rank + 16)
    torch.cuda.synchronize()
    ar_graph.replay()
    ag_graph.replay()
    moe_graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(ar_output, torch.full_like(ar_output, 44))
    torch.testing.assert_close(ag_output, _expected_all_gather(world_size, ag_shape, 8))
    torch.testing.assert_close(moe_output, moe_input)

    dist.barrier()
    ar_workspace.destroy()
    ag_workspace.destroy()
    return {
        "all_reduce": "eager+graph+post-restore-graph",
        "symmetric_all_gather": "eager+graph+post-restore-graph",
        "one_sided_moe": "remote-dispatch+combine+post-restore-graph",
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_WORLD_SIZE} ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    try:
        runtime = _validate_runtime()
        collectives = _run_collectives(rank, world_size)

        from vllm.distributed.nccl_audit import (
            assert_no_nccl_communicators,
            get_nccl_audit_events,
        )

        backends = assert_no_nccl_communicators()
        audit_events = get_nccl_audit_events()
        if audit_events:
            raise RuntimeError(f"NCCL creation events were recorded: {audit_events}")
        report = {
            "rank": rank,
            "local_rank": local_rank,
            "pid": os.getpid(),
            "device": torch.cuda.get_device_name(local_rank),
            "process_groups": backends,
            "nccl_creation_events": audit_events,
            "mapped_nccl_dsos": _mapped_nccl_dsos(),
            "collectives": collectives,
            **runtime,
        }
        reports: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(reports, report)
        if rank == 0:
            if any(item is None for item in reports):
                raise RuntimeError(f"Missing rank reports: {reports!r}")
            print(
                "FLASHINFER_NO_NCCL_PREFLIGHT=" + json.dumps(reports, sort_keys=True),
                flush=True,
            )
            print("FLASHINFER_NO_NCCL_PREFLIGHT_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
