# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank FlashInfer checkpoint lifecycle with an idle PyNCCL communicator."""

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

import torch
import torch.distributed as dist

EXPECTED_VLLM_SHA = os.environ["EXPECTED_VLLM_SHA"]
EXPECTED_FLASHINFER_SHA = "330cc8e1a09f59c1241084459f3df3204b9b8327"
EXPECTED_FLASHINFER_VERSION = "0.6.14"
EXPECTED_WORLD_SIZE = 8
EXPECTED_NCCL_ENV = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_RAS_ENABLE": "0",
    "NCCL_DEBUG": "INFO",
}


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
    expected_dynamo_sha = os.environ["EXPECTED_DYNAMO_SHA"]
    actual_dynamo_sha = os.environ.get("DYNAMO_COMMIT_SHA")
    if actual_dynamo_sha != expected_dynamo_sha:
        raise RuntimeError(
            f"Dynamo image commit is {actual_dynamo_sha!r}, "
            f"expected {expected_dynamo_sha!r}"
        )

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

    import vllm.envs as vllm_envs

    if importlib.util.find_spec("nccl_checkpoint") is not None:
        raise RuntimeError("NCCL checkpoint Python package must not be installed")
    if Path("/opt/nccl-checkpoint").exists():
        raise RuntimeError("NCCL checkpoint native prefix must not exist")
    preload = Path("/etc/ld.so.preload")
    if preload.is_file() and "nccl-checkpoint" in preload.read_text(encoding="utf-8"):
        raise RuntimeError("NCCL checkpoint shim must not be preloaded")
    if "nccl-checkpoint" in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError("NCCL checkpoint shim must not be in LD_PRELOAD")
    nccl_env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(("NCCL_", "TORCH_NCCL_"))
        or name == "DYN_SNAPSHOT_NCCL_KVS_ENDPOINT"
    }
    for name, expected in EXPECTED_NCCL_ENV.items():
        actual = nccl_env.get(name)
        if actual != expected:
            raise RuntimeError(f"{name}={actual!r}, expected {expected!r}")
    if "NCCL_NET" in nccl_env:
        raise RuntimeError("NCCL_NET must remain unset so SHM transport is available")
    if "DYN_SNAPSHOT_NCCL_KVS_ENDPOINT" in nccl_env:
        raise RuntimeError("NCCL checkpoint KVS must not be configured")
    if any(name.startswith("NCCL_CHECKPOINT_") for name in nccl_env):
        raise RuntimeError(f"NCCL checkpoint environment is present: {nccl_env}")
    if os.environ.get("VLLM_DISABLE_NCCL"):
        raise RuntimeError("VLLM_DISABLE_NCCL must remain unset")
    if vllm_envs.VLLM_DISABLE_PYNCCL:
        raise RuntimeError("PyNCCL must remain enabled")
    if not vllm_envs.VLLM_ALLREDUCE_USE_FLASHINFER:
        raise RuntimeError("FlashInfer all-reduce selection is disabled")
    from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory
    from vllm.plugins import load_general_plugins
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    if not callable(PyNcclCommunicator):
        raise RuntimeError("PyNCCL communicator is unavailable")
    if not callable(MoeAlltoAll.checkpoint_restore):
        raise RuntimeError("One-sided MoE checkpoint API is missing")
    if hasattr(Worker, "checkpoint_prepare") or hasattr(Worker, "checkpoint_restore"):
        raise RuntimeError("GPU worker must not contain snapshot lifecycle overrides")
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
    native_modules["torch.libtorch_cuda"] = str(
        Path(torch.__file__).parent / "lib/libtorch_cuda.so"
    )
    return {
        "dynamo_commit_sha": actual_dynamo_sha,
        "provenance": provenance,
        "versions": versions,
        "nccl_env": nccl_env,
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "native_modules": native_modules,
        "elf_nccl_dependencies": _elf_nccl_dependencies(native_modules),
    }


def _assert_raises_detached(operation: Any, label: str) -> None:
    try:
        operation()
    except RuntimeError:
        return
    raise RuntimeError(f"{label} unexpectedly ran with detached workspace")


def _run_collectives(rank: int, world_size: int) -> dict[str, Any]:
    from flashinfer.comm import MoeAlltoAll
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        checkpoint_prepare_fi_ar_workspaces,
        checkpoint_restore_fi_ar_workspaces,
    )
    from vllm.distributed.device_communicators.mnnvl_compat import (
        CustomCommunicator,
    )

    control_group = dist.group.WORLD
    parallel_state._ENABLE_CUSTOM_ALL_REDUCE = False
    parallel_state._NODE_COUNT = 1
    communicator = CudaCommunicator(
        cpu_group=control_group,
        device=torch.device("cuda", rank),
        device_group=control_group,
        unique_name="tp:flashinfer-pynccl-shm-preflight",
    )
    pynccl = communicator.pynccl_comm
    if pynccl is None or not pynccl.available or pynccl.disabled:
        raise RuntimeError("PyNCCL communicator was not initialized")
    pynccl_handle_before = int(pynccl.comm.value or 0)
    if pynccl_handle_before == 0:
        raise RuntimeError("PyNCCL communicator has an empty NCCL handle")

    pynccl_model_fallbacks = {"all_reduce": 0}
    pynccl_all_reduce = pynccl.all_reduce

    def unexpected_pynccl_all_reduce(*args, **kwargs):
        pynccl_model_fallbacks["all_reduce"] += 1
        raise RuntimeError("Model all-reduce unexpectedly fell through to PyNCCL")

    pynccl.all_reduce = unexpected_pynccl_all_reduce

    ar_token_num = max(4, world_size * 2)
    ar_input = torch.full(
        (ar_token_num, 4096),
        rank + 1,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def all_reduce() -> torch.Tensor:
        return communicator.all_reduce(ar_input)

    ar_output = all_reduce()
    torch.cuda.synchronize()
    initial_sum = world_size * (world_size + 1) // 2
    torch.testing.assert_close(ar_output, torch.full_like(ar_output, initial_sum))
    ar_graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(ar_graph):
        ar_graph_output = all_reduce()

    if getattr(communicator, "fi_ag_workspaces", {}):
        raise RuntimeError("FlashInfer all-gather workspace was unexpectedly created")

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
    moe_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(moe_output, moe_input)

    dist.barrier()
    torch.cuda.synchronize()
    communicator.checkpoint_prepare()
    checkpoint_prepare_fi_ar_workspaces()
    moe_workspace.checkpoint_prepare()
    torch.cuda.synchronize()
    _assert_raises_detached(all_reduce, "all-reduce")
    _assert_raises_detached(moe_round_trip, "one-sided MoE")

    dist.barrier()
    torch.cuda.synchronize()
    checkpoint_restore_fi_ar_workspaces()
    communicator.checkpoint_restore()
    moe_workspace.checkpoint_restore(CustomCommunicator(control_group))
    torch.cuda.synchronize()
    dist.barrier()

    ar_input.fill_(rank + 2)
    moe_input.fill_(rank + 16)
    torch.cuda.synchronize()
    ar_graph.replay()
    moe_graph.replay()
    torch.cuda.synchronize()

    restored_sum = world_size * (world_size + 3) // 2
    torch.testing.assert_close(
        ar_graph_output, torch.full_like(ar_graph_output, restored_sum)
    )
    torch.testing.assert_close(moe_output, moe_input)

    pynccl.all_reduce = pynccl_all_reduce
    pynccl_diagnostic_input = torch.full(
        (1,), rank + 1, dtype=torch.float32, device="cuda"
    )
    pynccl_diagnostic_output = pynccl.all_reduce(pynccl_diagnostic_input)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        pynccl_diagnostic_output,
        torch.full_like(pynccl_diagnostic_output, initial_sum),
    )
    pynccl_handle_after = int(pynccl.comm.value or 0)
    if pynccl_handle_after != pynccl_handle_before:
        raise RuntimeError(
            "PyNCCL communicator handle changed during FlashInfer restore"
        )
    if any(pynccl_model_fallbacks.values()):
        raise RuntimeError(
            f"Unexpected PyNCCL model fallback: {pynccl_model_fallbacks}"
        )

    dist.barrier()
    result = {
        "model_collectives": {
            "all_reduce": "vllm-flashinfer-eager+graph+post-restore-graph",
            "one_sided_moe": "remote-dispatch+combine+post-restore-graph",
            "pynccl_fallbacks": pynccl_model_fallbacks,
        },
        "topology_exclusions": {
            "multi_rank_model_all_gather": "not exercised",
            "flashinfer_all_gather_workspace_count": len(
                getattr(communicator, "fi_ag_workspaces", {})
            ),
        },
        "pynccl": {
            "class": type(pynccl).__name__,
            "nccl_version": pynccl.nccl_version,
            "handle_stable": pynccl_handle_after == pynccl_handle_before,
            "constructor_warmup": "completed",
            "post_restore_diagnostic_all_reduce": "passed",
        },
    }
    communicator.destroy()
    return result


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

        report = {
            "rank": rank,
            "local_rank": local_rank,
            "pid": os.getpid(),
            "device": torch.cuda.get_device_name(local_rank),
            "torch_process_group": str(dist.get_backend(dist.group.WORLD)),
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
                "FLASHINFER_PYNCCL_SHM_PREFLIGHT="
                + json.dumps(reports, sort_keys=True),
                flush=True,
            )
            print("FLASHINFER_PYNCCL_SHM_PREFLIGHT_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
