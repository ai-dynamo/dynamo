# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from contextlib import contextmanager

import torch
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import register_module_tensors
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

from .runtime import DYNAMO_BIN, REPO_ROOT

SGLANG_BIN = REPO_ROOT / "dynamo-sglang" / "bin"


def get_external_weight_writer_command(backend: str) -> list[str]:
    return [
        "python",
        "-m",
        "tests.gms.harness.external_weight_writer",
        "--backend",
        backend,
    ]


def get_external_weight_writer_env(backend: str) -> dict[str, str]:
    if backend == "sglang":
        return {
            **os.environ,
            "PATH": f"/usr/local/cuda/bin:{SGLANG_BIN}:{os.environ.get('PATH', '')}",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "PYTHONPATH": str(REPO_ROOT),
        }
    return {
        **os.environ,
        "PATH": f"{DYNAMO_BIN}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(REPO_ROOT),
    }


def run_external_weight_writer(backend: str) -> None:
    command = get_external_weight_writer_command(backend)
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=get_external_weight_writer_env(backend),
        check=True,
    )


def _get_writer_manager(device: int, tag: str) -> GMSClientMemoryManager:
    return get_or_create_gms_client_memory_manager(
        get_socket_path(device, tag),
        device,
        RequestedLockType.RW,
        tag=tag,
    )


def _publish_model(manager: GMSClientMemoryManager, model: torch.nn.Module) -> None:
    register_module_tensors(manager, model)
    torch.cuda.synchronize()
    manager.commit()
    manager.close()


@contextmanager
def _vllm_single_rank_distributed(device: int):
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=device,
        distributed_init_method=f"tcp://{host}:{port}",
        backend="gloo",
    )
    ensure_model_parallel_initialized(1, 1)
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@contextmanager
def _sglang_single_rank_distributed(device: int):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=device,
        distributed_init_method=f"tcp://{host}:{port}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1, 1, backend="gloo")
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _publish_vllm_dummy_weights(device: int, tag: str) -> None:
    from vllm.config import (
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch.cuda.set_device(device)
    model_config = ModelConfig(model=FAULT_TOLERANCE_MODEL_NAME, enforce_eager=True)
    load_config = LoadConfig(load_format="dummy")
    device_config = DeviceConfig(device="cuda")
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        load_config=load_config,
    )
    target_device = torch.device("cuda", device)
    manager = _get_writer_manager(device, tag)

    with set_current_vllm_config(vllm_config):
        with _vllm_single_rank_distributed(device):
            with set_default_torch_dtype(model_config.dtype):
                with gms_use_mem_pool(tag, target_device):
                    with target_device:
                        model = initialize_model(
                            vllm_config=vllm_config,
                            model_config=model_config,
                        )
                    DummyModelLoader(load_config).load_weights(model, model_config)
                    process_weights_after_loading(model, model_config, target_device)

    _publish_model(manager, model)


def _publish_sglang_dummy_weights(device: int, tag: str) -> None:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_loader.loader import LoadConfig, get_model_loader
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    torch.cuda.set_device(device)
    model_config = ModelConfig(FAULT_TOLERANCE_MODEL_NAME)
    device_config = DeviceConfig(device="cuda", gpu_id=device)
    load_config = LoadConfig(load_format="dummy")
    loader = get_model_loader(load_config, model_config)
    manager = _get_writer_manager(device, tag)
    server_args = ServerArgs(
        model_path=FAULT_TOLERANCE_MODEL_NAME,
        load_format="dummy",
        device="cuda",
    )
    set_global_server_args_for_scheduler(server_args)

    with _sglang_single_rank_distributed(device):
        initialize_dp_attention(server_args, model_config)
        with gms_use_mem_pool(tag, torch.device("cuda", device)):
            model = loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )

    _publish_model(manager, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tag", default="weights")
    args = parser.parse_args()

    if args.backend == "vllm":
        _publish_vllm_dummy_weights(args.device, args.tag)
        return
    _publish_sglang_dummy_weights(args.device, args.tag)


if __name__ == "__main__":
    main()
