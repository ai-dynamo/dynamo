#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check both idle GPU views and one two-rank NCCL all-reduce before model load."""

import datetime
import inspect
import json
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import request_memory


def rank_main(rank, port):
    expected = json.loads(os.environ["TP2_EXPECTED_MAP"])
    memory = int(os.environ["TP2_MEMORY_MIB"]) << 20
    assert len(expected) == torch.cuda.device_count() == 2
    assert os.getenv("CUDA_MPS_SM_PARTITION") is None
    torch.cuda.set_device(rank)
    props = torch.cuda.get_device_properties(rank)
    gpu = "GPU-" + str(props.uuid).removeprefix("GPU-")
    assert props.multi_processor_count == expected[gpu]
    assert torch.cuda.can_device_access_peer(rank, 1 - rank)
    snap = MemorySnapshot(device=torch.device("cuda", rank))
    assert snap.total_memory == memory
    assert (
        snap.free_memory >= memory * 0.8
    ), "Insufficient free quota for the proposed model budget"
    print(
        json.dumps(
            {
                "event": "vllm_memory_snapshot",
                "rank": rank,
                "uuid": gpu,
                "sms": props.multi_processor_count,
                "total_bytes": snap.total_memory,
                "free_bytes": snap.free_memory,
                "budget_bytes": int(memory * 0.8),
                "euid": os.geteuid(),
            }
        ),
        flush=True,
    )
    print("REQUEST_MEMORY_SOURCE " + inspect.getsource(request_memory), flush=True)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize(rank)
    assert value.item() == 3
    print(
        json.dumps({"event": "nccl_result", "rank": rank, "result": value.item()}),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(rank_main, args=(port,), nprocs=2, join=True)
