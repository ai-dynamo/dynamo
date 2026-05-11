# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal NCCL broadcast SENDER (rank 0) for the RL admin smoke test.

Mirrors the wire protocol of prime_rl's `NCCLWeightBroadcastSender` so that
prime_rl's `NCCLWeightUpdateWorker` (rank >= 1) can receive the broadcast and
load it via `model.load_weights(state_iter)`.

Coordination:
  1. Process starts and loads the model.
  2. Calls StatelessProcessGroup.create(rank=0, world_size=2) — BLOCKS until
     the inference worker also calls create (via POST init_weights_update_group).
  3. Once the NCCL communicator is ready, reads a line from stdin. Sending "GO"
     triggers the broadcast; anything else aborts.

Usage:
    python nccl_broadcaster.py --host 127.0.0.1 --port 29501 \\
        --world-size 2 --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Generator

# Force offline so transformers/HF don't try to reach the hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

LAYER_PREFIX = "model.layers."


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    t = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(t, src=0)


def broadcast_state_dict(state_dict: dict[str, Tensor], communicator: PyNcclCommunicator) -> None:
    """Wire protocol matches prime_rl.trainer.rl.broadcast.nccl.broadcast_state_dict.

    Sends, in order:
      1. integer = byte-length of pickled metadata
      2. raw pickle bytes (state_tensor of uint8)
      3. for each dtype group: one concatenated tensor (flatten + cat)
    """
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        dtype_groups.setdefault(value.dtype, []).append((key, value))

    metadata = {dt: [(k, v.shape, v.numel()) for k, v in items] for dt, items in dtype_groups.items()}
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    for dtype, items in dtype_groups.items():
        flat = [v.flatten() for _, v in items]
        concatenated = torch.cat(flat)
        communicator.broadcast(concatenated, src=0)
        del concatenated


def filter_state_dict_by_layers(
    state_dict: dict[str, Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, Tensor]], None, None]:
    """Yield (-1, non-layer weights), then (i, layer_i weights) for each i."""
    yield -1, {k: v for k, v in state_dict.items() if not k.startswith(LAYER_PREFIX)}
    for i in range(num_layers):
        yield i, {k: v for k, v in state_dict.items() if k.startswith(f"{LAYER_PREFIX}{i}.")}


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    max_layer = -1
    for key in state_dict:
        if key.startswith(LAYER_PREFIX):
            tail = key[len(LAYER_PREFIX) :].split(".", 1)[0]
            if tail.isdigit():
                max_layer = max(max_layer, int(tail))
    return max_layer + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--world-size", type=int, default=2,
                    help="Total ranks (trainer + inference workers). Default 2 = 1 trainer + 1 worker.")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--timeout", type=int, default=120, help="StatelessProcessGroup store timeout (s)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"[broadcaster] Loading {args.model} (dtype={args.dtype}) on cuda ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda().eval()
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    num_layers = get_max_layer_num(sd)
    print(f"[broadcaster] Model loaded: {len(sd)} tensors, {num_layers} layers", flush=True)

    print(
        f"[broadcaster] Creating StatelessProcessGroup({args.host}:{args.port}, "
        f"rank=0, world_size={args.world_size}) — blocks until peers join ...",
        flush=True,
    )
    pg = StatelessProcessGroup.create(
        host=args.host,
        port=args.port,
        rank=0,
        world_size=args.world_size,
        store_timeout=args.timeout,
    )
    print("[broadcaster] StatelessProcessGroup created — all peers joined", flush=True)
    communicator = PyNcclCommunicator(pg, device=torch.cuda.current_device())
    print("[broadcaster] PyNcclCommunicator ready", flush=True)

    print("[broadcaster] Waiting for 'GO' on stdin ...", flush=True)
    line = sys.stdin.readline().strip()
    if line != "GO":
        print(f"[broadcaster] aborting: expected 'GO', got {line!r}", flush=True)
        sys.exit(2)

    num_state_dict_to_send = num_layers + 1  # non-layer chunk + one per layer
    print(f"[broadcaster] Broadcasting {num_state_dict_to_send} state-dict chunks ...", flush=True)
    broadcast_integer(num_state_dict_to_send, communicator)
    for layer_idx, layer_sd in filter_state_dict_by_layers(sd, num_layers):
        layer_sd = {k: v.to(dtype) for k, v in layer_sd.items()}
        broadcast_state_dict(layer_sd, communicator)
        print(f"[broadcaster] sent chunk {layer_idx} ({len(layer_sd)} tensors)", flush=True)

    print("[broadcaster] Broadcast complete", flush=True)


if __name__ == "__main__":
    main()
