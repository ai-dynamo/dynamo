# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rung 2: does the prefix cache survive a GMS sleep/wake in ONE process?

A single engine, persist-KV on. GMS grants ``RW_DATA`` when an engine re-attaches
a layout that is already sealed, so a sleep/wake exercises the whole
adopt-and-replay path with none of the failover machinery -- no flock, no second
engine, no frontend, no etcd.

**Two cycles, deliberately.** GMS does not own the KV pool until the first
``wake_up``: at startup the pool is built client-side and the server has no
session. So cycle 1 is what *creates and seals* the layout, and only cycle 2
re-attaches an already-sealed one and is granted ``RW_DATA``. Measuring across
cycle 1 would report a miss for a reason that has nothing to do with the index.

Run it twice:
  DYN_KV_INDEX=0   control    -- bytes survive, index does not, hits go to zero
  DYN_KV_INDEX=1   treatment  -- index is carried, hits survive the wake
"""

from __future__ import annotations

import json
import os
import socket
import sys

# Registers the `gms` load format and applies the GMS patches in THIS process.
# Without it `load_format="gms"` fails config validation before a worker exists.
import gpu_memory_service.integrations.vllm.worker  # noqa: F401
from gpu_memory_service.common.utils import get_socket_path  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("RUNG2_MODEL", "Qwen/Qwen3-0.6B")
# Long enough to span many blocks so a hit is unmistakable (block_size 16).
PROMPT = "The quick brown fox jumps over the lazy dog. " * 90
GREEDY = SamplingParams(temperature=0.0, max_tokens=24, seed=1234)
DEVICE = 0


def probe_gms(label: str) -> dict:
    """Read GMS runtime state over the lock-free inspection path.

    ``GetRuntimeState`` never enters the lock FSM, so this observes the server
    without perturbing whatever the engine is doing.
    """
    from gpu_memory_service.common.protocol.messages import (
        GetRuntimeStateRequest,
        GetRuntimeStateResponse,
    )
    from gpu_memory_service.common.protocol.wire import (
        recv_message_sync,
        send_message_sync,
    )

    path = get_socket_path(DEVICE, "kv_cache")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
        send_message_sync(sock, GetRuntimeStateRequest())
        resp, _fd, _buf = recv_message_sync(sock, bytearray())
    finally:
        sock.close()

    assert isinstance(resp, GetRuntimeStateResponse), resp
    state = {
        "state": resp.state,
        "allocations": resp.allocation_count,
        "layout_committed": getattr(resp, "layout_committed", None),
        "hash": (resp.memory_layout_hash or "")[:16],
    }
    print(f"[gms ] {label:<20} {state}", flush=True)
    return state


def run(llm: LLM, label: str) -> dict:
    out = llm.generate([PROMPT], GREEDY)[0]
    result = {
        "phase": label,
        "prompt_tokens": len(out.prompt_token_ids),
        "cached_tokens": out.num_cached_tokens,
        "text": out.outputs[0].text,
        "token_ids": list(out.outputs[0].token_ids),
    }
    print(
        f"[rung2] {label:<20} prompt={result['prompt_tokens']:>5} "
        f"cached={result['cached_tokens']:>5}",
        flush=True,
    )
    return result


def cycle(llm: LLM, n: int) -> None:
    print(f"[rung2] --- sleep/wake cycle {n} ---", flush=True)
    llm.sleep(level=1)
    probe_gms(f"after sleep {n}")
    llm.wake_up()
    probe_gms(f"after wake {n}")


def main() -> int:
    llm = LLM(
        model=MODEL,
        load_format="gms",
        # Dotted, not the `module:Class` form the docs show for `dynamo.vllm`:
        # vLLM's resolver splits on the last dot.
        worker_cls="gpu_memory_service.integrations.vllm.worker.GMSWorker",
        enable_sleep_mode=True,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("RUNG2_GPU_UTIL", "0.35")),
        max_model_len=4096,
        enable_prefix_caching=True,
    )

    probe_gms("at startup")
    cold = run(llm, "cold")

    # Cycle 1 creates and seals the GMS-owned layout; nothing to adopt yet.
    cycle(llm, 1)

    # Populate the index on the pool GMS now owns. This is the state a failover
    # would find, and the baseline the post-wake number is compared against.
    seed = run(llm, "seed (post-cycle1)")
    warm = run(llm, "warm (same proc)")

    # Cycle 2 re-attaches an already-sealed layout => RW_DATA => adoption.
    cycle(llm, 2)
    after = run(llm, "after wake 2")

    results = {"cold": cold, "seed": seed, "warm": warm, "after_wake": after}
    print(
        "\n"
        + json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "token_ids"}
                for k, v in results.items()
            },
            indent=2,
        )
    )

    ok = True
    if warm["cached_tokens"] <= 0:
        print("FAIL: prefix caching is not working at all before the sleep")
        ok = False
    if after["token_ids"] != warm["token_ids"]:
        print("FAIL: output changed across sleep/wake -- KV corruption")
        ok = False

    enabled = bool(os.environ.get("GMS_KV_INDEX_PATH"))
    if enabled:
        if after["cached_tokens"] <= 0:
            print("FAIL: index did not survive the wake (treatment)")
            ok = False
        else:
            print(f"PASS: index survived -- {after['cached_tokens']} cached tokens")
    else:
        print(
            f"CONTROL: cached_tokens after wake = {after['cached_tokens']} "
            "(expected 0 -- the gap this work closes)"
        )

    with open(os.environ.get("RUNG2_OUT", "/tmp/rung2.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
