# SPDX-License-Identifier: Apache-2.0
"""Two-process GMS cross-process byte-sharing probe (no vLLM).

Reproduces the failover's KV shape: N_ALLOC scratch mappings (all aliased to ONE
shared scratch granule, like vLLM's per-layer KV), then scratch->reallocate->remap
on the writer and scratch->remap on the reader. Each allocation is filled with a
distinct value so cross-wiring / aliasing is visible.
"""
import json
import os
import sys
import time

import torch
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.locks import RequestedLockType

N_ALLOC = int(os.getenv("N_ALLOC", "28"))
SIZE = int(os.getenv("ALLOC_SIZE", str(512 << 20)))  # 512 MiB per allocation
N = SIZE // 4  # float32 elements


def gms_tensor(va):
    return _tensor_from_pointer(va, [N], [1], torch.float32, 0)


def _cuda_ctx():
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()


def writer_scratch(sock, alloc_file, hang=False):
    _cuda_ctx()
    mgr = GMSClientMemoryManager(sock, device=0)
    vas = [
        mgr.create_scratch_mapping(size=SIZE, tag="kv_cache") for _ in range(N_ALLOC)
    ]
    mgr.unmap_all_vas()
    mgr.connect(RequestedLockType.RW)
    mgr.prepare_scratch_for_reallocation()
    mgr.reallocate_all_handles(tag="kv_cache")
    mgr.remap_all_vas()
    recs = []
    for i, va in enumerate(vas):
        gms_tensor(va).fill_(float(i + 1))  # distinct constant per allocation
        recs.append({"alloc": mgr.mappings[va].allocation_id, "val": i + 1})
    torch.cuda.synchronize()
    # verify the writer reads back its own writes
    bad = [
        i
        for i, va in enumerate(vas)
        if abs(float(gms_tensor(va)[0].item()) - (i + 1)) > 0.5
    ]
    print(f"[A] {N_ALLOC} allocs, writer self-readback bad={bad}", flush=True)
    json.dump(recs, open(alloc_file, "w"))
    print("[A] published", flush=True)
    if hang:
        print("[A] hanging (kill me)", flush=True)
        time.sleep(600)
    else:
        mgr.abort()


def reader_scratch(sock, alloc_file):
    _cuda_ctx()
    recs = json.load(open(alloc_file))
    mgr = GMSClientMemoryManager(sock, device=0)
    vas = [
        mgr.create_scratch_mapping(size=SIZE, tag="kv_cache") for _ in range(len(recs))
    ]
    mgr.unmap_all_vas()
    mgr.connect(RequestedLockType.RW)
    mgr.prepare_scratch_for_reallocation()
    mgr.remap_all_vas()
    fails = 0
    for i, va in enumerate(vas):
        got = float(gms_tensor(va)[0].item())
        exp = recs[i]["val"]
        ok = abs(got - exp) < 0.5
        if not ok:
            fails += 1
            if fails <= 5:
                print(f"[B] alloc{i} read {got} expected {exp}  FAIL", flush=True)
    print(f"[B] {len(recs)} allocs, {fails} mismatched", flush=True)
    print("RESULT", "PASS" if fails == 0 else "FAIL", flush=True)


def reader_scratch_concurrent(sock, alloc_file, go_file):
    """Like the standby: reserve+unmap scratch and stay ALIVE (colocated) through the
    writer's serving, then reattach only after the writer dies (go_file appears)."""
    _cuda_ctx()
    mgr = GMSClientMemoryManager(sock, device=0)
    vas = [
        mgr.create_scratch_mapping(size=SIZE, tag="kv_cache") for _ in range(N_ALLOC)
    ]
    mgr.unmap_all_vas()  # "sleep": VAs reserved, scratch physical dropped, process alive
    print(
        f"[B] {N_ALLOC} scratch reserved+unmapped, waiting (colocated) ...", flush=True
    )
    while not os.path.exists(go_file):
        time.sleep(0.5)
    mgr.connect(RequestedLockType.RW)
    mgr.prepare_scratch_for_reallocation()
    mgr.remap_all_vas()
    recs = json.load(open(alloc_file))
    fails = 0
    for i, va in enumerate(vas):
        got = float(gms_tensor(va)[0].item())
        if abs(got - recs[i]["val"]) > 0.5:
            fails += 1
            if fails <= 5:
                print(
                    f"[B] alloc{i} read {got} expected {recs[i]['val']}  FAIL",
                    flush=True,
                )
    print(f"[B] {len(recs)} allocs, {fails} mismatched", flush=True)
    print("RESULT", "PASS" if fails == 0 else "FAIL", flush=True)


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "reader_scratch_concurrent":
        reader_scratch_concurrent(sys.argv[2], sys.argv[3], sys.argv[4])
    elif role == "writer_scratch":
        writer_scratch(
            sys.argv[2], sys.argv[3], hang=(len(sys.argv) > 4 and sys.argv[4] == "hang")
        )
    elif role == "reader_scratch":
        reader_scratch(sys.argv[2], sys.argv[3])
