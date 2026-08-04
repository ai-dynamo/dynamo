# SPDX-License-Identifier: Apache-2.0
"""Independent external GMS client (no vLLM): import the engine's KV allocation at a
fresh VA and fingerprint it. MATCH => the engine's KV is persisted in GMS memory and
readable by an outside process; MISMATCH => it is not externally visible."""
import json
import sys

import torch
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path

info = json.load(open(sys.argv[1]))
alloc = info["alloc"]
expected = float(info["fingerprint"])
numel = int(info["numel"])
dtype = getattr(torch, str(info["dtype"]).split(".")[-1])

torch.zeros(1, device="cuda:0")  # establish a CUDA context on device 0
torch.cuda.synchronize()

sock = get_socket_path(0, "kv_cache")
mgr = GMSClientMemoryManager(sock, device=0)
mgr.connect(RequestedLockType.RW)  # uncommitted alloc; persist-on-abort adopts it
va = mgr.create_mapping(allocation_id=alloc)  # fresh import at a brand-new VA
t = _tensor_from_pointer(va, [numel], [1], dtype, 0)
torch.cuda.synchronize()
got = float(t.float().abs().sum().item())

match = abs(got - expected) < max(1.0, abs(expected) * 1e-4)
print(
    f"[EXT] alloc={alloc[:8]} expected_fingerprint={expected:.1f} got={got:.1f}",
    flush=True,
)
print("RESULT", "MATCH" if match else "MISMATCH", flush=True)
