import faulthandler
import json
import os
import time

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
import subprocess
from pathlib import Path

uuids = [json.loads(Path(f"/gms/rank-{r}.json").read_text())["uuid"] for r in range(8)]
rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True
)
ordered = [
    u.strip()
    for u, b in sorted(
        (line.split(",") for line in rows.splitlines()), key=lambda x: x[1].strip()
    )
]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(ordered.index(u)) for u in uuids)
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)


def event(name, **fields):
    print(
        json.dumps({"rank": rank, "time": time.time(), "event": name, **fields}),
        flush=True,
    )


event("begin")
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
event("group")
from gpu_memory_service.common.vmm import VMMDeviceType, init_vmm
from gpu_memory_service.v1.client.mempool import TorchMempoolMemoryClient

init_vmm(VMMDeviceType.CUDA)
client = TorchMempoolMemoryClient()
with client.weight_region():
    model = torch.nn.Linear(1024, 1024, bias=False, device=f"cuda:{rank}")
client.publish_weights([model])
event("gms-published")
x = symm.empty((1048576,), dtype=torch.uint8, device=f"cuda:{rank}")
event("allocated")
h = symm.rendezvous(x, group=dist.group.WORLD)
event("rendezvous", multicast_ptr=h.multicast_ptr)
x.fill_(rank)
torch.cuda.synchronize()
dist.barrier()
event("done")
dist.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
