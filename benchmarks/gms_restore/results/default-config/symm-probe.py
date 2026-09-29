import faulthandler
import json
import os
import time

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
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
