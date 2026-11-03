<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill workers discover decode workers

A prefill worker regularly checks which decode workers are running, gets an
address that each decode worker publishes, and runs an operation against it.
Here the operation is connecting to a small TCP server on the decode worker.

- `decode.py` starts a TCP server, a stand-in for whatever the prefill side
  needs to reach (for example, a KV receiver). It serves two endpoints from one
  process: `decode.generate` (mock output) and `decode.side_channel`, which
  returns the server's `host` and `port`.
- `prefill.py` serves `prefill.generate` (mock output). Every `--interval`
  seconds it lists the decode workers with `client.instances()`. For each new
  decode worker, it calls `client.direct(request, instance_id)` on
  `side_channel` to get the address, then connects. When a decode worker
  leaves the list, the prefill worker drops it.

## Run

On one machine, the file discovery backend needs no etcd or NATS. Start each
command in its own terminal, from this directory:

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/pd-discovery DYN_LOG=warn

python3 decode.py
python3 decode.py
python3 prefill.py
python3 prefill.py
```

Without `DYN_DISCOVERY_BACKEND`, the workers use etcd at `ETCD_ENDPOINTS`.

Each prefill worker connects to every decode worker, and prints:

```text
prefill 3e4623350c2d3fd: decode 2dd62b4d9d3f923 at 192.168.1.242:39333 -> 'ack from decode 2dd62b4d9d3f923'
```

Start another decode worker and every prefill worker connects to it within one
poll. Stop one and every prefill worker prints:

```text
prefill 3e4623350c2d3fd: decode 2dd62b4d9d3f923 at 192.168.1.242:39333 left
```

## Notes

- `client.instances()` reads a list that Dynamo keeps up to date from
  discovery, so polling it makes no network calls. The address lookup and the
  connection run once per new decode worker.
- A worker that shuts down gracefully leaves the list right away. A worker that
  is killed, which includes Ctrl-C in this example, leaves when its discovery
  lease expires: 10 seconds with the file backend, 30 seconds with etcd
  (`ETCD_LEASE_TTL`).
- All endpoints served by one process share its instance ID, so a prefill
  worker can also watch `decode.generate` and look up the address with the
  same ID.
- `--host` sets the address that the decode worker publishes. The default, the
  machine's hostname resolved to an IP, works on one machine. Across machines,
  pass an IP that the prefill workers can reach.
