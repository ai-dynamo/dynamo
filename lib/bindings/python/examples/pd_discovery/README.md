<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill workers discover decode workers

A prefill worker keeps track of the running decode workers, reads an address
that each decode worker publishes in discovery, and runs an operation against
it. Here the operation is connecting to a small TCP server on the decode
worker. The prefill worker never asks a decode worker for its address.

- `decode.py` starts a TCP server, a stand-in for whatever the prefill side
  needs to reach (for example, a KV receiver). It publishes the server's
  address in the runtime config of its model card, with
  `ModelRuntimeConfig.set_engine_specific("side_channel", ...)` and
  `register_model`. A real decode worker already registers its model, so only
  the `set_engine_specific` call is new.
- `prefill.py` serves `prefill.generate` (mock output). Every two seconds it
  calls `runtime_configs()` on the decode endpoint, which returns the
  `ModelRuntimeConfig` of each decode worker from discovery, keyed by instance
  ID, and reads the address from `config.runtime_data["side_channel"]`. It
  connects to each new decode worker and drops the ones that leave.

`Endpoint.runtime_configs()` is new, so a released or nightly `ai-dynamo-runtime`
may not have it yet. Build the Python bindings from this
source tree first, as described in
[Building from Source](../../../../../docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md).

## Run locally

On one machine, the file discovery backend needs no etcd or NATS. Start each
command in its own terminal, from this directory:

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/pd-discovery DYN_LOG=warn

python3 decode.py
python3 decode.py
python3 prefill.py
python3 prefill.py
```

The scripts don't depend on the discovery backend. Without
`DYN_DISCOVERY_BACKEND`, the workers use etcd at `ETCD_ENDPOINTS`; with
`DYN_DISCOVERY_BACKEND=kubernetes`, they use Kubernetes discovery like any other
Dynamo worker.

Each prefill worker connects to every decode worker, and prints:

```text
prefill 5a0c58c0314098a0: decode 124b86db51ca2566 at 192.168.1.242:40807 -> 'ack from decode 124b86db51ca2566'
```

Start another decode worker and every prefill worker connects to it within one
check. Stop one and every prefill worker prints:

```text
prefill 5a0c58c0314098a0: decode 7391da11afde812d at 192.168.1.242:38553 left
```

## Notes

- Each `runtime_configs()` call lists the model cards from discovery. With Kubernetes
  discovery it reads a local cache; with etcd it is one read of the model card
  keys. A failed connection is retried on the next check.
- A worker that shuts down gracefully leaves the list right away. A worker that
  is killed, which includes Ctrl-C in this example, leaves when its discovery
  lease expires: 10 seconds with the file backend, 30 seconds with etcd
  (`ETCD_LEASE_TTL`). With Kubernetes discovery, a worker leaves when its pod
  stops being ready, for example when the pod starts terminating.
- Each decode worker publishes a new `incarnation` value when it starts. With
  Kubernetes discovery, the instance ID comes from the pod name, so a container
  that restarts in its pod keeps its ID; the new incarnation tells the prefill
  worker to treat it as a new worker.
- The model card checksum that the frontend compares across the workers of a
  model ignores `runtime_data` keys such as this one, so per-worker values do
  not split the workers into separate sets.
- The decode worker publishes its hostname resolved to an IP, which is the pod
  IP on Kubernetes. Across machines without Kubernetes, change it to an address
  that the prefill workers can reach.
