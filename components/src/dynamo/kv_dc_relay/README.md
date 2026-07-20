<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DC KV Relay

The DC KV Relay maintains compact, data-center-local KV-cache routing state for one or more
serving endpoints. It discovers workers through the Dynamo runtime, consumes their ordered KV
events, and maintains one endpoint-scoped Cuckoo-filter producer for each compatible cache domain.

The Relay is the producer side of the multi-DC routing boundary. It can provide barrier snapshots
and sequenced absolute bucket-image deltas to an endpoint-scoped global CKF consumer. The current
component hosts that producer lifecycle and local diagnostic endpoints; cross-DC transport and
request forwarding are separate global-router integration work.

## Usage

```bash
python -m dynamo.kv_dc_relay --dc-id <stable-dc-id>
```

`--dc-id` must be stable for the logical data center across Relay process restarts. Optional
discovery filters can limit the endpoints supervised by one Relay:

```bash
python -m dynamo.kv_dc_relay \
  --dc-id us-west \
  --namespace-filter dynamo \
  --endpoint-prefix dynamo.backend
```

`DYN_NAMESPACE` controls the namespace used for the Relay's own runtime endpoints and defaults to
`dynamo`.

## Runtime endpoints

The component always exposes a health endpoint. Builds with the Rust `ckf-diagnostics` feature
also expose Relay statistics and an endpoint-specific producer snapshot. Endpoint component names
include a stable digest of `dc_id`, allowing several DC Relay processes to share a runtime
namespace without colliding.

These diagnostic endpoints are not the WAN publication protocol, and the Relay does not proxy
inference requests. A production global router is expected to transport published state, choose a
DC-local serving pool, and forward requests to that pool.
