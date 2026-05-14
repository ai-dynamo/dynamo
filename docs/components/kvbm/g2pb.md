---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM G2PB
---

# KVBM G2PB

`G2PB` is a peer-backed remote cache for KVBM.

It lets one KVBM-enabled worker copy eligible KV blocks to other live peers and
later query or fetch them by `sequence_hash`. It is a cache, not a source of
truth: misses are allowed and fall back to normal recompute or local paths.

## What It Does

- keeps the normal KVBM local tiers in place
- adds a remote peer cache behind the same KVBM stack
- routes each block to one owner peer with rendezvous hashing
- fetches remote hits back through the existing host-to-device onboarding path

## What It Does Not Do

- it does not add a separate discovery system just for G2PB
- it does not expose peer-local disk layout to callers
- it does not provide strong consistency or durable ownership guarantees
- it does not turn remote disk into a directly addressable GPU transfer tier

## How It Is Enabled

You enable KVBM the normal way through the runtime connector.

Then you opt into native G2PB admission with:

```bash
export DYN_KVBM_G2PB_ADMISSION_POLICY=after_first_reuse
```

Supported values today:

- `after_first_reuse`: admit a block to G2PB after it is seen again
- `eager`: admit immediately
- `disabled`: do not use G2PB admission (default)

This is a KVBM policy surface, not a separate serving mode.

## How Routing and Discovery Works

Each live peer advertises:

- a live `instance_id` used for request routing
- a `hostname`, used to derive a stable `routing_id`

Workers use rendezvous hashing on `(sequence_hash, routing_id)` to pick the
owner peer for a block. This keeps routing deterministic while avoiding a
central per-block index. In the future we may offer replication to pick two of n owners.

The important split is:

- `routing_id`: stable shard identity
- `instance_id`: current live process identity

If a peer restarts but comes back with the same hostname, it keeps the same
logical shard identity even if its live `instance_id` changes.

## How Discovery Works

G2PB uses normal Dynamo component discovery.

The service registers the G2PB endpoint at:

```text
kvbm-g2pb/service/g2pb
```

The worker-side client:

1. watches the live Dynamo-discovered instances for that endpoint
2. sends a `Health` RPC to each instance
3. builds a local peer snapshot
4. refreshes that snapshot when discovery changes

There is no separate G2PB registry layered on top of Dynamo.

## Data Flow

The current flow is intentionally simple:

1. KVBM local G2 decides a block is eligible for G2PB remote admission.
2. The worker hashes the block to its owner peer.
3. The worker offers metadata to that peer via dynamo/tcp message.
4. If accepted, the a metadata message from KVBM offers the G2PB service agent, which pulls the block(s) through NIXL staging.
5. Later, a worker looking for a block hash can query or fetch the block from the owning peer.
6. A fetched block is registered locally in G2 and onboarded back to G1 device memory by the trt-llm framework. 

Control traffic goes through the Dynamo request plane. Bulk block movement stays
on the NIXL transfer pull path.

## TRT-LLM Async Loading

TRT-LLM can use a small leader-side prefetch hook before scheduling:
the KV cache connector leader method `await kvbm.advise_async_loading(token_ids=[], timeout=0.200) -> int`.

The intent is:

1. inspect a new request before scheduling
2. allow a short bounded grace period, for example `200ms`
3. prefetch a useful contiguous remote prefix into local (G2) KVBM host cache
4. run normal scheduling, with regular guarantees

Important properties:

- this is best-effort only and is not a promise that remote KV will be ready
- it stops at the first remote miss in the ordered block sequence
- it may continue in the background after the initial grace period expires. Storage system does not require a SLA.
- it prefetches into local host cache first, then relies on the existing
  host-to-device onboarding path later. 
- background is showing a block to trt-llm scheduler must be instant and atomic. kv onboarding is a guarantee and cannot fail. A distributed system cannot deliver such SLA. 

This keeps G2PB as a cache-assist mechanism rather than making remote peer hits
part of the correctness contract for scheduling.

## Operational Notes

- G2PB is best treated as a cache extension, not a durability layer.
- Misses are normal and degrade to local miss handling. Blocks are not discovered in Routing. 
- Brief peer churn should cause limited remapping because ownership uses
  rendezvous hashing.
- Stable peer hostnames matter. Ephemeral container IDs are poor shard
  identities.

## Tradeoffs

- approx design space
- designed for ~32 Nodes of 2TB host, offering 500GB G2 each, and 1.5 TB G2PB
- llm service: 40 RPS with 3000 mean blocks per query => 1280 G2PB qps and 120k lookup hashes/s

## Current Scope

Today, the implemented surface is:

- peer discovery through Dynamo
- owner selection through rendezvous hashing
- remote `offer`, `query`, `fetch`, `stage_put`, and `commit_put`
- NIXL-backed staged transfers
- KVBM-native admission policy via `DYN_KVBM_G2PB_ADMISSION_POLICY`
