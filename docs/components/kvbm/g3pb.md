---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM G3PB
---

# KVBM G3PB

`G3PB` is a peer-backed remote cache for KVBM.

It lets one KVBM-enabled worker copy eligible KV blocks to other live peers and
later query or fetch them by `sequence_hash`. It is a cache, not a source of
truth: misses are allowed and fall back to normal recompute or local paths.

## What It Does

- keeps the normal KVBM local tiers in place
- adds a remote peer cache behind the same KVBM stack
- routes each block to one owner peer with rendezvous hashing
- fetches remote hits back through the existing host-to-device onboarding path

## What It Does Not Do

- it does not add a separate discovery system just for G3PB
- it does not expose peer-local disk layout to callers
- it does not provide strong consistency or durable ownership guarantees
- it does not turn remote disk into a directly addressable GPU transfer tier

## How It Is Enabled

You enable KVBM the normal way through the runtime connector.

Then you opt into native G3PB admission with:

```bash
export DYN_KVBM_G3PB_ADMISSION_POLICY=after_first_reuse
```

Supported values today:

- `after_first_reuse`: admit a block to G3PB after it is seen again
- `eager`: admit immediately
- `disabled`: do not use G3PB admission

This is a KVBM policy surface, not a separate serving mode.

## How Routing Works

Each live peer advertises:

- a live `instance_id` used for request routing
- a `hostname`, used to derive a stable `routing_id`

Workers use rendezvous hashing on `(sequence_hash, routing_id)` to pick the
owner peer for a block. This keeps routing deterministic while avoiding a
central per-block index.

The important split is:

- `routing_id`: stable shard identity
- `instance_id`: current live process identity

If a peer restarts but comes back with the same hostname, it keeps the same
logical shard identity even if its live `instance_id` changes.

## How Discovery Works

G3PB uses normal Dynamo component discovery.

The backend registers the peer-cache endpoint at:

```text
kvbm-g3pb/peer-cache/g3pb
```

The worker-side client:

1. watches the live Dynamo-discovered instances for that endpoint
2. sends a `Health` RPC to each instance
3. builds a local peer snapshot
4. refreshes that snapshot when discovery changes

There is no separate G3PB registry layered on top of Dynamo.

## Data Flow

The current flow is intentionally simple:

1. KVBM decides a block is eligible for remote admission.
2. The worker hashes the block to its owner peer.
3. The worker offers metadata to that peer.
4. If accepted, the payload is transferred through NIXL staging.
5. Later, a worker can query or fetch the block from the owning peer.
6. A fetched block is registered locally and onboarded back to device memory.

Control traffic goes through the Dynamo request plane. Bulk block movement stays
on the NIXL transfer path.

## Operational Notes

- G3PB is best treated as a cache extension, not a durability layer.
- Misses are normal and degrade to local miss handling.
- Brief peer churn should cause limited remapping because ownership uses
  rendezvous hashing.
- Stable peer hostnames matter. Ephemeral container IDs are poor shard
  identities.

## Current Storage Choice

The current peer-local storage backend uses `foyer`.

We chose it for practical reasons:

- it already gives us a hybrid cache structure that fits the peer-backed model
- it supports `io_uring`
- it supports sharding across multiple disks
- it supports persistence across process restart
- it provides a more capable LRU-style eviction policy based on recent access

This was informed by previous good experience with it in earlier Chroma
workplane work.

The tradeoff is that this does not give us direct remote-disk access as a KVBM
transfer tier.

That is also partly intentional:

- remote disks are often slow or operationally unreliable
- some remote media may be spinning or network-backed
- holding remote GPU-visible staging for the duration of a slow disk read would
  tie up remote memory until local onboard completes

So the current choice is to keep peer distribution and cache ownership simple
first, and defer direct remote-disk integration.

This is not meant to be permanent. We are open to replacing or removing
`foyer` later if a better disk/storage design emerges.

## Current Scope

Today, the implemented surface is:

- peer discovery through Dynamo
- owner selection through rendezvous hashing
- remote `offer`, `query`, `fetch`, `stage_put`, and `commit_put`
- NIXL-backed staged transfers
- KVBM-native admission policy via `DYN_KVBM_G3PB_ADMISSION_POLICY`
