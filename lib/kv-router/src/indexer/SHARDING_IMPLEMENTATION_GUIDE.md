<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharding Implementation Guide

This note explains the three sharded-indexer implementations currently being
benchmarked in `kv-router`, how they differ, and how to think about them in DEP
language.

The important distinction is not just "sharded" vs "not sharded." The three
implementations make different tradeoffs around:

- whether request routing is fully deterministic
- whether shard ownership is mutable at runtime
- how much control-plane state the shard router must own
- how well the design adapts when workload balance drifts over time

## Naming Translation

The code uses concise implementation names:

- `BranchShardedIndexer`
- `PrefixShardedIndexer`
- `VirtualShardShardedIndexer`

In design discussion, it is often clearer to think of them as:

- **stateful least-loaded branch placement**
- **fully deterministic direct prefix routing**
- **deterministic logical routing with stateful ownership**

Those descriptive names are used below, but each section also points back to the
concrete Rust type.

## 1. Stateful Least-Loaded Branch Placement

Implementation: `BranchShardedIndexer`

### Routing model

`BranchShardedIndexer` builds an explicit routing table:

- `branch_key -> physical_shard`

The branch key is derived from the first `prefix_depth` block hashes. When a new
branch key is first observed, the indexer assigns it to a physical shard and
stores that mapping in `branch_to_shard`.

### Why it is stateful

This implementation is not fully deterministic from request-visible data alone.
When a new branch key appears, the target shard is chosen using live shard load:

- primary signal: `shard_block_counts`
- tie-breaker: branch count

After that first assignment, the mapping is stable. The same branch key always
routes to the same shard unless a future rebalancing system explicitly migrates
it.

### Strengths

- good near-term balance without needing a second indirection layer
- single-shard reads in steady state
- straightforward fit for the current `BranchShardedIndexer` design
- usually the strongest default benchmark result today

### Weaknesses

- shard placement is order-dependent at first observation
- the shard router must own fine-grained mutable placement state
- warm restart / replication / multi-process ownership is more complex
- imbalance can still emerge later because branch placement is permanent unless
  migration is added

### Best mental model

This is the most direct sharded extension of the current router/indexer path:

- route by branch key
- place new branches on the least-loaded physical shard
- keep reads single-shard

## 2. Fully Deterministic Direct Prefix Routing

Implementation: `PrefixShardedIndexer`

### Routing model

`PrefixShardedIndexer` routes directly from prefix hash to physical shard:

- `branch_key -> physical_shard`

The target shard is computed by hashing the first `prefix_depth` blocks and
taking `hash % num_shards`.

For continuation events, the implementation can inherit the parent shard so the
chain remains internally consistent.

### Why it is deterministic

There is no mutable placement table for new branches. Given:

- the same prefix blocks
- the same `prefix_depth`
- the same shard count

the routing decision is the same everywhere.

### Strengths

- simplest routing rule
- easiest to replicate
- easiest to explain and reason about operationally
- best fit if the long-term goal is mostly stateless shard-router replicas

### Weaknesses

- if the routing key is too coarse, skew is unavoidable
- long shared prefixes can collapse traffic onto too few shards
- recovering from a bad routing rule usually means changing the routing rule
  itself, not just moving ownership
- repartitioning is awkward without adding another layer of indirection

### Best mental model

This is the cleanest "pure hash" version:

- deterministic request-time routing
- minimal control-plane state
- best when prefix keys are already a good partitioning of the workload

## 3. Deterministic Logical Routing With Stateful Ownership

Implementation: `VirtualShardShardedIndexer`

### Routing model

`VirtualShardShardedIndexer` splits routing into two layers:

- `branch_key -> virtual_shard` deterministically
- `virtual_shard -> physical_shard` through a shared ownership table

This is a hybrid design. The logical routing key stays deterministic, but
physical ownership can still change over time.

### Why this is a hybrid

Unlike `PrefixShardedIndexer`, this implementation does maintain mutable
ownership state. But that state lives at the **virtual-shard** level rather than
as one branch-to-physical-shard mapping per branch.

That makes it a middle ground between:

- fully stateful least-loaded placement
- fully deterministic direct routing

### Current migration model

The current benchmark implementation supports a simple rebalance path:

1. detect an imbalanced physical shard
2. pick a hot virtual shard on that physical shard
3. replay the virtual shard into a cooler destination
4. temporarily dual-write / dual-read during the transition
5. switch `virtual_shard -> physical_shard` ownership

This is intended to model the kind of ownership transition a more productionized
control plane would need.

### Strengths

- deterministic request-time routing
- ownership can change without changing the logical routing function
- better long-term bridge to multi-process, Kubernetes, and multi-node
- closest of the three to a Cassandra-like "deterministic keyspace plus movable
  ownership" model

### Weaknesses

- more machinery than direct deterministic routing
- still needs migration protocol, ownership fencing, and recovery rules
- does not fix skew if the logical routing key itself is too coarse
- current benchmark implementation uses hardcoded rebalance defaults and is not
  yet a finalized production control-plane design

### Best mental model

This is the "virtual shards" option:

- deterministic at the logical layer
- stateful at the ownership layer
- rebalancing cost paid mostly during migration events rather than steady-state
  request routing

### Potential improvements

The current benchmark prototype is useful as a first ownership-migration model,
but there are a few obvious next steps if this direction continues to look
promising:

- **better logical routing keys**: virtual shards cannot fix a routing key that
  is fundamentally too coarse, so this design likely pairs best with either a
  deeper prefix rule or node-depth routing
- **virtual-shard splitting**: if one virtual shard becomes persistently hot,
  split it into child virtual shards instead of only moving it as a whole
- **cleanup after migration**: remove or compact stale copied state on the old
  owner after cutover so balance metrics reflect true serving ownership
- **smarter cutover**: replace the current time-based dual-write window with a
  replay-until-caught-up protocol and cut over once lag reaches zero

## Comparison Table

| Design view | Rust type | Request-time routing | Mutable ownership | Main benefit | Main risk |
|---|---|---|---|---|---|
| Stateful least-loaded branch placement | `BranchShardedIndexer` | branch key -> physical shard | yes, per-branch routing table | strong near-term balance | stateful control plane |
| Fully deterministic direct prefix routing | `PrefixShardedIndexer` | branch key -> physical shard | no | simplest replication / recovery story | skew if prefix key is too coarse |
| Deterministic logical routing with stateful ownership | `VirtualShardShardedIndexer` | branch key -> virtual shard -> physical shard | yes, at virtual-shard layer | deterministic routing plus controlled rebalance | extra migration / ownership machinery |

## Relation To Node-Depth Sharding

These three implementations mainly differ in how they map a routing key to
shards. `NodeDepthShardedIndexer` explores a different question:

- whether the routing key itself should be derived from a deeper point in the
  compressed trie rather than from the earliest prefix blocks

That means node-depth can be thought of as a **better logical routing key**,
not necessarily as a replacement for every ownership model above.

In particular, a future design could combine:

- node-depth routing for the logical key
- virtual shards for movable ownership

## Practical Summary

If the main goal is near-term benchmark wins, `BranchShardedIndexer` is the
current strongest default.

If the main goal is the simplest operational story, `PrefixShardedIndexer` is
the cleanest design.

If the main goal is a durable long-term architecture that keeps deterministic
routing while still allowing controlled rebalance, `VirtualShardShardedIndexer`
is the most promising direction of the three.
