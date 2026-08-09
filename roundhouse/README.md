<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Roundhouse

A stateful agentic front-end that sits **in front of** Dynamo.

A coding agent today re-uploads its entire conversation on every turn. At 100k+
tokens of context and hundreds of turns, that is the dominant cost of agentic
work — in bytes on the wire, in prefill FLOPs, and in dollars. Roundhouse owns
the conversation instead, so the client sends only deltas.

Once the service owns the context, a second capability falls out for free:
because it knows the exact token prefix, it can ask *which engine already has
this prefix cached* and route accordingly. Statefulness and routing are not two
features — the first is what makes the second possible.

> **Status: exploratory walking skeleton.** The session core, routing layer, and
> embedded Dynamo integration are real and tested. The HTTP/SSE, WebSocket, and
> gRPC transports and the Redis store are not yet implemented.

`roundhouse` is a **separate Cargo workspace**, deliberately excluded from the
Dynamo workspace. Extracting it into its own repository means changing the two
`dynamo-*` path dependencies to versions — nothing else.

## Layout

| Crate | Contents |
|---|---|
| `roundhouse-core` | Session state machine, event log, lease, context assembly, routing vocabulary and policies |
| `roundhouse-fleet` | Local Dynamo fleet (embedded selection service) and frontier providers |
| `roundhouse-store-redis` | Redis Streams `SessionStore` *(not yet implemented)* |
| `roundhouse-server` | Turn engine and binary; HTTP transports *(not yet implemented)* |

## Design

**One append-only event log per session, with a monotonic `seq`.** That single
structure serves three otherwise separate needs: Responses-API streaming
resumption (`starting_after`), reconnect replay for the bidirectional
transports, and the routing audit trail.

**Conversation items and the routing ledger are projections of that log**, not
separately stored collections. There is one write path, so nothing can disagree
after a crash — and `SessionStore` collapses to six methods as a result.

**A single-writer lease with fencing.** Every mutating call takes a `Lease`. An
owner that stalled, was partitioned, or died and came back fails its next append
rather than interleaving with its successor.

**Incremental tokenization.** Routing on cache locality means knowing the
prompt's block hashes before dispatch. Recomputing them each turn would cost
O(context) per turn and O(context × turns) per session — more work than the
routing decision can save. Because the conversation is append-only and Dynamo's
block hashes are computed per fixed-size block from tokens alone, only newly
completed blocks are hashed. Verified against a full recompute using Dynamo's own
hashing functions.

**One comparison axis.** "Serve this from our Llama on worker 7" and "send it to
Anthropic" become comparable once both are expressed as cache-adjusted expected
prefill. The two sides get that number very differently:

- **Local** — `SelectionService::select` is query-only and returns
  `effective_prefill_tokens`, the scheduler's own cache-credit-weighted prefill
  cost, *without booking anything*.
- **Frontier** — no provider exposes its cache, so it is modelled from the
  routing ledger: what we last sent, when, and under which provider TTL.

## Why embed the selection service

Dynamo's `SelectionService` exposes every HTTP endpoint of
`python -m dynamo.select_service` as a plain async method, so it can be called
in-process. Embedding removes the TCP round trip *and* the JSON serialization of
the prompt, which dominates per-call cost at long context.

Queries carry **block and sequence hashes, never token ids** — for a 100k-token
context, the difference between shipping a 400 KB array and a few kilobytes.

The **select/reserve split** is what makes cross-provider routing possible at
all: price the local option, compare it against a frontier model, and only book
if local wins. An abandoned quote costs nothing; the pending selection expires.

The reservation lifecycle (`prefill_complete` → `release`) is **mandatory** — a
leaked reservation permanently inflates the router's view of a worker and
silently distorts every later decision.

### Scaling shape

The selection service is stateful but **replicated, not sharded**. Every replica
holds the complete radix tree and processes the complete KV event firehose, so
neither index memory nor event-ingest CPU divides by N. Replica sync is
best-effort by design (no sequencing, acknowledgement, replay, or
resynchronization), and output-block growth is deliberately not synced — so each
replica underestimates load driven by its peers, and it worsens as N grows.

Embedding at N ≈ 3–10 is comfortable. The `select`→`reserve` stickiness that
bites multi-replica HTTP deployments is free here, since one process does both.

**At N = 1 replica sync is simply off** — `replica_sync` is never called, so no
PUB socket is bound and none of the O(N²) peer-mesh configuration applies. The
only remaining ZMQ is worker→selector KV-event ingest, which is an engine-side
wire contract.

## Switchyard

Kept behind our own `RoutingPolicy` trait rather than wired in directly.
Switchyard's `Algorithm` trait is a good fit — the algorithm emits
`Step::CallLlm` with a semantic target and the *host* executes it — but
`libsy::State` is in-memory with no pluggable persistence, which collides with
the requirement to survive process death. Behind the trait, it is an option
rather than a dependency.

## Build and test

Requires the Rust toolchain pinned in `rust-toolchain.toml` and system
`libzmq3-dev` (pulled in transitively by `dynamo-kv-router/standalone-selection`).

```bash
apt-get install -y libzmq3-dev
cargo test --workspace
```

No GPUs, no worker processes, and no network are needed: the selection plane
runs inside the test binary.

## What the tests establish

- **Incremental hashing is exact** — matches a full recompute across unaligned
  appends, and a shared prefix yields an identical sequence-hash chain.
- **Client bytes stay flat while context grows** — over 20 turns, per-turn client
  bytes vary by ≤ 2 while server-side context grows more than 10×.
- **Pricing books no load** — two `select` calls yield distinct pending
  selections and zero booked prefill.
- **A quote can be abandoned for free** — the fleet prices identically afterwards.
- **Routing reacts to cache state** — a warmed target is priced below its prompt
  length on the following turn.
- **Reservations settle** — load returns to zero; a consumed `selection_id`
  cannot be booked twice.
- **Failover loses nothing** — killing the owner mid-session, a successor claims
  the lease, replays the log, and continues with contiguous sequence numbers.
- **Retries do not regenerate** — a re-sent `turn_id` replays the existing
  response instead of opening a second turn.

## Not yet built

HTTP/SSE, WebSocket, and gRPC transports; the Redis store; a real tokenizer
(the trait exists, with a byte-level test implementation); real provider
clients for OpenAI and Anthropic; and resuming an interrupted generation from
its partial output — the partial is already durable in the log, so the
groundwork is there.
