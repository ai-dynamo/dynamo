# Examples by principle

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

These illustrate each principle in [SKILL.md](../SKILL.md). They are not an exhaustive list. New code
can hit a principle in ways none of these show; judge it by the mechanism and the rate, not by
resemblance to an example.

Each entry gives the mechanism, a simplified before/after shape (not the PR's code), the cost the PR
measured, and the PR. Measured costs come from each PR's own setup and are there to show scale, not to
predict your change.

## 1. Shared state touched per item

**Shared counter written per token.** Every stream increments one model-wide atomic for every output
token, so the cache line bounces between all cores serving streams. The stall shows up as self time
on the line after the increment.

```rust
// before: one shared atomic, every stream, every token
counters.output_tokens.inc_by(n);
// after: accumulate per request; publish every 64 tokens, at finish, and in Drop
self.pending += n;
if self.pending >= 64 { counters.output_tokens.inc_by(std::mem::take(&mut self.pending)); }
```

Shared adds fell from ~2M/s to ~35K/s per frontend.
[#17067](https://github.com/ai-dynamo/dynamo/pull/17067). The same shape for an ITL histogram cut that
path's cost by 72% ([#11569](https://github.com/ai-dynamo/dynamo/pull/11569)).

**Read-mostly map iterated per request.** Iterating a `DashMap` read-locks every shard on every
request. The data changes only when workers join or leave.

```rust
// before
let ids: Vec<_> = self.workers.iter().map(|e| *e.key()).collect();
// after: immutable snapshot; writers clone, modify, and swap under a mutex
let ids = self.workers.load();            // ArcSwap<HashMap<..>>
```

`DashMap` CPU fell from 6.41 to 0.17 ms per request; frontend CPU fell 10%.
[#16119](https://github.com/ai-dynamo/dynamo/pull/16119).

**One cancellation token shared by every request.** Clones of a single `CancellationToken` (or one
`Notify`) polled on every message serialize all streams on its internal lock.

```rust
let cancel = shutdown.clone();        // before: every stream polls the same token
let cancel = shutdown.child_token();  // after: one child per request or connection
```

TRT-LLM sidecar throughput went from 851K to 1.73M tok/s
([#16082](https://github.com/ai-dynamo/dynamo/pull/16082)). Frontend threads blocked on the shared
shutdown notifier went from 98 to 1.9 of 176
([#16728](https://github.com/ai-dynamo/dynamo/pull/16728)). The vLLM and SGLang sidecars got the same
change in [#14726](https://github.com/ai-dynamo/dynamo/pull/14726) and
[#14727](https://github.com/ai-dynamo/dynamo/pull/14727).

**Single mailbox lock for all producers.** Every producer pushed through one mutex. 131 of 352
threads were blocked on it. Sixteen cache-line-aligned shards keyed by worker cut blocked time from
52.2 to 0.02 ms per request. [#16118](https://github.com/ai-dynamo/dynamo/pull/16118).

**`ArcSwap::store` per chunk.** Each store makes the old value's drop walk every thread's debt list
(`Debt::pay_all` in profiles). A monotonic cursor moved to an `AtomicU64`, and the snapshot is swapped
only on lifecycle changes. The state agent went from 47.3 to 22.4 cores.
[#16935](https://github.com/ai-dynamo/dynamo/pull/16935).

## 2. Work done once, at the coarsest correct granularity

**Work no consumer reads.** Per-token text was decoded for every token but read only when the request
asked for logprobs. Gating on that saved ~0.35 ms of frontend CPU per request.
[#17349](https://github.com/ai-dynamo/dynamo/pull/17349).

**Lookups repeated per token.** `with_label_values` hashed labels and took a lock for every token.
Resolving the metric handle once per stream fixed it. Together with a compile-time log cap
([#10286](https://github.com/ai-dynamo/dynamo/pull/10286)) and an `Arc` in place of a prefix copy,
throughput rose 45% on a 4-core frontend. [#10273](https://github.com/ai-dynamo/dynamo/pull/10273).

```rust
// before: per token
metrics.with_label_values(&[model, endpoint]).inc();
// after: per stream
let counter = metrics.with_label_values(&[model, endpoint]);   // once
counter.inc();                                                  // per token
```

**Rebuild on every event.** The overload set was rebuilt from all workers on every load event. An
incremental tracker halved that profile bucket (9.0% to 4.8%).
[#10645](https://github.com/ai-dynamo/dynamo/pull/10645).

**One operation per element where a run would do.** Blocks were removed from a compressed radix tree
one hash at a time. Grouping the runs that share a node cut that path's cycles 56%.
[#12863](https://github.com/ai-dynamo/dynamo/pull/12863). Per-event messages batched with caps that
never split an event are the wire-side version
([#11776](https://github.com/ai-dynamo/dynamo/pull/11776)).

**Feature work while the feature is off.** A per-pass snapshot was built even with queueing disabled.
Gating it raised throughput 8.5%. [#15423](https://github.com/ai-dynamo/dynamo/pull/15423).

## 3. Bytes you don't need to move

**Intermediate JSON tree.** The response was converted from typed values to `serde_json::Value`,
patched, and serialized again, which took 37% of cycles on `/generate`.
[#11483](https://github.com/ai-dynamo/dynamo/pull/11483). The same shape in the Python request plane
was fixed by transcoding straight to Python objects
([#11104](https://github.com/ai-dynamo/dynamo/pull/11104)), and in chat metrics with a typed
`#[serde(skip)]` field ([#10512](https://github.com/ai-dynamo/dynamo/pull/10512)).

```rust
// before
let mut v = serde_json::to_value(&resp)?; patch(&mut v); serde_json::to_vec(&v)?
// after: put the field in the type and serialize once
serde_json::to_vec(&resp)?
```

**`serde(flatten)` on request parsing.** Flatten buffers every field into serde's internal `Content`
before dispatching it. That was 38–40% of the selection service's CPU. Flat wire structs converted
with `#[serde(from = ...)]` took p50 from 6.5 to 1.8 ms.
[#17238](https://github.com/ai-dynamo/dynamo/pull/17238).

**Frame copies.** Received ZMQ frames were copied into `Vec<Vec<u8>>` and cloned again. Wrapping them
with `Bytes::from_owner` removed the `memcpy` entirely.
[#11574](https://github.com/ai-dynamo/dynamo/pull/11574). Decoding envelope payloads as `Bytes`
slices did the same for MessagePack events ([#11539](https://github.com/ai-dynamo/dynamo/pull/11539)).

**Retained or copied prompts.** Prompt token arrays were kept alive after dispatch and copied once more
with `to_vec`. That held ~1.5 GiB per frontend. Releasing them after the last use and taking the
buffer by value brought it to ~0.
[#16223](https://github.com/ai-dynamo/dynamo/pull/16223). Copying the full prompt into each decoder
cost 33–48 µs per decoder lifecycle ([#15234](https://github.com/ai-dynamo/dynamo/pull/15234)).

**Unset wire fields.** Worker response frames serialized seven nil keys. `skip_serializing_if` took
each frame from 138 to 45 bytes and kept it wire-compatible.
[#17110](https://github.com/ai-dynamo/dynamo/pull/17110).

## 4. Per-element memory at block × worker scale

**Padding.** A `u64` key next to a `u32` pads to 16 bytes.

```rust
struct Slot { key: u64, val: u32 }          // 16 bytes
struct Slot { key: [u32; 2], val: u32 }     // 12 bytes
const _: () = assert!(std::mem::size_of::<Slot>() == 12);
```

Indexer tables went from 63.4 to 47.8 GiB. [#17109](https://github.com/ai-dynamo/dynamo/pull/17109).

**Enums sized by their rare variant, `Option` tags, and wide IDs.** A lower-tier edge slot was 113
bytes. Three changes brought it to 33 bytes, cutting memory per entry by over half:
- boxing the rare variants;
- a sentinel in place of `Option`;
- interning 24-byte owner IDs to `u32`.

[#17143](https://github.com/ai-dynamo/dynamo/pull/17143).

**An `Arc` per element.** Every request held an `Arc` per block, so every add and release did atomic
refcount traffic. A slot-map arena with plain `NonZeroU32` counts cut tracker cycles 43%.
[#11508](https://github.com/ai-dynamo/dynamo/pull/11508).

**Container that doesn't fit the access pattern.** A `BTreeMap` used as an LRU took 14.7% self time.
An intrusive doubly linked list cut replay time 28%
([#12244](https://github.com/ai-dynamo/dynamo/pull/12244)). Ordered maps keyed by monotonic IDs gave
way to a direct-indexed `Vec` ([#12242](https://github.com/ai-dynamo/dynamo/pull/12242)).

**Dead state kept around.** Dead radix nodes were swept every five minutes, so 73% of nodes were dead.
Reclaiming leaves as they die cut heap from 4.49 to 2.28 GiB.
[#16313](https://github.com/ai-dynamo/dynamo/pull/16313).

**Storage allocated before it is needed.** Child storage was allocated on node creation, not on the
first child. Allocating lazily raised write capacity 15–49%.
[#17259](https://github.com/ai-dynamo/dynamo/pull/17259).

## 5. Scaling with the number of workers

**A context per socket.** A `zmq::Context` per subscriber socket brings its own I/O and reaper
threads. At 1,024 workers the process had 4,118 threads; a shared context pool brought that to 32.
[#17253](https://github.com/ai-dynamo/dynamo/pull/17253).

**Per-publisher sockets and quadratic startup.** One SUB socket per publisher and a full re-plan on
every discovery change meant 2,048 SUBs per router and ~8.8M startup RPCs. Pooled fan-in and
incremental reconcile brought those to 32 and ~22K.
[#16568](https://github.com/ai-dynamo/dynamo/pull/16568).

**One lossy hop for every publisher.** A shared 1,024-entry broadcast channel in front of a serial
consumer applied 1.7% of KV batches. Per-publisher bounded lanes went from 3.7K to 228K batches/s.
[#14167](https://github.com/ai-dynamo/dynamo/pull/14167).

## 6. Per-item async, syscall and thread overhead

**A task per chunk.** The SGLang handler raced every chunk against cancellation with a new task and
`asyncio.wait`.

```python
# before: per chunk
nxt = asyncio.ensure_future(it.__anext__())
done, _ = await asyncio.wait({nxt, cancelled}, return_when=asyncio.FIRST_COMPLETED)
# after: one reader task per request feeds the stream; cancellation is checked once
```

Cost fell from 38.1 to 26.3 µs per token. [#16351](https://github.com/ai-dynamo/dynamo/pull/16351).
In Rust, cancellation futures rebuilt every frame and then pinned once per stream saved 6% CPU per
request ([#11149](https://github.com/ai-dynamo/dynamo/pull/11149)).

**A syscall object per wait.** Creating and closing a timerfd for every precise sleep made kernel
locks 29% of kernel samples. A reused per-actor timer cut mocker system CPU per request from 141 to
9.8 ms. [#14340](https://github.com/ai-dynamo/dynamo/pull/14340). Skipping the timer entirely when
the deadline has passed is the cheap half of the same idea
([#11025](https://github.com/ai-dynamo/dynamo/pull/11025)).

**Async mutex around short work.** A tokio mutex on the TCP stream registry, held through pre-send
preparation, formed a convoy. A `parking_lot` mutex with the preparation moved outside the lock took the
average pre-send time from 14.24 to 0.05 ms. [#11065](https://github.com/ai-dynamo/dynamo/pull/11065).

**A lock taken where none was needed.** A dynamic log filter took a span read-write lock on every
span even when every directive was static: 52% of frontend samples. Using a static filter in that case
halved frontend CPU per request. [#11820](https://github.com/ai-dynamo/dynamo/pull/11820).

## 7. The Python boundary

**Generic extraction.** Converting a Python `list[int]` through the generic sequence path costs a
refcount and an iterator step per element while holding the GIL. An exact-`list` fast path, with the
old path for everything else, went from 7.9 to 3.4 ns per token.
[#17337](https://github.com/ai-dynamo/dynamo/pull/17337).

**Temporary strings per frame.** Envelope keys were created as new Python strings for every frame.
Interning six keys removed 6,000 of 59,088 object allocations in the measured workload.
[#11331](https://github.com/ai-dynamo/dynamo/pull/11331).

## 8. Hot-path dependencies

**Swapping the HTTP server.** [#15564](https://github.com/ai-dynamo/dynamo/pull/15564) moved the
frontend listener to a server crate whose connection tasks polled shared notifiers on every wakeup.
98 of 176 worker threads ended up blocked there.
[#16728](https://github.com/ai-dynamo/dynamo/pull/16728) replaced it with a Dynamo-owned accept loop.

**Upstream fixes.** When the cost is inside a maintained dependency, fix it upstream with that
project's own benchmarks, then bump: tokio
([#16813](https://github.com/ai-dynamo/dynamo/pull/16813)) and quinn
([#14876](https://github.com/ai-dynamo/dynamo/pull/14876)).

## 9. Exact fast paths and skips

**Guarded fast path.** A scalar walk for the common one-worker, one-hint-source case was checked at
every step, with the general walk as fallback. Cost per matched block fell 48% with identical results.
[#17149](https://github.com/ai-dynamo/dynamo/pull/17149). The base64 data-URI fast path in
[#15874](https://github.com/ai-dynamo/dynamo/pull/15874) has an exhaustive parity test against the
scalar path.

**Unsound skip.** A skip that compared only set sizes overcounted: a fuzz test found 85,149
overcounts on main. Requiring size *and* an XOR fingerprint to match fixed it.
[#16265](https://github.com/ai-dynamo/dynamo/pull/16265).
