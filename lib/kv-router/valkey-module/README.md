<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Valkey KV index module

`dynkv.so` is a loadable Valkey module for Dynamo's persistent device-tier KV
prefix index. It owns a native in-memory radix-like index under one Valkey key;
RDB snapshots, AOF rewrites, and native Valkey replication preserve that state.

It stores KV metadata only: block hashes, prefix ownership, worker/rank state,
and event ordering. It does **not** store GPU KV tensors. NIXL remains the GPU
KV transfer data plane.

## Build and validate

```bash
export VALKEY_REPO="$HOME/src/valkey"
make -C "$VALKEY_REPO" -j"$(nproc)"
make -C lib/kv-router/valkey-module test \
  VALKEY_SRC="$VALKEY_REPO/src"
```

The black-box test covers prefix matching, malformed-event atomicity,
worker-wide DP clears, retirement/recovery, conditional tree-dump fencing, RDB
and AOF durability, authoritative admission, worker-owner crash expiry,
owner-aware event fencing, module replication/full sync, and `WAIT`
acknowledgement on a single client connection. It also resumes partial lease
cleanup across RDB, AOF rewrite/replay, and late full sync, and exercises a
legal 1,048,576-block epoch whose footprint exceeds one maximum GC budget.
The implementation is split by storage, matching, persistence, admission,
worker lifecycle, garbage collection, and command registration. Link-time
optimization preserves cross-file optimization while the linker exports only
the Valkey module entry point.

## Primary/replica topology

`--router-valkey-urls` is a primary-first topology, not a client-side shard
list. Load the same module on both servers and configure Valkey replication;
legacy commands replicate verbatim, while owner/admission/GC paths propagate
deterministic internal apply records.

```bash
MODULE="$PWD/lib/kv-router/valkey-module/dynkv.so"
VALKEY="$VALKEY_REPO/src/valkey-server"

"$VALKEY" --port 6380 --bind 127.0.0.1 --dir /var/lib/dynkv-primary \
  --appendonly yes --appendfsync everysec --repl-diskless-sync-delay 0 \
  --min-replicas-to-write 1 --min-replicas-max-lag 5 \
  --loadmodule "$MODULE"

"$VALKEY" --port 6381 --bind 127.0.0.1 --dir /var/lib/dynkv-replica \
  --appendonly yes --appendfsync everysec --repl-diskless-sync-delay 0 \
  --replicaof 127.0.0.1 6380 --replica-read-only yes \
  --replica-serve-stale-data no --loadmodule "$MODULE"
```

Configure all frontends with the same ordered endpoints:

```bash
python -m dynamo.frontend --router-mode kv --router-replica-sync \
  --router-valkey-urls valkey://valkey-primary:6379,valkey://valkey-replica:6379 \
  --router-valkey-allow-insecure-plaintext \
  --router-valkey-index-scope my-dgd-frontend
```

> [!CAUTION]
> The current client transport is plaintext and unauthenticated. The explicit
> insecure flag is accepted only for a separate tenant-isolated trusted network
> protected by NetworkPolicy or equivalent firewall rules. Do not expose these
> endpoints to a shared or untrusted network.

The client writes and reads the elected primary only. For a two-endpoint
topology it sends `DYNKV.*` followed by `WAIT 1 3000` on the **same persistent
TCP connection**. The replica is never a blind write or read fallback.
Production primary-endpoint reads should use `DYNKV.MATCH_PRIMARY`: it checks
the local Valkey role without another network round trip and returns
`DYNKV_NOT_PRIMARY` if an established connection still points at a demoted
primary. `DYNKV.MATCH` remains replica-readable for diagnostics and replication
verification.

Legacy lifecycle commands replicate verbatim. Admission commands instead append
a private deterministic `DYNKV.ADMIT_APPLY` transition containing the primary's
absolute lease deadline and chosen worker/rank. Worker-owner commands similarly
replicate `DYNKV.WORKER_LEASE_APPLY`, and owner-aware events replicate
`DYNKV.APPLY_OWNED_AT` with the primary's event-acceptance time. Replicas and
AOF replay never consult their local clock to decide lease validity.

Snapshot/RDB encoding 14 and exact-GC wire version 2 are backward-readable by
this module, but older modules cannot read the new formats or apply new GC
chunks. A rolling module upgrade must therefore update **all replicas first**,
wait for them to resynchronize, and update the primary last. Do not run GC from
the upgraded primary or fail over to an upgraded primary while any old replica
remains eligible for promotion.

Two servers do not provide leader discovery or split-brain fencing. Production
failover needs a Sentinel/operator/VIP-style primary endpoint and a fenced
promotion workflow; clients must reconnect to that promoted primary.

## Direct worker event mirroring

Workers can normalize and batch GPU KV events through the existing
`KvEventPublisher`, then write them directly to the primary index. Enable it
in both frontend and worker environments:

```bash
# frontend flag (also exports the intended deployment setting)
--router-valkey-worker-events

# set in every worker container/process
export DYN_ROUTER_VALKEY_URLS='valkey://valkey-primary:6379,valkey://valkey-replica:6379'
export DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT=true
export DYN_ROUTER_VALKEY_WORKER_EVENTS=true
export DYN_ROUTER_VALKEY_INDEX_SCOPE='my-dgd-frontend'
```

`DYN_ROUTER_VALKEY_INDEX_SCOPE` must be identical in every participating
container. It prevents DGD `prefill`/`decode` components from writing a
different index key than their `frontend` readers.

In direct-worker mode Valkey is the authority for GPU-tier routing metadata.
Frontend recovery subscribers do not replay GPU events or issue independent
rank resets; any legacy event-plane relay is compatibility-only for other
consumers. Replayed `STORE` events are idempotent in the module and return
`NOOP` without a replication round trip. An unowned `REMOVE` deliberately
commits a fence ticket, because it can invalidate a stale worker dump even
when it has no local prefix owner to change. A failed primary/replica
acknowledgement retains the ordered event and retries it until shutdown.

Frontend `MATCH` calls use a bounded 512-entry, one-second read-through cache
for repeated prompts. It is an affinity optimization only: a stale result can
lose a cache hit but cannot make a request incorrect, and worker lifecycle
operations invalidate it immediately.

Disaggregated decode workers intentionally publish no KV ownership events, but
they are still authoritative-admission targets. Worker startup therefore uses
an awaited registration-only path for every decode DP rank before model
discovery is advertised. This path sends one
leased `DYNKV.REGISTER_WORKER_RANKS` batch (and one replica `WAIT` barrier) without
constructing an event source or emitting fake `DYNKV.APPLY` data. Prefill and
aggregated ranks use the same startup gate.

Each worker process generates one nonzero, random 64-bit owner nonce and shares
it across every DP-rank publisher. Direct publishers use
`DYNKV.APPLY_OWNED`, never legacy `DYNKV.APPLY`, so a prior process cannot
publish after its lease expires or a successor claims the same discovery
worker ID. The process renews its worker lease periodically and sends an
owner-conditional unregister during graceful shutdown. The nonce identifies a
process incarnation; it is not a secret or authentication credential.

## Recovery fencing API

Direct worker events and recovery tree dumps can race: an old dump must not
erase a newer `DYNKV.APPLY`. Each worker/rank therefore exposes a persisted,
opaque mutation-generation ticket. Direct rank mutations advance that rank's
ticket; worker-wide `CLEAR` and removal-all advance a worker epoch folded into
the same ticket, so even an unseen sibling rank is fenced. An unowned direct
`REMOVE` also advances its rank ticket.

Recovery must use this sequence on the primary connection:

1. Read `DYNKV.RANK_GENERATION key worker_id dp_rank`.
2. Fetch the worker's tree dump.
3. Call `DYNKV.REPLACE_RANK_IF_GENERATION` with the generation from step 1 and
   that dump.
4. On `DYNKV_STALE_GENERATION`, fetch a fresh dump and retry.

`worker_id`, `dp_rank`, and the expected opaque ticket use the same fixed-width
big-endian binary encodings as the rest of the module (`u64`, `u32`, and `u64`,
respectively). The replacement dump is:

```text
u8 version (= 1)
u32 event_count
event_count × (u32 event_length, raw DYNKV.APPLY event)
```

Every embedded event must be a `STORE` for exactly the target worker/rank.
The module validates the entire dump before resetting the rank, then atomically
replaces it only if the ticket still matches. It returns the new ticket as an
eight-byte big-endian bulk string. Tickets, worker epochs, and clear-deduplication
watermarks are included in RDB/AOF state and module replication.
`DYNKV.RESET_WORKER` remains a legacy, unconditional primitive and is not safe
for racing recovery dumps.

The Rust worker tree-dump recovery call site is not wired to this API yet. If
that path is enabled for direct-worker mode, it must use this protocol rather
than the legacy `RESET_WORKER` plus replay sequence.

## Authoritative admission API

Workers must register each active rank before it can be admitted:

```text
DYNKV.REGISTER_WORKER key <u64 worker_id> <u32 dp_rank>
```

It is idempotent for an active rank. It deliberately refuses a retired rank;
registration never acts as a reset or recovery operation. It returns `OK` when
it creates a rank and `NOOP` when that active rank was already registered.
`DYNKV.APPLY` can record KV ownership for an unregistered rank, but that rank
is not admission-eligible until this command succeeds. A rank revived from a
retirement must register again.

Worker startup can register a complete DP rank set atomically:

```text
DYNKV.REGISTER_WORKER_RANKS key <u64 worker_id> <payload>

u8  version (= 1)
u32 rank_count                         # 1 through 65,536
rank_count × u32 dp_rank
```

All integers are big-endian. Ranks must be unique. The module validates the
complete payload, the worker-wide retirement fence, and every listed rank's
retirement fence before it creates or changes any rank. A malformed payload or
one fenced rank rejects the entire command with no partial registration. It
returns `OK` when at least one rank is created or newly marked registered and
`NOOP` when every listed rank was already active and registered. One successful
batch increments the index mutation counter once and is one verbatim
replication/AOF record, independent of rank count. The existing single-rank
command remains wire-compatible.

Version 1 is the unleased compatibility form. Version 2 is the original leased
form and remains accepted for rolling upgrades, but it has no replay-safe
registration CAS and is therefore legacy-tainted and not epoch-GCable. New
worker startup first reads an eight-byte big-endian lifecycle token:

```text
DYNKV.REGISTRATION_GENERATION key <u64 worker_id>
```

It then uses replay-safe version 3 to atomically claim the worker ID for one
process incarnation:

```text
DYNKV.REGISTER_WORKER_RANKS key <u64 worker_id> <payload>

u8  version (= 3)
u64 owner_nonce                        # nonzero, shared by all local DP ranks
u64 lease_ms                           # 1 through 600,000
u64 expected_lifecycle_generation      # from REGISTRATION_GENERATION
u32 rank_count                         # 1 through 65,536
rank_count × u32 dp_rank               # unique, complete current rank set
```

The lifecycle counter is separate from the high-rate KV-event generation. For
an absent worker, the query returns `lifecycle_counter + 1`; many workers may
concurrently receive the same prospective value. Their registrations all
serialize successfully and install distinct lifecycle revisions, unless epoch
GC advances the persisted registration floor first, in which case the caller
re-queries and retries. Deleting an epoch advances that floor, so a delayed
pre-GC version-3 registration cannot recreate its old owner.

The first call claims the worker and registers the entire rank set. Every
non-idempotent active rank-set transition advances the per-worker lifecycle
revision. An exact retry with the prior expected token is accepted only for
the same live owner and the exact current rank set; this makes an ambiguous
write/replica-ack result safe to retry without allowing an older rank set to
roll back a newer one. Omitted ranks become inadmissible, their KV owners are
deactivated, and their reservations are revoked. A different owner receives
`DYNKV_WORKER_OWNED` while the lease is live. An expired-owner query returns
the next lifecycle revision. Registration returns
`DYNKV_WORKER_CLEANUP_PENDING` until bounded GC finishes the old incarnation;
the Rust acquisition path actively runs configured-budget `GC CURRENT` calls,
waits for replica acknowledgement, refreshes the token, and retries with
cancellation-aware backoff. This prevents a successor from racing partially
cleaned rank, owner, or reservation state. A leased worker
permits legacy single-rank `REGISTER_WORKER` only as an idempotent `NOOP` for
an existing member; accepting that legacy call taints the epoch for safety.

Heartbeat and graceful unregister payloads include the worker ID so one shared
connection can manage multiple workers:

```text
DYNKV.RENEW_WORKER_LEASE key <payload>
u8 version (= 1) | u64 worker_id | u64 owner_nonce | u64 lease_ms

DYNKV.UNREGISTER_WORKER key <payload>
u8 version (= 1) | u64 worker_id | u64 owner_nonce
```

Both are owner-conditional. A stale nonce returns
`DYNKV_STALE_WORKER_OWNER` and cannot renew, remove, or alter a successor.
Unregister is deliberately not `REMOVE_WORKER[_ALL]`: it clears ephemeral
membership and allows the same worker ID to be claimed again after sleep or a
process restart. It deactivates all of the worker's KV owners, marks every rank
inadmissible, advances admission incarnations and the worker generation fence,
and revokes active reservations.

Crash expiry has the same cleanup semantics. `MATCH` and stateless `SELECT`
filter an expired absolute deadline immediately. `SELECT_RESERVE` may clean at
most one expired owner when its conservatively charged work fits a 256-item
inline budget; larger cleanup is left to bounded lifecycle GC. GC first
persists a cleanup fence and removes the lease from the expiry heap, then
drains reservations, admission incarnations, node owners, and rank state in
individually budgeted exact chunks before clearing the owner tuple. The expired
owner is inadmissible throughout. Cleanup generation markers persist through
RDB, AOF rewrite, incremental AOF, replication, and full sync; heap positions,
semantic-marker partitions, and per-epoch reference indexes are runtime-derived
and rebuilt on load.

Direct event publishing uses:

```text
DYNKV.APPLY_OWNED key <u64 owner_nonce> <DYNKV.APPLY event>
```

The event's worker/rank must be in the active owner's registered set and the
worker lease must be unexpired. The module records the primary's acceptance
time in its internal replicated transition, so a replica cannot reject an
event merely because it executes later. Legacy `DYNKV.APPLY` remains available
for frontend recovery and rolling compatibility, but direct worker writers
must use the owned command to receive stale-process fencing.

For lifecycle-managed, non-legacy ranks, an accepted owned `REMOVE` deletes
only the owner records named in that event and prunes newly ownerless leaves
and ancestors. Its work is O(blocks in the event), independent of total active
cache size. Owned `CLEAR` performs the corresponding full-rank cleanup. The
rank/worker generations remain as recovery fences, so active direct-worker
STORE/REMOVE churn does not retain every historical prefix. Legacy-tainted
owners keep inactive records conservatively for rolling compatibility.

Owner expiry/unregister retains inactive `WorkerState` and `WorkerEpoch`
tombstones: radix nodes contain worker pointers, event-ordering generations
must fence delayed recovery, and legacy `DYNKV.APPLY` can still arrive during a
rolling upgrade. `DYNKV.LIFECYCLE_STATS key` returns
`[active_owner_leases, retained_rank_tombstones, ownerless_worker_epochs]` so
operators can bound and alert on that retained state.

Bounded lifecycle GC is explicit and generation-fenced:

```text
DYNKV.GC key CURRENT <u32 max_items>
DYNKV.GC key <u64 generation_watermark> <u32 max_items>   # diagnostic/manual
```

`CURRENT` resolves the watermark inside the serialized primary command with no
preceding O(N) diagnostic scan. A manual watermark is an unsigned big-endian
u64; `max_items` is an unsigned big-endian u32 between 1 and 1,048,576. The
budget bounds records examined and exact object transitions. An expired lease
never needs to fit in one call: BEGIN advances its generation/lifecycle fences
once, and later calls consume at most the supplied number of reservation,
admission-rank, node-owner, rank, and FINALIZE transitions. Pending cleanup is
prioritized on subsequent ticks, so even a legal epoch larger than the maximum
single-call budget makes progress. A watermark greater than the current index
generation is rejected. A runtime cursor cycles through rank,
admission-authority, radix-node, and worker-epoch records. The reply is:

```text
[examined, transitions, owners, admission_ranks, nodes, ranks,
 epoch_or_expiry_transitions, next_phase]
```

A rank is eligible only after an owner lease ended, its tombstone generation
is nonzero and no newer than `generation_watermark`, and it was exclusively
managed through the leased/owner-fenced protocol. GC first removes inactive
node-owner records and unused admission authority. It deletes the rank only
after node memberships, reservations, and admission references reach zero.
Ownerless radix leaves are then pruned; parent child counts prevent removal of
shared prefixes or non-leaf parents. Expired owner cleanup chunks carry the
expected cleanup generation and original nonce/deadline; replicas never
consult their clock. Exact apply accepts a semantic target anywhere in the
unprocessed partition and swaps it to the tail in O(1), so RDB/full-sync array
rebuild order cannot invalidate later primary chunks. The worker epoch is last
and requires no live owner or remaining rank/admission/reservation references.
Epoch deletion advances both the recovery-generation floor and the
registration GC floor. Registering
a successor clears rank tombstones, while delayed pre-GC registration and
recovery CAS payloads remain stale rather than observing an ABA.

`DYNKV.GC_STATS key` is an O(N) operator diagnostic, not part of the periodic
control loop. It returns:

```text
[generation, direct_rank_tombstones, retained_legacy_ranks,
 direct_ownerless_epochs, retained_legacy_epochs,
 inactive_owner_records, ownerless_nodes, next_phase]
```

`direct_ownerless_epochs` and `retained_legacy_epochs` include expired or
cleanup-pending owner tuples, not only already-finalized tuples.

Any accepted legacy `DYNKV.APPLY`, unowned registration, or recovery/lifecycle
mutation permanently taints that worker incarnation for conservative rolling
upgrade safety. State loaded from a pre-GC snapshot is also legacy-tainted.
Such state is reported but not automatically reclaimed: an ownerless legacy
epoch is the fence that prevents a delayed legacy event or tree dump from
resurrecting compacted state. Operators should stop legacy/recovery writers,
wait their maximum delivery/retry window, and only then advance the watermark.
Owner nonces must remain process-unique, as required by the lease protocol.

The primary converts each successful call into a private, exact
`DYNKV.GC_APPLY` plan. Replicas and incremental AOF replay apply the listed
owner/node/rank identities and expected generations instead of independently
scanning or consulting a local clock. It rejects duplicate identities and
preflights the complete plan before mutation; the primary queues propagation
before committing the already-validated plan. RDB, AOF rewrite, and full sync
persist the compacted semantic state, partial-cleanup markers, and lifecycle
floors; child/reference indexes and the scan cursor are runtime-derived.
`DYNKV.GC_APPLY` rejects normal clients.

The Rust direct-worker publisher schedules `DYNKV.GC key CURRENT budget` on
the fenced primary (60 seconds and 256 inspection units by default), only while
its ordered writer is healthy and idle. It uses the same configured replica
acknowledgement policy as owner events and deliberately does not retry an
ambiguous GC result, because a retry is safe but may consume another complete
budget. Configure or disable the loop with
`DYN_ROUTER_VALKEY_GC_INTERVAL_MS` (`0` disables) and
`DYN_ROUTER_VALKEY_GC_INSPECTION_BUDGET`. Automatic reclamation remains
restricted to version-3 owner-fenced state that has never accepted legacy or
recovery mutations. When switching away from legacy/recovery mode, use an
external time-based quiescence window; legacy-tainted records stay retained.
The initial worker-nonce jitter spreads phases, but every publisher still owns
a periodic GC loop. This is modest for the tested four-worker topology; large
fleets should centralize GC ownership or add a leader/rate cap to avoid
O(fleet) command pressure.

`DYNKV.SELECT_RESERVE key payload` atomically ranks, books, and leases one
registered non-retired rank. All fields are big-endian:

```text
u8  version (= 1)
u32 admission_domain_length
bytes admission_domain                 # for example "prefill" or "decode"
u64 client_nonce
u64 request_nonce
u64 lease_ms                           # 1 through 600000
u32 prefix_hash_count                 # at most 1,048,576
prefix_hash_count × u64 local_hash
u32 candidate_count
candidate_count × (u64 worker_id, u32 dp_rank, u32 capacity)
```

`capacity` must be nonzero and less than `2^32 - 1`; the all-ones value is
reserved for conservative migration of legacy leases that did not persist a
capacity.

The identity is `(admission_domain, client_nonce, request_nonce)`. The first
accepted reservation latches capacity for that
`(admission_domain, worker_id, dp_rank)` tuple; later requests in the same
domain must supply the same capacity. Capacity and active-reservation count
are intentionally independent across domains, so a `prefill` lease does not
consume a `decode` slot. Unknown, expired, or retired candidates are skipped,
because independently updated frontend discovery snapshots may briefly retain
one stale rank; another healthy candidate in the same atomic request remains
eligible. Invalid capacity values and capacity conflicts on a live rank are
rejected. Ranking is longest device-prefix overlap, then module-held
domain-scoped active reservation count, then worker/rank. Frontend-supplied
load is not part of admission.

The reply is `u8 version | u8 status`. Status `0` is `NO_CAPACITY`. Status `1`
adds:

```text
u64 client_nonce
u64 request_nonce
u64 worker_id
u32 dp_rank
u64 expires_at_ms
u32 matched_blocks
u32 active_reservations_at_grant
```

An exact duplicate reserve returns the same active reservation without taking a
second slot, even if a different candidate was retired after the original
grant. If that reservation's worker owner has expired or entered cleanup, the
retry deterministically releases the stale reservation and returns
`NO_CAPACITY`; it never dispatches to the fenced worker. A different payload
with the same identity returns
`DYNKV_REQUEST_CONFLICT`. Duplicate reserve also emits a replicated no-op
apply transition, so a retry followed by `WAIT` establishes a fresh replication
offset.

`DYNKV.RELEASE` has:

```text
u8 version | u32 domain_length | domain | u64 client_nonce | u64 request_nonce |
u64 expected_expires_at_ms
```

It returns `u8 version | u8 status`, where status `1` released the lease and
`0` was already absent. `DYNKV.RENEW` adds `u64 lease_ms` after the expected
deadline and returns the successful reserve reply. The expected deadline is a
lease token: stale release or renew calls cannot affect a renewed reservation.
A retry of the most recent renew using its prior deadline is idempotent and
returns the already-renewed grant, making an ambiguous write/`WAIT` failure
safe to retry. Renewals never shorten an existing lease deadline.
Expired leases are cleaned during the next authoritative admission mutation.
`RESET_WORKER`, `REMOVE_WORKER`, `REMOVE_WORKER_ALL`, and fenced rank
replacement revoke matching leases and advance their persisted admission
incarnations; `REMOVE_WORKER_ALL` also keeps its unseen-rank retirement fence.
`DYNKV.ADMISSION_STATS key` returns the current number of stored leases for
diagnostics. `DYNKV.ADMIT_APPLY` is internal to replication/AOF and rejects
normal client calls.

## Module commands

The Rust client is the normal interface. The binary protocols are versioned;
do not handcraft them in production.

- `DYNKV.APPLY key event`: store/remove/clear a worker's device KV metadata.
- `DYNKV.APPLY_OWNED key owner_nonce event`: direct-worker mutation fenced by
  the active worker-owner lease.
- `DYNKV.MATCH key local_hashes`: return prefix overlap for all active
  worker/rank owners.
- `DYNKV.MATCH_PRIMARY key local_hashes`: the same response, but only on the
  current Valkey primary; replicas return `DYNKV_NOT_PRIMARY`.
- `DYNKV.SELECT key query_and_candidates`: choose from a frontend-filtered
  candidate set by longest device-prefix overlap, then supplied load, then
  worker/rank. This is a read-only deterministic accelerator; it does not yet
  make a booking.
- `DYNKV.REGISTER_WORKER` and `DYNKV.REGISTER_WORKER_RANKS`: compatible
  single-rank, unleased batch, and worker-owned atomic rank-set registration.
- `DYNKV.REGISTRATION_GENERATION`: read the lifecycle CAS token used by
  replay-safe version-3 registration.
- `DYNKV.RENEW_WORKER_LEASE` and `DYNKV.UNREGISTER_WORKER`: owner-conditional
  heartbeat and non-permanent lifecycle release.
- `DYNKV.SELECT_RESERVE`, `DYNKV.RELEASE`, and `DYNKV.RENEW`: registered-rank,
  lease-based authoritative admission.
- `DYNKV.RESET_WORKER`, `DYNKV.REMOVE_WORKER`, and
  `DYNKV.REMOVE_WORKER_ALL`: recovery and lifecycle primitives.
- `DYNKV.RANK_GENERATION` and `DYNKV.REPLACE_RANK_IF_GENERATION`:
  generation-fenced recovery replacement primitives.
- `DYNKV.RESTORE`: internal AOF-load support only; it rejects normal client
  calls so it cannot be used as an unconditional rollback.
- `DYNKV.GC`: generation-watermarked, max-inspection-budget lifecycle
  compaction. `DYNKV.GC_APPLY` is its replication/AOF-only exact plan.
- `DYNKV.GC_STATS`: GC generation/cursor diagnostics and direct-versus-legacy
  retained-state counts.
- `DYNKV.STATS`: diagnostic node/worker/mutation counts.
- `DYNKV.LIFECYCLE_STATS`: active owner leases and retained lifecycle
  tombstones.

Dynamo's existing Rust scheduler remains the production selection policy: it
applies topology constraints, lower-tier/shared-cache credit, and queue/load
policy after `MATCH`. `SELECT` is available for the fast common prefix/load
ranking path and for the next authoritative-routing stage.

## Current boundary

This module provides one replicated, fleet-wide prefix index, a per-rank
tree-dump mutation fence, and expiring process-incarnation ownership for a
worker ID. It also holds globally atomic active admission leases and
per-admission-domain rank capacity accounting. It does not yet hold global
queue state, authenticate owner nonces, elect one fleet-wide GC leader, or
provide automatic primary failover. The Rust publisher schedules bounded GC,
but callers must still use primary fencing, replica acknowledgement, upgrade
gating, and dispatch failure/retry handling around a granted lease.
