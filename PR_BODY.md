## Overview:

A Dynamo namespace answers requests only when its full worker topology is alive: an aggregated
deployment needs one live worker, a P/D or E/P/D deployment needs every role covered, and a LoRA
request additionally needs a live worker that actually advertises the adapter. Today the KV DC
Relay tracks KV-cache state per pool but has no answer to "can this namespace serve this model
right now, and how loaded is it" — the facts a cross-DC router needs before sending a request to
a data center.

This PR derives those serving facts inside the relay, engine-agnostically, from what is already
discovered: deployment cards, instance availability, and the `ActiveLoad` stream. Three layers:

- **Query semantics** — how a consumer must compute probe hashes for a pool (KV block size plus
  a versioned, eagle-aware hash format), resolved per pool and fenced on conflict instead of
  guessed.
- **Serving topology** — per-namespace model readiness (`Ready`/`Unavailable`/`Unknown`) derived
  with the exact semantics of the core namespace evaluation: the same `needs` DNF over live
  worker roles, the same legacy fallback for cards without worker types (exposed as an explicit
  fact rather than silently changing behavior), and the same ambiguity gate — a non-Aggregated
  role served by more than one live endpoint is not ready, with the duplicated roles reported as
  the explaining fact. LoRA readiness is nested under its base entry and counts only live
  workers advertising the adapter.
- **Pool load** — latest-wins load observations per worker rank (KV blocks used, active decode
  blocks, active prefill tokens) with capacities from runtime configurations and explicit
  coverage degradation; missing data is never reported as zero load.

Everything is consumable in-process through four watch/snapshot handles on `KvDcRelay`. There is
no transport here — no protobuf, no server, no publication machinery.

Part of #12102. Relates to #11225.

## Details

- Readiness follows instance availability through a supervised watch with bounded retry and
  reports `Unknown` until availability is authoritative: an empty snapshot is authoritative
  evidence of no live workers, a missing watch is not.
- Discovery projects worker topology (typed roles plus `needs` alternatives) and adapter
  membership into `EndpointMembership`, with worker- and binding-level conflicts.
- The registry enforces one pool per endpoint: a second attach is rejected, and endpoint
  reassignment waits for the prior pool's removal.
- Mixing standard and eagle workers under one endpoint fences the pool (they hash differently);
  a zero KV block size is a hard materialization conflict.
- Aggregated scale-out stays ready: the ambiguity gate covers only Prefill/Decode/Encode,
  matching the core evaluation.
- Stale-generation load collectors cannot update replacement pools, and an unexpected collector
  failure fences the generation through the existing fenced-withdrawal path.
- Cost model: serving facts are derived unconditionally — a relay without external consumers
  pays for one availability watch and one load collector per endpoint, nothing more.

### Validation

- `cargo test -p dynamo-llm --lib kv_dc_relay` — 102 passed; 103 with `ckf-diagnostics`.
- `cargo clippy --no-deps -p dynamo-llm --all-targets -- -D warnings` clean, with and without
  `ckf-diagnostics`; `cargo fmt --all -- --check` clean.
- New tests cover the readiness matrix (legacy single-role endpoints, disaggregated `needs`
  topologies, prefill/decode on distinct endpoints materializing independently, base-ready with
  LoRA-unavailable, adapters on non-eligible workers, `Unknown` before authoritative
  availability), the ambiguity gate (duplicate P/D endpoints, duplicate Encode endpoints,
  aggregated scale-out staying ready), standard/eagle fencing, endpoint exclusivity and
  reassignment, and load semantics (partial merges, duplicate reporters, unknown ranks ignored,
  coverage degradation, stale-generation rejection).

## Where should the reviewer start?

1. `lib/llm/src/kv_dc_relay/topology.rs` — the readiness derivation and its test matrix; worth
   checking against `Model::evaluate_namespace`, since verbatim parity (legacy fallback and
   ambiguity gate included) is a design goal.
2. `lib/llm/src/kv_dc_relay/discovery.rs` — the worker-topology and adapter projection.
3. `lib/llm/src/kv_dc_relay/identity.rs` / `resolution.rs` — `KvQuerySemantics` and its conflicts.
4. `lib/llm/src/kv_dc_relay/load.rs` — the latest-wins load state and coverage rules.
5. `lib/llm/src/kv_dc_relay/pool_registry.rs` — endpoint exclusivity and the serving-facts state.

## Related Issues

**🔗 This PR is linked to an issue:**

- Relates to #11225

🤖 Generated with [Claude Code](https://claude.com/claude-code)
