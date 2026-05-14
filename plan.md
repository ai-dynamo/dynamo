# G2PB Plan

## Goal

Deliver a remote `G2PB` offloading tier that integrates cleanly into the
TRT-LLM workflow.

Done means all of the following are true:

1. the remote tier is consistently named `G2PB`
2. the long-running service binary is `kvbm_g2pb_service`
3. the end-to-end client smoke is `kvbm_g2pb_client_smoke`
4. the service stores committed payloads only in large host-backed `G2`
5. `foyer` is fully removed from the active runtime path
6. payload integrity is enforced end-to-end with mandatory `xxh3_64` checksums
7. the live smoke path remains green after refactors
8. the service/client implementation is simplified enough to be a stable TRT-LLM integration target

## Status

Completed:

1. removed `foyer` from the service storage path
2. removed `foyer` CLI/config from `kvbm_g2pb_service`
3. removed the direct `foyer` dependency from `lib/llm/Cargo.toml`
4. split the old monolithic distributed implementation into:
   - `g2pb.rs` for shared protocol/types
   - `g2pb/g2pb_service.rs` for service/storage logic
   - `g2pb/g2pb_client.rs` for client/discovery/routing logic
5. completed the active `G3PB` -> `G2PB` rename across:
   - module names
   - binaries
   - active docs/config/env/metrics surfaces
6. added mandatory payload integrity checks using `xxh3_64`:
   - sender computes and attaches checksum
   - service validates before commit/store
   - service validates again before fetch returns payload
   - client validates fetched payload again before onboard
   - checksum mismatch degrades to `NotFound`
7. revalidated the live smoke path after the rename and checksum work:
   - `kvbm_g2pb_service`
   - `kvbm_g2pb_client_smoke`

Remaining focus:

1. simplify `G2PB` service/client internals where old `G3PB` layering still leaks through
2. simplify the TRT-LLM integration back toward `main` by removing connector over-engineering around remote prefetch
3. introduce a narrow `advise_async_onboarding` path that takes request tokens, hashes them to blocks, and best-effort advises remote onboarding before scheduling
4. keep that advisory path outside the core TRT-LLM connector metadata flow when possible, for example via a global hook registered once KVBM is ready
5. remove remaining hybrid-tier or host-to-disk wording from active code/docs
6. tighten TRT-LLM integration surfaces around `G2PB` as the remote offloading tier
7. keep the smoke path green while making those changes

## Non-Goals

- no protocol redesign for `offer/query/fetch/stage_put/commit_put/load_remote`
- no new storage layer beyond the service-managed `G2` tier
- no broad KVBM cleanup unrelated to `G2PB`

## Current Model

Today the remote tier is:

- a long-running `G2PB` service reachable through discovery
- a staged NIXL transfer path for remote payload movement
- service-managed pinned host memory (`G2`) as the committed storage tier
- mandatory `xxh3_64` payload integrity checks across transfer boundaries

This is no longer a hybrid `G2 + G3` cache. It is a remote `G2` cache with
staged transfers.

## Target Model

The target operating model is:

- peer-backed remote host-memory cache
- committed remote payloads live only in service-managed `G2`
- query/fetch operate against metadata plus `G2`
- corruption degrades to cache miss semantics
- the service/client path is straightforward enough to slot into TRT-LLM without special-case storage assumptions

## Phase 1: Keep The Runtime Green

Files expected to change:

- `lib/llm/src/bin/kvbm_g2pb_service.rs`
- `lib/llm/src/bin/kvbm_g2pb_client_smoke.rs`
- `lib/llm/src/bin/kvbm_nixl_transfer_smoke.rs`

Tasks:

1. keep re-running the existing smoke path during refactors
2. preserve discovery, offer, staged upload, query, fetch, and onboard behavior
3. fix regressions only in service of the `G2PB` integration goal

## Phase 2: Simplify The Active G2PB Path

Files expected to change:

- `lib/llm/src/block_manager/distributed/g2pb.rs`
- `lib/llm/src/block_manager/distributed/g2pb/g2pb_service.rs`
- `lib/llm/src/block_manager/distributed/g2pb/g2pb_client.rs`

Tasks:

1. remove remaining abstraction or naming leftovers that still reflect the old `G3PB` layering
2. simplify service-side storage and metadata flow where the `G2`-only model allows it
3. simplify client-side fetch and routing logic where old hybrid-tier assumptions remain
4. keep the shared protocol/types surface minimal and explicit

## Phase 3: Tighten TRT-LLM Integration

Files expected to change include:

- `lib/bindings/kvbm/src/block_manager/vllm/connector/trtllm_leader.rs`
- `lib/bindings/kvbm/python/kvbm/trtllm_integration/connector/kvbm_connector_leader.py`
- focused TRT-LLM connector surfaces
- focused `G2PB` docs
- active config and environment wiring as needed

Tasks:

1. remove TRT-LLM-specific leader-side remote prefetch orchestration that does not need to live inside the core connector lifecycle
2. introduce a narrow `advise_async_onboarding` path that:
   - takes request identity plus request tokens
   - hashes tokens into sequence blocks
   - performs best-effort remote query/fetch/local host onboard before scheduling
   - never promises availability to the scheduler
3. keep that advisory path side-band to the main connector flow when possible, for example via a globally registered hook once KVBM is initialized
4. pull the TRT-LLM leader back toward `main` so the remaining delta is small and reviewable
5. remove stale wording that implies a disk-backed or hybrid remote cache
6. document only the runtime assumptions that still matter:
   - service must be running
   - `ETCD_ENDPOINTS` must be set for discovery
   - NIXL runtime library must be loadable
   - smoke and integration path should be exercised against a known-good environment

## Guardrails

- do not redesign the worker/service protocol unless integration truly requires it
- do not reintroduce another persistence layer while simplifying `G2PB`
- do not broaden scope into unrelated KVBM cleanup
- keep documentation aligned with the actual runtime behavior

## Immediate Execution Order

1. keep `kvbm_g2pb_service` + `kvbm_g2pb_client_smoke` green
2. simplify `G2PB` internals without changing protocol semantics
3. remove stale wording and assumptions from the active path
4. tighten TRT-LLM workflow integration around the remote `G2PB` tier
