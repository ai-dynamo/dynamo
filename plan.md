# G2PB Migration Plan

## Goal

Replace the current `G2PB` service storage model with a large host-memory-backed
remote cache and remove `foyer` entirely.

The implementation must stay focused on these outcomes:

1. keep the current service/client remote protocol working
2. remove `foyer` from the service storage path
3. make the service operational using only large `G2` capacity
4. complete the `G2PB` rename only after the storage/service change is stable

The migration is done only when:

1. the service/client implementation is simplified to the current `G2`-only model
2. the full remote tier is consistently named `G2PB`
3. the service binary is `kvbm_g2pb_service`
4. the client smoke is `kvbm_g2pb_client_smoke`
5. the live smoke path still passes after the rename

## Status

Completed:

1. removed `foyer` from the backend storage implementation
2. removed `foyer` CLI/config from `kvbm_g2pb_service`
3. removed the `foyer` dependency from `lib/llm/Cargo.toml`
4. split the monolithic `g2pb.rs` into:
   - `g2pb.rs` for shared protocol/types
   - `g2pb/g2pb_service.rs` for backend storage/service logic
   - `g2pb/g2pb_client.rs` for client/discovery/routing logic
5. revalidated the live smoke path after the refactor:
   - `kvbm_g2pb_service`
   - `kvbm_g2pb_client_smoke`

Remaining focus:

1. keep the service/client path green while completing the `G2PB` rename
2. remove remaining host-to-disk wording and assumptions from the operational path
3. align the remote tier with the TRT-LLM integration workflow without broad unrelated cleanup

## Non-Goals For The First Pass

- no protocol redesign for `offer/query/fetch/stage_put/commit_put/load_remote`
- no endpoint rename during the foyer-removal phase
- no admission-policy rename during the foyer-removal phase
- no broad cleanup unrelated to backend storage removal

## Current Model

Today the remote service is structured as:

- worker-side routing and request-plane discovery
- service-side staged NIXL transfer path
- service pinned host memory (`G2`) for staged/onboarded data
- service `foyer` storage as the persistent and overflow tier

The key problem is that `foyer` depends on local SSD reliability, which is not
acceptable for the target machines. The replacement model is:

- large service-managed `G2` capacity
- no `foyer`
- no SSD dependency in the remote cache path

## Target Model

After the first implementation phase, the remote cache should behave as:

- peer-backed remote host-memory cache
- all committed remote payloads live in service-managed `G2`
- query/fetch operate only against metadata plus `G2`
- eviction is based only on host-memory capacity and access policy

This means the service is no longer a hybrid `G2 + G3` cache. It becomes a
remote `G2` cache with staged NIXL transfers.

## Phase 1: Remove Foyer From Storage Core

Files expected to change:

- `lib/llm/src/block_manager/distributed/g2pb.rs`
- `lib/llm/src/bin/kvbm_g2pb_service.rs`

Tasks:

1. Remove `foyer` imports, config fields, cache locations, helper methods, and
   background assumptions from `g2pb.rs`.
2. Collapse storage location tracking to `G2` only.
3. Remove logic that:
   - inserts payloads into `foyer`
   - promotes payloads from `foyer` into `G2`
   - validates and prunes stale `foyer` metadata
   - falls back to `foyer` on query/fetch
4. Keep metadata and capacity management centered on host-memory-backed blocks.
5. Preserve current request/response protocol and error behavior as much as
   possible.

## Phase 2: Rework Backend Configuration Around Large G2

Files expected to change:

- `lib/llm/src/bin/kvbm_g2pb_service.rs`
- `lib/llm/src/block_manager/distributed/g2pb.rs`

Tasks:

1. Remove CLI/config surfaces for:
   - `--foyer-dir`
   - `--foyer-memory-bytes`
   - `--foyer-disk-bytes`
2. Keep and clarify the `G2` sizing surface.
3. Make backend capacity planning explicitly about remote host memory.
4. Ensure comments and help text describe the backend as a large host-memory
   cache, not a host-plus-disk cache.

## Phase 3: Restore Operational Backend Behavior

Files expected to change:

- `lib/llm/src/bin/kvbm_g2pb_service.rs`
- `lib/llm/src/bin/kvbm_g2pb_client_smoke.rs`
- `lib/llm/src/bin/kvbm_nixl_transfer_smoke.rs`

Tasks:

1. Re-run the existing smoke path:
   - `kvbm_nixl_transfer_smoke`
   - `kvbm_g2pb_service` + `kvbm_g2pb_client_smoke`
2. Verify that the worker can still:
   - discover the backend
   - offer remote blocks
   - perform staged upload
   - query remote hits
   - fetch staged descriptors
   - onboard back into local device memory
3. Fix regressions only in service of the large-`G2` backend migration.
4. simplify G2pb.rs since it has introduced enums around G2 or G3Foyer, now its so simple
5. refactor the monolithic `g2pb.rs` into a shared module plus explicit service/client modules
6. make sure its everything works.


## Phase 4: Rename G2PB To G2PB

This phase starts only after the foyer-free backend is operational.

Files expected to change include:

- distributed module names
- backend and smoke binary names
- docs
- metrics
- env vars
- comments
- endpoint and namespace naming

Tasks:

1. Rename symbols and files from `G3PB` to `G2PB`.
2. Use role-based names for binaries and docs:
   - `kvbm_g3pb_backend` -> `kvbm_g2pb_service`
   - `kvbm_g3pb_worker_smoke` -> `kvbm_g2pb_client_smoke`
   This avoids overloaded `backend` and `worker` terminology.
3. Rename endpoint and namespace strings to the final `G2PB` form.
4. Update documentation to describe the service as a remote host-memory cache.

## Guardrails

- do not rename first
- do not redesign the worker/backend protocol first
- do not introduce new storage layers while removing `foyer`
- do not broaden the scope into unrelated KVBM cleanup

## Immediate Execution Order

1. Remove `foyer` from `g2pb.rs`
2. Remove `foyer` CLI/config from `kvbm_g2pb_service.rs`
3. Restore green smoke coverage for backend + worker
4. Rename `G2PB` to `G2PB`
