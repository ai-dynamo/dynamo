# Phase 4: Test Porting - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Port connector-specific tests from `ryan/kvbm-next:lib/kvbm/src/v2/testing/` and `lib/kvbm/src/v2/integrations/connector/worker/tests.rs` into `lib/kvbm-connector/`. Tests must compile and pass under `--features testing`. No new connector features or test infrastructure beyond what's in the source branch.

</domain>

<decisions>
## Implementation Decisions

### Scope of files to port
- Port **all files** from `ryan/kvbm-next:lib/kvbm/src/v2/testing/` (all 25+ files including subdirectories: e2e/, offloading/, scheduler/)
- Also port `lib/kvbm/src/v2/integrations/connector/worker/tests.rs` — it contains inline unit tests for ConnectorWorker and uses connector testing utilities
- Start with full verbatim copy; disable (comment out or `#[ignore]`) tests that don't compile or don't make sense after migration

### DRY — sub-crate testing exports take priority
- kvbm-logical, kvbm-physical, kvbm-engine each already have `src/testing/` modules exported under their `testing` feature
- When a v2/testing/ file duplicates what a sub-crate testing module already provides (e.g. managers.rs, token_blocks.rs, physical.rs), **use the sub-crate export instead** — do not copy duplicates into kvbm-connector
- Only add to `lib/kvbm-connector/src/testing/` what is genuinely connector-specific (ConnectorTestConfig, TestConnectorInstance, connector scheduler mocks, connector e2e fixtures)

### Test file location
- Connector-specific testing utilities: `lib/kvbm-connector/src/testing/` module, gated behind `#[cfg(feature = "testing")]`
- worker/tests.rs inline unit tests: `lib/kvbm-connector/src/connector/worker/tests.rs` (mirrors source structure), included via `mod tests` in worker/mod.rs under `#[cfg(test)]`

### Import migration strategy
- Port verbatim first, then fix imports iteratively (same approach as Phase 2)
- Apply same import mappings: `crate::v2::integrations::connector::*` → local paths, `crate::v2::testing::*` → sub-crate testing exports or local testing module
- `dynamo_kvbm_config` → `kvbm_config`, `crate::v2::*` → workspace crate paths

### Nova → Velo in tests
- Apply the same nova→velo mapping established in Phase 2: `dynamo_nova::Nova` → `velo::...`, `dynamo_nova_backend::WorkerAddress` → `velo::...`, `runtime.nova` → `runtime.messenger`
- If a test file uses a nova API with **no known velo equivalent**, stop and report — do not silently disable or guess the mapping

### Testing feature (TEST-01)
- `testing` feature already declared in `kvbm-connector/Cargo.toml` (added in Phase 1) with `kvbm-engine/testing`, `kvbm-logical/testing`, `kvbm-physical/testing` — TEST-01 is already satisfied; confirm and document

### Disabled tests
- Tests that don't make sense after migration: wrap in `#[cfg(TODO)]` or `#[ignore]` with a comment explaining why. Do not delete — they may become relevant later.

</decisions>

<specifics>
## Specific Ideas

- "Bring in all the tests, then we will disable those that don't make sense" — start broad, prune after
- "DRY dry dry" — the sub-crates already export their testing infra publicly; connector should consume it, not re-implement it
- Remove duplications found in sub-crate testing modules when porting — don't copy what already exists elsewhere

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets (sub-crate testing exports)
- `kvbm_logical::testing::*` — block managers, sequences, token_blocks, config, pools, blocks (src/testing/ in kvbm-logical)
- `kvbm_physical::testing::*` — physical layouts, test agents (src/testing/ in kvbm-physical)
- `kvbm_engine::testing::*` — distributed fixtures, events, managers, messenger, offloading, physical, token_blocks (src/testing/ in kvbm-engine)

### Connector-Specific (need to port to kvbm-connector/src/testing/)
- `connector.rs` — ConnectorTestConfig (builder + JSON API), TestConnectorInstance (1 leader + N workers fixture), TestConnectorWorker
- `scheduler/connector_tests.rs` — scheduler integration tests for the connector
- `scheduler/mock/` — mock engine, model, connector e2e tests, abort tests
- `worker/tests.rs` — unit tests for ConnectorWorker flag lifecycle (intra-pass onboard/offload)

### Possibly Redundant (check against sub-crate exports first)
- `managers.rs` — likely covered by kvbm_engine::testing::managers or kvbm_logical::testing::managers
- `physical.rs` — likely covered by kvbm_engine::testing::physical
- `token_blocks.rs` — likely covered by kvbm_logical::testing::token_blocks
- `distributed.rs` — likely covered by kvbm_engine::testing::distributed
- `events.rs` — likely covered by kvbm_engine::testing::events
- `nova.rs` — needs velo migration; check kvbm_engine::testing::messenger for equivalent
- `offloading/` — check kvbm_engine::testing::offloading
- `e2e/` — may be connector-specific enough to port

### Integration Points
- `lib/kvbm-connector/src/connector/worker/mod.rs` — add `mod tests` under `#[cfg(test)]` to include worker/tests.rs
- `lib/kvbm-connector/src/lib.rs` — add `pub mod testing` under `#[cfg(feature = "testing")]`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-test-porting*
*Context gathered: 2026-03-11*
