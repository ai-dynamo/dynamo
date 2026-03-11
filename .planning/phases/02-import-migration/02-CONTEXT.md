# Phase 2: Import Migration - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace every broken import path inside `lib/kvbm-connector/src/` with the correct workspace crate path, and swap all nova transport types for velo equivalents. This is pure import surgery — no logic changes, no new features, no refactoring beyond what is required to resolve broken paths.

</domain>

<decisions>
## Implementation Decisions

### Nova module rename
- Rename directory `src/connector/worker/nova/` → `src/connector/worker/velo/`
- Apply a full rename inside the files: update type names, docstrings, and comments that reference "Nova" to say "Velo"
- This is a complete nova→velo sweep, not just import surgery

### Migration execution order
- Fix one import namespace at a time, running `cargo check -p kvbm-connector` between each pass
- Order: simple-to-complex by mapping confidence:
  1. `crate::logical::*` → `kvbm_logical::*`
  2. `crate::physical::*` → `kvbm_physical::*`
  3. `crate::distributed::*` → `kvbm_engine::*` (distributed subtree)
  4. `crate::v2::*` → correct workspace crate per type
  5. `crate::integrations::*` self-refs → `crate::*` (kvbm-connector's root lib.rs IS the old integrations module)
  6. `nova`→`velo` transport swap (including directory rename, type renames, docstrings)
- Each namespace fix is committed atomically — one commit per namespace step

### Blocker escalation
- If a type from a broken import path cannot be found in any workspace crate after inspection: flag and hard-stop. Document the missing type and stop the plan for manual resolution. Nothing silently wrong.
- Testing imports (`crate::v2::testing::connector::*` in `connector/worker/tests.rs`) must be gated behind `#[cfg(test)]` and/or the `testing` feature. These types should be well-established in `kvbm-engine`'s testing infrastructure — reference `ryan/kvbm-next:lib/kvbm/src/v2/testing` to verify. If they exist in the testing feature, wire correctly; if not, note as a Phase 4 concern.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `kvbm-engine::worker`: exports `VeloWorkerClient`, `VeloWorkerService` — direct replacements for `NovaWorkerClient`
- `velo` crate: re-exports `Velo` (replaces `dynamo_nova::Nova`), `Handler`/`TypedUnaryHandlerBuilder` builders (replace `NovaHandler`), `TypedUnaryResult` (replace `dynamo_nova::am::TypedUnaryResult`), `PeerInfo`/`WorkerAddress` (replace `dynamo_nova_backend::*`)
- `velo::Messenger` / `TypedUnaryBuilder` / `UnaryBuilder` — available as direct drop-ins for the nova AM call patterns

### Established Patterns
- `crate::integrations::*` maps to `crate::*` — kvbm-connector's root `lib.rs` is the extracted integrations module; `crate::integrations::connector::worker::ConnectorWorkerClient` becomes `crate::connector::worker::ConnectorWorkerClient`
- `NovaWorkerClient` → `VeloWorkerClient` (confirmed via `kvbm-engine::worker::mod.rs` pub use)
- Nova handler registration pattern: `NovaHandler::typed_unary_async(name, closure).build()` → velo equivalent builder chain

### Integration Points
- `connector/leader/init.rs` uses both `dynamo_nova_backend::PeerInfo` and `NovaWorkerClient` — both have confirmed velo equivalents
- `connector/worker/nova/mod.rs` declares handler name constants (e.g., `kvbm.connector.worker.*`) — these string constants stay unchanged
- Testing infra gating: `connector/worker/tests.rs` — must be `#[cfg(test)]` + `#[cfg(feature = "testing")]`

</code_context>

<specifics>
## Specific Ideas

- "nova → velo is the same API, name change + minor updates" — treat as find-and-replace, don't redesign
- For testing imports: check `ryan/kvbm-next:lib/kvbm/src/v2/testing` as the source of truth for what types exist and where they now live in the workspace

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-import-migration*
*Context gathered: 2026-03-11*
