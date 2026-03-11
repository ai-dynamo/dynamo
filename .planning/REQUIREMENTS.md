# Requirements: kvbm-connector Port

**Defined:** 2026-03-11
**Core Value:** kvbm-connector compiles as a workspace member with all imports resolved against current crate structure

## v1 Requirements

### Workspace Integration

- [ ] **WS-01**: `lib/kvbm-connector` is listed as a member in root `Cargo.toml`
- [ ] **WS-02**: `kvbm-connector/Cargo.toml` declares all required workspace crate dependencies
- [ ] **WS-03**: `kvbm-connector` is listed in `[workspace.dependencies]` for downstream use

### Import Migration

- [ ] **IMP-01**: All `crate::v2::*` imports replaced with correct workspace crate paths (`kvbm_logical`, `kvbm_engine`, `kvbm_common`)
- [ ] **IMP-02**: All `crate::distributed::*` imports replaced with `kvbm_engine` paths
- [ ] **IMP-03**: All `crate::logical::*` imports replaced with `kvbm_logical::*`
- [ ] **IMP-04**: All `crate::physical::*` imports replaced with `kvbm_physical::*`
- [ ] **IMP-05**: All `crate::integrations::*` self-referential imports resolved (moved to correct module paths or local re-exports)

### Nova → Velo Migration

- [ ] **VELO-01**: `src/connector/worker/nova/client.rs` updated to use `velo` transport types
- [ ] **VELO-02**: `src/connector/worker/nova/service.rs` updated to use `velo` types
- [ ] **VELO-03**: `src/connector/worker/nova/protocol.rs` updated for velo protocol
- [ ] **VELO-04**: All `nova` module imports across the codebase updated to velo equivalents
- [ ] **VELO-05**: `velo` dependency declared in `kvbm-connector/Cargo.toml`

### Compilation

- [ ] **COMP-01**: `cargo check -p kvbm-connector` passes with zero errors
- [ ] **COMP-02**: `cargo check -p kvbm-connector` passes with zero warnings (or warnings are pre-existing and documented)
- [ ] **COMP-03**: Existing workspace members still compile after changes (`cargo check --workspace` passes)

### Test Porting

- [ ] **TEST-01**: `testing` feature enabled in `kvbm-connector/Cargo.toml` pulling in test infra from kvbm-logical, kvbm-physical, kvbm-engine
- [ ] **TEST-02**: Connector-specific tests ported from `ryan/kvbm-next:lib/kvbm/src/v2/testing` into the current workspace
- [ ] **TEST-03**: Ported tests compile with `cargo test -p kvbm-connector --features testing`
- [ ] **TEST-04**: Ported tests pass with `cargo test -p kvbm-connector --features testing`

## v2 Requirements

### Integration

- **INT-01**: vLLM Python component wired to kvbm-connector
- **INT-02**: TRT-LLM backend wired to kvbm-connector
- **INT-03**: SGLang backend wired to kvbm-connector
- **INT-04**: kvbm-connector accessible from Python bindings

## Out of Scope

| Feature | Reason |
|---------|--------|
| New kvbm-connector features | Port only — no scope expansion |
| Refactoring beyond compile fixes | Minimize diff, compile first |
| Python/vLLM integration | Separate milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WS-01 | Phase 1 | Pending |
| WS-02 | Phase 1 | Pending |
| WS-03 | Phase 1 | Pending |
| IMP-01 | Phase 2 | Pending |
| IMP-02 | Phase 2 | Pending |
| IMP-03 | Phase 2 | Pending |
| IMP-04 | Phase 2 | Pending |
| IMP-05 | Phase 2 | Pending |
| VELO-01 | Phase 2 | Pending |
| VELO-02 | Phase 2 | Pending |
| VELO-03 | Phase 2 | Pending |
| VELO-04 | Phase 2 | Pending |
| VELO-05 | Phase 2 | Pending |
| COMP-01 | Phase 3 | Pending |
| COMP-02 | Phase 3 | Pending |
| COMP-03 | Phase 3 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| TEST-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-11*
*Last updated: 2026-03-11 after roadmap creation*
