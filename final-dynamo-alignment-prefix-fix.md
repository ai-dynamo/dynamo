# Final Dynamo alignment and prefix fixture fix

## Scope

Applied the remaining Dynamo ABI hardening from the final Graham re-review.

- Dynamo Steppable `submit_batch` rejects a non-empty `DirectRequestV1` slice
  whose pointer is not aligned for `DirectRequestV1`, before constructing a
  Rust slice.
- Placement ABI typed-slice validation now checks alignment in addition to
  nullness and bounded byte extent. This covers mutation and KV-event batches,
  nested stored-block and removed-hash accessors, admission identity slices,
  and topology/capacity typed slices.
- Added misaligned `DirectRequestV1`, `PlacementMutationV1`, KV event/nested
  block, and worker-topology regressions.
- Extended the real prefix-sized Steppable V1 allocation fixture with all
  required callbacks, a descriptor, and `validate_descriptor_v1` coverage;
  optional-tail readers still prove the prefix does not expose newer fields.

## Verification

All checks passed:

- `cargo test -p aisimulate-placement-abi`
- `cargo test -p aiperf-steppable-abi --test layout`
- `cargo test -p dynamo-steppable-provider --test abi`
- `rustfmt --edition 2024 --check` on all four touched Rust files
- `git diff --check`

The provider test target also includes the concurrently landed batch-panic
poison regression; its full ABI test target passed.
