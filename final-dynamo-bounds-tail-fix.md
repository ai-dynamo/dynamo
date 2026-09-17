# Final Dynamo bounds and legacy-tail fix

## Changes

- Bound every Dynamo steppable and placement raw FFI slice constructor by
  `isize::MAX / size_of::<T>()`, including `u32`, `DirectRequestV1`,
  `RequestIdV1`, event/fact records, topology records, and byte slices.
- Apply the same bound before reconstructing provider-owned boxed slices during
  release, and reject oversized batches before the submit wrapper clears output
  memory.
- Make malformed oversized nested placement event slices return `None` from
  accessors instead of constructing an invalid slice.
- Correct the V1 descriptor safety contract to require only the readable legacy
  `PluginVTableV1Prefix`; optional tails remain extent-gated. The regression
  test allocates only the prefix and verifies neither optional helper reads it.

## Verification

- `cargo test -p 'aisimulate-placement-abi@0.12.0-dynamo-bench' -p aiperf-steppable-abi`
  — passed (2 placement ABI unit tests and 3 steppable ABI layout tests).
- `cargo test -p dynamo-steppable-provider --test abi ffi_rejects_slice_lengths_over_the_isize_byte_bound`
  — blocked while the parallel AISim core batch API change was mid-edit; the
  dependency reported an unrelated missing `PlacementPolicy` method.
- `cargo fmt --all -- --check` — repository-wide check reports pre-existing
  formatting diffs in the parallel AISim worktree; all touched files were
  formatted directly with rustfmt and `git diff --check` passes.

## Atomic batch API follow-up

- Updated the Dynamo provider for the owned `submit_batch(Vec<DirectRequest>)`
  API and map poisoned batch failures to `StatusV1::INTERNAL`; the ABI output
  remains zeroed on every failed batch.
- The focused provider test was retried after the API update but remains
  blocked until the parallel AISim core and mocker edits are coherent.
