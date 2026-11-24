# lib/router/kv-indexer/src/concurrent_radix_tree_compressed

Concurrent Radix Tree Compressed is a hot-path KV indexer. Read `README.md`
before changing locking, versioning, split, remove, or lookup-repair behavior.

## Guardrails

The crate's `AGENTS.md` has the rules for the whole indexer; these are CRTC's.
PR numbers point to worked examples; open one only when you need to see how a
rule was applied.

- **Changing locking, shape gates, version validation, or retry logic, even in
  passing.** Tell the user, and offer to run the CRTC benchmark or remind them
  to. Any hot-path change to locking, versioning, lookup repair, split/remove,
  child insertion, or read traversal needs before/after numbers in the PR.
- **Taking a child lock in a new place.** Follow the lock order in `README.md`
  ("Concurrency Model"): never a parent's child lock while holding a child's
  `state` or child lock (#18574).
- **Changing split.** Add a white-box test that proves the suffix keeps the
  original children.
- **Changing removal.** Don't split or merge nodes structurally. Keep hashes
  grouped by node, and take the state write lock only when coverage changes
  (#10858, #16134).
- **Touching `internal` or leaf extension.** Without explicit approval and race
  tests, keep `internal` sticky and don't let a node that has ever had children
  extend as a leaf.
- **Changing lookup repair.** Keep it lazy, direction-aware, and batched. Switch
  to eager or global repair only with benchmark numbers.
- **Changing `find_matches` or read traversal.** It may undercount during races,
  but must never overcount past a valid reachable prefix, except through a
  fingerprint collision in the equal-set skip (about 2^-64 per comparison; see
  `README.md`). Add no new overcount path.
- **Adding per-node state.** Most nodes are leaves, so allocate nothing for it
  on a leaf: make it lazy, as `NodeChildren` holds `None` until the first child.
  Don't put it behind an `ArcSwap`; the `NOTE(perf)` in `children.rs` says why
  (#17259, #18574).
- **Touching the store path.** Most events are one-block decode appends, which
  `try_append_at_parent_tail` handles under one lock. Keep it ahead of the
  general path, add no work in front of it, and update the guards in
  `Node::try_append_at_tail` if you change what makes a node an anchor or
  internal (#18108).
- **Needing a node on an event lane.** Borrow the lane's hold with
  `LaneLookup::held`; don't clone or drop the node's `Arc` per block
  (#16134, #18108).
- **Adding a bench-only metric or debug scan.** Keep it behind
  `feature = "bench"` or in tests.
- **Before the PR.** Use the overloaded replay for a write-path change and a
  keep-up replay for a read-path change; the overloaded headline can't show a
  read-path regression. The commands and parameters are in
  `lib/bench/kv_router/INDEXER_BENCH.md`.
