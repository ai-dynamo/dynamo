# lib/router/kv-indexer/src/concurrent_radix_tree_compressed

Concurrent Radix Tree Compressed is a hot-path KV indexer. Read `README.md`
before changing locking, versioning, split, remove, or lookup-repair behavior.

## Guardrails

- Do not casually change the indexer's lock or versioning semantics.
- If a change tangentially affects locking, shape gates, version validation, or
  retry logic, say so explicitly and ask whether to run the relevant benchmark,
  or remind the user to run it, to check for regressions.
- Do not change split semantics without a white-box test proving the suffix
  keeps original children.
- Do not change remove semantics to structurally split or merge nodes.
- Do not remove sticky `internal` behavior or allow leaf extension after a node
  has ever been childful without explicit approval and race tests.
- Do not replace lazy lookup repair with eager/global repair without
  benchmarking. Direction-aware and batched repair are intentional.
- `find_matches` may undercount during races, but must never overcount past a
  valid reachable prefix, except through a fingerprint collision in the
  equal-set skip (about 2^-64 per comparison; see `README.md`). Do not add new
  overcount paths.
- Any hot-path change to locking, versioning, lookup repair, split/remove, child
  insertion, or read traversal must include before/after benchmark numbers in
  the PR.
- Bench-only metrics and debug scans must stay behind `feature = "bench"` or
  tests.
- Preferred CRTC benchmark setup: full Mooncake trace, 128 inference workers,
  trace duplication factor 20, trace length factor 4, 8 event workers, and 20
  fresh-process runs. Measure write-path changes with the 750 ms overloaded
  replay (`achieved_block_ops_per_sec`). Measure read-path changes with a
  keep-up replay (`kept_up=true`, negligible drain) and compare
  `query_service` percentiles; the overloaded headline cannot detect read-path
  regressions. See `lib/bench/kv_router/INDEXER_BENCH.md`.

## Performance

The crate's `AGENTS.md` has the rules for the whole indexer; these are CRTC's.

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
- **Changing removal.** Keep hashes grouped by node, and take the state write
  lock only when coverage changes (#10858, #16134).
- **Taking a child lock in a new place.** Follow the lock order in `README.md`
  ("Concurrency Model"): never a parent's child lock while holding a child's
  `state` or child lock (#18574).
