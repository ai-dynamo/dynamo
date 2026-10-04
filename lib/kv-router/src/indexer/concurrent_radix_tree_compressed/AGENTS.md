# lib/kv-router/src/indexer/concurrent_radix_tree_compressed

Concurrent Radix Tree Compressed is a hot-path KV indexer. Read `README.md`
before changing locking, versioning, split, remove, or lookup-repair behavior.

## Guardrails

- Do not casually change the indexer's lock or versioning semantics.
- If a change tangentially affects locking, shape gates, version validation, or
  retry logic, confirm with PeaBrane first and explicitly ask whether to run the
  relevant benchmark, or remind the user to run it, to check for regressions.
- Do not change split semantics without a white-box test proving the suffix
  keeps original children.
- Do not change remove semantics to structurally split or merge nodes.
- Do not remove sticky `internal` behavior or allow leaf extension after a node
  has ever been childful without explicit approval and race tests.
- Do not replace lazy lookup repair with eager/global repair without
  benchmarking. Direction-aware and batched repair are intentional.
- `find_matches` may undercount during races, but must never overcount past a
  valid reachable prefix. Do not add overcount paths, such as skipping the
  per-hop coverage intersection when set sizes match (see `README.md`).
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
