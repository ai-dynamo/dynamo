# lib/kv-router

KV-router contains hot-path routing, indexing, scheduling, and active-sequence
state. Keep edits scoped and read the more specific `CLAUDE.md` in subdirectories
when one exists.

## Dependencies

- `validator` and `rand` carry a disproportionate transitive footprint for the
  small amount of functionality KV-router needs. Prefer local validation and
  `fastrand`; do not reintroduce them without a measured need.
- Keep `axum` and `reqwest` default features disabled and enable only features
  used by KV-router. Their unused feature sets add substantial build weight.

## Hash Collections

- Use `FxHashMap` / `FxHashSet` when possible for internal numeric keys and hot
  paths.
- Do not use `FxHashMap` / `FxHashSet` for text keys or externally controlled
  values such as `request_id`; use the standard hash collections there.
