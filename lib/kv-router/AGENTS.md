# lib/kv-router

KV-router contains hot-path routing, indexing, scheduling, and active-sequence
state. Keep edits scoped and read the more specific `CLAUDE.md` in subdirectories
when one exists.

When router configuration is serialized inside a model deployment card, it is
part of the N-2 worker/frontend wire contract described in
[`lib/llm/CLAUDE.md`](../llm/CLAUDE.md). Keep compatibility handling narrow and
give deprecated wire fields a versioned removal TODO.

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

## Engine Hash Invariant

- Within one indexer/hash domain, engine-published local block hashes and
  external sequence hashes are deterministic and consistent across workers: a
  unique local block chain maps to one unique external sequence-hash chain, and
  vice versa.
- Treat a violation as a producer, configuration, or protocol error. Do not
  revalidate this invariant by scanning per-worker external-hash ownership on
  request lookup or scheduling hot paths.
