# lib/kv-router

KV-router contains hot-path routing, indexing, scheduling, and active-sequence
state. Keep edits scoped and read the more specific `CLAUDE.md` in subdirectories
when one exists.

## Hash Collections

- Use `FxHashMap` / `FxHashSet` when possible for internal numeric keys and hot
  paths.
- Do not use `FxHashMap` / `FxHashSet` for text keys or externally controlled
  values such as `request_id`; use the standard hash collections there.
