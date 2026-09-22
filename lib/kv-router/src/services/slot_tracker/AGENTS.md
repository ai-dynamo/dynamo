# lib/kv-router/src/services/slot_tracker

This module is a thin HTTP adapter over the runtime-independent sequence tracker.

## Guardrails

- Keep this module usable without Dynamo runtime or LLM-layer dependencies.
- Do not add a write actor, lifecycle queue, batching layer, or service-level admission policy.
- Keep request lifecycle operations inline on `ActiveSequencesMultiWorker`.
- Treat load reads as advisory snapshots from the existing derived read model.
- Preserve the core tracker's arrival-ordered lifecycle semantics unless a separate design explicitly changes them.
