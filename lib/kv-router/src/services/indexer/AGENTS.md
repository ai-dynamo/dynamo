# lib/kv-router/src/services/indexer

Standalone indexer must remain usable without Dynamo runtime or LLM-layer
dependencies.

## Guardrails

- Never introduce `lib/runtime` / `dynamo-runtime` dependencies.
- Never introduce `lib/llm` / `dynamo-llm` dependencies, gated or ungated.
