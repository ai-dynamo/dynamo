# lib/kv-router/src/standalone_indexer

Standalone indexer must remain usable without Dynamo runtime or LLM-layer
dependencies.

## Guardrails

- Never introduce `lib/runtime` / `dynamo-runtime` dependencies unless they are
  explicitly feature-gated.
- Never introduce `lib/llm` / `dynamo-llm` dependencies, gated or ungated.
