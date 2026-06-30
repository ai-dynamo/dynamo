# dynamo-data-gen

Shared schemas and primitives for Dynamo data generation. Currently hosts the
Mooncake replay JSONL row, the rolling block-hash-to-id mapper, the token-block
hashing helper, the JSONL writer, and the shared Dynamo request-trace loader and
transient row lowering used to build replay models in memory.

The crate is producer- and consumer-agnostic on purpose: it centralizes the
Mooncake schema and request-trace ingestion primitives so replay consumers do
not duplicate either representation. Dynamo request-trace replay does not emit
an intermediate Mooncake file.

## Guardrails

- `MooncakeRow` mirrors the externally-authored Mooncake trace format.
  `timestamp` and `delay` are `f64` milliseconds, and deserialization accepts
  the upstream aliases (`input_tokens`, `output_tokens`, `created_time`,
  `delay_ms`). Don't rename the fields or drop the aliases — third-party
  traces in the wild use the upstream names.
- Keep this crate dependency-light. Anything tokenizer- or HTTP-shaped does
  not belong here; it belongs in `dynamo-bench` or the relevant producer.
- Changes to the schema or the hash helper are de facto wire-format changes.
  Add round-trip tests for both canonical and aliased field names, and check
  that existing producer/consumer tests still cover the new shape.
