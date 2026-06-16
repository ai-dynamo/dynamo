# dynamo-data-gen

Shared schemas and primitives for Dynamo data generation. Currently hosts the
Mooncake replay JSONL row, the rolling block-hash-to-id mapper, the token-block
hashing helper, and the JSONL writer.

The crate is producer- and consumer-agnostic on purpose: it owns the schema
once so that producers (`dynamo-bench`'s Claude exporter and
`agent_trace_to_mooncake` binary) and consumers (`dynamo-mocker`'s load
generator) can never drift.

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
