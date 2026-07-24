# Dynamo OpenEngine sidecar

This crate consumes generated `openengine-proto` 0.3.0 bindings from immutable
OpenEngine commit `a66ff6f73a65e262a7c3edd5ea6fd0d8701d402f`.

The `dynamo-openengine-sidecar` binary is intentionally engine-neutral. The
same artifact discovers and serves TRT-LLM, vLLM, and SGLang endpoints without
engine-name dispatch. `--expected-engine` and `--expected-schema-release` are
optional deployment assertions; compatible schema revision ranges are
negotiated independently of the release assertion.

OpenEngine advertises the canonical model and tokenizer separately. Dynamo
uses the tokenizer source for preprocessing, registers the primary served name
plus aliases, forwards context-first handoff attributes without interpreting
engine-specific profiles, and creates typed bootstrap sessions only when the
connector advertises `supports_client_bootstrap`.

Launch and DGD examples live under each engine's `examples/backends/<engine>`
tree. In Kubernetes, mount the same tokenizer/model cache into the CPU sidecar
and GPU engine containers, and mount one shared LoRA cache across P/D pods.

`Cargo.toml` uses the pinned Git dependency so clean CI and release builds do
not depend on a sibling checkout. To develop against a local OpenEngine
worktree, add an uncommitted Cargo patch with an absolute path:

```toml
[patch."https://github.com/ai-dynamo/openengine.git"]
openengine-proto = { path = "/absolute/path/to/openengine/packages/rust/openengine-proto" }
```

Remove the override before generating `Cargo.lock` or publishing changes.
