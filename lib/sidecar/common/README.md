# Sidecar common

Shared infrastructure for Rust sidecars:

- gRPC transport arguments and defaults
- plaintext endpoint validation
- connection pooling and startup retries
- gRPC-to-Dynamo error mapping

Engine protocols and request conversion stay in each sidecar crate.

## Shared sidecar unit contracts

The `testing` feature provides four shared contracts. Each runs against the real
`LLMEngine::generate` implementation in vLLM, SGLang, and TensorRT-LLM: twelve
test entries in total. Only native RPC opening and response reads are scripted.
No engine process, model, GPU, socket, or discovery service is used by these tests.

| Report row | Prototype coverage |
|---|---|
| R09: streaming | Exact token sequence, one final length terminal, prompt/completion/total usage, ignored payload after terminal. SGLang fixtures include cumulative repetition and regression; TensorRT-LLM includes trailing zero counters. |
| R11: transport failure | Open failure, token then early EOF, token then read error; preserve prior tokens, return one typed error, stop consuming. |
| R12: cancellation | Before open, during pending open, after a token while a read is pending; cancelled terminal, observed usage, local resource release. |
| R13: cleanup | Generation before start fails; cleanup before start and during a pending read is repeatable; active generation ends cancelled. |

Run from the repository root:

```bash
cargo test --locked \
  -p dynamo-vllm-sidecar \
  -p dynamo-sglang-sidecar \
  -p dynamo-trtllm-sidecar \
  --lib conformance
```

`src/testing.rs` owns the scenarios, cancellation ordering, and assertions.
Each adapter's `src/conformance.rs` implements `SidecarFixture` with its native
protobuf messages and a constructed engine, then invokes
`sidecar_contract_tests!`. Add a shared contract there once; add native fixture
data in the adapters when their protocols differ. Enable `testing` only in
`[dev-dependencies]`.

Tests poll pending operations explicitly and use paused Tokio time for the
failure deadline. They preserve existing adapter differences: SGLang opens its
RPC lazily and classifies early EOF as `EngineShutdown`; vLLM and TensorRT-LLM
open during `generate` and classify early EOF as `Unknown`.

This is partial coverage of those report rows. Other finish reasons, logprobs,
disaggregation, startup/discovery failures, remote abort, and request isolation
remain outside this prototype. Resource checks establish local stream release
after draining, not cancellation of remote GPU work. Existing socket tests
remain separate transport integration coverage.
