# dynamo-mocker

`dynamo-mocker` is a GPU-free simulation crate for Dynamo's LLM scheduling and KV-cache behavior.
It is used for testing, replay, and benchmarking workflows where you want realistic scheduler and
cache behavior without running a real inference engine.

## What This Crate Provides

- `MockEngineArgs` for configuring a simulated engine
- `engine::create_engine` for building a vLLM-style or SGLang-style mock scheduler
- `KvEventPublishers` hooks for emitting router-visible KV cache events
- `loadgen` and `replay` modules for synthetic and trace-driven experiments

## Basic Rust Usage

```rust
use dynamo_mocker::common::protocols::{
    DirectRequest, KvEventPublishers, MockEngineArgs,
};
use dynamo_mocker::engine::create_engine;

let args = MockEngineArgs::builder()
    .block_size(16)
    .num_gpu_blocks(1024)
    .max_num_seqs(Some(32))
    .max_num_batched_tokens(Some(4096))
    .build()
    .unwrap();

let engine = create_engine(args, 0, None, KvEventPublishers::default(), None);

engine.receive(DirectRequest {
    tokens: vec![1, 2, 3, 4],
    max_output_tokens: 16,
    uuid: None,
    dp_rank: 0,
    arrival_timestamp_ms: None,
});
```

This crate is also the foundation for Dynamo's higher-level mocker CLI and replay tooling. In many
deployments you will interact with it indirectly through the Python entry points rather than
embedding it directly as a standalone Rust dependency.

## Further Reading

- Mocker guide:
  <https://github.com/ai-dynamo/dynamo/blob/main/docs/mocker/mocker.md>
- Trace replay guide:
  <https://github.com/ai-dynamo/dynamo/blob/main/docs/benchmarks/mocker-trace-replay.md>
- Python component README:
  <https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/mocker/README.md>
