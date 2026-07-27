# dynamo-mocker

`dynamo-mocker` is the shared engine-simulation crate behind live Mocker workers and offline
DynoSim replay. It models scheduler, KV-cache, timing, handoff, and lower-tier behavior without
executing model inference on GPUs.

## What This Crate Provides

- `MockEngineArgs` for configuring a simulated engine
- `engine::create_engine` for building vLLM, SGLang, or TensorRT-LLM engine behavior
- `KvEventPublishers` hooks for emitting router-visible KV cache events
- `loadgen` and `replay` modules for synthetic and trace-driven experiments
- `kvbm_offload` for multi-tier KV movement and bandwidth sharing

## Basic Rust Usage

```rust
use dynamo_mocker::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs,
};
use dynamo_mocker::engine::create_engine;

let args = MockEngineArgs::builder()
    .block_size(16)
    .num_gpu_blocks(1024)
    .max_num_seqs(Some(32))
    .max_num_batched_tokens(Some(4096))
    .build()
    .unwrap();

let engine = create_engine(
    args,
    0,
    None,
    KvEventPublishers::default(),
    None,
    FpmPublisher::default(),
);

engine.receive(DirectRequest {
    tokens: vec![1, 2, 3, 4],
    max_output_tokens: 16,
    dp_rank: 0,
    ..DirectRequest::default()
});
```

Most users interact with the crate through `python -m dynamo.mocker` for live workers or
`python -m dynamo.replay` for deterministic offline experiments.

## Further Reading

- Mocker guide:
  [../../docs/dynosim/mocker.md](../../docs/dynosim/mocker.md)
- DynoSim runs guide:
  [../../docs/dynosim/runs.md](../../docs/dynosim/runs.md)
- Simulation model:
  [../../docs/dynosim/modeling.md](../../docs/dynosim/modeling.md)
- Python component README:
  [../../components/src/dynamo/mocker/README.md](../../components/src/dynamo/mocker/README.md)
