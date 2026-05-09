---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mocker Backend (Rust)
---

# Mocker Backend (Rust)

Reference Rust backend for Dynamo. Wraps the `dynamo-mocker` scheduler
in the `LLMEngine` contract from
[`dynamo-backend-common`](../../../lib/backend-common/), giving a
full-fidelity mocked engine — one shared forward-pass loop paces all
in-flight requests (continuous batching), not independent per-request
timers. Use it as a template for writing your own Rust backend, a
stand-in engine for AIPerf / end-to-end pipeline tests, and as the
forcing function that keeps the mocker scheduler tied to the
`LLMEngine` contract.

See the `LLMEngine` trait's doc comments in
[`engine.rs`](../../../lib/backend-common/src/engine.rs) for the
authoritative contract.

## Quick demo (docker compose)

One command brings up NATS, etcd, the Dynamo frontend, and the mocker
backend — all built from source in this repo:

```bash
cd lib/backend-common/examples/mocker
docker compose up --build
```

First `up` is slow — it builds the Rust image and downloads the Qwen3
tokenizer from HuggingFace. Subsequent runs reuse Docker's layer cache
and a named volume for the HF cache.

In another terminal, send a chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "mocker-model",
          "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 32
        }'
```

The response carries random token IDs detokenized through Qwen's
vocabulary — not meaningful text, but it proves every stage of the
pipe is connected end-to-end.

Tear down with `docker compose down` (add `-v` to drop the HF cache
volume).

Set `HF_TOKEN` in your shell if you hit HuggingFace rate limits.

## Build and run locally

```bash
cargo build -p dynamo-mocker-backend --release

# For chat/completions endpoints the frontend needs a tokenizer + chat
# template, so point --model-path at an open HF repo. For tensor/prefill
# endpoints (no tokenization), omit --model-path for name-only mode.
./target/release/dynamo-mocker-backend --model-path Qwen/Qwen3-0.6B
```

Requires the infra services (NATS, etcd) you normally run with Dynamo
reachable via `NATS_SERVER` / `ETCD_ENDPOINTS` env vars.

## Writing your own Rust backend

1. New crate depending on `dynamo-backend-common`; place under `lib/`.
2. Implement
   [`LLMEngine`](../../../lib/backend-common/src/engine.rs)
   plus an inherent
   `from_args(argv) -> Result<(Self, WorkerConfig), DynamoError>`.
3. Mirror the mocker example's three-line `main.rs`.
4. Run the conformance kit in your tests:

   ```toml
   [dev-dependencies]
   dynamo-backend-common = { workspace = true, features = ["testing"] }
   ```

   ```rust
   #[tokio::test]
   async fn my_engine_satisfies_contract() {
       let engine = MyEngine::new_for_test();
       dynamo_backend_common::testing::run_conformance(engine)
           .await
           .expect("conformance");
   }
   ```

## Layout

```text
lib/backend-common/examples/mocker/
├── Cargo.toml
├── Dockerfile              # builds the mocker backend binary
├── Dockerfile.frontend     # builds the Dynamo frontend from source
├── docker-compose.yml      # one-command infra + frontend + backend
└── src/
    ├── main.rs             # 3-line entry point
    └── engine.rs           # MockerBackend wrapping dynamo-mocker scheduler
```

## References

- Crate: [`lib/backend-common/`](../../../lib/backend-common/)
- Example source: [`lib/backend-common/examples/mocker/`](../../../lib/backend-common/examples/mocker/)
- Conformance kit: [`lib/backend-common/src/testing.rs`](../../../lib/backend-common/src/testing.rs)
