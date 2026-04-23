---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Rotating Tokens (Rust Backend Example)
---

# Rotating Tokens — Rust Backend Example (`rotating_tokens`)

Reference implementation of Dynamo's Rust backend interface. Generates
rotating token IDs on a timer — not a production engine, but every
lifecycle path a real Rust backend needs is wired up end-to-end. Use
it as a template for writing your own Rust backend, or as a stand-in
engine for testing the runtime.

Rust backends implement the `LLMEngine` trait from
[`dynamo-backend-common`](../../../lib/backend-common/) — see the trait's
doc comments in [`engine.rs`](../../../lib/backend-common/src/engine.rs)
for the authoritative contract.

## Quick demo (docker compose)

One command brings up NATS, etcd, the Dynamo frontend, and this
backend — all built from source in this repo:

```bash
cd lib/backend-common/examples/rotating_tokens
docker compose up --build
```

First `up` is slow — it builds two Rust images and downloads the
Qwen3 tokenizer from HuggingFace. Subsequent runs reuse Docker's
layer cache and a named volume for the HF cache.

In another terminal, send a chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "sample-model",
          "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 32
        }'
```

Expect a response whose tokens are rotating IDs 1..32 detokenized
through Qwen's vocabulary — not meaningful text, but it proves every
stage of the pipe is connected end-to-end.

Tear down with `docker compose down` (add `-v` to drop the HF cache
volume).

Set `HF_TOKEN` in your shell if you hit HuggingFace rate limits.

## Build and run locally

```bash
cargo build -p dynamo-rotating-tokens --release

# For chat/completions endpoints the frontend needs a tokenizer + chat
# template, so point --model-path at an open HF repo. For tensor/prefill
# endpoints (no tokenization), omit --model-path for name-only mode.
./target/release/dynamo-rotating-tokens \
    --model-path Qwen/Qwen3-0.6B
```

Requires the infra services (NATS, etcd) you normally run with Dynamo
reachable via `NATS_SERVER` / `ETCD_ENDPOINTS` env vars.

## Writing your own Rust backend

1. New crate depending on `dynamo-backend-common`; place under `lib/`.
2. Implement
   [`LLMEngine`](../../../lib/backend-common/src/engine.rs)
   plus an inherent
   `from_args(argv) -> Result<(Self, WorkerConfig), DynamoError>`.
3. Mirror `rotating_tokens`'s three-line `main.rs`.
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

```
lib/backend-common/examples/rotating_tokens/
├── Cargo.toml
├── Dockerfile              # builds the rotating_tokens backend binary
├── Dockerfile.frontend     # builds the Dynamo frontend from source
├── docker-compose.yml      # one-command infra + frontend + backend
├── nats-server.conf
└── src/
    ├── main.rs             # 3-line entry point
    └── engine.rs           # Args, LLMEngine impl, tests
```

## References

- Crate: [`lib/backend-common/`](../../../lib/backend-common/)
- Example source: [`lib/backend-common/examples/rotating_tokens/`](../../../lib/backend-common/examples/rotating_tokens/)
- Conformance kit: [`lib/backend-common/src/testing.rs`](../../../lib/backend-common/src/testing.rs)
