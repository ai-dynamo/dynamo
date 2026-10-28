<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Frontend

An HTTP entry point that forwards requests to explicitly configured Dynamo pool
frontends. It reuses Dynamo's HTTP service and performs no tokenization; the pool
frontend owns preprocessing and generation. Shared pool identity and description
types live in `dynamo-global-common`.

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --pool site-a/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool site-b/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002
```

Each `--pool` uses `<site>/<namespace>/<DGD name>=<model>@<frontend URL>`.
Repeat the flag for different pools. Comma-separated URLs after `@` configure
multiple frontends for the same pool. Pool identities are derived from the
site, namespace, and DGD name; duplicate identities are rejected.

Requests rotate across pools serving the requested model, then across that
pool's frontends. `/v1/models` lists the configured models. Chat completions and
completions support streaming and non-streaming requests; the existing Responses
API translation goes through the chat forwarding engine. Non-streaming requests
ask the pool for a stream including usage, which the HTTP service aggregates.
The `x-dynamo-session-id` header is forwarded to the selected pool.

This first implementation uses static configuration. The catalog interface also
accepts snapshots for embedding and tests, but there is no discovery client,
polling, or registration. There is no automatic failover or retry: unreachable
pools return 503. Upstream HTTP errors use the existing frontend error policy:
404 remains 404, while an upstream 502 is classified as internal and returns
500; diagnostic bodies are sanitized. The existing SSE
codec does not distinguish a clean upstream EOF from `[DONE]`; premature clean
EOF handling remains a limitation of this draft. Request-ID/trace propagation
and authentication header forwarding are not implemented.

Run CPU-only tests with:

```bash
cargo test -p dynamo-global-common -p dynamo-global-frontend --no-default-features
```

The common `PoolId` identifies a DGD using its deployment coordinates. It is
distinct from `dynamo_kv_router::identity::PoolId`, which identifies an indexer
domain/DC lane; Global View integration will need an explicit mapping.
