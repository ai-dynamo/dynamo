<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared sidecar tests with Mocker

The vLLM and SGLang `conformance` integration tests compile the same four scenarios
from `common/scenarios.rs`. Each fixture starts its existing CPU-only Mocker gRPC
service on an ephemeral localhost port and connects the real Rust sidecar to it.
No engine installation, model, GPU, or external discovery service is required.

```bash
cargo test --locked -p dynamo-vllm-mocker -p dynamo-sglang-mocker --test conformance
```

| Scenario | Checks |
|---|---|
| Streaming (R09) | Sidecar tokens match native Mocker output; one final length completion, correct usage, and ignored data after completion. |
| Failure (R11) | Open failure, early EOF, and read failure preserve delivered tokens and return the expected error. |
| Cancellation (R12) | Cancellation before submission, while opening, and while waiting for another token. |
| Cleanup (R13) | Generation before startup fails; repeated cleanup succeeds; cleanup cancels an active stream. |

`common/mod.rs` contains fixture interfaces, observations, and test-only fault
controls. Each framework's `tests/conformance.rs` delegates to its existing Mocker
service and interprets native token fields. Requests and assertions live in the
shared scenarios; no production sidecar or Mocker hooks are added.

Failures and pending operations are injected around Mocker responses. Explicit
barriers ensure a token reaches the sidecar before injecting a read failure or
EOF. These socket tests use bounded real-time waits, without sleep-based ordering
or paused Tokio time. Cancellation checks observe the remote RPC being dropped;
the existing `sidecar` integration tests retain scheduler-cancellation and
prefill/decode handoff coverage.

TensorRT-LLM is outside this suite because it has no corresponding Mocker server.
The suite does not exercise real model inference or KV-cache data transfer.
