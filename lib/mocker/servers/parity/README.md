<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker engine parity

Each mocker server declares the engine release it matches in `ENGINE_VERSION`
(`<engine>/src/server.rs`). The `parity_with_live_engine` test in
`<engine>/tests/sidecar.rs` sends the same requests through a sidecar connected
to the real engine and through one connected to the mocker, then compares the
results:

| Scenario | Compared |
|---|---|
| Version | `ENGINE_VERSION` against the installed engine |
| Registration | Every field the sidecar registers, except values that depend on GPU memory |
| Length, stop tokens, logprobs | Token counts, finish and stop reasons, usage, and logprob layout |
| Abort, overlong prompt, recovery | Error type, and that the next request succeeds |
| KV events (vLLM, SGLang) | A stored prompt reaches a router index |
| Prefill handoff (vLLM, TensorRT-LLM) | Handoff fields and their types |

Token values, logprob values and timing are not compared.

## When it runs

PR CI runs `.github/workflows/shared-mocker-parity.yml` for an engine when its
CUDA runtime image in `container/context.yaml` changes, when its mocker server
under `lib/mocker/servers/<engine>/` changes, or when the shared parity code
changes. `plan.py` makes that selection.

## After an engine bump

The test fails until `ENGINE_VERSION` names the new release. Update it, then fix
any other difference the test reports in the mocker, so the mocker matches the
new engine.

## Run it locally

Build the test binary, then run it inside the engine's image on a GPU:

```bash
cargo test -p dynamo-vllm-mocker --test sidecar --no-run
docker run --rm --gpus all --network host -v "$PWD:/ws" -w /ws vllm/vllm-openai:v0.33.0 \
  python3 lib/mocker/servers/parity/run.py --engine vllm --binary <test binary>
```

For TensorRT-LLM, first install the OpenEngine bindings with
`trtllm_ensure_openengine_bindings` from `lib/sidecar/trtllm/launch/common.sh`.
