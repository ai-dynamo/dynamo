<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (what it is, engine setup, flags,
one launch example, limits). Do not edit it unless the user explicitly asks
you to.
-->

# TileRT sidecar

> [!WARNING]
> **Experimental.** This sidecar is a first integration of [TileRT](https://github.com/tile-ai/TileRT). It is not part of the `dynamo-sidecar` image or the `ai-dynamo` wheel, and its flags and behavior may change without notice.

`dynamo-tilert-sidecar` serves one vLLM prefill and one TileRT decode server as a single Dynamo worker. For each request it runs a one-token generation on vLLM over vLLM's native gRPC API, with the decode server's address attached so that TileRT's KV connector pushes the prompt KV there, and then streams the rest of the reply from the decode server's `/pd/decode`.

The two engines sit behind one worker because TileRT's prefill has to be told which decode server to push to before it starts. Dynamo's prefill/decode routing picks the prefill first, so it cannot make that choice today.

## Engines

Works with TileRT 0.1.8 only (`tilert.pd_vllm`). TileRT 0.2 changed the decode server's API, and the sidecar fails at startup against it.

| Engine | Requirements |
|--------|--------------|
| vLLM prefill | Native gRPC server enabled (`VLLM_USE_RUST_FRONTEND=1`, `--grpc-port`), and TileRT's KV connector as the producer: `--kv-transfer-config '{"kv_connector":"TileRTConnector","kv_connector_module_path":"tilert.pd_vllm.prefill_connector","kv_role":"kv_producer",...}'` |
| TileRT decode | `python -m tilert.pd_vllm.decode_server` with its HTTP port (`--http-port`), control port (`--ctrl-port`) and `--max-seq-len` |

Pair one sidecar with one decode server. The decode server holds one request's KV at a time and nothing else may send it requests.

## Build and run

```bash
cargo build --release -p dynamo-tilert-sidecar

export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
target/release/dynamo-tilert-sidecar \
  --grpc-endpoint http://prefill-node:50061 \
  --decode-http-endpoint http://decode-node:5557 \
  --decode-ctrl-port 5556 \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --context-length 1048576 \
  --probe-token-ids 1,2,3
```

Start `python -m dynamo.frontend` as for any other worker.

| Flag | Environment variable | Meaning |
|------|----------------------|---------|
| `--grpc-endpoint` | `DYN_SIDECAR_GRPC_ENDPOINT` | vLLM prefill's gRPC address |
| `--decode-http-endpoint` | `TILERT_DECODE_HTTP_ENDPOINT` | Decode server's HTTP base URL |
| `--decode-kv-host` | `TILERT_DECODE_KV_HOST` | Host the prefill pushes KV to. Defaults to the host in `--decode-http-endpoint` |
| `--decode-ctrl-port` | `TILERT_DECODE_CTRL_PORT` | Decode server's control port. Default `5556` |
| `--model-path` | | Hugging Face model ID or local path, for tokenization and the chat template |
| `--served-model-name` | `DYN_SERVED_MODEL_NAME` | Public model name. Defaults to `--model-path` |
| `--context-length` | `TILERT_CONTEXT_LENGTH` | Maximum sequence length; match the decode server's `--max-seq-len` |
| `--kv-transfer-timeout-secs` | `TILERT_KV_TRANSFER_TIMEOUT_SECS` | How long the decode server waits for a request's KV. Default `10` |
| `--probe-token-ids` | | Prompt token ids for one short request run through both engines before the worker registers. A broken KV handoff then fails startup. Empty skips the check |

The shared sidecar flags (`--namespace`, `--dyn-reasoning-parser`, `--dyn-tool-call-parser` and the rest) are listed by `--help`.

The worker withdraws from Dynamo and the sidecar exits when vLLM stops serving or the decode server fails three health checks in a row, so that a supervisor can restart it.

## Limits

- One request at a time for each worker, and one prefill paired with one decode server. There is no routing across several of either.
- Aggregated mode only, with no KV-aware routing.
- One completion per request, without logprobs.
- The decode stage samples by temperature only. Requests that set `top_p` below 1, `top_k`, `min_p`, `frequency_penalty`, `logit_bias`, `min_tokens`, a thinking token budget or structured output are refused. Repetition and presence penalties and `ignore_eos` are accepted when the decode server's `/capabilities` reports them.
- A request that omits `temperature` samples at 1.0; the model's generation defaults are not read.
- Text prompts only.
- CI runs the sidecar against in-process fakes of both engines. Runs against real engines are manual.
