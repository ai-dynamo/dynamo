<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LPX sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Flags and behavior may change without notice.

`dynamo-lpx-sidecar` connects a Dynamo worker to the OpenEngine gRPC server of
an LPX engine (hybrid: aggregated or disaggregated decode; LPU-only:
aggregated). The executable ships only in the
[sidecar image](../README.md#docker), as `dynamo-sidecar lpx`; it is not part
of the Python package.
[LPX Engines](../../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/sidecar-backends.md#lpx-engines)
describes how to deploy and operate it.

## Run

```bash
# Hybrid (with --request-plane openengine) or LPU-only engine, aggregated
dynamo-sidecar lpx --grpc-endpoint unix:///run/openengine/openengine.sock \
  --model-path meta-llama/Llama-3.2-1B-Instruct

# Hybrid engine (also with --use-disagg-prefill), decode behind a vLLM prefill worker
dynamo-sidecar lpx --grpc-endpoint unix:///run/openengine/openengine.sock \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --disaggregation-mode decode
```

`--grpc-endpoint` takes `host:port`, an `http://` or `grpc://` URL, or
`unix:///ABSOLUTE/PATH`.
[OpenEngine on a Unix Socket](../../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/sidecar-backends.md#openengine-on-a-unix-socket)
covers mounting the socket and its permissions.

## Metrics

The sidecar polls `Control.GetLoad` on the serving rank to publish router load
and KV cache gauges, and planner forward-pass metrics. An engine without
`GetLoad` publishes neither.

## Errors

| Failure | Frontend sees | Migrated with `--migration-limit` > 0 |
|---|---|---|
| Request the sidecar or the engine rejects | 4xx | No |
| Engine unavailable before the response stream: starting, paused, draining, or its queue full | Retry on another worker | Yes |
| Engine ends a stream with UNAVAILABLE: it restarts, drains or shuts down | Retry on another worker | Yes; on a decode leg, prefill re-runs |
| Decode KV transfer failure | 5xx with the engine's "KV transfer failed" message | No |

A migratable error also takes the failing worker out of routing for a few
seconds (`DYN_RUNTIME_INHIBITED_DURATION_SECS`), so a migration needs a second
worker of the same kind.

## Limitations

- No LoRA, KV events, or data-parallel ranks beyond one. The sidecar rejects
  logprobs, guided decoding, multimodal input, prompt embeddings, more than one
  sequence, visible stop token IDs, `include_stop_str_in_output`,
  `max_thinking_tokens`, cache namespaces, priorities, any `allowed_token_ids`
  or `logprob_token_ids`, and nonempty `bad_words_token_ids` with a 4xx.
- `no_stop_trim=true` is rejected with a 4xx when client stop token IDs are
  nonempty or `skip_special_tokens=false`. Stop strings remain supported.
