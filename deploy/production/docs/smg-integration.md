<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SMG, Dynamo, and HiSparse Integration Boundary

This document records what the production SMG path actually wires today. The SMG integration is HTTP-only. It does not add a gRPC integration path and it does not move Dynamo or SGLang request-processing features into SMG.

## Request Chain

```text
client -> SMG router -> Dynamo Frontend -> SGLang prefill/decode -> HiSparse / IndexCache / TurboQuant
```

The Dynamo Frontend remains the KV-aware routing owner. It consumes SGLang ZMQ KV events from the prefill and decode workers and decides which worker owns each prefix.

## Responsibility Split

| Layer | Owns | Does not own |
|---|---|---|
| SMG (`addons/smg/`) | OpenAI-compatible HTTP gateway, static backend forwarding to Dynamo Frontend, retry, circuit breaker, backend health checks, request timeout settings, Prometheus metrics, OpenTelemetry export, structured logs, Grafana dashboard export | Tokenization, detokenization, reasoning parsing, tool-call parsing, multimodal processing, MCP tool orchestration, chat history, prefix routing, KV state, GPU orchestration |
| Dynamo Frontend (`examples/deepseek-v32-reap-sglang.yaml` Frontend service) | KV-aware routing with `--router-mode kv --router-kv-events`, request preprocessing for the Dynamo HTTP path, disaggregation orchestration, per-request scheduling | Client TLS termination, external auth, MCP tool execution, durable chat history |
| SGLang prefill/decode workers | Forward pass, KV cache management, SMC-SD, HiSparse top-k selection on decode, IndexCache, TurboQuant, model-specific parser configuration exported to Dynamo | Routing decisions across workers, gateway policy, client-facing HTTP retries |
| HiSparse | Decode-side sparse attention kernel behavior | Anything above the kernel boundary |

## Parser And Tokenizer Boundary

SMG does not perform tokenization, detokenization, reasoning parsing, or tool-call parsing in this profile. Parser configuration stays on the Dynamo/SGLang path:

```yaml
- --dyn-tool-call-parser
- deepseek_v3_2
- --dyn-reasoning-parser
- deepseek_r1
```

Those worker flags let Dynamo's frontend-side preprocessing path use the parser metadata while preserving KV routing. Do not document parser movement to SMG unless the SMG chart and deployment are changed to prove that behavior end to end.

## No Multimodal Or MCP Claim

This profile does not configure SMG multimodal handling or MCP tool execution. If multimodal support is needed, it must be configured and verified in Dynamo/SGLang. If MCP tool orchestration is needed, it must be added as a separate, tested application-layer integration instead of being implied by SMG.

## No Chat History Claim

This profile does not deploy an SMG Postgres database and does not configure SMG chat history or audit-log persistence. Durable conversation storage must be provided by a separate application service if needed.

## Why SMG Uses `round_robin`

SMG's `cache_aware` policy and Dynamo Frontend's KV router both maintain prefix state. Dynamo sees the actual SGLang KV-cache state through ZMQ KV events, so this profile keeps SMG as a simple gateway and leaves prefix routing to Dynamo.

When the cluster grows to more than one Dynamo Frontend, SMG may use a non-cache-aware policy across Dynamo Frontends. Each Dynamo Frontend should still own prefix routing within its own fleet.

## HiSparse Boundary

`--enable-hisparse` requires `--disable-radix-cache`, but Dynamo's KV-aware router does not depend on SGLang's radix tree. It consumes KV events from SGLang, so Dynamo routing remains correct over the HiSparse-enabled decode fleet.

SMG sits above that path and never sees HiSparse, IndexCache, TurboQuant, or SMC-SD state.

## What Did Not Change

- 4-GPU prefill plus 4-GPU decode shape.
- Dynamo Frontend args: `--router-mode kv --router-kv-events --router-reset-states`.
- HiSparse, IndexCache, TurboQuant, and SMC-SD worker settings.
- The SGLang runtime image.
- The single-node B200 A4 placement constraints.

## Verifying The HTTP Chain

After Argo CD has synced the SMG app and the Dynamo platform plus DynamoGraphDeployment are healthy:

```bash
./tests/smg-roundtrip.sh
```

The script port-forwards SMG's ClusterIP service, sends an OpenAI-compatible chat request, and asserts a 200 with a `chat.completion` response. It also checks SMG's `/health` and the Dynamo Frontend's `/health` separately.

## Resource Footprint

| Pod | CPU | RAM | GPUs | Storage |
|---|---|---|---|---|
| SMG router | 1-4 | 2-4 GiB | 0 | None |
| Dynamo Frontend | Operator default | Operator default | 0 | None |
| SGLang prefill | Operator default | 120 GiB shm | 4 B200 | None |
| SGLang decode | Operator default | 120 GiB shm | 4 B200 | None |

The 8 B200 GPUs remain split 4+4 prefill/decode. SMG is CPU-only and does not contend with the SGLang workers for GPUs.
