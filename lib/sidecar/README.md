<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (installation and one example).
Do not edit it unless the user explicitly asks you to.
-->

# Sidecars

Rust sidecars connect Dynamo to vLLM, SGLang, and TensorRT-LLM engines over
their native gRPC APIs. The engine runs in its own process; the sidecar
registers it with Dynamo and serves its requests.

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

Engine-specific guides: [vLLM](vllm/README.md), [SGLang](sglang/README.md),
[TensorRT-LLM](trtllm/README.md).

## Installation

### pip

The sidecars ship in the `ai-dynamo` wheel:

```bash
pip install ai-dynamo
python -m dynamo.vllm.sidecar --help    # also dynamo.sglang.sidecar, dynamo.trtllm.sidecar
```

### Docker

The CPU-only `dynamo-sidecar` image contains all three sidecars and is
published to NGC:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version>
docker run --rm nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version> vllm --help
```

To build it from the repository root instead:

```bash
docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.6.0-dev .
```

## Operational limits

Dynamo runtime metrics, logs, and request spans are implemented. Scrape native engine metrics separately; the sidecars do not provide full engine-internal tracing or parity with existing dashboards.

A supported service-level agreement (SLA) autoscaling workflow is not available for these sidecars. Kubernetes worker discovery is implemented, but built-in per-iteration Forward Pass Metrics (FPM) emission/relay is missing, so default load-based scaling has no input and cannot make scaling decisions. A separate predictive-scaling path does not establish an integrated sidecar workflow. See [Planner Limitations](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/overview.md#current-limitations).

The Kubernetes operator rejects sidecar mode combined with GPU Memory Service (GMS), enabled checkpointing, or shadow-engine failover. See the [sidecar deployment contract](../../docs/fern/pages/reference/kubernetes-api/dynamo-component-deployment.mdx#dynamo-sidecar-mode).

Enable request migration with `--migration-limit > 0`; it is off by default. Eligible requests can retry connection failures, timeouts, and supported interrupted streams, continuing from delivered tokens. Structured-output requests and explicit worker pins are excluded, and not all engine errors are migratable. See each engine's support notes for error-specific restrictions. Request migration is distinct from shadow-engine failover.

The shared worker unregisters, waits, and cleans up during shutdown. The sidecars do not report engine quiescence, so prefill draining uses a bounded wait rather than proving that engine KV transfers have completed.

Configure cache offloading in the native engine, with any required plugin installed in the engine container. Sidecar separation does not prevent engine-local caching. Offload-aware routing, storage tiers, and combinations with prefill/decode transfer require separate qualification; a prefix-cache hit or KV event alone does not prove them. Implementation support does not certify every model and feature combination.
