<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Inference Gateway (GAIE)

Integrate Dynamo with the Gateway API Inference Extension for intelligent KV-aware request routing at the gateway layer.

See [Gateway API Inference Extension documentation](../../docs/kubernetes/gateway-api/README.mdx) for setup instructions, configuration options, and deployment examples.

## Dynamo EPP Image

Dynamo ships a native Rust Endpoint Picker Plugin (EPP) image (`dynamo-epp`) built from
[`deploy/inference-gateway/ext-proc/`](ext-proc/DEVEL.md).

> [!IMPORTANT]
> **Deprecation:** Dynamo no longer provides the Go-based EPP plugin for its router. The former
> `deploy/inference-gateway/epp/` Go plugin tree has been removed. Use the Rust EPP image instead.

The Rust EPP implements the Gateway API Inference Extension
[Lightweight Endpoint Picker (LW-EPP)](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/lwepp/README.md)
`ext_proc` interface. It can be called from an InferenceGateway / GAIE deployment and runs the full
Dynamo KV-aware router natively in Rust (no CGO bridge).

The upstream GAIE project also publishes a minimal LW-EPP reference implementation for conformance
testing. Dynamo's EPP follows that interface contract while adding full Dynamo routing:

- [LW-EPP README](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/lwepp/README.md)
- [LW-EPP source](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/lwepp)

Build and development workflows are documented in [`ext-proc/DEVEL.md`](ext-proc/DEVEL.md).
