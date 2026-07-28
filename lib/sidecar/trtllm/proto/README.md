<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored TensorRT-LLM gRPC protocol

- Source package: `smg-grpc-proto`
- Version: `0.4.14`
- File: `smg_grpc_proto/proto/trtllm_service.proto`
- Upstream SHA-256: `ec2f93de3af59047e3779283369eba89c2e6981b27849553144671b1fbc8eea5`

`trtllm_service.proto` here is derived from the upstream file above with a
single, mechanical edit: the `import "common.proto"` line and the two RPCs that
reference `smg.grpc.common` types — `GetTokenizer` and `SubscribeKvEvents` — are
removed. The sidecar drives only `Generate`, so dropping the tokenizer/KV-event
RPCs lets the contract compile standalone without vendoring `common.proto`.
Every message the sidecar exchanges is preserved byte-for-byte.

TensorRT-LLM loads the matching generated stubs from `smg-grpc-proto` at
runtime (`tensorrt_llm/grpc/__init__.py`), first shipped in TRT-LLM
`1.3.0rc21`. Update the version, checksum, and this note together when bumping
the protocol.
