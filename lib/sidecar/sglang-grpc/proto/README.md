<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

This copy is temporary while Dynamo waits for SGLang to include
`sglang/srt/grpc/sglang.proto` in a release wheel. Once the contract is
available there, Dynamo should remove this directory, pin and install the
matching `sglang` wheel as a build dependency, and compile the packaged proto
instead.

The contract was copied from SGLang commit
[`cc7d6659fd68694797892d0d863b2549a5b61b69`](https://github.com/sgl-project/sglang/blob/cc7d6659fd68694797892d0d863b2549a5b61b69/proto/sglang/runtime/v1/sglang.proto).
The upstream file's SHA-256 is
`a2e14952ddb2b34b6e22cbbc4e76d76d70c44f2dbf087cb9918aed3399d9ef42`.
The local file adds SPDX and temporary-copy comments and applies Dynamo's
`clang-format` style; these changes do not alter the protobuf descriptor. The
shared `dynamo-sglang-grpc` crate generates both the sidecar client and the
Mocker server from this one copy. The formatted local source is pinned by the
test checksum `f3d5bf6c18dd95248c311f1368a77631862d9c9f0febe748d19964b7e1154f07`.
