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
`clang-format` style. The engine-state work from
[Dynamo #14985](https://github.com/ai-dynamo/dynamo/pull/14985) additionally
introduces `WatchEngineState` and its snapshot messages, following
[SGLang #39915](https://github.com/sgl-project/sglang/pull/39915). These are
additive protocol changes; the hash above identifies the original base, not
the extended contract. Existing generation wire tags remain unchanged. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.
