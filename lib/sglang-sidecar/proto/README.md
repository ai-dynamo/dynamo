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
[`cc7d6659fd68694797892d0d863b2549a5b61b69`](https://github.com/sgl-project/sglang/blob/cc7d6659fd68694797892d0d863b2549a5b61b69/python/sglang/srt/grpc/sglang.proto).
The upstream file's SHA-256 is
`a2e14952ddb2b34b6e22cbbc4e76d76d70c44f2dbf087cb9918aed3399d9ef42`.
The local file adds only the SPDX and temporary-copy comments above the
upstream content.
