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

The contract was copied from SGLang v0.5.21, commit
[`e00930c5489053f26d86b179cee0d087f846acbb`](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/proto/sglang/runtime/v1/sglang.proto).
The upstream file's SHA-256 is
`14538b3c0a7114f9cc91a59e2166db6e08877ee4802bcfa40d28bbde7d1293a5`.
The local file adds SPDX and temporary-copy comments and applies Dynamo's
`clang-format` style; these changes do not alter the protobuf descriptor. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.

The local file also adds `GenerateRequest` fields 18 to 21 (`image_data`,
`video_data`, `audio_data`, and `mm_hashes`) from SGLang pull requests that are
not merged yet:
[sgl-project/sglang#42833](https://github.com/sgl-project/sglang/pull/42833)
adds the three media fields, and a follow-up change adds `mm_hashes` and the
`multimodal_inputs` entry in the `GetServerInfo` `grpc_features` list. Until
these changes merge, the local descriptor does not match the commit above. When
they merge, copy the file again from the SGLang merge commit and update the
commit and the SHA-256. The sidecar sends media only to servers that list
`multimodal_inputs`, because older servers drop unknown fields and text-only
models ignore media, both silently.
