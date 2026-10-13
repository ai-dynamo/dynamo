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

The contract was copied from the SGLang v0.5.22 release branch, commit
[`baa9befa381511cd4d56f83980951e5a45fffc6d`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/proto/sglang/runtime/v1/sglang.proto).
The upstream file's SHA-256 is
`7852d2dbff67c35889d4592516c9c17a9681b6559a514959c36b6acac157c2ff`.
The local file adds SPDX and temporary-copy comments and applies Dynamo's
`clang-format` style; these changes do not alter the protobuf descriptor. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.

The local file also adds `GenerateRequest` fields 18 to 21 (`image_data`,
`video_data`, `audio_data`, and `mm_hashes`) from SGLang pull requests that are
not merged yet:
[sgl-project/sglang#42833](https://github.com/sgl-project/sglang/pull/42833)
adds the three media fields, and
[sgl-project/sglang#43886](https://github.com/sgl-project/sglang/pull/43886)
adds `mm_hashes` and the `GetServerInfo` `grpc_features` list. Until
these changes merge, the local descriptor does not match the commit above. When
they merge, copy the file again from the SGLang merge commit and update the
commit and the SHA-256. The sidecar sends each media kind only to servers that
list it in `grpc_features` (`image_inputs`, `video_inputs`, or `audio_inputs`),
because older servers drop unknown fields and a model processor drops each
media kind it has no placeholder for, both silently.
