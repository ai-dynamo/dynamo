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

The contract was copied from SGLang main, commit
[`3acfa2efcc51a0c473930e60614e7f7c39863fad`](https://github.com/sgl-project/sglang/blob/3acfa2efcc51a0c473930e60614e7f7c39863fad/proto/sglang/runtime/v1/sglang.proto).
The upstream file's SHA-256 is
`3e7a78c9d2327530ea22369b2c6159886f720a9887d41a4f2983aebe7f257d2f`.
The local file adds SPDX and temporary-copy comments and applies Dynamo's
`clang-format` style; these changes do not alter the protobuf descriptor. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.

That commit merged
[sgl-project/sglang#42833](https://github.com/sgl-project/sglang/pull/42833),
which adds the `GenerateRequest` media fields 19 to 21 (`image_data`,
`video_data`, and `audio_data`); no SGLang release has them yet. The sidecar
sends each media kind only to servers that list it in the `GetServerInfo`
`grpc_features` list (`image_inputs`, `video_inputs`, or `audio_inputs`),
because older servers drop unknown fields and a model processor drops each
media kind it has no placeholder for, both silently.
[sgl-project/sglang#46882](https://github.com/sgl-project/sglang/pull/46882)
adds that list and is not merged yet. Until a server has it, the sidecar
refuses all media.
