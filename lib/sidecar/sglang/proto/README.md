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

The contract was copied from the SGLang v0.5.23 release branch, commit
[`297e54fd105e974cdf401e5b4d20253a22c16f90`](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/proto/sglang/runtime/v1/sglang.proto).
The upstream file's SHA-256 is
`3e7a78c9d2327530ea22369b2c6159886f720a9887d41a4f2983aebe7f257d2f`.
The local file adds SPDX and temporary-copy comments and applies Dynamo's
`clang-format` style; these changes do not alter the protobuf descriptor. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.

SGLang 0.5.23 adds the `GenerateRequest` media fields 19 to 21 (`image_data`,
`video_data`, and `audio_data`) and the `GetServerInfo` `grpc_features` list.
The sidecar sends each media kind only to a server that lists it in
`grpc_features` (`image_inputs`, `video_inputs`, or `audio_inputs`). An older
server drops unknown fields, and a model processor drops each media kind that
it has no placeholder for. Both drops are silent, so the sidecar refuses each
media kind that the server does not list.
