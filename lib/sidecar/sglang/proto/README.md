<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

Dynamo main uses the published language-service bindings. This branch stages
the new fields until `sglang-api-types` 0.2.0 is published. The standalone
encoder uses the separate `sglang.grpc.encoder` service. [SGLang schema](https://github.com/sgl-project/sglang/pull/51754) · [Dynamo crate migration](https://github.com/ai-dynamo/dynamo/pull/19156)

The vendored language service is based on SGLang `37f3b2730bca1cc875126f8b94c4487ec1331a95`.
Its staged extensions and the standalone encoder schema come from SGLang commit
[`b2accdf7856ba359b204f1cd5135ae0ac8efdd14`](https://github.com/sgl-project/sglang/commit/b2accdf7856ba359b204f1cd5135ae0ac8efdd14).

- [GenerateRequest](https://github.com/sgl-project/sglang/blob/b2accdf7856ba359b204f1cd5135ae0ac8efdd14/proto/sglang/api/v1/generate.proto) adds the signed handoff in field 41.
- [GetServerInfoResponse](https://github.com/sgl-project/sglang/blob/b2accdf7856ba359b204f1cd5135ae0ac8efdd14/proto/sglang/api/v1/service.proto) advertises the staged layout, key and transfer backend in fields 17–24.
- [The standalone encoder schema](https://github.com/sgl-project/sglang/blob/b2accdf7856ba359b204f1cd5135ae0ac8efdd14/proto/sglang/encoder/v1/encoder.proto) is copied verbatim, including the Stage, Deliver and Release RPCs and the staged layout and transport discovery fields. DescribeResponse field 4 reports E's send timeout for NIXL write recovery.

The table lists SHA-256 digests for the vendored files.

| File | SHA-256 |
|------|---------|
| `sglang/api/v1/common.proto` | `6c2dd838206329d897567d60e9fbdfe9e66f9baba438e8aa34d3b3e3b9be40cf` |
| `sglang/api/v1/finish.proto` | `a8069d9b7be196224a0e7011e4d69e514e50b0833a038338dbfc82fc296fa40c` |
| `sglang/api/v1/generate.proto` | `ffc9ee7f071e883d3226c30122ff7afaa867847646bf424b3b0058343c4d93d0` |
| `sglang/api/v1/hicache.proto` | `3224ff3d64234d23678d1a6f899c35b131ebc73eddeabf4a6b25fa983852e736` |
| `sglang/api/v1/sampling.proto` | `64621faa7bc31835a52973db9ce8e961c14e66c7d95b21282399d5049531fa6f` |
| `sglang/api/v1/service.proto` | `2229836152336b143eca9fc281b350a82cc52a9f924d93f8538d412c0b3e8f0b` |
| `sglang/encoder/v1/encoder.proto` | `201f4d5c6d7e4895c6f70e6b9190ade65ca2ed1c37ce136b160ef01bb523620c` |
| `sglang/json/v1/options.proto` | `d1aa73d1a5ee39b180ccc5967dae55d061a4e6e10f9a01d4c116d3a269b111fe` |

All of them are excluded from Dynamo's formatting and copyright-header checks.
The SGLang sidecar generates both client and server types and temporarily
exposes them to the Mocker server.
