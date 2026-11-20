<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

This copy is temporary while Dynamo waits for SGLang to publish
`sglang-api-types` 0.1.1 on crates.io (sgl-project/sglang#50028). Once it is
published, Dynamo should remove this directory and the build script and use
the crate instead.

The language service is based on SGLang `37f3b2730bca1cc875126f8b94c4487ec1331a95`.
The staged-encoder extension adds the encoder service, GenerateRequest field 41,
and GetServerInfoResponse field 17. Unrelated newer SGLang service methods are
omitted from this snapshot. The SHA-256 digests of the vendored files follow.

| File | SHA-256 |
|------|---------|
| `sglang/api/v1/common.proto` | `6c2dd838206329d897567d60e9fbdfe9e66f9baba438e8aa34d3b3e3b9be40cf` |
| `sglang/api/v1/finish.proto` | `a8069d9b7be196224a0e7011e4d69e514e50b0833a038338dbfc82fc296fa40c` |
| `sglang/api/v1/generate.proto` | `55eecf96a77fe3ce4cd4bc2c0d7e812dfca30832c72584dfece18d14a507d5a5` |
| `sglang/api/v1/hicache.proto` | `3224ff3d64234d23678d1a6f899c35b131ebc73eddeabf4a6b25fa983852e736` |
| `sglang/api/v1/sampling.proto` | `64621faa7bc31835a52973db9ce8e961c14e66c7d95b21282399d5049531fa6f` |
| `sglang/api/v1/service.proto` | `570a287ac1c47c9963f43dd97cdd8fbe512453af3e469f48cfac59c7d3db5b1f` |
| `sglang/encoder/v1/encoder.proto` | `76d5060508a03ddbc59161c5a027b8432d247dc26804bb88978e19501d2f77ee` |
| `sglang/json/v1/options.proto` | `d1aa73d1a5ee39b180ccc5967dae55d061a4e6e10f9a01d4c116d3a269b111fe` |

All of them are excluded from Dynamo's formatting and copyright-header checks.
The SGLang sidecar generates both client and server types and temporarily
exposes them to the Mocker server.
