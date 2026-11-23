<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

This copy is temporary while Dynamo waits for SGLang to publish
`sglang-api-types` 0.1.1 on crates.io (sgl-project/sglang#50028). Once it is
published, Dynamo should remove this directory and the build script and use
the crate instead.

The files are unchanged copies of SGLang upstream `main`
`37f3b2730bca1cc875126f8b94c4487ec1331a95`. Their SHA-256s are:

| File | SHA-256 |
|------|---------|
| `sglang/api/v1/common.proto` | `6c2dd838206329d897567d60e9fbdfe9e66f9baba438e8aa34d3b3e3b9be40cf` |
| `sglang/api/v1/finish.proto` | `a8069d9b7be196224a0e7011e4d69e514e50b0833a038338dbfc82fc296fa40c` |
| `sglang/api/v1/generate.proto` | `f18cb71127ad9760efae059796f382e5c4d015a148bb42cc1ca7fa00eeb8f4cc` |
| `sglang/api/v1/hicache.proto` | `3224ff3d64234d23678d1a6f899c35b131ebc73eddeabf4a6b25fa983852e736` |
| `sglang/api/v1/sampling.proto` | `64621faa7bc31835a52973db9ce8e961c14e66c7d95b21282399d5049531fa6f` |
| `sglang/api/v1/service.proto` | `c2794fb9d4fdb8dc489a75c5f22066d2b5b455c77e3606137785c2278521ac11` |
| `sglang/json/v1/options.proto` | `d1aa73d1a5ee39b180ccc5967dae55d061a4e6e10f9a01d4c116d3a269b111fe` |

All of them are excluded from Dynamo's formatting and copyright-header checks.
The SGLang sidecar generates both client and server types and temporarily
exposes them to the Mocker server.
