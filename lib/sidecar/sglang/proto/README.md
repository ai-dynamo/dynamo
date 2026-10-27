<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

This copy is temporary while Dynamo waits for SGLang to include
the `sglang.api.v1` protos in a release wheel. Once the contract is
available there, Dynamo should remove this directory, pin and install the
matching `sglang` wheel as a build dependency, and compile the packaged proto
instead.

The contract was copied from SGLang upstream `main`
`8cd3c494a3aa3ae56dfd76f51bab6912234914dc` plus sgl-project/sglang#45859,
#45883, #45996, #46635, #46636 and #46637, which are not yet merged upstream.
The upstream files' SHA-256s are:

| File | SHA-256 |
|------|---------|
| `sglang/api/v1/common.proto` | `32e12cdc19b047a9b0df0a29b33744f92616927c7f0120514b8228c721e39fb5` |
| `sglang/api/v1/finish.proto` | `a8069d9b7be196224a0e7011e4d69e514e50b0833a038338dbfc82fc296fa40c` |
| `sglang/api/v1/generate.proto` | `4b7b7779d2f616441507f4b4d9c30cb31edf7922d1994c97f07076b9cf83be45` |
| `sglang/api/v1/hicache.proto` | `3224ff3d64234d23678d1a6f899c35b131ebc73eddeabf4a6b25fa983852e736` |
| `sglang/api/v1/sampling.proto` | `64621faa7bc31835a52973db9ce8e961c14e66c7d95b21282399d5049531fa6f` |
| `sglang/api/v1/service.proto` | `bcfa79785b69ef6ec0091fe63e10cd01c577c094ba1a4bf6ee971e4f13f5d683` |
| `sglang/json/v1/options.proto` | `d1aa73d1a5ee39b180ccc5967dae55d061a4e6e10f9a01d4c116d3a269b111fe` |

The local files are unchanged copies, excluded from Dynamo's formatting and
copyright-header checks. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.
