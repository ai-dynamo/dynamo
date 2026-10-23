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

The contract was copied from SGLang commit
`0c093264ad185e8f6e791f0e5e39a17ad8512d0f` (upstream `main` `7d9bc10493` plus
the sidecar changes not yet merged upstream).
The upstream files' SHA-256s are:

| File | SHA-256 |
|------|---------|
| `sglang/api/v1/common.proto` | `32e12cdc19b047a9b0df0a29b33744f92616927c7f0120514b8228c721e39fb5` |
| `sglang/api/v1/finish.proto` | `a8069d9b7be196224a0e7011e4d69e514e50b0833a038338dbfc82fc296fa40c` |
| `sglang/api/v1/generate.proto` | `5f515b13ab34dd2cbc454fb79e7da9534ca89c09f3cccefbc9bdf901f7a4df73` |
| `sglang/api/v1/sampling.proto` | `64621faa7bc31835a52973db9ce8e961c14e66c7d95b21282399d5049531fa6f` |
| `sglang/api/v1/service.proto` | `7c300b87f72fc956eb0d13b72ad1434a242bc35f5e23bfeddf7974c6eddd5f12` |
| `sglang/json/v1/options.proto` | `d1aa73d1a5ee39b180ccc5967dae55d061a4e6e10f9a01d4c116d3a269b111fe` |

The local files are unchanged copies, excluded from Dynamo's formatting and
copyright-header checks. The
SGLang sidecar generates both client and server types and temporarily exposes
them to the Mocker server.
