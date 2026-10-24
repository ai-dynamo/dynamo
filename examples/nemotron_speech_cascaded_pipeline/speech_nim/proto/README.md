---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speech NIM Protocol Sources
---

Unmodified sources from [nvidia-riva/common](https://github.com/nvidia-riva/common/tree/268890b7286031a6d4950e34f7ce13ed0d4ce621) at commit `268890b7286031a6d4950e34f7ce13ed0d4ce621`. The upstream license is included in `LICENSE`.

`build.rs` generates the Rust messages and services into Cargo's output directory. Builds do not download protobuf definitions. Both streaming RPCs use bidirectional streams.

| Source | SHA-256 |
| --- | --- |
| `riva/proto/riva_asr.proto` | `e9b9443de9fd02c96935bf9fe32a7d459e5ca89c6f7731735de23d6e83dfd41e` |
| `riva/proto/riva_tts.proto` | `c219215676a94576c3f19c00e0bdb52eeef0e538edfdeefc798b98330df38853` |
| `riva/proto/riva_audio.proto` | `7ab2939dfe9f5ed121eba1c6b86cad0cd561fa483acfc653762a8b8af0b2a123` |
| `riva/proto/riva_common.proto` | `07d668e74d7c2c4e01fa5fee5323b9f6aff2f57f86f4f00d51656b9cb2a3a377` |
| `LICENSE` | `9c0bb53336319c13d061a2246eea4f3f216bad31cdeb1fc9ff0fd8d8308dd543` |

To update the contract, replace the upstream sources and license together, update this provenance, and rerun the Dynamo Speech NIM sidecar tests against the targeted Speech NIM images.
