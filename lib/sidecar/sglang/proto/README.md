<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

The SGLang `0.5.18` release wheel does not package the native gRPC contract, so Dynamo vendors the fields consumed by the sidecar. The local schema is wire-compatible with SGLang `0.5.18`; additive release fields that the sidecar does not consume are omitted.

The local file's SHA-256 is `f3d5bf6c18dd95248c311f1368a77631862d9c9f0febe748d19964b7e1154f07`. It adds SPDX and vendoring comments and applies Dynamo's `clang-format` style. The SGLang sidecar generates both client and server types and exports them to the Mocker server.
