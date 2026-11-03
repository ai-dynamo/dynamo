<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-error

The semantic error contract Dynamo components classify failures with (DEP #14354): `DynamoError`, `ErrorClass`, `ErrorReason`, `Diagnostic`, `PublicDetails`, and the builder. The component closest to a failure classifies it once; protocol boundaries render the class.

A leaf crate: it depends on serde and serde_json and nothing else, so crates below the runtime can classify with the same type. `dynamo-runtime` re-exports it as `dynamo_runtime::error`.
