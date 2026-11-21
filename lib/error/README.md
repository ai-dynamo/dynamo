<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-error

`dynamo-error` is the error type Dynamo components use to classify failures. A `DynamoError` carries an `ErrorClass` (what kind of failure), an `ErrorReason` from a fixed catalog (why), and the `PublicDetails` that may reach a client; the component closest to the cause classifies it once, and protocol boundaries render the class. It depends only on `serde` and `serde_json`, and `dynamo-runtime` re-exports it as `dynamo_runtime::error`. Design: DEP #14354.
