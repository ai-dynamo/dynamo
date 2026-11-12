<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-error

`dynamo-error` is the error type Dynamo components use to classify failures. A `DynamoError` carries an `ErrorClass` (what kind of failure), an `ErrorReason` from a fixed catalog (why), and the `PublicDetails` that may reach a client; the component closest to the cause classifies it once, and protocol boundaries render the class. It depends only on `serde` and `serde_json`, and `dynamo-runtime` re-exports it as `dynamo_runtime::error`. Design: DEP #14354.

The optional `http` feature adds the `http` module, the HTTP rendering of the contract that every HTTP host shares: the total class-to-status policy (`http_action_for_class`, `http_action_for_error`, `ClientErrorAction`), the lookup that picks the error a response is rendered from (`find_canonical_error_in_chain`), the configured overload status (`overload_status_code`, read from `DYN_HTTP_OVERLOAD_STATUS_CODE`), and the sanitized response vocabulary (`SanitizedError`, `BackendStatusAction`). Its only added dependency is the `http` crate, for `StatusCode`.
