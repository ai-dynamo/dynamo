// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;
use serde_json::Value;

use super::DispatchError;

pub(super) fn classify(finish: &Value) -> Option<DispatchError> {
    if finish.get("type").and_then(Value::as_str) != Some("abort") {
        return None;
    }
    Some(
        if finish.get("status_code").and_then(Value::as_u64) == Some(429) {
            DispatchError {
                status: StatusCode::TOO_MANY_REQUESTS,
                public_message: "SGLang scoring worker capacity is exhausted",
                detail: "native scoring worker capacity exhausted".to_string(),
                retry_after: true,
            }
        } else {
            // Dynamo constructs the scoring request; a backend 4xx does not establish client fault.
            // Cancellation is determined by the parent context, not the native abort metadata.
            DispatchError {
                retry_after: true,
                ..DispatchError::unavailable("native scoring worker aborted")
            }
        },
    )
}
