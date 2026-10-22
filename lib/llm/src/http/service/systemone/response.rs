// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn decision_error(error: DecisionError) -> Response {
    (
        StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(error.response_body()),
    )
        .into_response()
}

pub(super) fn error(dialect: Dialect, status: StatusCode, message: impl Into<String>) -> Response {
    let status = if status == StatusCode::UNPROCESSABLE_ENTITY && dialect != Dialect::Jev {
        StatusCode::BAD_REQUEST
    } else {
        status
    };
    let code = match status.as_u16() {
        400 | 413 | 415 | 422 => "invalid_request_error",
        404 => "model_not_found",
        429 | 529 => "rate_limit_exceeded",
        499 => "request_cancelled",
        _ => "server_error",
    };
    decision_error(DecisionError::new(dialect, status.as_u16(), code, message))
}

pub(super) fn attach_request_ids(response: &mut Response, request_id: &str) {
    let value =
        HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("invalid"));
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-request-id"), value.clone());
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-typesafe-request-id"), value);
}

pub(super) async fn dispatch_with_deadline(
    task: tokio::task::JoinHandle<Response>,
    deadline: tokio::time::Instant,
    parent: Arc<dyn AsyncEngineContext>,
    dialect: Dialect,
) -> Response {
    match decision_lifecycle::dispatch_with_deadline(task, deadline, parent).await {
        Ok(response) => response,
        Err(decision_lifecycle::DispatchWaitError::Task(cause)) => {
            tracing::error!(error = %cause, "decision dispatch task failed");
            error(
                dialect,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error",
            )
        }
        Err(decision_lifecycle::DispatchWaitError::Deadline) => error(
            dialect,
            StatusCode::GATEWAY_TIMEOUT,
            "decision request deadline exceeded",
        ),
    }
}
