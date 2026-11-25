// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;
use dynamo_llm::generate_format::{ErrorPhase, WorkerError};
use dynamo_llm::http::service::{error::SanitizedError, metrics::*};
use dynamo_runtime::error::DynamoError;

pub(super) fn classify(
    error: &(dyn std::error::Error + 'static),
    phase: ErrorPhase,
) -> WorkerError {
    let mut current = Some(error);
    let mut invalid = None;
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<DynamoError>()
            && matches!(
                error.reason().as_str(),
                "backend.invalid_argument" | "request.invalid_argument"
            )
        {
            invalid = Some(error.message().to_string());
            break;
        }
        current = error.source();
    }
    if matches!(phase, ErrorPhase::Dispatch) {
        return WorkerError::dispatch(error, invalid);
    }
    // Both unary folding and SSE used this ordering. In unary, cancellation
    // (including a queue deadline) is 499; SSE has already committed HTTP 200.
    let cancelled = StatusCode::from_u16(499).unwrap();
    if request_deadline_exceeded(error) {
        WorkerError::new(ErrorType::Cancelled, cancelled, "request deadline exceeded")
    } else if request_was_rejected(error) {
        WorkerError::new(
            ErrorType::Overload,
            StatusCode::SERVICE_UNAVAILABLE,
            SanitizedError::Overloaded,
        )
    } else if request_was_unavailable(error) {
        WorkerError::new(
            ErrorType::Unavailable,
            StatusCode::SERVICE_UNAVAILABLE,
            SanitizedError::Unavailable,
        )
    } else if let Some(message) = invalid {
        WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, message)
    } else if request_was_cancelled(error) {
        WorkerError::new(ErrorType::Cancelled, cancelled, SanitizedError::Cancelled)
    } else {
        WorkerError::new(
            ErrorType::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            SanitizedError::Internal,
        )
    }
}
