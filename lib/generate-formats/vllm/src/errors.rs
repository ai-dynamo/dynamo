// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;
use dynamo_llm::generate_format::{ErrorPhase, WorkerError};
use dynamo_llm::http::service::{error::SanitizedError, metrics::*};

pub(super) fn classify(
    error: &(dyn std::error::Error + 'static),
    phase: ErrorPhase,
) -> WorkerError {
    let invalid = WorkerError::canonical_invalid_request_message(error);
    if matches!(phase, ErrorPhase::Dispatch) {
        return WorkerError::dispatch(error, invalid);
    }
    // Preserve vLLM's unary-fold policy: deadline and overload markers alone
    // were internal stream failures, unlike errors returned before dispatch.
    if matches!(phase, ErrorPhase::Stream { cancelled: true }) || request_was_cancelled(error) {
        WorkerError::new(
            ErrorType::Cancelled,
            StatusCode::from_u16(499).unwrap(),
            "request was cancelled",
        )
    } else if request_was_unavailable(error) {
        WorkerError::new(
            ErrorType::Unavailable,
            StatusCode::SERVICE_UNAVAILABLE,
            SanitizedError::Unavailable,
        )
    } else if let Some(message) = invalid {
        WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, message)
    } else {
        WorkerError::new(
            ErrorType::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal server error",
        )
    }
}
