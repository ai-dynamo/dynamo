// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;
use thiserror::Error;

/// Implementation of the Completion Engines served by the HTTP service should
/// map their custom errors to to this error type if they wish to return error
/// codes besides 500.
#[derive(Debug, Error)]
#[error("HTTP Error {code}: {message}")]
pub struct HttpError {
    pub code: u16,
    pub message: String,
}

/// Admission-control rejection hint appended to overload responses.
pub const ADMISSION_CONTROL_REJECTION_HINT: &str =
    "If this rejection is not intended, consider passing --admission-control none to the frontend.";

/// Canonical sanitized error responses returned at the HTTP boundary.
///
/// Each variant fixes the `(status, public message, protocol error_type)`
/// triple so call sites stop duplicating literals. The protocol-specific
/// mappings (OpenAI `error_type` string, Anthropic `error_type`) and the
/// `Display` impl that produces the user-safe message all live on this
/// enum — clients see exactly what the enum says, never a backend error
/// chain, file path, or panic stack.
#[derive(Debug, Clone, Copy)]
pub enum SanitizedError {
    /// 499 Client Closed Request.
    Cancelled,
    /// 503 Service Unavailable, with admission-control hint.
    Overloaded,
    /// 500 Internal Server Error.
    Internal,
    /// Preserve a backend-reported 5xx status code while replacing the
    /// body with the generic internal-error message. Clients still see
    /// the original status (so 503 retry semantics survive); only the
    /// payload is sanitized.
    PreserveServerError(StatusCode),
}

impl SanitizedError {
    pub fn status(self) -> StatusCode {
        match self {
            // 499 is not IANA-registered but is widely used (nginx).
            SanitizedError::Cancelled => StatusCode::from_u16(499).unwrap(),
            SanitizedError::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            SanitizedError::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            SanitizedError::PreserveServerError(code) => code,
        }
    }

    /// Anthropic `error.type` for this category.
    pub fn anthropic_type(self) -> &'static str {
        match self {
            SanitizedError::Cancelled => "request_cancelled",
            SanitizedError::Overloaded => "overloaded_error",
            SanitizedError::Internal | SanitizedError::PreserveServerError(_) => "api_error",
        }
    }

    /// OpenAI-style snake_case `type` field used in inline error frames.
    pub fn openai_type_slug(self) -> &'static str {
        match self {
            SanitizedError::Cancelled => "request_cancelled",
            SanitizedError::Overloaded => "service_unavailable",
            SanitizedError::Internal | SanitizedError::PreserveServerError(_) => {
                "internal_server_error"
            }
        }
    }

    /// Whether to log this category at `error!` (true) or `debug!` (false).
    /// Cancellations are client-driven and routinely fire on disconnect, so
    /// they stay at debug to avoid drowning real errors.
    pub fn log_as_error(self) -> bool {
        !matches!(self, SanitizedError::Cancelled)
    }
}

impl std::fmt::Display for SanitizedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SanitizedError::Cancelled => f.write_str("Request cancelled"),
            SanitizedError::Overloaded => write!(
                f,
                "Service temporarily overloaded. {ADMISSION_CONTROL_REJECTION_HINT}"
            ),
            SanitizedError::Internal | SanitizedError::PreserveServerError(_) => {
                f.write_str("Internal server error")
            }
        }
    }
}
