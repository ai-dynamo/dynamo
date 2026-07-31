// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use axum::http::StatusCode;
use dynamo_runtime::config::environment_names::llm as env_llm;
use thiserror::Error;

/// Overload / admission-control rejection status. Reads
/// `DYN_HTTP_OVERLOAD_STATUS_CODE` (default 529); cached since env is fixed at
/// runtime and this is on the rejection path.
pub(crate) fn overload_status_code() -> StatusCode {
    static CODE: LazyLock<StatusCode> = LazyLock::new(|| {
        let default = StatusCode::from_u16(529).expect("529 is a valid HTTP status code");
        std::env::var(env_llm::DYN_HTTP_OVERLOAD_STATUS_CODE)
            .ok()
            .and_then(|s| s.trim().parse::<u16>().ok())
            .and_then(|n| StatusCode::from_u16(n).ok())
            .unwrap_or(default)
    });
    *CODE
}

/// Implementation of the Completion Engines served by the HTTP service should
/// map their custom errors to to this error type if they wish to return error
/// codes besides 500.
#[derive(Debug, Error)]
#[error("HTTP Error {code}: {message}")]
pub struct HttpError {
    pub code: u16,
    pub message: String,
}

/// Canonical sanitized error responses returned at the HTTP boundary.
///
/// Each variant fixes the `(status, public message, protocol error_type)`
/// triple so call sites stop duplicating literals. The protocol-specific
/// mappings (OpenAI `error_type` string, Anthropic `error_type`) and the
/// `Display` impl that produces the user-safe message all live on this
/// enum — clients see exactly what the enum says, never a backend error
/// chain, file path, or panic stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SanitizedError {
    /// 499 Client Closed Request.
    Cancelled,
    /// 529 Site Is Overloaded.
    Overloaded,
    /// 503 Service Unavailable.
    Unavailable,
    /// 500 Internal Server Error.
    Internal,
    /// Preserve a backend-reported 5xx status code while replacing the
    /// body with the generic internal-error message. Clients still see
    /// the original status (so 503 retry semantics survive); only the
    /// payload is sanitized.
    ///
    /// Invariant: the inner status MUST be in the 500–599 range. Construct
    /// via [`SanitizedError::for_backend_status`] to enforce this.
    PreserveServerError(StatusCode),
}

impl SanitizedError {
    /// Classify a backend-supplied HTTP status into the right sanitized
    /// variant. Returns `None` to mean "forward this 4xx (non-499)
    /// message as-is" — that case is the protocol contract for client
    /// errors and is the caller's responsibility to handle.
    ///
    /// The single source of truth for the status → variant mapping;
    /// every site that triages a backend status code should call this
    /// instead of inlining the if-chain.
    pub fn for_backend_status(status: StatusCode) -> Option<Self> {
        if status.as_u16() == 499 {
            Some(SanitizedError::Cancelled)
        } else if status.is_client_error() {
            // 4xx (non-499) is the protocol contract; caller forwards.
            None
        } else if status.is_server_error() {
            Some(SanitizedError::PreserveServerError(status))
        } else {
            // 1xx/2xx/3xx asserted by a backend payload — coerce to 500.
            Some(SanitizedError::Internal)
        }
    }

    pub fn status(self) -> StatusCode {
        match self {
            // 499 is not IANA-registered but is widely used (nginx).
            SanitizedError::Cancelled => StatusCode::from_u16(499).unwrap(),
            SanitizedError::Overloaded => overload_status_code(),
            SanitizedError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            SanitizedError::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            SanitizedError::PreserveServerError(code) => {
                debug_assert!(
                    code.is_server_error(),
                    "PreserveServerError requires a 5xx status; got {code}"
                );
                code
            }
        }
    }

    /// Anthropic `error.type` for this category. For `PreserveServerError`
    /// the inner status is consulted so a backend 503/529 is reported as
    /// `overloaded_error` (matching the Anthropic spec) rather than the
    /// generic `api_error`.
    pub fn anthropic_type(self) -> &'static str {
        match self {
            SanitizedError::Cancelled => "request_cancelled",
            SanitizedError::Overloaded => "overloaded_error",
            SanitizedError::Unavailable => "overloaded_error",
            SanitizedError::Internal => "api_error",
            SanitizedError::PreserveServerError(status) => match status.as_u16() {
                503 | 529 => "overloaded_error",
                _ => "api_error",
            },
        }
    }

    /// OpenAI-style snake_case `type` field used in inline error frames.
    pub fn openai_type_slug(self) -> &'static str {
        match self {
            SanitizedError::Cancelled => "request_cancelled",
            SanitizedError::Overloaded => "service_unavailable",
            SanitizedError::Unavailable => "service_unavailable",
            SanitizedError::Internal => "internal_server_error",
            SanitizedError::PreserveServerError(status) => match status.as_u16() {
                503 | 529 => "service_unavailable",
                _ => "internal_server_error",
            },
        }
    }

    /// Whether to log this category at `error!` (true) or `debug!` (false).
    /// Cancellations are client-driven and routinely fire on disconnect, so
    /// they stay at debug to avoid drowning real errors.
    pub fn log_as_error(self) -> bool {
        !matches!(self, SanitizedError::Cancelled)
    }
}

/// Retry-semantics allowlist for a **worker-asserted** 5xx status.
///
/// Only two server-error codes keep their identity on Dynamo's status line:
/// `503 Service Unavailable` and whatever [`overload_status_code`] resolves to
/// (`DYN_HTTP_OVERLOAD_STATUS_CODE`, 529 by default). Both are codes Dynamo
/// already generates from its own admission control and already advertises in
/// its OpenAPI document, so forwarding them promises the client nothing new.
/// Every other engine-chosen 5xx — 500, 501, 502, 504, 507, vendor extensions —
/// is coerced to a generic 500, because Dynamo's public contract should not
/// change shape just because a particular backend picked an unusual code.
///
/// Deriving the allowlist from the env knob rather than hard-coding 529 means
/// an operator who sets `DYN_HTTP_OVERLOAD_STATUS_CODE=503` gets one coherent
/// overload code regardless of whether the signal came from Dynamo's router or
/// from the worker itself.
///
/// Scope boundary: this decides only what goes *on the wire*. Whether a
/// worker-local overload should instead be retried against a healthy replica is
/// a routing-layer question tracked by
/// <https://github.com/ai-dynamo/dynamo/issues/12383>. Today it cannot happen
/// here: `ErrorType::ResourceExhausted` sits in `NON_MIGRATABLE` in
/// `crate::migration` and `migration_limit` defaults to 0, so the request has
/// already failed by the time this mapping runs — suppressing the status would
/// not produce a successful response, only a less legible failure.
fn keeps_retry_semantics(status: StatusCode) -> bool {
    status == StatusCode::SERVICE_UNAVAILABLE || status == overload_status_code()
}

/// What to do with a status a backend worker asserted for itself.
///
/// Returned by [`BackendStatusAction::triage`]. Callers must match every arm,
/// so the coercion policy cannot be silently skipped at a new call site the way
/// a bare predicate could be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatusAction {
    /// Non-499 4xx: the protocol contract. Forward the backend's status *and*
    /// message verbatim — client errors are the engine's to describe.
    ForwardClientError,
    /// Answer with this sanitized variant, keeping the status it carries.
    Sanitize(SanitizedError),
    /// Answer `500 Internal Server Error`. The inner status is what the engine
    /// asserted; callers tunnel it into the response body as a bare number so
    /// the information survives for debugging without the status line
    /// promising semantics Dynamo cannot honour.
    CoerceToInternal(StatusCode),
}

impl BackendStatusAction {
    /// Triage a worker-asserted status. Layers the retry-semantics allowlist
    /// on top of [`SanitizedError::for_backend_status`], which stays the base
    /// status → variant mapping shared with the streaming preflight and the
    /// Anthropic surface.
    pub fn triage(status: StatusCode) -> Self {
        match SanitizedError::for_backend_status(status) {
            // 4xx (non-499): caller forwards.
            None => BackendStatusAction::ForwardClientError,
            Some(SanitizedError::PreserveServerError(asserted))
                if !keeps_retry_semantics(asserted) =>
            {
                BackendStatusAction::CoerceToInternal(asserted)
            }
            Some(variant) => BackendStatusAction::Sanitize(variant),
        }
    }
}

impl std::fmt::Display for SanitizedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SanitizedError::Cancelled => f.write_str("Request cancelled"),
            SanitizedError::Overloaded => f.write_str("Service temporarily overloaded"),
            SanitizedError::Unavailable => f.write_str("Service temporarily unavailable"),
            SanitizedError::Internal | SanitizedError::PreserveServerError(_) => {
                f.write_str("Internal server error")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_statuses_distinguish_overload_from_unavailable() {
        assert_eq!(SanitizedError::Overloaded.status().as_u16(), 529);
        assert_eq!(
            SanitizedError::Unavailable.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn preserve_server_error_503_maps_to_overload_types() {
        // Backend-asserted 503 must surface as the spec-correct overload
        // type on both protocols, not as a generic api_error /
        // internal_server_error.
        let err = SanitizedError::PreserveServerError(StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.anthropic_type(), "overloaded_error");
        assert_eq!(err.openai_type_slug(), "service_unavailable");
    }

    #[test]
    fn preserve_server_error_529_maps_to_overload_types() {
        // Anthropic uses 529 as an alternative overload signal; mirror
        // the 503 mapping so clients can apply the same backoff.
        let err = SanitizedError::PreserveServerError(StatusCode::from_u16(529).unwrap());
        assert_eq!(err.anthropic_type(), "overloaded_error");
        assert_eq!(err.openai_type_slug(), "service_unavailable");
    }

    #[test]
    fn preserve_server_error_500_remains_generic() {
        let err = SanitizedError::PreserveServerError(StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.anthropic_type(), "api_error");
        assert_eq!(err.openai_type_slug(), "internal_server_error");
    }

    #[test]
    fn for_backend_status_classifies_correctly() {
        // 499 → Cancelled
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(499).unwrap()),
            Some(SanitizedError::Cancelled)
        ));
        // 5xx → PreserveServerError preserving the code
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::SERVICE_UNAVAILABLE),
            Some(SanitizedError::PreserveServerError(s)) if s == StatusCode::SERVICE_UNAVAILABLE
        ));
        // Non-499 4xx → None (forward as-is)
        assert!(SanitizedError::for_backend_status(StatusCode::BAD_REQUEST).is_none());
        assert!(SanitizedError::for_backend_status(StatusCode::NOT_FOUND).is_none());
        // 1xx/2xx/3xx asserted by backend → Internal
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(399).unwrap()),
            Some(SanitizedError::Internal)
        ));
    }

    #[test]
    fn triage_keeps_only_retry_bearing_5xx_on_the_status_line() {
        // 503 and the configured overload code are the codes Dynamo itself
        // generates and advertises, so a worker asserting them says something
        // Dynamo can honour; they stay on the status line.
        for status in [StatusCode::SERVICE_UNAVAILABLE, overload_status_code()] {
            assert_eq!(
                BackendStatusAction::triage(status),
                BackendStatusAction::Sanitize(SanitizedError::PreserveServerError(status)),
                "{status} should keep its status line"
            );
        }
    }

    #[test]
    fn triage_coerces_other_5xx_and_reports_the_asserted_status() {
        // Arbitrary engine-chosen server errors must not reach the client's
        // status line, but the asserted code has to survive for the caller to
        // tunnel into the body.
        for code in [500u16, 501, 502, 504, 507, 599] {
            let status = StatusCode::from_u16(code).unwrap();
            assert_eq!(
                BackendStatusAction::triage(status),
                BackendStatusAction::CoerceToInternal(status),
                "{code} should be coerced to 500 with the asserted status reported"
            );
        }
    }

    #[test]
    fn triage_leaves_client_errors_and_cancellation_alone() {
        // The allowlist governs 5xx only; 4xx (non-499) still forwards and
        // 499 still classifies as cancellation.
        assert_eq!(
            BackendStatusAction::triage(StatusCode::BAD_REQUEST),
            BackendStatusAction::ForwardClientError
        );
        assert_eq!(
            BackendStatusAction::triage(StatusCode::UNSUPPORTED_MEDIA_TYPE),
            BackendStatusAction::ForwardClientError
        );
        assert_eq!(
            BackendStatusAction::triage(StatusCode::from_u16(499).unwrap()),
            BackendStatusAction::Sanitize(SanitizedError::Cancelled)
        );
        // A backend asserting a non-error status is still nonsense → 500.
        assert_eq!(
            BackendStatusAction::triage(StatusCode::from_u16(399).unwrap()),
            BackendStatusAction::Sanitize(SanitizedError::Internal)
        );
    }
}
