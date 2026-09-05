// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use axum::http::StatusCode;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};
use thiserror::Error;

fn parse_overload_status_code(value: Option<&str>) -> StatusCode {
    let default = StatusCode::from_u16(529).expect("529 is a valid HTTP status code");
    value
        .and_then(|s| s.trim().parse::<u16>().ok())
        .and_then(|n| StatusCode::from_u16(n).ok())
        .filter(|status| !status.is_informational())
        .unwrap_or(default)
}

/// Overload / admission-control rejection status. Reads
/// `DYN_HTTP_OVERLOAD_STATUS_CODE` (default 529) on first use; cached since the
/// environment is fixed at runtime and this is on the rejection path.
pub(crate) fn overload_status_code() -> StatusCode {
    static CODE: LazyLock<StatusCode> = LazyLock::new(|| {
        let value = std::env::var(env_llm::DYN_HTTP_OVERLOAD_STATUS_CODE).ok();
        parse_overload_status_code(value.as_deref())
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

/// Frontend-owned HTTP disposition derived only from semantic error class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClientErrorAction {
    NoDelivery,
    Respond {
        status: StatusCode,
        public_message: &'static str,
    },
}

/// Total class-to-HTTP policy. Backends never select these status codes.
pub(crate) fn http_action_for_class(class: dynamo_runtime::error::ErrorClass) -> ClientErrorAction {
    use dynamo_runtime::error::{BackendError, ErrorClass};

    let response = |status, public_message| ClientErrorAction::Respond {
        status,
        public_message,
    };

    match class {
        ErrorClass::InvalidArgument
        | ErrorClass::InvalidRequest
        | ErrorClass::Backend(BackendError::InvalidArgument) => {
            response(StatusCode::BAD_REQUEST, "Invalid request")
        }
        ErrorClass::Unauthenticated => {
            response(StatusCode::UNAUTHORIZED, "Authentication required")
        }
        ErrorClass::PermissionDenied => response(StatusCode::FORBIDDEN, "Permission denied"),
        ErrorClass::NotFound => response(StatusCode::NOT_FOUND, "Resource not found"),
        ErrorClass::Conflict => response(StatusCode::CONFLICT, "Request conflict"),
        ErrorClass::PayloadTooLarge => {
            response(StatusCode::PAYLOAD_TOO_LARGE, "Request payload too large")
        }
        ErrorClass::UnsupportedMedia => {
            response(StatusCode::UNSUPPORTED_MEDIA_TYPE, "Unsupported media type")
        }
        ErrorClass::RateLimited => response(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded"),
        ErrorClass::ResourceExhausted
        | ErrorClass::WorkerOverloaded
        | ErrorClass::CapacityExhausted => response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Service temporarily unavailable",
        ),
        ErrorClass::CannotConnect
        | ErrorClass::Disconnected
        | ErrorClass::Unavailable
        | ErrorClass::Backend(BackendError::CannotConnect)
        | ErrorClass::Backend(BackendError::Disconnected)
        | ErrorClass::Backend(BackendError::EngineShutdown)
        | ErrorClass::Backend(BackendError::StreamIncomplete) => response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Service temporarily unavailable",
        ),
        ErrorClass::BackendProtocol => response(StatusCode::BAD_GATEWAY, "Bad gateway"),
        ErrorClass::ConnectionTimeout
        | ErrorClass::ResponseTimeout
        | ErrorClass::DeadlineExceeded
        | ErrorClass::Backend(BackendError::ConnectionTimeout)
        | ErrorClass::Backend(BackendError::ResponseTimeout) => {
            response(StatusCode::GATEWAY_TIMEOUT, "Request deadline exceeded")
        }
        ErrorClass::NotImplemented => {
            response(StatusCode::NOT_IMPLEMENTED, "Operation not implemented")
        }
        ErrorClass::Cancelled | ErrorClass::Backend(BackendError::Cancelled) => {
            ClientErrorAction::NoDelivery
        }
        ErrorClass::Unknown | ErrorClass::Internal | ErrorClass::Backend(BackendError::Unknown) => {
            response(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
        }
    }
}

pub(crate) fn http_action_for_error(error: &DynamoError) -> ClientErrorAction {
    http_action_for_class(error.class())
}

/// Return the first DynamoError only when it already uses the canonical class
/// vocabulary. Legacy variants continue through compatibility handling.
pub(crate) fn find_canonical_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
) -> Option<&'a DynamoError> {
    use dynamo_runtime::error::ErrorClass;

    let mut current = Some(err);
    let mut fallback = None;
    while let Some(error) = current {
        if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
            let raw_class = dynamo_error.error_type();
            if raw_class == raw_class.normalized() && raw_class != ErrorClass::Cancelled {
                if dynamo_error.reason().as_str() == "runtime.unclassified" {
                    fallback.get_or_insert(dynamo_error);
                } else {
                    return Some(dynamo_error);
                }
            }
        }
        current = error.source();
    }
    fallback
}

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(DynamoErrorType::InvalidArgument)
        .message(message)
        .build()
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

/// Whether a worker-asserted 5xx keeps its own status on the wire.
///
/// Only 503 and the configured overload code (`DYN_HTTP_OVERLOAD_STATUS_CODE`,
/// 529 by default) do. Dynamo already generates and documents both, so
/// forwarding them tells a client nothing it could not already receive; any
/// other 5xx becomes a 500. Reading the overload code from the env rather than
/// hard-coding 529 keeps one overload status whether the signal came from the
/// router or from the worker.
///
/// This only decides what goes on the wire. Retrying a worker-local overload
/// against another replica is
/// <https://github.com/ai-dynamo/dynamo/issues/12383>.
fn keeps_retry_semantics(status: StatusCode) -> bool {
    status == StatusCode::SERVICE_UNAVAILABLE || status == overload_status_code()
}

/// What to do with a status a backend worker asserted for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatusAction {
    /// Non-499 4xx. Forward the backend's status and message verbatim.
    ForwardClientError,
    /// Answer with this sanitized variant, keeping the status it carries.
    Sanitize(SanitizedError),
    /// Answer 500. The inner status is what the engine asserted; callers put it
    /// in the response body so it survives for debugging.
    CoerceToInternal(StatusCode),
}

impl BackendStatusAction {
    /// Triage a worker-asserted status, applying [`keeps_retry_semantics`] on
    /// top of [`SanitizedError::for_backend_status`].
    ///
    /// `DYN_HTTP_OVERLOAD_STATUS_CODE` is not restricted to 5xx (see
    /// [`parse_overload_status_code`]), so a configured non-5xx overload code
    /// (e.g. 429) would otherwise fall through
    /// [`SanitizedError::for_backend_status`]'s 4xx branch and be forwarded
    /// as a plain client error instead of sanitized and preserved as the
    /// overload response. Check it first, before that classification runs.
    pub fn triage(status: StatusCode) -> Self {
        if status == overload_status_code() && !status.is_server_error() {
            return BackendStatusAction::Sanitize(SanitizedError::Overloaded);
        }
        match SanitizedError::for_backend_status(status) {
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
    fn parses_configured_overload_status_code() {
        for value in [
            None,
            Some(""),
            Some("not-a-code"),
            Some("99"),
            Some("100"),
            Some("199"),
            Some("1000"),
        ] {
            assert_eq!(
                parse_overload_status_code(value).as_u16(),
                529,
                "expected {value:?} to fall back to 529"
            );
        }

        for value in [200_u16, 503, 529, 600, 999] {
            let configured = value.to_string();
            assert_eq!(
                parse_overload_status_code(Some(&configured)).as_u16(),
                value,
                "expected {value} to be preserved"
            );
        }
    }

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
        // A non-error status from a backend is nonsense, so it sanitizes too.
        assert_eq!(
            BackendStatusAction::triage(StatusCode::from_u16(399).unwrap()),
            BackendStatusAction::Sanitize(SanitizedError::Internal)
        );
    }

    #[test]
    fn canonical_lookup_skips_legacy_wrapper() {
        use dynamo_runtime::error::{ErrorClass, ErrorReason};

        let inner = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .reason(ErrorReason::new("request.invalid").unwrap())
            .build();
        let outer = DynamoError::builder()
            .error_type(DynamoErrorType::Unknown)
            .cause(inner)
            .build();

        assert_eq!(
            find_canonical_error_in_chain(&outer).map(DynamoError::class),
            Some(ErrorClass::InvalidRequest)
        );
    }

    #[test]
    fn canonical_lookup_retains_unclassified_as_fallback() {
        let error = DynamoError::builder()
            .class(dynamo_runtime::error::ErrorClass::Internal)
            .reason(dynamo_runtime::error::ErrorReason::new("runtime.unclassified").unwrap())
            .build();

        assert_eq!(
            find_canonical_error_in_chain(&error).map(|error| error.reason().as_str()),
            Some("runtime.unclassified")
        );
    }

    #[test]
    fn semantic_classes_have_total_http_policy() {
        use dynamo_runtime::error::ErrorClass;

        let cases = [
            (ErrorClass::InvalidRequest, Some(StatusCode::BAD_REQUEST)),
            (ErrorClass::Unauthenticated, Some(StatusCode::UNAUTHORIZED)),
            (ErrorClass::PermissionDenied, Some(StatusCode::FORBIDDEN)),
            (ErrorClass::NotFound, Some(StatusCode::NOT_FOUND)),
            (ErrorClass::Conflict, Some(StatusCode::CONFLICT)),
            (
                ErrorClass::PayloadTooLarge,
                Some(StatusCode::PAYLOAD_TOO_LARGE),
            ),
            (
                ErrorClass::UnsupportedMedia,
                Some(StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ),
            (ErrorClass::RateLimited, Some(StatusCode::TOO_MANY_REQUESTS)),
            (
                ErrorClass::CapacityExhausted,
                Some(StatusCode::SERVICE_UNAVAILABLE),
            ),
            (ErrorClass::BackendProtocol, Some(StatusCode::BAD_GATEWAY)),
            (
                ErrorClass::Unavailable,
                Some(StatusCode::SERVICE_UNAVAILABLE),
            ),
            (
                ErrorClass::DeadlineExceeded,
                Some(StatusCode::GATEWAY_TIMEOUT),
            ),
            (
                ErrorClass::NotImplemented,
                Some(StatusCode::NOT_IMPLEMENTED),
            ),
            (
                ErrorClass::Internal,
                Some(StatusCode::INTERNAL_SERVER_ERROR),
            ),
            (ErrorClass::Cancelled, None),
        ];

        for (class, expected_status) in cases {
            match (http_action_for_class(class), expected_status) {
                (ClientErrorAction::Respond { status, .. }, Some(expected)) => {
                    assert_eq!(status, expected, "{class}");
                }
                (ClientErrorAction::NoDelivery, None) => {}
                (actual, expected) => panic!("{class}: got {actual:?}, expected {expected:?}"),
            }
        }
    }

    #[test]
    fn legacy_classes_normalize_before_http_policy() {
        use dynamo_runtime::error::ErrorClass;

        for (class, expected) in [
            (ErrorClass::InvalidArgument, StatusCode::BAD_REQUEST),
            (
                ErrorClass::WorkerOverloaded,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (ErrorClass::Unknown, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            assert!(matches!(
                http_action_for_class(class),
                ClientErrorAction::Respond { status, .. } if status == expected
            ));
        }
    }
}
