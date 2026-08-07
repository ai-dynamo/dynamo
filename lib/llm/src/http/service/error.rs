// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use axum::http::StatusCode;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType as DynamoErrorType};
use serde_json::Value;
use thiserror::Error;

use super::metrics::ErrorType as MetricErrorType;
use crate::types::Annotated;

pub(crate) const INTERNAL_ERROR_MESSAGE: &str = "Internal server error";

#[derive(serde::Deserialize)]
struct BackendHttpPayload {
    code: u16,
    message: String,
}

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
pub fn overload_status_code() -> StatusCode {
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

/// Compatibility model used by endpoint handlers that have not yet migrated
/// to the shared classifier. The next PR in the stack removes this once every
/// HTTP surface uses the shared classifier.
#[derive(Debug, Clone, Copy)]
pub enum SanitizedError {
    Cancelled,
    Overloaded,
    Unavailable,
    Internal,
    PreserveServerError(StatusCode),
}

impl SanitizedError {
    pub fn for_backend_status(status: StatusCode) -> Option<Self> {
        if status.as_u16() == 499 {
            Some(Self::Cancelled)
        } else if status.is_client_error() {
            None
        } else if status.is_server_error() {
            Some(Self::PreserveServerError(status))
        } else {
            Some(Self::Internal)
        }
    }

    pub fn status(self) -> StatusCode {
        match self {
            Self::Cancelled => StatusCode::from_u16(499).expect("499 is a valid status code"),
            Self::Overloaded => overload_status_code(),
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            Self::PreserveServerError(code) => code,
        }
    }

    pub fn anthropic_type(self) -> &'static str {
        match self {
            Self::Cancelled => "request_cancelled",
            Self::Overloaded | Self::Unavailable => "overloaded_error",
            Self::PreserveServerError(status) if matches!(status.as_u16(), 503 | 529) => {
                "overloaded_error"
            }
            Self::Internal | Self::PreserveServerError(_) => "api_error",
        }
    }

    pub fn openai_type_slug(self) -> &'static str {
        match self {
            Self::Cancelled => "request_cancelled",
            Self::Overloaded | Self::Unavailable => "service_unavailable",
            Self::PreserveServerError(status) if matches!(status.as_u16(), 503 | 529) => {
                "service_unavailable"
            }
            Self::Internal | Self::PreserveServerError(_) => "internal_server_error",
        }
    }

    pub fn log_as_error(self) -> bool {
        !matches!(self, Self::Cancelled)
    }
}

impl std::fmt::Display for SanitizedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => f.write_str("Request cancelled"),
            Self::Overloaded => f.write_str("Service temporarily overloaded"),
            Self::Unavailable => f.write_str("Service temporarily unavailable"),
            Self::Internal | Self::PreserveServerError(_) => f.write_str(INTERNAL_ERROR_MESSAGE),
        }
    }
}

/// Protocol-neutral error categories used for metrics and protocol rendering.
#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HttpProblemKind {
    Validation,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    Cancelled,
    Overloaded,
    Unavailable,
    NotImplemented,
    Internal,
}

/// A fully classified HTTP failure.
///
/// Classification owns the status, public message, metrics category, and
/// diagnostic. Protocol modules only translate this model into their wire
/// envelope; they do not re-classify errors.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
pub(crate) struct HttpProblem {
    kind: HttpProblemKind,
    status: StatusCode,
    message: String,
    diagnostic: String,
    details: Option<Box<Value>>,
}

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
#[allow(dead_code)] // Used by the protocol-validation PR later in this stack.
pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(DynamoErrorType::InvalidArgument)
        .message(message)
        .build()
}

#[derive(Debug, Clone, Copy)]
enum ValidationScope {
    Chain,
    Outermost,
}

#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
impl HttpProblem {
    /// Classify an arbitrary error through the same logical nested-error flow
    /// used by `from_annotated`; generic callers retain chain-wide validation
    /// lookup because untyped context wrappers may surround the typed failure.
    pub(crate) fn from_error(
        err: &(dyn std::error::Error + 'static),
        internal_message: &str,
    ) -> Self {
        Self::from_error_with_validation_scope(err, internal_message, ValidationScope::Chain)
    }

    fn from_error_with_validation_scope(
        err: &(dyn std::error::Error + 'static),
        internal_message: &str,
        validation_scope: ValidationScope,
    ) -> Self {
        let diagnostic = format_error_chain(err);
        let mut current = Some(err);
        while let Some(error) = current {
            if let Some(rejection) =
                error.downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>()
            {
                return Self {
                    kind: HttpProblemKind::Overloaded,
                    status: overload_status_code(),
                    message: rejection.to_string(),
                    diagnostic,
                    details: serde_json::to_value(rejection).ok().map(Box::new),
                };
            }
            current = error.source();
        }

        if let Some(dynamo_error) = select_dynamo_error_in_chain(err, validation_scope) {
            return Self::from_dynamo_error(dynamo_error, internal_message, diagnostic);
        }

        // All other typed failures retain outermost-error classification.
        let mut current = Some(err);
        while let Some(error) = current {
            if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
                return Self::from_dynamo_error(dynamo_error, internal_message, diagnostic);
            }
            if let Some(http_error) = error.downcast_ref::<HttpError>() {
                return Self::from_explicit_http_error(http_error, diagnostic);
            }
            current = error.source();
        }

        Self::internal(internal_message, diagnostic)
    }

    /// Classify a backend stream event through the same logical nested-error
    /// flow as `from_error`; the event's outer typed validation is authoritative,
    /// while nested operational failures retain their retry/cancellation meaning.
    pub(crate) fn from_annotated<T>(event: &Annotated<T>) -> Option<Self> {
        #[derive(serde::Deserialize)]
        struct ErrorPayload {
            message: Option<String>,
            code: Option<u16>,
        }

        if event.is_error() {
            if let Some(error) = event.error.as_ref() {
                return Some(Self::from_error_with_validation_scope(
                    error,
                    INTERNAL_ERROR_MESSAGE,
                    ValidationScope::Outermost,
                ));
            }

            let diagnostic = event
                .comment
                .as_ref()
                .map(|comments| comments.join(", "))
                .filter(|message| !message.trim().is_empty())
                .unwrap_or_else(|| "unspecified error".to_string());

            // Compatibility for legacy error events that carried a JSON body
            // in their comment instead of the structured `error` field.
            if let Ok(payload) = serde_json::from_str::<ErrorPayload>(&diagnostic) {
                let status = payload
                    .code
                    .and_then(|code| StatusCode::from_u16(code).ok())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let message = payload.message.unwrap_or_else(|| diagnostic.clone());
                return Some(Self::from_backend_status(status, message, diagnostic));
            }

            return Some(Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic));
        }

        // Legacy backends used a comment-only, otherwise empty annotation as
        // an error marker. Preserve that signal, but never expose or interpret
        // the untyped comment as a client-safe status/message.
        if event.data.is_none()
            && event.event.is_none()
            && let Some(comments) = event.comment.as_ref()
            && !comments.is_empty()
        {
            return Some(Self::internal(INTERNAL_ERROR_MESSAGE, comments.join(", ")));
        }

        None
    }

    pub(crate) fn from_backend_status(
        status: StatusCode,
        message: impl Into<String>,
        diagnostic: impl Into<String>,
    ) -> Self {
        let message = message.into();
        let diagnostic = diagnostic.into();
        match status {
            status if status.as_u16() == 499 => Self::classified(
                HttpProblemKind::Cancelled,
                status,
                "Request cancelled",
                diagnostic,
            ),
            status if status.is_client_error() => Self {
                kind: kind_for_status(status),
                status,
                message,
                diagnostic,
                details: None,
            },
            status if status.is_server_error() => Self::classified(
                kind_for_status(status),
                status,
                INTERNAL_ERROR_MESSAGE,
                diagnostic,
            ),
            _ => Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic),
        }
    }

    pub(crate) fn internal(message: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            kind: HttpProblemKind::Internal,
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            diagnostic: diagnostic.into(),
            details: None,
        }
    }

    fn from_dynamo_error(error: &DynamoError, internal_message: &str, diagnostic: String) -> Self {
        if let Some((status, message)) = backend_http_payload(error) {
            return Self::from_backend_status(status, message, diagnostic);
        }

        match error.error_type() {
            DynamoErrorType::InvalidArgument
            | DynamoErrorType::Backend(BackendError::InvalidArgument) => Self {
                kind: HttpProblemKind::Validation,
                status: StatusCode::BAD_REQUEST,
                message: error.message().to_string(),
                diagnostic,
                details: None,
            },
            DynamoErrorType::ResourceExhausted => Self::classified(
                HttpProblemKind::Overloaded,
                overload_status_code(),
                "Service temporarily overloaded",
                diagnostic,
            ),
            DynamoErrorType::Unavailable => Self::classified(
                HttpProblemKind::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable",
                diagnostic,
            ),
            DynamoErrorType::Cancelled | DynamoErrorType::Backend(BackendError::Cancelled) => {
                Self::classified(
                    HttpProblemKind::Cancelled,
                    StatusCode::from_u16(499).expect("499 is a valid HTTP status code"),
                    "Request cancelled",
                    diagnostic,
                )
            }
            _ => Self::internal(internal_message, diagnostic),
        }
    }

    fn from_explicit_http_error(error: &HttpError, diagnostic: String) -> Self {
        let Ok(status) = StatusCode::from_u16(error.code) else {
            return Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic);
        };
        if status.as_u16() == 499 {
            Self::classified(
                HttpProblemKind::Cancelled,
                status,
                "Request cancelled",
                diagnostic,
            )
        } else if status.is_client_error() {
            Self {
                kind: kind_for_status(status),
                status,
                message: error.message.clone(),
                diagnostic,
                details: None,
            }
        } else {
            Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic)
        }
    }

    fn classified(
        kind: HttpProblemKind,
        status: StatusCode,
        message: impl Into<String>,
        diagnostic: String,
    ) -> Self {
        Self {
            kind,
            status,
            message: message.into(),
            diagnostic,
            details: None,
        }
    }

    pub(crate) fn kind(&self) -> HttpProblemKind {
        self.kind
    }

    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn diagnostic(&self) -> &str {
        &self.diagnostic
    }

    pub(crate) fn details(&self) -> Option<Box<Value>> {
        self.details.clone()
    }

    pub(crate) fn metric_type(&self) -> MetricErrorType {
        metric_type_for_kind(self.kind)
    }

    pub(crate) fn metric_type_for_status(status: StatusCode) -> MetricErrorType {
        metric_type_for_kind(kind_for_status(status))
    }
}

/// Decode the deliberate Python-boundary `{code, message}` compatibility
/// envelope without treating arbitrary typed error messages as HTTP policy.
fn backend_http_payload(error: &DynamoError) -> Option<(StatusCode, String)> {
    let payload = serde_json::from_str::<BackendHttpPayload>(error.message()).ok()?;
    let status = StatusCode::from_u16(payload.code).ok()?;

    match error.error_type() {
        DynamoErrorType::Backend(BackendError::InvalidArgument) if status.is_client_error() => {}
        DynamoErrorType::Backend(BackendError::Unknown) if status.is_server_error() => {}
        _ => return None,
    }

    Some((status, payload.message))
}

fn select_dynamo_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
    validation_scope: ValidationScope,
) -> Option<&'a DynamoError> {
    // Preserve operational signals through wrappers before considering the
    // validation scope selected by the calling boundary.
    for error_type in [
        DynamoErrorType::ResourceExhausted,
        DynamoErrorType::Unavailable,
    ] {
        if let Some(error) = find_dynamo_error_in_chain(err, error_type) {
            return Some(error);
        }
    }

    match validation_scope {
        ValidationScope::Chain => {
            for error_type in [
                DynamoErrorType::InvalidArgument,
                DynamoErrorType::Backend(BackendError::InvalidArgument),
            ] {
                if let Some(error) = find_dynamo_error_in_chain(err, error_type) {
                    return Some(error);
                }
            }
        }
        ValidationScope::Outermost => {
            if let Some(error) = find_outermost_dynamo_error_in_chain(err)
                && matches!(
                    error.error_type(),
                    DynamoErrorType::InvalidArgument
                        | DynamoErrorType::Backend(BackendError::InvalidArgument)
                )
            {
                return Some(error);
            }
        }
    }

    for error_type in [
        DynamoErrorType::Cancelled,
        DynamoErrorType::Backend(BackendError::Cancelled),
    ] {
        if let Some(error) = find_dynamo_error_in_chain(err, error_type) {
            return Some(error);
        }
    }

    None
}

fn find_outermost_dynamo_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
) -> Option<&'a DynamoError> {
    let mut current = Some(err);
    while let Some(error) = current {
        if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
            return Some(dynamo_error);
        }
        current = error.source();
    }
    None
}

fn find_dynamo_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
    error_type: DynamoErrorType,
) -> Option<&'a DynamoError> {
    let mut current = Some(err);
    while let Some(error) = current {
        if let Some(dynamo_error) = error.downcast_ref::<DynamoError>()
            && dynamo_error.error_type() == error_type
        {
            return Some(dynamo_error);
        }
        current = error.source();
    }
    None
}

#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
fn metric_type_for_kind(kind: HttpProblemKind) -> MetricErrorType {
    match kind {
        HttpProblemKind::Validation
        | HttpProblemKind::Authentication
        | HttpProblemKind::Permission => MetricErrorType::Validation,
        HttpProblemKind::NotFound => MetricErrorType::NotFound,
        HttpProblemKind::RateLimit | HttpProblemKind::Overloaded => MetricErrorType::Overload,
        HttpProblemKind::Unavailable => MetricErrorType::Unavailable,
        HttpProblemKind::Cancelled => MetricErrorType::Cancelled,
        HttpProblemKind::NotImplemented => MetricErrorType::NotImplemented,
        HttpProblemKind::Internal => MetricErrorType::Internal,
    }
}

impl std::fmt::Display for HttpProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HttpProblem {}

#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
fn kind_for_status(status: StatusCode) -> HttpProblemKind {
    match status {
        StatusCode::UNAUTHORIZED => HttpProblemKind::Authentication,
        StatusCode::FORBIDDEN => HttpProblemKind::Permission,
        StatusCode::NOT_FOUND => HttpProblemKind::NotFound,
        StatusCode::TOO_MANY_REQUESTS => HttpProblemKind::RateLimit,
        StatusCode::NOT_IMPLEMENTED => HttpProblemKind::NotImplemented,
        StatusCode::SERVICE_UNAVAILABLE => HttpProblemKind::Unavailable,
        status if status.as_u16() == 499 => HttpProblemKind::Cancelled,
        status if status.as_u16() == 529 => HttpProblemKind::Overloaded,
        status if status.is_client_error() => HttpProblemKind::Validation,
        _ => HttpProblemKind::Internal,
    }
}

#[allow(dead_code)] // Used by the endpoint-integration PR later in this stack.
fn format_error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = Vec::new();
    let mut current = Some(err);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<DynamoError>() {
            parts.push(error.message().to_string());
        } else {
            parts.push(error.to_string());
        }
        current = error.source();
    }
    parts.join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::protocols::maybe_error::MaybeError;

    #[test]
    fn typed_validation_is_safe_and_unknown_errors_are_sanitized() {
        for error_type in [
            DynamoErrorType::InvalidArgument,
            DynamoErrorType::Backend(BackendError::InvalidArgument),
        ] {
            let error = DynamoError::builder()
                .error_type(error_type)
                .message("temperature must be between 0 and 2")
                .build();
            let problem = HttpProblem::from_error(&error, "request failed");
            assert_eq!(problem.kind(), HttpProblemKind::Validation);
            assert_eq!(problem.status(), StatusCode::BAD_REQUEST);
            assert_eq!(problem.message(), "temperature must be between 0 and 2");
        }

        let error = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message("panic at /srv/worker.py:42")
            .build();
        let problem = HttpProblem::from_error(&error, "request failed");
        assert_eq!(problem.kind(), HttpProblemKind::Internal);
        assert_eq!(problem.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(problem.message(), "request failed");
        assert!(!problem.message().contains("/srv/worker.py"));
    }

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
    fn nested_operational_errors_preserve_classification() {
        for (error_type, expected_kind, expected_status) in [
            (
                DynamoErrorType::ResourceExhausted,
                HttpProblemKind::Overloaded,
                overload_status_code(),
            ),
            (
                DynamoErrorType::Unavailable,
                HttpProblemKind::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                DynamoErrorType::Cancelled,
                HttpProblemKind::Cancelled,
                StatusCode::from_u16(499).unwrap(),
            ),
            (
                DynamoErrorType::Backend(BackendError::Cancelled),
                HttpProblemKind::Cancelled,
                StatusCode::from_u16(499).unwrap(),
            ),
        ] {
            let error = DynamoError::builder()
                .error_type(DynamoErrorType::Unknown)
                .message("outer wrapper")
                .cause(
                    DynamoError::builder()
                        .error_type(error_type)
                        .message("nested operational failure")
                        .build(),
                )
                .build();

            let problem = HttpProblem::from_error(&error, "request failed");
            assert_eq!(problem.kind(), expected_kind);
            assert_eq!(problem.status(), expected_status);

            let problem = HttpProblem::from_annotated(&Annotated::<()>::from_err(error)).unwrap();
            assert_eq!(problem.kind(), expected_kind);
            assert_eq!(problem.status(), expected_status);
        }
    }

    #[test]
    fn annotated_validation_uses_the_outermost_typed_error() {
        for error_type in [
            DynamoErrorType::InvalidArgument,
            DynamoErrorType::Backend(BackendError::InvalidArgument),
        ] {
            let wrapped_invalid = || {
                DynamoError::builder()
                    .error_type(DynamoErrorType::Unknown)
                    .message("outer internal failure")
                    .cause(
                        DynamoError::builder()
                            .error_type(error_type)
                            .message("nested invalid argument")
                            .build(),
                    )
                    .build()
            };

            let problem = HttpProblem::from_error(&wrapped_invalid(), "request failed");
            assert_eq!(problem.kind(), HttpProblemKind::Validation);
            assert_eq!(problem.status(), StatusCode::BAD_REQUEST);
            assert_eq!(problem.message(), "nested invalid argument");

            let problem =
                HttpProblem::from_annotated(&Annotated::<()>::from_err(wrapped_invalid())).unwrap();
            assert_eq!(problem.kind(), HttpProblemKind::Internal);
            assert_eq!(problem.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(problem.message(), INTERNAL_ERROR_MESSAGE);
        }
    }

    #[test]
    fn local_statuses_distinguish_overload_from_unavailable() {
        let overloaded = HttpProblem::from_dynamo_error(
            &DynamoError::builder()
                .error_type(DynamoErrorType::ResourceExhausted)
                .message("busy")
                .build(),
            INTERNAL_ERROR_MESSAGE,
            "busy".to_string(),
        );
        assert_eq!(overloaded.status().as_u16(), 529);
        assert_eq!(overloaded.kind(), HttpProblemKind::Overloaded);
        let unavailable = HttpProblem::from_dynamo_error(
            &DynamoError::builder()
                .error_type(DynamoErrorType::Unavailable)
                .message("down")
                .build(),
            INTERNAL_ERROR_MESSAGE,
            "down".to_string(),
        );
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(unavailable.kind(), HttpProblemKind::Unavailable);
    }

    #[test]
    fn backend_statuses_forward_client_messages_and_sanitize_everything_else() {
        let client =
            HttpProblem::from_backend_status(StatusCode::BAD_REQUEST, "bad prompt", "bad prompt");
        assert_eq!(client.status(), StatusCode::BAD_REQUEST);
        assert_eq!(client.message(), "bad prompt");

        let server = HttpProblem::from_backend_status(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker host leaked",
            "worker host leaked",
        );
        assert_eq!(server.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(server.message(), INTERNAL_ERROR_MESSAGE);

        let invalid_status = HttpProblem::from_backend_status(
            StatusCode::from_u16(399).unwrap(),
            "not an error",
            "not an error",
        );
        assert_eq!(invalid_status.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(invalid_status.message(), INTERNAL_ERROR_MESSAGE);
    }

    #[test]
    fn typed_backend_http_payload_preserves_status_policy() {
        let invalid = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::InvalidArgument))
            .message(r#"{"code":415,"message":"unsupported media type"}"#)
            .build();
        let problem = HttpProblem::from_annotated(&Annotated::<()>::from_err(invalid)).unwrap();
        assert_eq!(problem.kind(), HttpProblemKind::Validation);
        assert_eq!(problem.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(problem.message(), "unsupported media type");

        let unavailable = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message(r#"{"code":503,"message":"worker host leaked"}"#)
            .build();
        let problem = HttpProblem::from_annotated(&Annotated::<()>::from_err(unavailable)).unwrap();
        assert_eq!(problem.kind(), HttpProblemKind::Unavailable);
        assert_eq!(problem.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(problem.message(), INTERNAL_ERROR_MESSAGE);
    }

    #[test]
    fn typed_annotated_errors_reject_untrusted_json_shaped_messages() {
        let invalid = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .message(r#"{"code":500,"message":"wrong"}"#)
            .build();
        let problem = HttpProblem::from_annotated(&Annotated::<()>::from_err(invalid)).unwrap();
        assert_eq!(problem.status(), StatusCode::BAD_REQUEST);
        assert_eq!(problem.message(), r#"{"code":500,"message":"wrong"}"#);

        let unknown = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message(r#"{"code":400,"message":"secret /srv/worker"}"#)
            .build();
        let problem = HttpProblem::from_annotated(&Annotated::<()>::from_err(unknown)).unwrap();
        assert_eq!(problem.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(problem.message(), INTERNAL_ERROR_MESSAGE);
    }

    #[test]
    fn legacy_server_statuses_keep_protocol_error_types() {
        for (status, anthropic_type, openai_type) in [
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "overloaded_error",
                "service_unavailable",
            ),
            (
                StatusCode::from_u16(529).unwrap(),
                "overloaded_error",
                "service_unavailable",
            ),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "internal_server_error",
            ),
        ] {
            let error = SanitizedError::PreserveServerError(status);
            assert_eq!(error.anthropic_type(), anthropic_type);
            assert_eq!(error.openai_type_slug(), openai_type);
        }
    }

    #[test]
    fn legacy_backend_status_classification_remains_compatible() {
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(499).unwrap()),
            Some(SanitizedError::Cancelled)
        ));
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::SERVICE_UNAVAILABLE),
            Some(SanitizedError::PreserveServerError(status))
                if status == StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(SanitizedError::for_backend_status(StatusCode::BAD_REQUEST).is_none());
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(399).unwrap()),
            Some(SanitizedError::Internal)
        ));
    }
}
