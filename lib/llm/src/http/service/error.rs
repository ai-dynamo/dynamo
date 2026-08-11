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

/// Protocol-neutral error categories used for metrics and protocol rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HttpErrorKind {
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
pub(crate) struct ClassifiedHttpError {
    kind: HttpErrorKind,
    status: StatusCode,
    message: String,
    diagnostic: String,
    details: Option<Box<Value>>,
}

pub(crate) fn drain_converted_stream_events<'a, T>(
    events: &'a mut Vec<Result<T, anyhow::Error>>,
    protocol: &'a str,
) -> impl Iterator<Item = Result<T, ClassifiedHttpError>> + 'a {
    events.drain(..).map(move |event| {
        event.map_err(|error| {
            ClassifiedHttpError::internal(
                INTERNAL_ERROR_MESSAGE,
                format!("{protocol} stream event conversion failed: {error:#}"),
            )
        })
    })
}

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
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

impl ClassifiedHttpError {
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
                    kind: HttpErrorKind::Overloaded,
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
    /// Ordinary data and tagged annotation metadata are never interpreted as
    /// errors.
    pub(crate) fn from_annotated<T>(event: &Annotated<T>) -> Option<Self> {
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

            return Some(Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic));
        }

        None
    }

    /// Preserve the untagged comment error shape recognized by the pre-commit
    /// stream check without interpreting ordinary mid-stream comments as errors.
    pub(crate) fn from_precommit_annotated<T>(event: &Annotated<T>) -> Option<Self> {
        if let Some(problem) = Self::from_annotated(event) {
            return Some(problem);
        }
        if event.data.is_none()
            && event.event.is_none()
            && let Some(comments) = event.comment.as_ref()
            && !comments.is_empty()
        {
            let diagnostic = comments.join(", ");
            return Some(Self::internal(INTERNAL_ERROR_MESSAGE, diagnostic));
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
                HttpErrorKind::Cancelled,
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
            kind: HttpErrorKind::Internal,
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            diagnostic: diagnostic.into(),
            details: None,
        }
    }

    fn from_dynamo_error(error: &DynamoError, internal_message: &str, diagnostic: String) -> Self {
        if let Some((status, message)) = backend_http_metadata(error) {
            return Self::from_backend_status(status, message, diagnostic);
        }
        if error.http_error().is_some() {
            return Self::internal(internal_message, diagnostic);
        }

        match error.error_type() {
            DynamoErrorType::InvalidArgument
            | DynamoErrorType::Backend(BackendError::InvalidArgument) => Self {
                kind: HttpErrorKind::Validation,
                status: StatusCode::BAD_REQUEST,
                message: error.message().to_string(),
                diagnostic,
                details: None,
            },
            DynamoErrorType::ResourceExhausted | DynamoErrorType::WorkerOverloaded => {
                Self::classified(
                    HttpErrorKind::Overloaded,
                    overload_status_code(),
                    "Service temporarily overloaded",
                    diagnostic,
                )
            }
            DynamoErrorType::Unavailable => Self::classified(
                HttpErrorKind::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable",
                diagnostic,
            ),
            DynamoErrorType::Cancelled | DynamoErrorType::Backend(BackendError::Cancelled) => {
                Self::classified(
                    HttpErrorKind::Cancelled,
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
                HttpErrorKind::Cancelled,
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
        kind: HttpErrorKind,
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

    pub(crate) fn kind(&self) -> HttpErrorKind {
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

    pub(crate) fn details(&self) -> Option<&Value> {
        self.details.as_deref()
    }

    pub(crate) fn metric_type(&self) -> MetricErrorType {
        metric_type_for_kind(self.kind)
    }

    pub(crate) fn metric_type_for_status(status: StatusCode) -> MetricErrorType {
        metric_type_for_kind(kind_for_status(status))
    }
}

fn backend_http_metadata(error: &DynamoError) -> Option<(StatusCode, String)> {
    let http_error = error.http_error()?;
    validate_backend_http_metadata(error, http_error.code(), http_error.message())
}

fn validate_backend_http_metadata(
    error: &DynamoError,
    code: u16,
    message: &str,
) -> Option<(StatusCode, String)> {
    let status = StatusCode::from_u16(code).ok()?;

    match error.error_type() {
        DynamoErrorType::Backend(BackendError::InvalidArgument) if status.is_client_error() => {}
        DynamoErrorType::Backend(BackendError::Unknown) if status.is_server_error() => {}
        _ => return None,
    }

    Some((status, message.to_string()))
}

fn select_dynamo_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
    validation_scope: ValidationScope,
) -> Option<&'a DynamoError> {
    const OPERATIONAL: &[DynamoErrorType] = &[
        DynamoErrorType::ResourceExhausted,
        DynamoErrorType::WorkerOverloaded,
        DynamoErrorType::Unavailable,
    ];
    const VALIDATION: &[DynamoErrorType] = &[
        DynamoErrorType::InvalidArgument,
        DynamoErrorType::Backend(BackendError::InvalidArgument),
    ];
    const CANCELLATION: &[DynamoErrorType] = &[
        DynamoErrorType::Cancelled,
        DynamoErrorType::Backend(BackendError::Cancelled),
    ];

    // Preserve operational signals through wrappers before considering the
    // validation scope selected by the calling boundary.
    find_dynamo_error_in_chain(err, OPERATIONAL)
        .or_else(|| match validation_scope {
            ValidationScope::Chain => find_dynamo_error_in_chain(err, VALIDATION),
            ValidationScope::Outermost => find_dynamo_error_in_chain(err, &[])
                .filter(|error| VALIDATION.contains(&error.error_type())),
        })
        .or_else(|| find_dynamo_error_in_chain(err, CANCELLATION))
}

/// Search error types in preference order. An empty slice returns the
/// outermost `DynamoError` in the chain.
fn find_dynamo_error_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
    error_types: &[DynamoErrorType],
) -> Option<&'a DynamoError> {
    if error_types.is_empty() {
        let mut current = Some(err);
        while let Some(error) = current {
            if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
                return Some(dynamo_error);
            }
            current = error.source();
        }
        return None;
    }

    for error_type in error_types {
        let mut current = Some(err);
        while let Some(error) = current {
            if let Some(dynamo_error) = error.downcast_ref::<DynamoError>()
                && dynamo_error.error_type() == *error_type
            {
                return Some(dynamo_error);
            }
            current = error.source();
        }
    }
    None
}

fn metric_type_for_kind(kind: HttpErrorKind) -> MetricErrorType {
    match kind {
        HttpErrorKind::Validation | HttpErrorKind::Authentication | HttpErrorKind::Permission => {
            MetricErrorType::Validation
        }
        HttpErrorKind::NotFound => MetricErrorType::NotFound,
        HttpErrorKind::RateLimit | HttpErrorKind::Overloaded => MetricErrorType::Overload,
        HttpErrorKind::Unavailable => MetricErrorType::Unavailable,
        HttpErrorKind::Cancelled => MetricErrorType::Cancelled,
        HttpErrorKind::NotImplemented => MetricErrorType::NotImplemented,
        HttpErrorKind::Internal => MetricErrorType::Internal,
    }
}

impl std::fmt::Display for ClassifiedHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ClassifiedHttpError {}

fn kind_for_status(status: StatusCode) -> HttpErrorKind {
    match status {
        StatusCode::UNAUTHORIZED => HttpErrorKind::Authentication,
        StatusCode::FORBIDDEN => HttpErrorKind::Permission,
        StatusCode::NOT_FOUND => HttpErrorKind::NotFound,
        StatusCode::TOO_MANY_REQUESTS => HttpErrorKind::RateLimit,
        StatusCode::NOT_IMPLEMENTED => HttpErrorKind::NotImplemented,
        StatusCode::SERVICE_UNAVAILABLE => HttpErrorKind::Unavailable,
        status if status.as_u16() == 499 => HttpErrorKind::Cancelled,
        status if status.as_u16() == 529 => HttpErrorKind::Overloaded,
        status if status.is_client_error() => HttpErrorKind::Validation,
        _ => HttpErrorKind::Internal,
    }
}

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
    fn stream_event_conversion_failures_are_classified_without_reallocation() {
        let mut events: Vec<Result<(), anyhow::Error>> = Vec::with_capacity(4);
        events.push(Err(anyhow::anyhow!("serializer failed")));
        let capacity = events.capacity();

        let problem = {
            let mut drained = drain_converted_stream_events(&mut events, "Responses");
            drained.next().unwrap().unwrap_err()
        };

        assert!(events.is_empty());
        assert_eq!(events.capacity(), capacity);
        assert_eq!(problem.kind(), HttpErrorKind::Internal);
        assert_eq!(problem.message(), INTERNAL_ERROR_MESSAGE);
        assert_eq!(
            problem.diagnostic(),
            "Responses stream event conversion failed: serializer failed"
        );
    }

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
            let classified = ClassifiedHttpError::from_error(&error, "request failed");
            assert_eq!(classified.kind(), HttpErrorKind::Validation);
            assert_eq!(classified.status(), StatusCode::BAD_REQUEST);
            assert_eq!(classified.message(), "temperature must be between 0 and 2");
        }

        let error = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message("panic at /srv/worker.py:42")
            .build();
        let classified = ClassifiedHttpError::from_error(&error, "request failed");
        assert_eq!(classified.kind(), HttpErrorKind::Internal);
        assert_eq!(classified.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(classified.message(), "request failed");
        assert!(!classified.message().contains("/srv/worker.py"));
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
                HttpErrorKind::Overloaded,
                overload_status_code(),
            ),
            (
                DynamoErrorType::Unavailable,
                HttpErrorKind::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                DynamoErrorType::Cancelled,
                HttpErrorKind::Cancelled,
                StatusCode::from_u16(499).unwrap(),
            ),
            (
                DynamoErrorType::Backend(BackendError::Cancelled),
                HttpErrorKind::Cancelled,
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

            let classified = ClassifiedHttpError::from_error(&error, "request failed");
            assert_eq!(classified.kind(), expected_kind);
            assert_eq!(classified.status(), expected_status);

            let classified =
                ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(error)).unwrap();
            assert_eq!(classified.kind(), expected_kind);
            assert_eq!(classified.status(), expected_status);
        }
    }

    #[test]
    fn nested_operational_errors_use_type_precedence() {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::Unavailable)
            .message("outer unavailable")
            .cause(
                DynamoError::builder()
                    .error_type(DynamoErrorType::ResourceExhausted)
                    .message("nested overload")
                    .build(),
            )
            .build();

        let classified = ClassifiedHttpError::from_error(&error, "request failed");
        assert_eq!(classified.kind(), HttpErrorKind::Overloaded);
        assert_eq!(classified.status(), overload_status_code());
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

            let classified = ClassifiedHttpError::from_error(&wrapped_invalid(), "request failed");
            assert_eq!(classified.kind(), HttpErrorKind::Validation);
            assert_eq!(classified.status(), StatusCode::BAD_REQUEST);
            assert_eq!(classified.message(), "nested invalid argument");

            let classified =
                ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(wrapped_invalid()))
                    .unwrap();
            assert_eq!(classified.kind(), HttpErrorKind::Internal);
            assert_eq!(classified.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(classified.message(), INTERNAL_ERROR_MESSAGE);
        }
    }

    #[test]
    fn http_json_in_annotation_metadata_is_not_classified_as_an_error() {
        let event = Annotated::<()> {
            data: None,
            id: None,
            event: Some("legacy.backend.annotation".to_string()),
            comment: Some(vec![
                r#"{"code":400,"message":"bad prompt","type":"Bad Request"}"#.to_string(),
            ]),
            error: None,
        };

        assert!(ClassifiedHttpError::from_annotated(&event).is_none());
    }

    #[test]
    fn untagged_comments_are_errors_only_during_precommit_inspection() {
        let event = Annotated::<()> {
            data: None,
            id: None,
            event: None,
            comment: Some(vec!["Connection timeout".to_string()]),
            error: None,
        };

        assert!(ClassifiedHttpError::from_annotated(&event).is_none());
        let problem = ClassifiedHttpError::from_precommit_annotated(&event).unwrap();
        assert_eq!(problem.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(problem.message(), INTERNAL_ERROR_MESSAGE);
        assert_eq!(problem.diagnostic(), "Connection timeout");
    }

    #[test]
    fn http_looking_data_is_not_classified_as_an_error() {
        let data = Annotated::from_data(serde_json::json!({
            "code": 415,
            "message": "unsupported media type",
        }));
        assert!(ClassifiedHttpError::from_annotated(&data).is_none());
    }

    #[test]
    fn local_statuses_distinguish_overload_from_unavailable() {
        let overloaded = ClassifiedHttpError::from_dynamo_error(
            &DynamoError::builder()
                .error_type(DynamoErrorType::ResourceExhausted)
                .message("busy")
                .build(),
            INTERNAL_ERROR_MESSAGE,
            "busy".to_string(),
        );
        assert_eq!(overloaded.status(), overload_status_code());
        assert_eq!(overloaded.kind(), HttpErrorKind::Overloaded);
        let unavailable = ClassifiedHttpError::from_dynamo_error(
            &DynamoError::builder()
                .error_type(DynamoErrorType::Unavailable)
                .message("down")
                .build(),
            INTERNAL_ERROR_MESSAGE,
            "down".to_string(),
        );
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(unavailable.kind(), HttpErrorKind::Unavailable);
    }

    #[test]
    fn backend_statuses_forward_client_messages_and_sanitize_everything_else() {
        let client = ClassifiedHttpError::from_backend_status(
            StatusCode::BAD_REQUEST,
            "bad prompt",
            "bad prompt",
        );
        assert_eq!(client.status(), StatusCode::BAD_REQUEST);
        assert_eq!(client.message(), "bad prompt");

        let server = ClassifiedHttpError::from_backend_status(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker host leaked",
            "worker host leaked",
        );
        assert_eq!(server.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(server.message(), INTERNAL_ERROR_MESSAGE);

        let invalid_status = ClassifiedHttpError::from_backend_status(
            StatusCode::from_u16(399).unwrap(),
            "not an error",
            "not an error",
        );
        assert_eq!(invalid_status.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(invalid_status.message(), INTERNAL_ERROR_MESSAGE);
    }

    #[test]
    fn typed_backend_http_metadata_preserves_status_policy() {
        let invalid = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::InvalidArgument))
            .message("backend diagnostic")
            .http_error(415, "unsupported media type")
            .build();
        let classified =
            ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(invalid)).unwrap();
        assert_eq!(classified.kind(), HttpErrorKind::Validation);
        assert_eq!(classified.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(classified.message(), "unsupported media type");

        let unavailable = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message("backend diagnostic")
            .http_error(503, "worker host leaked")
            .build();
        let classified =
            ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(unavailable)).unwrap();
        assert_eq!(classified.kind(), HttpErrorKind::Unavailable);
        assert_eq!(classified.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(classified.message(), INTERNAL_ERROR_MESSAGE);

        let inconsistent = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::InvalidArgument))
            .message("secret diagnostic")
            .http_error(503, "secret public message")
            .build();
        let classified =
            ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(inconsistent)).unwrap();
        assert_eq!(classified.kind(), HttpErrorKind::Internal);
        assert_eq!(classified.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(classified.message(), INTERNAL_ERROR_MESSAGE);
    }

    #[test]
    fn typed_annotated_errors_reject_untrusted_json_shaped_messages() {
        let invalid = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .message(r#"{"code":500,"message":"wrong"}"#)
            .build();
        let classified =
            ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(invalid)).unwrap();
        assert_eq!(classified.status(), StatusCode::BAD_REQUEST);
        assert_eq!(classified.message(), r#"{"code":500,"message":"wrong"}"#);

        let unknown = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .message(r#"{"code":400,"message":"secret /srv/worker"}"#)
            .build();
        let classified =
            ClassifiedHttpError::from_annotated(&Annotated::<()>::from_err(unknown)).unwrap();
        assert_eq!(classified.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(classified.message(), INTERNAL_ERROR_MESSAGE);
    }
}
