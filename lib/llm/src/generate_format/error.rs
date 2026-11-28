// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;

use dynamo_error::http::SanitizedError;
use dynamo_http_server::metrics::{
    ErrorType, REQUEST_DEADLINE_EXCEEDED_MESSAGE, request_deadline_exceeded, request_was_cancelled,
    request_was_rejected, request_was_unavailable,
};

use crate::http::service::error::first_dynamo_error;

/// Whether dispatch failed or a returned response stream failed.
#[derive(Clone, Copy)]
pub enum ErrorPhase {
    Dispatch,
    Stream { cancelled: bool },
}

/// Frontend-authored defaults kept separate from a format's mutable message.
enum TrustedFallback {
    Static(&'static str),
    Sanitized(SanitizedError),
}

impl TrustedFallback {
    fn message(&self) -> String {
        match self {
            Self::Static(message) => (*message).to_string(),
            Self::Sanitized(error) => error.to_string(),
        }
    }
}

/// A format's public worker error and its request-metric category.
pub struct WorkerError {
    pub kind: ErrorType,
    pub status: StatusCode,
    pub message: String,
    fallback: Option<TrustedFallback>,
}

impl WorkerError {
    pub fn new(kind: ErrorType, status: StatusCode, message: impl ToString) -> Self {
        Self {
            kind,
            status,
            message: message.to_string(),
            fallback: None,
        }
    }

    /// Construct an error with an explicitly frontend-authored static fallback.
    pub fn new_static(kind: ErrorType, status: StatusCode, message: &'static str) -> Self {
        Self {
            fallback: Some(TrustedFallback::Static(message)),
            ..Self::new(kind, status, message)
        }
    }

    /// Construct an error with an existing sanitized frontend fallback.
    pub fn new_sanitized(kind: ErrorType, status: StatusCode, error: SanitizedError) -> Self {
        Self {
            fallback: Some(TrustedFallback::Sanitized(error)),
            ..Self::new(kind, status, error)
        }
    }

    /// Apply frontend disclosure while preserving the format's status and classification.
    pub(crate) fn with_client_policy(mut self, error: &(dyn std::error::Error + 'static)) -> Self {
        use crate::http::service::error::{Protocol, backend_http_error_class, public_error_body};

        if let Some(error) = first_dynamo_error(error) {
            let body = public_error_body(error, Protocol::OpenAi);
            self.message = body.message.filter(|_| !body.blocked).unwrap_or_else(|| {
                if let Some(fallback) = &self.fallback {
                    return fallback.message();
                }
                // Formats may select a different status from the source's class.
                let class = if self.kind == ErrorType::Overload {
                    dynamo_runtime::error::ErrorClass::CapacityExhausted
                } else if self.status == StatusCode::INTERNAL_SERVER_ERROR {
                    dynamo_runtime::error::ErrorClass::Internal
                } else {
                    backend_http_error_class(self.status)
                };
                match dynamo_error::http::http_action_for_class(class) {
                    dynamo_error::http::ClientErrorAction::Respond { public_message, .. } => {
                        public_message.to_string()
                    }
                    dynamo_error::http::ClientErrorAction::NoDelivery => {
                        "Request cancelled".to_string()
                    }
                }
            });
        }
        self
    }

    /// Public validation detail from the canonical error chain, with private
    /// diagnostics redacted. Used by formats following the OpenAI policy.
    pub fn canonical_invalid_request_message(
        error: &(dyn std::error::Error + 'static),
    ) -> Option<String> {
        use crate::http::service::error::{Protocol, public_message_for};

        dynamo_error::http::find_canonical_error_in_chain(error)
            .filter(|error| error.class() == dynamo_runtime::error::ErrorClass::InvalidRequest)
            .map(|canonical| {
                public_message_for(
                    first_dynamo_error(error).unwrap_or(canonical),
                    Protocol::OpenAi,
                )
                .unwrap_or_else(|| "Invalid request".to_string())
            })
    }

    /// Common pre-stream policy of the native Generate endpoints. The format
    /// supplies its validation message; transport failures take precedence.
    /// Dispatch counts `Overload` as an unavailable request plus a rejection.
    pub fn dispatch(error: &(dyn std::error::Error + 'static), invalid: Option<String>) -> Self {
        if request_deadline_exceeded(error) {
            Self::new_static(
                ErrorType::Cancelled,
                StatusCode::TOO_MANY_REQUESTS,
                REQUEST_DEADLINE_EXCEEDED_MESSAGE,
            )
        } else if request_was_cancelled(error) {
            Self::new_static(
                ErrorType::Cancelled,
                StatusCode::from_u16(499).unwrap(),
                "request was cancelled",
            )
        } else if request_was_rejected(error) {
            Self::new_static(
                ErrorType::Overload,
                StatusCode::SERVICE_UNAVAILABLE,
                "engine rejected the request",
            )
        } else if request_was_unavailable(error) {
            Self::new_sanitized(
                ErrorType::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
                SanitizedError::Unavailable,
            )
        } else if let Some(message) = invalid {
            Self::new(ErrorType::Validation, StatusCode::BAD_REQUEST, message)
        } else {
            Self::new_static(
                ErrorType::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::{DynamoError, ErrorClass, ErrorReason, UpstreamError};

    #[test]
    fn native_generate_body_policy_forwards_unknown_without_changing_status_or_kind() {
        let error = DynamoError::builder()
            .upstream_error(UpstreamError::new("unfamiliar engine failure"))
            .diagnostic("PRIVATE_DYNAMO_WRAPPER")
            .build();
        let selected = WorkerError::new(ErrorType::Internal, StatusCode::IM_A_TEAPOT, "plugin")
            .with_client_policy(&error);
        assert_eq!(selected.message, "unfamiliar engine failure");
        assert_eq!(selected.status, StatusCode::IM_A_TEAPOT);
        assert_eq!(selected.kind, ErrorType::Internal);
    }

    #[test]
    fn native_generate_body_policy_masks_public_copies_and_keeps_private_fallback_private() {
        let private = "PRIVATE_MEDIA_URL_AND_CONNECTION_DETAIL";
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new(private).family("media_fetch"))
            .public_message(private)
            .build();
        let selected = WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, private)
            .with_client_policy(&error);
        assert_eq!(selected.message, "Invalid request");

        let legacy = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic(private)
            .build();
        let selected = WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, private)
            .with_client_policy(&legacy);
        assert_eq!(selected.message, "Invalid request");
    }

    #[test]
    fn trusted_fallback_is_independent_of_replaced_messages() {
        let private = "PRIVATE_PLUGIN_OR_MEDIA_DETAIL";
        let absent = DynamoError::builder().diagnostic(private).build();
        let blocked = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new(private).family("media_fetch"))
            .public_message(private)
            .build();
        for source in [absent, blocked] {
            for (mut selected, expected) in [
                (
                    WorkerError::new_static(
                        ErrorType::Cancelled,
                        StatusCode::TOO_MANY_REQUESTS,
                        REQUEST_DEADLINE_EXCEEDED_MESSAGE,
                    ),
                    REQUEST_DEADLINE_EXCEEDED_MESSAGE,
                ),
                (
                    WorkerError::new_sanitized(
                        ErrorType::Unavailable,
                        StatusCode::SERVICE_UNAVAILABLE,
                        SanitizedError::Unavailable,
                    ),
                    "Service temporarily unavailable",
                ),
            ] {
                let status = selected.status;
                let kind = selected.kind.clone();
                selected.message = private.to_string();
                let selected = selected.with_client_policy(&source);
                assert_eq!(selected.message, expected);
                assert_eq!(selected.status, status);
                assert_eq!(selected.kind, kind);
            }
        }
    }

    #[test]
    fn approved_public_and_received_messages_precede_trusted_fallbacks() {
        let received = "engine validation explanation";
        for (source, expected) in [
            (
                DynamoError::builder()
                    .upstream_error(UpstreamError::new(received))
                    .build(),
                received,
            ),
            (
                DynamoError::builder()
                    .upstream_error(UpstreamError::new(received))
                    .public_message("authored public replacement")
                    .build(),
                "authored public replacement",
            ),
        ] {
            for selected in [
                WorkerError::new_static(
                    ErrorType::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error",
                ),
                WorkerError::new_sanitized(
                    ErrorType::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    SanitizedError::Internal,
                ),
            ] {
                let selected = selected.with_client_policy(&source);
                assert_eq!(selected.message, expected);
                assert_eq!(selected.status, StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(selected.kind, ErrorType::Internal);
            }
        }
    }

    #[test]
    fn native_generate_body_policy_stops_at_invalid_outer_identity() {
        let source = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("MUST_NOT_RECOVER_CAUSE"))
            .build();
        let invalid = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .reason(ErrorReason::new("request.rate_limited").unwrap())
            .cause(source)
            .build();
        let selected = WorkerError::new(
            ErrorType::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            "plugin",
        )
        .with_client_policy(&invalid);
        assert_eq!(selected.message, "Internal server error");
        assert_eq!(
            WorkerError::canonical_invalid_request_message(&invalid),
            None
        );
    }

    #[test]
    fn native_generate_masked_and_absent_messages_use_the_selected_status_fallback() {
        let blocked = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("PRIVATE_MEDIA_SOURCE").family("media_fetch"))
            .build();
        let absent = DynamoError::builder()
            .class(ErrorClass::BackendProtocol)
            .diagnostic("missing opaque SGLang response")
            .build();
        for source in [blocked, absent] {
            let selected = WorkerError::new(
                ErrorType::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "PRIVATE_PLUGIN_DIAGNOSTIC",
            )
            .with_client_policy(&source);
            assert_eq!(selected.message, "Internal server error");
            assert_eq!(selected.status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(selected.kind, ErrorType::Internal);
        }
    }

    #[test]
    fn received_http_status_keeps_format_owned_status_and_metric_kind() {
        for status in [408, 418, 422, 431, 503, 529] {
            let source = DynamoError::builder()
                .class(ErrorClass::InvalidRequest)
                .upstream_error(UpstreamError::new("engine explanation").http_status(status))
                .build();
            let selected =
                WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, "plugin")
                    .with_client_policy(&source);
            assert_eq!(selected.status, StatusCode::BAD_REQUEST);
            assert_eq!(selected.kind, ErrorType::Validation);
        }
        let native_code = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("native engine code").code("503"))
            .build();
        assert_eq!(
            WorkerError::new(ErrorType::Validation, StatusCode::BAD_REQUEST, "plugin")
                .with_client_policy(&native_code)
                .status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn local_cancellation_and_queue_deadline_keep_their_own_status() {
        let source = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("source response").http_status(422))
            .build();
        for status in [
            StatusCode::from_u16(499).unwrap(),
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            let selected = WorkerError::new(ErrorType::Cancelled, status, "local cancellation")
                .with_client_policy(&source);
            assert_eq!(selected.status, status);
            assert_eq!(selected.kind, ErrorType::Cancelled);
        }
        let local_499 = WorkerError::new(
            ErrorType::Validation,
            StatusCode::from_u16(499).unwrap(),
            "plugin-local cancellation",
        )
        .with_client_policy(&source);
        assert_eq!(local_499.status.as_u16(), 499);
    }
}
