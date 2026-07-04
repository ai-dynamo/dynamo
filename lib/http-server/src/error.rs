// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use serde::Serialize;
use thiserror::Error;

/// Maximum number of leading annotation frames buffered while looking for a
/// backend error. This bounds per-request memory if a backend emits only
/// annotations.
const MAX_LEADING_ANNOTATIONS: usize = 16;

pub fn overload_status_code() -> StatusCode {
    StatusCode::from_u16(529).expect("529 is a valid HTTP status code")
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

/// Checks whether an annotated event carries a backend error.
///
/// Returns the HTTP status and backend message. Callers remain responsible for
/// applying their protocol-specific sanitization and response envelope.
pub fn extract_backend_error_if_present<T: Serialize>(
    event: &Annotated<T>,
) -> Option<(StatusCode, String)> {
    #[derive(serde::Deserialize)]
    struct ErrorPayload {
        message: Option<String>,
        code: Option<u16>,
    }

    if event.event.as_deref() == Some("error") {
        // Prefer the typed error chain. `DynamoError::message()` omits the
        // error-type prefix, preserving JSON error payloads for parsing.
        let error_str = if let Some(ref dynamo_err) = event.error {
            let mut parts = Vec::new();
            let mut current: Option<&dyn std::error::Error> = Some(dynamo_err);
            while let Some(error) = current {
                if let Some(error) = error.downcast_ref::<dynamo_runtime::error::DynamoError>() {
                    parts.push(error.message().to_string());
                } else {
                    parts.push(error.to_string());
                }
                current = error.source();
            }
            parts.join(", ")
        } else {
            event
                .comment
                .as_ref()
                .map(|comments| comments.join(", "))
                .unwrap_or_else(|| "Unknown error".to_string())
        };

        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&error_str) {
            let status = error_payload
                .code
                .and_then(|code| StatusCode::from_u16(code).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return Some((status, error_payload.message.unwrap_or(error_str)));
        }

        return Some((StatusCode::INTERNAL_SERVER_ERROR, error_str));
    }

    if let Some(data) = &event.data
        && let Ok(json_value) = serde_json::to_value(data)
        && let Ok(error_payload) = serde_json::from_value::<ErrorPayload>(json_value.clone())
        && let Some(code) = error_payload.code
        && code >= 400
    {
        let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let message = error_payload
            .message
            .unwrap_or_else(|| json_value.to_string());
        return Some((status, message));
    }

    if let Some(comments) = &event.comment
        && !comments.is_empty()
    {
        let comment = comments.join(", ");
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(&comment)
            && let Some(code) = error_payload.code
            && code >= 400
        {
            let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return Some((status, error_payload.message.unwrap_or(comment)));
        }

        if event.data.is_none() && event.event.is_none() {
            return Some((StatusCode::INTERNAL_SERVER_ERROR, comment));
        }
    }

    None
}

fn is_annotation_frame<T>(event: &Annotated<T>) -> bool {
    event.data.is_none()
        && event.error.is_none()
        && matches!(event.event.as_deref(), Some(tag) if tag != "error")
}

/// Inspects the first non-annotation stream item for a backend error.
///
/// On success, the returned stream replays all buffered items in their original
/// order. On failure, the raw backend detail is returned for logging and must be
/// sanitized by the protocol boundary before it is sent to a client.
pub async fn check_for_backend_error<T, S>(
    mut stream: S,
) -> Result<impl Stream<Item = Annotated<T>> + Send, (StatusCode, String)>
where
    T: Serialize + Send + 'static,
    S: Stream<Item = Annotated<T>> + Send + Unpin + 'static,
{
    let mut buffered = Vec::new();
    while let Some(event) = stream.next().await {
        if is_annotation_frame(&event) && buffered.len() < MAX_LEADING_ANNOTATIONS {
            buffered.push(event);
            continue;
        }
        if let Some(error) = extract_backend_error_if_present(&event) {
            return Err(error);
        }

        buffered.push(event);
        break;
    }
    Ok(futures::stream::iter(buffered).chain(stream))
}

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
    use futures::stream;

    fn annotation(index: usize) -> Annotated<serde_json::Value> {
        Annotated::from_annotation("request_id", &format!("request-{index}"))
            .expect("annotation should serialize")
    }

    fn error_event(message: &str, code: u16) -> Annotated<serde_json::Value> {
        Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![
                serde_json::json!({
                    "message": message,
                    "code": code,
                })
                .to_string(),
            ]),
            error: None,
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
        let err = SanitizedError::PreserveServerError(StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.anthropic_type(), "overloaded_error");
        assert_eq!(err.openai_type_slug(), "service_unavailable");
    }

    #[test]
    fn preserve_server_error_529_maps_to_overload_types() {
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
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(499).unwrap()),
            Some(SanitizedError::Cancelled)
        ));
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::SERVICE_UNAVAILABLE),
            Some(SanitizedError::PreserveServerError(s)) if s == StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(SanitizedError::for_backend_status(StatusCode::BAD_REQUEST).is_none());
        assert!(SanitizedError::for_backend_status(StatusCode::NOT_FOUND).is_none());
        assert!(matches!(
            SanitizedError::for_backend_status(StatusCode::from_u16(399).unwrap()),
            Some(SanitizedError::Internal)
        ));
    }

    #[tokio::test]
    async fn backend_error_inspection_returns_raw_detail() {
        let result =
            check_for_backend_error(stream::iter(vec![error_event("worker detail", 503)])).await;

        let (status, message) = match result {
            Ok(_) => panic!("error event should be detected"),
            Err(error) => error,
        };
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(message, "worker detail");
    }

    #[tokio::test]
    async fn backend_error_inspection_replays_annotations_in_order() {
        let first = annotation(0);
        let second = Annotated::from_data(serde_json::json!({"token": "ok"}));
        let returned = check_for_backend_error(stream::iter(vec![first, second]))
            .await
            .expect("normal stream should pass");
        let returned: Vec<_> = returned.collect().await;

        assert_eq!(returned.len(), 2);
        assert_eq!(returned[0].event.as_deref(), Some("request_id"));
        assert_eq!(returned[1].data, Some(serde_json::json!({"token": "ok"})));
    }

    #[tokio::test]
    async fn backend_error_inspection_bounds_annotation_buffer() {
        let mut events: Vec<_> = (0..=MAX_LEADING_ANNOTATIONS).map(annotation).collect();
        events.push(error_event("after cap", 500));

        let returned = check_for_backend_error(stream::iter(events))
            .await
            .expect("scanner stops after the bounded annotation prefix");
        let returned: Vec<_> = returned.collect().await;

        assert_eq!(returned.len(), MAX_LEADING_ANNOTATIONS + 2);
        for (index, event) in returned[..=MAX_LEADING_ANNOTATIONS].iter().enumerate() {
            assert_eq!(event.event.as_deref(), Some("request_id"));
            assert_eq!(
                event.comment.as_deref(),
                Some([format!("\"request-{index}\"")].as_slice())
            );
        }
        assert_eq!(returned.last().unwrap().event.as_deref(), Some("error"));
    }
}
