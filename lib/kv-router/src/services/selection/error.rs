// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::scheduling::KvSchedulerError;
use crate::sequences::SequenceError;

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotReady(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Internal(String),
    #[error(transparent)]
    Scheduler(#[from] KvSchedulerError),
    #[error(transparent)]
    Sequence(#[from] SequenceError),
}

impl SelectionError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotReady(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Scheduler(error) => scheduler_error_status(error),
            Self::Sequence(error) => sequence_error_status(error),
        }
    }

    /// HTTP-style status code for this error, for callers that consume the
    /// service in-process without an HTTP layer.
    pub fn status_code(&self) -> u16 {
        self.status().as_u16()
    }

    /// Stable, machine-readable category for this error.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::NotReady(_) => "not_ready",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::Internal(_) => "internal",
            Self::Scheduler(_) => "scheduler",
            Self::Sequence(_) => "sequence",
        }
    }
}

fn scheduler_error_status(error: &KvSchedulerError) -> StatusCode {
    match error {
        KvSchedulerError::NoEndpoints
        | KvSchedulerError::AllEligibleWorkersFiltered
        | KvSchedulerError::SubscriberShutdown
        | KvSchedulerError::InitFailed(_) => StatusCode::SERVICE_UNAVAILABLE,
        KvSchedulerError::WorkerSelectionPolicy(_) => StatusCode::INTERNAL_SERVER_ERROR,
        // Deadline expiry is deliberately 429, not 504: the deadline elapsed
        // while waiting for capacity, so it is backpressure the client should
        // respond to like the overloaded family, not a gateway timeout.
        KvSchedulerError::AllEligibleWorkersOverloaded
        | KvSchedulerError::PinnedWorkerOverloaded { .. }
        | KvSchedulerError::DeadlineExceeded => StatusCode::TOO_MANY_REQUESTS,
        KvSchedulerError::QueueRejected(_) => StatusCode::SERVICE_UNAVAILABLE,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => StatusCode::BAD_REQUEST,
        // A duplicate live request id is caller-induced, like `BookingFailed`
        // and `SequenceError::DuplicateRequest`.
        KvSchedulerError::BookingFailed(_)
        | KvSchedulerError::DuplicateClassificationRequestId(_) => StatusCode::CONFLICT,
        KvSchedulerError::RequestClassifierPanicked(_)
        | KvSchedulerError::RequestClassifierFailed(_)
        | KvSchedulerError::InvalidClassificationMetadata(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn sequence_error_status(error: &SequenceError) -> StatusCode {
    match error {
        SequenceError::WorkerNotFound { .. } | SequenceError::RequestNotFound { .. } => {
            StatusCode::NOT_FOUND
        }
        SequenceError::DuplicateRequest { .. } => StatusCode::CONFLICT,
        SequenceError::ReplicaSyncPublishFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

impl IntoResponse for SelectionError {
    fn into_response(self) -> Response {
        if matches!(
            &self,
            Self::Scheduler(
                KvSchedulerError::RequestClassifierPanicked(_)
                    | KvSchedulerError::RequestClassifierFailed(_)
            )
        ) {
            return (
                self.status(),
                Json(serde_json::json!({"error": "request classifier failed"})),
            )
                .into_response();
        }
        if let Self::Scheduler(KvSchedulerError::QueueRejected(rejection)) = &self {
            return (
                self.status(),
                Json(serde_json::json!({
                    "error": self.to_string(),
                    "details": rejection,
                })),
            )
                .into_response();
        }

        (
            self.status(),
            Json(serde_json::json!({"error": self.to_string()})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("private plugin detail")]
    struct PrivateClassifierError;

    #[test]
    fn filtered_workers_are_unavailable_not_overloaded() {
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersFiltered).status_code(),
            StatusCode::SERVICE_UNAVAILABLE.as_u16()
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersOverloaded).status_code(),
            StatusCode::TOO_MANY_REQUESTS.as_u16()
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded).status_code(),
            StatusCode::TOO_MANY_REQUESTS.as_u16()
        );
    }

    #[tokio::test]
    async fn classifier_error_response_is_sanitized() {
        let response = SelectionError::Scheduler(KvSchedulerError::RequestClassifierFailed(
            std::sync::Arc::new(PrivateClassifierError),
        ))
        .into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), br#"{"error":"request classifier failed"}"#);
    }
}
