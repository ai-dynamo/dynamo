// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::indexer::KvRouterError;
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
    /// The KV index lookup failed; an offline remote index is not ready,
    /// anything else is internal.
    #[error(transparent)]
    Indexer(#[from] KvRouterError),
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
            Self::Indexer(KvRouterError::IndexerOffline) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Indexer(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
            Self::Indexer(KvRouterError::IndexerOffline) => "not_ready",
            Self::Indexer(_) => "internal",
        }
    }

    /// Stable reason for a scheduler outcome that needs a distinct client
    /// response, if one is defined.
    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Self::Scheduler(KvSchedulerError::DoNotQueue { .. }) => Some("router.do_not_queue"),
            _ => None,
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
        | KvSchedulerError::QueueRejected(_)
        | KvSchedulerError::DoNotQueue { .. }
        | KvSchedulerError::DeadlineExceeded => StatusCode::TOO_MANY_REQUESTS,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => StatusCode::BAD_REQUEST,
        // A duplicate live request id, or a lifecycle the caller ended (or
        // re-registered) mid-classification, is caller-induced, like
        // `BookingFailed` and `SequenceError::DuplicateRequest`.
        KvSchedulerError::BookingFailed(_)
        | KvSchedulerError::DuplicateClassificationRequestId(_)
        | KvSchedulerError::ClassificationLifecycleEnded(_) => StatusCode::CONFLICT,
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
                    | KvSchedulerError::InvalidClassificationMetadata(_)
            )
        ) {
            // Plugin-produced detail (its error text, or the metadata it
            // returned) stays server-side: log it and return a fixed body.
            tracing::warn!(error = %self, "request classifier failure sanitized from response");
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
        if let Self::Scheduler(KvSchedulerError::DoNotQueue {
            policy_class,
            pending_count,
            pending_isl_tokens,
            pending_cached_tokens,
        }) = &self
        {
            return (
                self.status(),
                Json(serde_json::json!({
                    "error": self.to_string(),
                    "reason": self.reason(),
                    "details": {
                        "policy_class": policy_class,
                        "pending_count": pending_count,
                        "pending_isl_tokens": pending_isl_tokens,
                        "pending_cached_tokens": pending_cached_tokens,
                    },
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
    use crate::scheduling::{QueueLimitKind, QueueRejection};

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

    #[test]
    fn preserves_other_scheduler_error_statuses_and_identifies_do_not_queue() {
        let rejection = KvSchedulerError::QueueRejected(QueueRejection {
            policy_class: "default".to_string(),
            limit_kind: QueueLimitKind::Requests,
            current: 1,
            limit: 1,
        });
        assert_eq!(SelectionError::from(rejection).status_code(), 429);
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::NoEndpoints).status_code(),
            StatusCode::SERVICE_UNAVAILABLE.as_u16()
        );
        let error = SelectionError::Scheduler(KvSchedulerError::DoNotQueue {
            policy_class: "latency-sensitive".to_string(),
            pending_count: 2,
            pending_isl_tokens: 128,
            pending_cached_tokens: 64,
        });
        assert_eq!(error.status_code(), StatusCode::TOO_MANY_REQUESTS.as_u16());
        assert_eq!(error.kind(), "scheduler");
        assert_eq!(error.reason(), Some("router.do_not_queue"));
    }

    #[tokio::test]
    async fn do_not_queue_response_has_reason_and_safe_load_details() {
        let response = SelectionError::Scheduler(KvSchedulerError::DoNotQueue {
            policy_class: "latency-sensitive".to_string(),
            pending_count: 2,
            pending_isl_tokens: 128,
            pending_cached_tokens: 64,
        })
        .into_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["reason"], "router.do_not_queue");
        assert_eq!(body["details"]["policy_class"], "latency-sensitive");
        assert_eq!(body["details"]["pending_count"], 2);
        assert_eq!(body["details"]["pending_isl_tokens"], 128);
        assert_eq!(body["details"]["pending_cached_tokens"], 64);
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

    #[tokio::test]
    async fn invalid_classification_metadata_response_is_sanitized() {
        let response = SelectionError::Scheduler(KvSchedulerError::InvalidClassificationMetadata(
            "unknown policy class \"plugin-private-class\"".to_string(),
        ))
        .into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), br#"{"error":"request classifier failed"}"#);
    }
}
