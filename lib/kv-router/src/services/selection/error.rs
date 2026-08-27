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
        KvSchedulerError::WorkerSelectionPolicy(_)
        | KvSchedulerError::RequestClassifierPanicked(_)
        | KvSchedulerError::RequestClassifierReplacedRequest
        | KvSchedulerError::DuplicateClassificationRequestId(_)
        | KvSchedulerError::InvalidClassificationMetadata(_) => StatusCode::INTERNAL_SERVER_ERROR,
        KvSchedulerError::AllEligibleWorkersOverloaded
        | KvSchedulerError::PinnedWorkerOverloaded { .. }
        | KvSchedulerError::ClassifyPendingLimit { .. }
        | KvSchedulerError::QueueRejected(_)
        | KvSchedulerError::DueTimeExpired => StatusCode::TOO_MANY_REQUESTS,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => StatusCode::BAD_REQUEST,
        KvSchedulerError::BookingFailed(_) => StatusCode::CONFLICT,
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
            SelectionError::Scheduler(KvSchedulerError::ClassifyPendingLimit { limit: 1 })
                .status_code(),
            StatusCode::TOO_MANY_REQUESTS.as_u16()
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::DueTimeExpired).status_code(),
            StatusCode::TOO_MANY_REQUESTS.as_u16()
        );
    }
}
