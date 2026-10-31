// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::indexer::KvRouterError;
use crate::scheduling::KvSchedulerError;
use crate::sequences::SequenceError;

/// HTTP status numbers the kernel's errors map to. Plain integers: the kernel does
/// not link an HTTP crate; `kv-router`'s server turns them into responses.
mod status {
    pub(super) const BAD_REQUEST: u16 = 400;
    pub(super) const NOT_FOUND: u16 = 404;
    pub(super) const CONFLICT: u16 = 409;
    pub(super) const TOO_MANY_REQUESTS: u16 = 429;
    pub(super) const INTERNAL_SERVER_ERROR: u16 = 500;
    pub(super) const SERVICE_UNAVAILABLE: u16 = 503;
}

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
    /// The HTTP status class this error maps to, as a bare number; the HTTP
    /// adapter turns it into a response.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => status::BAD_REQUEST,
            Self::NotReady(_) => status::SERVICE_UNAVAILABLE,
            Self::NotFound(_) => status::NOT_FOUND,
            Self::Conflict(_) => status::CONFLICT,
            Self::Internal(_) => status::INTERNAL_SERVER_ERROR,
            Self::Scheduler(error) => scheduler_error_status(error),
            Self::Sequence(error) => sequence_error_status(error),
            Self::Indexer(KvRouterError::IndexerOffline) => status::SERVICE_UNAVAILABLE,
            Self::Indexer(_) => status::INTERNAL_SERVER_ERROR,
        }
    }

    /// HTTP-style status code for this error, for callers that consume the
    /// service in-process without an HTTP layer.
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
}

fn scheduler_error_status(error: &KvSchedulerError) -> u16 {
    match error {
        KvSchedulerError::NoEndpoints
        | KvSchedulerError::AllEligibleWorkersFiltered
        | KvSchedulerError::SubscriberShutdown
        | KvSchedulerError::InitFailed(_) => status::SERVICE_UNAVAILABLE,
        KvSchedulerError::WorkerSelectionPolicy(_) => status::INTERNAL_SERVER_ERROR,
        // Deadline expiry is deliberately 429, not 504: the deadline elapsed
        // while waiting for capacity, so it is backpressure the client should
        // respond to like the overloaded family, not a gateway timeout.
        KvSchedulerError::AllEligibleWorkersOverloaded
        | KvSchedulerError::PinnedWorkerOverloaded { .. }
        | KvSchedulerError::QueueRejected(_)
        | KvSchedulerError::DeadlineExceeded => status::TOO_MANY_REQUESTS,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => status::BAD_REQUEST,
        // A duplicate live request id, or a lifecycle the caller ended (or
        // re-registered) mid-classification, is caller-induced, like
        // `BookingFailed` and `SequenceError::DuplicateRequest`.
        KvSchedulerError::BookingFailed(_)
        | KvSchedulerError::DuplicateClassificationRequestId(_)
        | KvSchedulerError::ClassificationLifecycleEnded(_) => status::CONFLICT,
        KvSchedulerError::RequestClassifierPanicked(_)
        | KvSchedulerError::RequestClassifierFailed(_)
        | KvSchedulerError::InvalidClassificationMetadata(_) => status::INTERNAL_SERVER_ERROR,
    }
}

fn sequence_error_status(error: &SequenceError) -> u16 {
    match error {
        SequenceError::WorkerNotFound { .. } | SequenceError::RequestNotFound { .. } => {
            status::NOT_FOUND
        }
        SequenceError::DuplicateRequest { .. } => status::CONFLICT,
        SequenceError::ReplicaSyncPublishFailed(_) => status::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_workers_are_unavailable_not_overloaded() {
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersFiltered).status_code(),
            503
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersOverloaded).status_code(),
            429
        );
        assert_eq!(
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded).status_code(),
            429
        );
    }
}
