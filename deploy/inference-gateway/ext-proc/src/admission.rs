// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Maps router rejections to the EPP's client statuses.
//!
//! [`classify_router_error`] recovers the typed rejection from an
//! `anyhow::Error`; stringifying it would lose the reason and leak the router's
//! `Debug` text. The classification itself is [`KvSchedulerError::rejection`],
//! shared with the selection service.

use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_runtime::error::{DynamoError, ErrorType};

use crate::picker::PickError;

/// Why the embedded router refused a request; see [`KvSchedulerError::rejection`].
pub use dynamo_kv_router::scheduling::SchedulerRejection as RouterRejection;

/// How the EPP presents a [`RouterRejection`]: metric label and client error.
pub trait RouterRejectionExt {
    /// Stable, low-cardinality label for the rejection metric.
    fn metric_label(self) -> &'static str;

    /// Client-safe [`PickError`]; the router's own text is logged, never returned.
    fn into_pick_error(self) -> PickError;
}

impl RouterRejectionExt for RouterRejection {
    fn metric_label(self) -> &'static str {
        match self {
            Self::Overloaded => "overloaded",
            Self::QueueRejected => "queue_rejected",
            Self::Unavailable => "unavailable",
            Self::Conflict => "conflict",
            Self::BadRequest => "bad_request",
            Self::Internal => "internal",
        }
    }

    fn into_pick_error(self) -> PickError {
        match self {
            Self::Overloaded => PickError::RouterOverloaded,
            Self::QueueRejected => PickError::RouterQueueRejected,
            Self::Unavailable => PickError::NoEndpoints,
            Self::Conflict => PickError::RouterConflict,
            Self::BadRequest => PickError::InvalidRequest("request is not routable".to_string()),
            Self::Internal => PickError::RouterInternal,
        }
    }
}

/// Classify an error from a routing call. Walks the whole chain, so added
/// context cannot hide the rejection.
pub fn classify_router_error(error: &anyhow::Error) -> RouterRejection {
    for cause in error.chain() {
        if let Some(dynamo_error) = cause.downcast_ref::<DynamoError>()
            && let Some(rejection) = classify_error_class(dynamo_error.class())
        {
            return rejection;
        }
        if let Some(scheduler_error) = cause.downcast_ref::<KvSchedulerError>() {
            return scheduler_error.rejection();
        }
    }

    // Unknown upstream errors are 503, not 500: they are not EPP bugs.
    RouterRejection::Unavailable
}

/// `None` when the class has no routing meaning. Matches canonical classes,
/// since [`DynamoError::class`] normalizes legacy names like `ResourceExhausted`.
fn classify_error_class(class: ErrorType) -> Option<RouterRejection> {
    match class {
        // Both overload cases from `map_scheduler_error`.
        ErrorType::CapacityExhausted => Some(RouterRejection::Overloaded),
        ErrorType::Unavailable => Some(RouterRejection::Unavailable),
        ErrorType::InvalidRequest => Some(RouterRejection::BadRequest),
        // TODO(epp-deadline-429): queue deadlines (reason
        // `router.queue_deadline_exceeded`, 429 in the selection service) and
        // transport timeouts (504 elsewhere here) share `DeadlineExceeded`.
        // Split on the reason before mapping it; both fall back to 503 until then.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::WorkerId;

    fn anyhow_from(error: KvSchedulerError) -> anyhow::Error {
        error.into()
    }

    fn dynamo_error(error_type: ErrorType) -> anyhow::Error {
        DynamoError::builder()
            .error_type(error_type)
            .message("internal router detail that must not reach the client")
            .build()
            .into()
    }

    #[test]
    fn overloaded_family_classifies_as_overloaded() {
        assert_eq!(
            classify_router_error(&anyhow_from(KvSchedulerError::AllEligibleWorkersOverloaded)),
            RouterRejection::Overloaded
        );
        assert_eq!(
            classify_router_error(&anyhow_from(KvSchedulerError::PinnedWorkerOverloaded {
                worker_id: WorkerId::default(),
            })),
            RouterRejection::Overloaded
        );
    }

    #[test]
    fn dynamo_error_types_classify_without_a_scheduler_error() {
        // Legacy classes, as `map_scheduler_error` builds them today.
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::ResourceExhausted)),
            RouterRejection::Overloaded
        );
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::WorkerOverloaded)),
            RouterRejection::Overloaded
        );
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::Unavailable)),
            RouterRejection::Unavailable
        );
    }

    /// Canonical classes classify like their legacy names.
    #[test]
    fn canonical_error_classes_classify_the_same_as_legacy_ones() {
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::CapacityExhausted)),
            RouterRejection::Overloaded
        );
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::WorkerUnavailable)),
            RouterRejection::Unavailable
        );
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::InvalidRequest)),
            RouterRejection::BadRequest
        );
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::InvalidArgument)),
            RouterRejection::BadRequest
        );
    }

    #[test]
    fn unavailable_family_classifies_as_unavailable() {
        for error in [
            KvSchedulerError::NoEndpoints,
            KvSchedulerError::AllEligibleWorkersFiltered,
            KvSchedulerError::SubscriberShutdown,
            KvSchedulerError::InitFailed("boom".to_string()),
        ] {
            assert_eq!(
                classify_router_error(&anyhow_from(error)),
                RouterRejection::Unavailable
            );
        }
    }

    #[test]
    fn booking_and_pin_failures_keep_their_own_classes() {
        assert_eq!(
            classify_router_error(&anyhow_from(KvSchedulerError::BookingFailed(
                "duplicate".to_string()
            ))),
            RouterRejection::Conflict
        );
        assert_eq!(
            classify_router_error(&anyhow_from(KvSchedulerError::PinnedWorkerNotAllowed {
                worker_id: WorkerId::default(),
            })),
            RouterRejection::BadRequest
        );
    }

    #[test]
    fn classification_survives_added_context() {
        let wrapped = anyhow_from(KvSchedulerError::AllEligibleWorkersOverloaded)
            .context("decode selection failed");
        assert_eq!(classify_router_error(&wrapped), RouterRejection::Overloaded);
    }

    #[test]
    fn unrecognized_errors_stay_unavailable_not_internal() {
        let opaque = anyhow::anyhow!("something the EPP has never seen");
        assert_eq!(classify_router_error(&opaque), RouterRejection::Unavailable);
    }

    /// Pins `TODO(epp-deadline-429)`.
    #[test]
    fn deadline_exceeded_is_not_yet_a_429() {
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::DeadlineExceeded)),
            RouterRejection::Unavailable
        );
    }

    #[test]
    fn rejections_carry_no_router_internals() {
        let internal = "internal router detail that must not reach the client";
        for rejection in [
            RouterRejection::Overloaded,
            RouterRejection::QueueRejected,
            RouterRejection::Unavailable,
            RouterRejection::Conflict,
            RouterRejection::BadRequest,
            RouterRejection::Internal,
        ] {
            let message = rejection.into_pick_error().to_string();
            assert!(
                !message.contains(internal),
                "{rejection:?} leaked router internals: {message}"
            );
        }
    }

    #[test]
    fn metric_labels_are_distinct_and_low_cardinality() {
        let labels: Vec<&str> = [
            RouterRejection::Overloaded,
            RouterRejection::QueueRejected,
            RouterRejection::Unavailable,
            RouterRejection::Conflict,
            RouterRejection::BadRequest,
            RouterRejection::Internal,
        ]
        .into_iter()
        .map(RouterRejection::metric_label)
        .collect();

        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), labels.len(), "labels must be distinct");
    }
}
