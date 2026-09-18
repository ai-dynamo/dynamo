// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The EPP's half of the router admission contract.
//!
//! In full dynamo mode the EPP embeds the KV router in-process *and* does the
//! frontend's job for gateway traffic, including turning failures into client
//! status codes. [`classify_router_error`] recovers the typed rejection from an
//! `anyhow::Error` so the ext_proc boundary can pick the right status instead
//! of flattening everything to 503.
//!
//! The reason is always recoverable: `KvRouter::map_scheduler_error` converts
//! the overload family into a [`DynamoError`] and passes every other
//! [`KvSchedulerError`] through untouched. It is lost only when a caller
//! stringifies it — which is what the EPP used to do, so every routing failure
//! became a 503 carrying the router's `Debug` text.
//!
//! # Status classes
//!
//! A scheduler error is classified once, by [`KvSchedulerError::rejection`] in
//! `dynamo-kv-router`; this module only renders that classification in ext_proc's
//! status vocabulary. The selection service renders the same classification in
//! HTTP's, via `scheduler_error_status`, so the two cannot drift —
//! `epp_statuses_match_the_selection_service` pins that.
//!
//! That shared mapping answered a queue rejection with 503; this change moves it
//! to 429 alongside the overload family, per DEP #9755 (`dep:approved`,
//! `dep:implementing`):
//!
//! > Terminal router-side rejection **SHOULD** use downstream throttling
//! > semantics, such as `TooManyRequests` / HTTP 429.
//!
//! which names "router queue full" as an instance. The DEP's 503 allowance
//! covers the Frontend's *pre-tokenization* gate, not a router queue decision
//! taken after tokenization. The difference is behavioural: 503 invites a
//! gateway to fail over to another endpoint, which cannot help when the limit
//! is a fleet-wide policy-class setting.
//!
//! The Frontend still answers both families with `overload_status_code()`, 529
//! by default. That is a public configurable default across seven surfaces, so
//! it is left alone here; converging it is raised on ai-dynamo/dynamo#14176.

use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_runtime::error::{DynamoError, ErrorType};

use crate::picker::PickError;

/// Why the embedded router refused to place a request.
///
/// The router's own classification, reused rather than restated:
/// [`KvSchedulerError::rejection`] is the single place a scheduler error is
/// given a meaning, so the EPP and the selection service cannot drift apart on
/// what a refusal is. This crate only decides how to *present* it, in
/// [`RouterRejectionExt`].
pub use dynamo_kv_router::scheduling::SchedulerRejection as RouterRejection;

/// How the EPP presents a [`RouterRejection`] to a gateway client.
///
/// An extension trait because the enum belongs to `dynamo-kv-router`; the
/// status vocabulary and the metric namespace are this crate's.
pub trait RouterRejectionExt {
    /// Stable, low-cardinality label for the rejection metric.
    fn metric_label(self) -> &'static str;

    /// Client-safe [`PickError`] for this rejection.
    ///
    /// No variant carries router internals. The detailed cause is logged at the
    /// call site; the client sees only the category, matching how the tokenizer
    /// variants already behave.
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

/// Classify an `anyhow::Error` returned by a routing call.
///
/// Walks the chain, not just the outermost error, so added context cannot hide
/// the rejection. The [`DynamoError`] channel is checked first because
/// `map_scheduler_error` converts the overload family into that form;
/// everything it leaves alone arrives as a bare [`KvSchedulerError`].
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

    // Unavailable, not internal: keeps the pre-classification status (503) and
    // avoids reporting an unknown upstream error to the client as an EPP bug.
    RouterRejection::Unavailable
}

/// `None` when the class carries no routing meaning, so the caller keeps
/// walking the chain.
///
/// Matches *canonical* classes, because the caller reads
/// [`DynamoError::class`], which normalizes. `map_scheduler_error` still builds
/// the legacy `ResourceExhausted`/`WorkerOverloaded` names; matching those
/// directly would work today and break silently when that producer moves to
/// canonical classes, turning every overload into a 503 with no test failing.
fn classify_error_class(class: ErrorType) -> Option<RouterRejection> {
    match class {
        // Where `map_scheduler_error` lands both overload cases.
        ErrorType::CapacityExhausted => Some(RouterRejection::Overloaded),
        ErrorType::Unavailable => Some(RouterRejection::Unavailable),
        ErrorType::InvalidRequest => Some(RouterRejection::BadRequest),
        // TODO(epp-deadline-429): `DeadlineExceeded` is unmapped deliberately,
        // not because it is unreachable. It arrives today from transport
        // timeouts (`ErrorClass::normalized` folds `ConnectionTimeout` and
        // `ResponseTimeout` into it), which this crate answers with 504
        // elsewhere, while #14176 adds a queue-deadline producer it answers
        // with 429. One class, two right answers, so both keep the
        // `Unavailable` (503) fallback until they can be told apart.
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
        // `map_scheduler_error` converts the overload family into this form, so
        // no scheduler error is left in the chain. These are the legacy variants
        // it builds today, which classify only because `class()` normalizes.
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

    /// The other half of the pair above: a producer building canonical classes
    /// directly must classify identically, or migrating `map_scheduler_error`
    /// turns every 429 into a 503 with no test failing.
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
        // A caller adding context above the rejection must not erase it, which
        // is precisely the failure this module exists to prevent.
        let wrapped = anyhow_from(KvSchedulerError::AllEligibleWorkersOverloaded)
            .context("decode selection failed");
        assert_eq!(classify_router_error(&wrapped), RouterRejection::Overloaded);
    }

    #[test]
    fn unrecognized_errors_stay_unavailable_not_internal() {
        let opaque = anyhow::anyhow!("something the EPP has never seen");
        assert_eq!(classify_router_error(&opaque), RouterRejection::Unavailable);
    }

    /// Pins the `TODO(epp-deadline-429)` seam. A deadline is reachable today,
    /// so this fails if the arm is added without first splitting transport
    /// timeouts from queue deadlines.
    #[test]
    fn deadline_exceeded_is_not_yet_a_429() {
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::DeadlineExceeded)),
            RouterRejection::Unavailable
        );
    }

    #[test]
    fn rejections_carry_no_router_internals() {
        // Every classification must produce a client-safe message. The router's
        // own text is logged, never returned.
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
