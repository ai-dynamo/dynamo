// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The EPP's half of the router admission contract.
//!
//! In full dynamo mode the EPP embeds the KV router in-process *and* performs
//! the frontend duties for gateway traffic — tokenization, header handling, and
//! mapping failures to a client status code. The Dynamo Frontend performs the
//! same duties for its own traffic through `lib/llm/src/http/service/*`. When a
//! rejection reason is added to the router, both hosts have to learn it; this
//! module is the EPP's side.
//!
//! Two pieces live here:
//!
//! - [`classify_router_error`] turns an `anyhow::Error` from a routing call
//!   back into a typed rejection, so the ext_proc boundary can pick the right
//!   status class instead of flattening everything to 503.
//! - [`requested_ttft_slo`] reads the per-request TTFT budget that DEP #9755
//!   defines, which is the input the router needs to order by deadline.
//!
//! # Why classification is needed at all
//!
//! The router already preserves the reason. `KvRouter::map_scheduler_error`
//! (`lib/llm/src/kv_router.rs`) converts the overload family into a
//! `DynamoError` carrying an [`ErrorType`], and returns every other
//! [`KvSchedulerError`] unchanged. Both survive inside the `anyhow::Error`, so
//! the reason only disappears when a caller stringifies it. Until now the EPP
//! did exactly that, and every routing failure became a 503 whose body carried
//! the router's internal `Debug` text.
//!
//! The status classes below match the Dynamo selection service's own mapping in
//! `lib/kv-router/src/services/selection/error.rs`, so a rejection means the
//! same thing to a client whichever host produced it.

use std::time::Duration;

use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_llm::http::service::metadata::extract_metadata_from_header_pairs;
use dynamo_runtime::error::{DynamoError, ErrorType};

use crate::picker::PickError;

/// Metadata key naming the per-request time-to-first-token budget, in
/// milliseconds.
///
/// Rides the same Dynamo metadata-header mechanism as the policy class (see
/// [`crate::epp_router::requested_policy_class`]), so the wire header is this
/// key behind `DYNAMO_METADATA_HEADER_PREFIX_DEFAULT` —
/// `x-dynamo-meta-slo-ttft-ms` by default, and whatever
/// `DYNAMO_METADATA_HEADER_ENV` sets otherwise.
///
/// Note this differs from DEP #9755, which writes the header as
/// `x-dynamo-slo-ttft-ms`. The metadata prefix is the mechanism the codebase
/// actually implements and the one the policy class already uses, so the SLO
/// budget follows it rather than introducing a second, unprefixed convention
/// with a different trust boundary.
pub const SLO_TTFT_MS_KEY: &str = "slo-ttft-ms";

/// Largest TTFT budget the EPP will accept, bounding a value that reaches the
/// scheduler's ordering key. A budget beyond this is a configuration mistake
/// rather than a real latency target, and clamping silently would hide it.
const MAX_TTFT_SLO: Duration = Duration::from_secs(3600);

/// Why the embedded router refused to place a request.
///
/// Deliberately coarser than [`KvSchedulerError`]: the EPP only needs enough
/// resolution to choose a status class and a metric label, and a coarse enum
/// keeps a non-exhaustive upstream enum from breaking this crate every time a
/// variant is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterRejection {
    /// Downstream capacity is saturated. Retryable backpressure → 429.
    Overloaded,
    /// A policy-class queue-depth limit refused the request → 503.
    QueueRejected,
    /// No worker could serve the request right now → 503.
    Unavailable,
    /// The request contradicted router state, e.g. a duplicate booking → 409.
    Conflict,
    /// The request itself was not routable, e.g. a pin outside the allowed
    /// set → 400.
    BadRequest,
    /// A router-internal invariant failed → 500.
    Internal,
}

impl RouterRejection {
    /// Stable, low-cardinality label for the rejection metric.
    pub fn metric_label(self) -> &'static str {
        match self {
            Self::Overloaded => "overloaded",
            Self::QueueRejected => "queue_rejected",
            Self::Unavailable => "unavailable",
            Self::Conflict => "conflict",
            Self::BadRequest => "bad_request",
            Self::Internal => "internal",
        }
    }

    /// Client-safe [`PickError`] for this rejection.
    ///
    /// No variant carries router internals. The detailed cause is logged at the
    /// call site; the client sees only the category, matching how the tokenizer
    /// variants already behave.
    pub fn into_pick_error(self) -> PickError {
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
/// Walks the error chain rather than inspecting only the outermost error, so a
/// rejection stays classifiable if a caller adds context above it.
///
/// The `DynamoError` channel is checked first because
/// `KvRouter::map_scheduler_error` converts the overload family into that form;
/// everything it leaves alone arrives as a bare [`KvSchedulerError`].
pub fn classify_router_error(error: &anyhow::Error) -> RouterRejection {
    for cause in error.chain() {
        if let Some(dynamo_error) = cause.downcast_ref::<DynamoError>()
            && let Some(rejection) = classify_error_type(dynamo_error.error_type())
        {
            return rejection;
        }
        if let Some(scheduler_error) = cause.downcast_ref::<KvSchedulerError>() {
            return classify_scheduler_error(scheduler_error);
        }
    }

    // An unrecognized failure is treated as unavailable rather than internal:
    // it preserves the pre-classification behaviour (503) and keeps an unknown
    // upstream error from being reported to the client as an EPP bug.
    RouterRejection::Unavailable
}

/// `None` when the error type carries no routing meaning, so the caller keeps
/// walking the chain.
fn classify_error_type(error_type: ErrorType) -> Option<RouterRejection> {
    match error_type {
        // Set by `map_scheduler_error` for AllEligibleWorkersOverloaded and
        // PinnedWorkerOverloaded respectively.
        ErrorType::ResourceExhausted | ErrorType::WorkerOverloaded => {
            Some(RouterRejection::Overloaded)
        }
        ErrorType::Unavailable | ErrorType::WorkerUnavailable => Some(RouterRejection::Unavailable),
        ErrorType::InvalidArgument => Some(RouterRejection::BadRequest),
        // TODO(epp-deadline-429): ai-dynamo/dynamo#14176 adds
        // `KvSchedulerError::DeadlineExceeded` and maps it to
        // `ErrorType::DeadlineExceeded`, which that PR classifies as 429
        // alongside the overload family. `ErrorType` has no such variant on
        // main, so a deadline rejection currently falls through to
        // `Unavailable` (503) — the same status it gets today, so this is not a
        // regression. When #14176 merges, add:
        //
        //     ErrorType::DeadlineExceeded => Some(RouterRejection::Overloaded),
        //
        // and extend `deadline_rejection_is_not_yet_429` below.
        _ => None,
    }
}

fn classify_scheduler_error(error: &KvSchedulerError) -> RouterRejection {
    // Mirrors `scheduler_error_status` in
    // `lib/kv-router/src/services/selection/error.rs`. `KvSchedulerError` is
    // `#[non_exhaustive]`, so the wildcard is required from this crate; it is
    // also what keeps a newly added variant from failing the build here rather
    // than being classified conservatively.
    match error {
        KvSchedulerError::AllEligibleWorkersOverloaded
        | KvSchedulerError::PinnedWorkerOverloaded { .. } => RouterRejection::Overloaded,
        KvSchedulerError::QueueRejected(_) => RouterRejection::QueueRejected,
        KvSchedulerError::NoEndpoints
        | KvSchedulerError::AllEligibleWorkersFiltered
        | KvSchedulerError::SubscriberShutdown
        | KvSchedulerError::InitFailed(_) => RouterRejection::Unavailable,
        KvSchedulerError::PinnedWorkerNotAllowed { .. } => RouterRejection::BadRequest,
        KvSchedulerError::BookingFailed(_) => RouterRejection::Conflict,
        KvSchedulerError::WorkerSelectionPolicy(_) => RouterRejection::Internal,
        _ => RouterRejection::Unavailable,
    }
}

/// Read the per-request TTFT budget from the request's Dynamo metadata headers.
///
/// DEP #9755 defines the router's queue key as
/// `due_time = receive_time + selected_ttft_slo`, with a trusted per-request
/// override taking precedence over the policy class default. For gateway
/// traffic the EPP is the component that sees the headers, so it is the
/// component that has to read it.
///
/// Returns `Ok(None)` when the caller supplied no budget, which means the
/// policy class default applies.
///
/// # Trust
///
/// The value is only as trustworthy as the gateway in front of the EPP. It
/// rides the Dynamo metadata-header prefix, and `envoy_helpers` already strips
/// the client-spoofable `x-gateway-*` control headers on the way in, so a
/// deployment that lets clients set arbitrary `x-dynamo-*` headers is letting
/// them choose their own queue priority. That is the same exposure the policy
/// class already has.
pub fn requested_ttft_slo(headers: &[(String, String)]) -> Result<Option<Duration>, PickError> {
    let metadata =
        extract_metadata_from_header_pairs(headers.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .map_err(|e| PickError::MetadataHeadersInvalid(e.to_string()))?;

    let Some(raw) = metadata.get(SLO_TTFT_MS_KEY) else {
        return Ok(None);
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let millis: u64 = trimmed.parse().map_err(|_| {
        PickError::InvalidRequest(format!(
            "{SLO_TTFT_MS_KEY} must be a whole number of milliseconds"
        ))
    })?;

    if millis == 0 {
        return Err(PickError::InvalidRequest(format!(
            "{SLO_TTFT_MS_KEY} must be greater than zero"
        )));
    }

    let slo = Duration::from_millis(millis);
    if slo > MAX_TTFT_SLO {
        return Err(PickError::InvalidRequest(format!(
            "{SLO_TTFT_MS_KEY} must not exceed {} ms",
            MAX_TTFT_SLO.as_millis()
        )));
    }

    Ok(Some(slo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::WorkerId;

    fn header(key: &str, value: &str) -> Vec<(String, String)> {
        vec![(key.to_string(), value.to_string())]
    }

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
        // the scheduler error is no longer in the chain at all.
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

    #[test]
    fn deadline_rejection_is_not_yet_429() {
        // Documents the TODO(epp-deadline-429) seam. `ErrorType` gains a
        // `DeadlineExceeded` variant in ai-dynamo/dynamo#14176; until then a
        // deadline rejection is indistinguishable from any other unmapped
        // error and stays 503, exactly as it is today. When that PR lands this
        // test should assert `RouterRejection::Overloaded`.
        assert_eq!(
            classify_router_error(&dynamo_error(ErrorType::Cancelled)),
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

    #[test]
    fn ttft_slo_absent_is_not_an_error() {
        assert_eq!(requested_ttft_slo(&[]).unwrap(), None);
        assert_eq!(
            requested_ttft_slo(&header("content-type", "application/json")).unwrap(),
            None
        );
    }

    #[test]
    fn ttft_slo_parses_milliseconds() {
        let headers = header("x-dynamo-meta-slo-ttft-ms", "250");
        assert_eq!(
            requested_ttft_slo(&headers).unwrap(),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn ttft_slo_blank_value_is_treated_as_absent() {
        let headers = header("x-dynamo-meta-slo-ttft-ms", "   ");
        assert_eq!(requested_ttft_slo(&headers).unwrap(), None);
    }

    #[test]
    fn ttft_slo_rejects_malformed_values() {
        for value in ["abc", "-5", "1.5", "250ms"] {
            let headers = header("x-dynamo-meta-slo-ttft-ms", value);
            assert!(
                matches!(
                    requested_ttft_slo(&headers),
                    Err(PickError::InvalidRequest(_))
                ),
                "{value} should be rejected as a client error"
            );
        }
    }

    #[test]
    fn ttft_slo_rejects_zero_and_absurd_budgets() {
        assert!(matches!(
            requested_ttft_slo(&header("x-dynamo-meta-slo-ttft-ms", "0")),
            Err(PickError::InvalidRequest(_))
        ));
        assert!(matches!(
            requested_ttft_slo(&header("x-dynamo-meta-slo-ttft-ms", "3600001")),
            Err(PickError::InvalidRequest(_))
        ));
    }
}
