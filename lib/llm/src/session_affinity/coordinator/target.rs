// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::pipeline::{Error, SingleIn};

use super::{AffinityTarget, invalid_argument};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

pub(crate) const ADVISORY_DECODE_TARGET_CONTEXT_KEY: &str =
    "x-dynamo-internal-advisory-decode-target";

pub fn affinity_id(
    request: &SingleIn<PreprocessedRequest>,
) -> Result<Option<Arc<SessionAffinityId>>, Error> {
    request
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| invalid_argument(format!("invalid session affinity context: {message}")))
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    explicit_target_inner(request, phase, false)
}

pub(crate) fn explicit_target_for_routing(
    request: &SingleIn<PreprocessedRequest>,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    let advisory_decode_target = if request
        .routing
        .as_ref()
        .is_some_and(|routing| routing.decode_worker_id.is_some())
    {
        request
            .get_optional::<()>(ADVISORY_DECODE_TARGET_CONTEXT_KEY)
            .map_err(|message| {
                invalid_argument(format!("invalid advisory target context: {message}"))
            })?
            .is_some()
    } else {
        false
    };
    explicit_target_inner(request.content(), phase, advisory_decode_target)
}

fn explicit_target_inner(
    request: &PreprocessedRequest,
    phase: RequestPhase,
    advisory_decode_target: bool,
) -> Result<Option<AffinityTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let prefill_worker_id = routing.prefill_worker_id.or(routing.backend_instance_id);
    let prefill_dp_rank = if advisory_decode_target && prefill_worker_id.is_none() {
        None
    } else {
        routing.prefill_dp_rank.or(routing.dp_rank)
    };
    let decode_worker_id = if advisory_decode_target {
        routing.backend_instance_id
    } else {
        routing.decode_worker_id.or(routing.backend_instance_id)
    };
    let decode_dp_rank = if advisory_decode_target && routing.backend_instance_id.is_none() {
        None
    } else {
        routing.dp_rank
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (prefill_worker_id, prefill_dp_rank),
        RequestPhase::Decode => (decode_worker_id, decode_dp_rank),
        RequestPhase::Aggregated => (decode_worker_id, decode_dp_rank),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for session affinity",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget { worker_id, dp_rank }))
}

pub(super) fn validate_bound_target(
    session_id: &str,
    bound: AffinityTarget,
    requested: Option<AffinityTarget>,
) -> Result<(), Error> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if bound.worker_id != requested.worker_id {
        return Err(invalid_argument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, requested.worker_id
        )));
    }
    match (bound.dp_rank, requested.dp_rank) {
        (Some(bound), Some(requested)) if bound != requested => Err(invalid_argument(format!(
            "session {session_id} is bound to DP rank {bound}, not {requested}"
        ))),
        (None, Some(requested)) => Err(invalid_argument(format!(
            "session {session_id} has worker-only affinity and cannot add DP rank {requested}"
        ))),
        _ => Ok(()),
    }
}

/// Validates that a request was dispatched within an existing session binding.
///
/// Unlike an explicit requested target, a dispatch target may add a DP rank to a worker-only
/// binding because load-aware scheduling chooses that rank for this request only.
pub(super) fn validate_dispatch_target(
    session_id: &str,
    bound: AffinityTarget,
    dispatched: AffinityTarget,
) -> Result<(), Error> {
    if bound.worker_id != dispatched.worker_id {
        return Err(invalid_argument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, dispatched.worker_id
        )));
    }
    if let Some(bound_rank) = bound.dp_rank {
        match dispatched.dp_rank {
            Some(dispatched_rank) if dispatched_rank == bound_rank => {}
            Some(dispatched_rank) => {
                return Err(invalid_argument(format!(
                    "session {session_id} is bound to DP rank {bound_rank}, not {dispatched_rank}"
                )));
            }
            None => {
                return Err(invalid_argument(format!(
                    "session {session_id} is bound to DP rank {bound_rank}, but dispatch did not select a DP rank"
                )));
            }
        }
    }
    Ok(())
}
