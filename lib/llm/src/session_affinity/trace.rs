// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

use super::AffinityTarget;
use crate::protocols::common::timing::RequestPhase;

const TRACE_ENV: &str = "DYN_ROUTER_SESSION_AFFINITY_TRACE";
const TRACE_PREFIX_ENV: &str = "DYN_ROUTER_SESSION_AFFINITY_TRACE_PREFIX";

#[derive(Debug, Default, PartialEq, Eq)]
struct TraceConfig {
    enabled: bool,
    session_prefix: Option<String>,
}

impl TraceConfig {
    fn from_values(enabled: Option<&str>, session_prefix: Option<&str>) -> Self {
        Self {
            enabled: parse_enabled(enabled),
            session_prefix: session_prefix
                .map(str::trim)
                .filter(|prefix| !prefix.is_empty())
                .map(str::to_string),
        }
    }

    fn matches(&self, session_id: &str) -> bool {
        self.enabled
            && self
                .session_prefix
                .as_deref()
                .is_none_or(|prefix| session_id.starts_with(prefix))
    }
}

static TRACE_CONFIG: OnceLock<TraceConfig> = OnceLock::new();
static INITIALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
static COMMIT_COUNT: AtomicU64 = AtomicU64::new(0);
static BOUND_REUSE_COUNT: AtomicU64 = AtomicU64::new(0);
static BOUND_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static FALLBACK_REBIND_COUNT: AtomicU64 = AtomicU64::new(0);

fn parse_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn config() -> &'static TraceConfig {
    TRACE_CONFIG.get_or_init(|| {
        let enabled = std::env::var(TRACE_ENV).ok();
        let prefix = std::env::var(TRACE_PREFIX_ENV).ok();
        TraceConfig::from_values(enabled.as_deref(), prefix.as_deref())
    })
}

fn next(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed) + 1
}

fn fallback_rebind_count() -> u64 {
    FALLBACK_REBIND_COUNT.load(Ordering::Relaxed)
}

pub(super) fn initialize(
    session_id: &str,
    revision: u64,
    requested_target: Option<AffinityTarget>,
    reason: &str,
) {
    let config = config();
    if !config.matches(session_id) {
        return;
    }

    let requested_worker_bound = requested_target.is_some();
    let requested_worker_id = requested_target.map(|target| target.worker_id).unwrap_or(0);
    let requested_dp_rank = requested_target
        .and_then(|target| target.dp_rank)
        .map(i64::from)
        .unwrap_or(-1);
    let requested_dp_rank_bound = requested_dp_rank >= 0;
    let trace_prefix = config.session_prefix.as_deref().unwrap_or("");
    let initialize_count = next(&INITIALIZE_COUNT);
    let fallback_rebind_count = fallback_rebind_count();
    tracing::info!(
        target: "dynamo::session_affinity",
        session_affinity_event = "initialize",
        session_id,
        revision,
        reason,
        trace_prefix,
        requested_worker_bound,
        requested_worker_id,
        requested_dp_rank_bound,
        requested_dp_rank,
        initialize_count,
        fallback_rebind_count,
        "session affinity trace"
    );
}

pub(super) fn commit(session_id: &str, revision: u64, target: AffinityTarget) {
    let config = config();
    if !config.matches(session_id) {
        return;
    }

    let dp_rank = target.dp_rank.map(i64::from).unwrap_or(-1);
    let dp_rank_bound = dp_rank >= 0;
    let trace_prefix = config.session_prefix.as_deref().unwrap_or("");
    let commit_count = next(&COMMIT_COUNT);
    let fallback_rebind_count = fallback_rebind_count();
    tracing::info!(
        target: "dynamo::session_affinity",
        session_affinity_event = "commit",
        session_id,
        revision,
        trace_prefix,
        worker_id = target.worker_id,
        dp_rank_bound,
        dp_rank,
        commit_count,
        fallback_rebind_count,
        "session affinity trace"
    );
}

pub(super) fn bound_reuse(
    session_id: &str,
    revision: u64,
    target: AffinityTarget,
    active_leases: usize,
) {
    let config = config();
    if !config.matches(session_id) {
        return;
    }

    let dp_rank = target.dp_rank.map(i64::from).unwrap_or(-1);
    let dp_rank_bound = dp_rank >= 0;
    let trace_prefix = config.session_prefix.as_deref().unwrap_or("");
    let bound_reuse_count = next(&BOUND_REUSE_COUNT);
    let fallback_rebind_count = fallback_rebind_count();
    tracing::info!(
        target: "dynamo::session_affinity",
        session_affinity_event = "bound_reuse",
        session_id,
        revision,
        trace_prefix,
        worker_id = target.worker_id,
        dp_rank_bound,
        dp_rank,
        active_leases,
        bound_reuse_count,
        fallback_rebind_count,
        "session affinity trace"
    );
}

pub(super) fn bound_dispatch(
    session_id: &str,
    revision: u64,
    target: AffinityTarget,
    phase: RequestPhase,
    acquisition_kind: &str,
) {
    let config = config();
    if !config.matches(session_id) {
        return;
    }

    let dp_rank = target.dp_rank.map(i64::from).unwrap_or(-1);
    let dp_rank_bound = dp_rank >= 0;
    let role = match phase {
        RequestPhase::Prefill => "context",
        RequestPhase::Decode => "generation",
        RequestPhase::Aggregated => "aggregated",
    };
    let trace_prefix = config.session_prefix.as_deref().unwrap_or("");
    let bound_dispatch_count = next(&BOUND_DISPATCH_COUNT);
    let fallback_rebind_count = fallback_rebind_count();
    tracing::info!(
        target: "dynamo::session_affinity",
        session_affinity_event = "bound_dispatch",
        session_id,
        revision,
        trace_prefix,
        acquisition_kind,
        phase = %phase,
        role,
        worker_id = target.worker_id,
        dp_rank_bound,
        dp_rank,
        bound_dispatch_count,
        fallback_rebind_count,
        "session affinity trace"
    );
}

pub(super) fn bound_target_fallback_rebind(
    session_id: &str,
    revision: u64,
    target: AffinityTarget,
) {
    let config = config();
    if !config.matches(session_id) {
        return;
    }

    let dp_rank = target.dp_rank.map(i64::from).unwrap_or(-1);
    let dp_rank_bound = dp_rank >= 0;
    let trace_prefix = config.session_prefix.as_deref().unwrap_or("");
    let fallback_rebind_count = next(&FALLBACK_REBIND_COUNT);
    tracing::info!(
        target: "dynamo::session_affinity",
        session_affinity_event = "bound_target_fallback_rebind",
        session_id,
        revision,
        trace_prefix,
        worker_id = target.worker_id,
        dp_rank_bound,
        dp_rank,
        fallback_rebind_count,
        "session affinity trace"
    );
}

#[cfg(test)]
mod tests {
    use super::{TraceConfig, parse_enabled};

    #[test]
    fn session_affinity_trace_gate_is_opt_in_and_prefix_filtered() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("no"),
            Some("invalid"),
        ] {
            assert!(!parse_enabled(value), "unexpected enabled value: {value:?}");
        }
        for value in ["1", "true", "TRUE", " on ", "yes"] {
            assert!(parse_enabled(Some(value)), "disabled truthy value: {value}");
        }

        assert!(TraceConfig::from_values(Some("true"), None).matches("any-session"));
        assert!(TraceConfig::from_values(Some("true"), Some("  ")).matches("any-session"));

        let filtered = TraceConfig::from_values(Some("true"), Some(" dyn-affinity-trace- "));
        assert!(filtered.matches("dyn-affinity-trace-0001"));
        assert!(!filtered.matches("benchmark-session-0001"));

        let disabled = TraceConfig::from_values(Some("false"), Some("dyn-affinity-trace-"));
        assert!(!disabled.matches("dyn-affinity-trace-0001"));
    }
}
