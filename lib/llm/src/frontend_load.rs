// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend metric snapshots and recent-session hints published on the event plane.
//!
//! All fields in this first release are required. Fields added later must use
//! `#[serde(default)]`, and `Option` when absence does not mean zero.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::transports::event_plane::EventPublisher;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;
use crate::http::service::metrics::Metrics;
use crate::utils::retry::{Backoff, FailureStreak};

pub(crate) const FRONTEND_LOAD_TOPIC: &str = "frontend_load";
pub(crate) const FRONTEND_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const PUBLISHER_RETRY_INITIAL: Duration = Duration::from_millis(100);
const PUBLISHER_RETRY_MAX: Duration = Duration::from_secs(5);

/// Exact configured namespace, else `DYN_NAMESPACE`, else `dynamo`.
pub(crate) fn frontend_load_namespace(configured: Option<String>) -> String {
    configured
        .or_else(|| std::env::var("DYN_NAMESPACE").ok())
        .unwrap_or_else(|| "dynamo".to_string())
}

/// One frontend's complete load report. Each frame replaces the previous frame
/// from the same frontend incarnation; the Relay never merges partial frames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendLoadFrame {
    /// Stable across event-publisher reconnects.
    pub(crate) frontend_instance_id: u64,
    /// Identifies the lifetime of the counters; reconnecting does not reset it.
    pub(crate) incarnation: u64,
    /// Starts at 1 and strictly increases within one incarnation.
    pub(crate) sequence: u64,
    pub(crate) is_serving_ready: bool,
    pub(crate) models: Vec<FrontendModelLoad>,
}

/// Load for one canonical model on one frontend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendModelLoad {
    pub(crate) model: String,
    pub(crate) is_ready: bool,
    pub(crate) gauges: RequestGauges,
    pub(crate) totals: RequestTotals,
    /// Recent request observations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) sessions: Vec<SessionObservation>,
}

pub(crate) const MAX_SESSION_ID_BYTES: usize = 256;
pub(crate) const MAX_FRONTEND_SESSIONS: usize = 4_096;
pub(crate) const FRONTEND_SESSION_RETENTION: Duration =
    Duration::from_millis(crate::pool_relay::wan::grpc::protocol::SESSION_HINT_RETENTION_MS);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SessionObservation {
    pub(crate) session_id: String,
    pub(crate) last_seen_ms: u64,
}

struct SeenSession {
    observed_at: Instant,
    last_seen_ms: u64,
}

/// Recent sessions for frontend reports. Busy snapshots must not block requests.
#[derive(Default)]
pub(crate) struct FrontendSessions {
    enabled: AtomicBool,
    entries: Mutex<HashMap<(String, String), SeenSession>>,
}

impl FrontendSessions {
    pub(crate) fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn observe(&self, model: &str, session_id: &str) {
        let last_seen_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64;
        self.observe_at(model, session_id, Instant::now(), last_seen_ms);
    }

    fn observe_at(&self, model: &str, session_id: &str, now: Instant, last_seen_ms: u64) {
        if !self.is_enabled()
            || session_id.is_empty()
            || session_id.len() > MAX_SESSION_ID_BYTES
            || session_id.trim() != session_id
            || session_id.chars().any(char::is_control)
        {
            return;
        }
        let Some(mut entries) = self.entries.try_lock() else {
            return;
        };
        let key = (model.to_owned(), session_id.to_owned());
        if let Some(seen) = entries.get_mut(&key) {
            seen.observed_at = seen.observed_at.max(now);
            seen.last_seen_ms = seen.last_seen_ms.max(last_seen_ms);
        } else if entries.len() < MAX_FRONTEND_SESSIONS {
            entries.insert(
                key,
                SeenSession {
                    observed_at: now,
                    last_seen_ms,
                },
            );
        }
    }

    pub(crate) fn snapshot(&self) -> HashMap<String, Vec<SessionObservation>> {
        if !self.is_enabled() {
            return HashMap::new();
        }
        self.snapshot_at(Instant::now())
    }

    fn snapshot_at(&self, now: Instant) -> HashMap<String, Vec<SessionObservation>> {
        let mut entries = self.entries.lock();
        entries.retain(|_, seen| {
            now.saturating_duration_since(seen.observed_at) < FRONTEND_SESSION_RETENTION
        });
        let mut models = HashMap::<String, Vec<SessionObservation>>::new();
        for ((model, session_id), seen) in entries.iter() {
            models
                .entry(model.clone())
                .or_default()
                .push(SessionObservation {
                    session_id: session_id.clone(),
                    last_seen_ms: seen.last_seen_ms,
                });
        }
        models
    }
}

/// In-flight requests at publish time.
///
/// Prompt token counts are known once a request has been tokenized; a request
/// still being tokenized contributes zero input tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RequestGauges {
    /// Admitted requests that have not produced their first output token.
    pub(crate) requests_awaiting_first_token: u64,
    pub(crate) requests_generating: u64,
    /// Prompt tokens of `requests_awaiting_first_token`.
    pub(crate) awaiting_first_token_input_tokens: u64,
    /// Prompt tokens of every in-flight request.
    pub(crate) inflight_input_tokens: u64,
}

impl RequestGauges {
    pub(crate) fn add(&mut self, other: &Self) {
        self.requests_awaiting_first_token = self
            .requests_awaiting_first_token
            .saturating_add(other.requests_awaiting_first_token);
        self.requests_generating = self
            .requests_generating
            .saturating_add(other.requests_generating);
        self.awaiting_first_token_input_tokens = self
            .awaiting_first_token_input_tokens
            .saturating_add(other.awaiting_first_token_input_tokens);
        self.inflight_input_tokens = self
            .inflight_input_tokens
            .saturating_add(other.inflight_input_tokens);
    }
}

/// Cumulative request counters since the frontend incarnation started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RequestTotals {
    pub(crate) requests_started: u64,
    pub(crate) requests_completed: u64,
    pub(crate) requests_failed: u64,
    pub(crate) requests_cancelled: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
}

impl RequestTotals {
    pub(crate) fn add(&mut self, other: &Self) {
        self.requests_started = self.requests_started.saturating_add(other.requests_started);
        self.requests_completed = self
            .requests_completed
            .saturating_add(other.requests_completed);
        self.requests_failed = self.requests_failed.saturating_add(other.requests_failed);
        self.requests_cancelled = self
            .requests_cancelled
            .saturating_add(other.requests_cancelled);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
    }

    /// Growth since `previous`, from the same frontend incarnation. A total that
    /// went backwards (an alias unregistered, say) grows by nothing; a restart
    /// arrives as a new incarnation and starts from a fresh baseline instead.
    pub(crate) fn growth_since(&self, previous: &Self) -> Self {
        let grow = |current: u64, previous: u64| current.saturating_sub(previous);
        Self {
            requests_started: grow(self.requests_started, previous.requests_started),
            requests_completed: grow(self.requests_completed, previous.requests_completed),
            requests_failed: grow(self.requests_failed, previous.requests_failed),
            requests_cancelled: grow(self.requests_cancelled, previous.requests_cancelled),
            input_tokens: grow(self.input_tokens, previous.input_tokens),
            output_tokens: grow(self.output_tokens, previous.output_tokens),
        }
    }
}

fn session_hints_publishing_enabled() -> bool {
    crate::pool_relay::serving_load_publishing_enabled()
        && dynamo_runtime::config::env_is_truthy(
            dynamo_runtime::config::environment_names::llm::DYN_SESSION_HINTS_PUBLISH,
        )
}

/// Start periodic publishing after the caller checks `DYN_SERVING_LOAD_PUBLISH`.
pub(crate) fn start_frontend_load_publisher(
    runtime: DistributedRuntime,
    namespace: String,
    manager: Arc<ModelManager>,
    metrics: Arc<Metrics>,
    is_ready: impl Fn() -> bool + Send + 'static,
    cancel: CancellationToken,
) {
    if session_hints_publishing_enabled() {
        metrics.enable_session_observation();
    }
    runtime.runtime().secondary().spawn(async move {
        let namespace_handle = match runtime.namespace(namespace.clone()) {
            Ok(namespace) => namespace,
            Err(error) => {
                tracing::error!(%namespace, %error, "cannot create frontend load event namespace");
                return;
            }
        };
        let frontend_instance_id = runtime.connection_id();
        // Publish on the first tick, matching the scheduler-load publisher.
        let mut interval = tokio::time::interval(FRONTEND_LOAD_PUBLISH_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut retry = Backoff::new(PUBLISHER_RETRY_INITIAL, PUBLISHER_RETRY_MAX);
        let mut failures = FailureStreak::default();

        // Reconnect loop: a publish failure drops the publisher and builds a new one.
        loop {
            let publisher = tokio::select! {
                _ = cancel.cancelled() => return,
                result = EventPublisher::for_namespace(&namespace_handle, FRONTEND_LOAD_TOPIC) => result,
            };
            let error = match publisher {
                Ok(publisher) => loop {
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        _ = interval.tick() => {}
                    }
                    let frame = metrics.frontend_load(
                        frontend_instance_id,
                        is_ready(),
                        manager.committed_model_views(),
                    );
                    let result = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        result = publisher.publish(&frame) => result,
                    };
                    match result {
                        Ok(()) => {
                            retry.reset();
                            if let Some(failures) = failures.recover() {
                                tracing::info!(%namespace, failures, "frontend load publishing recovered");
                            }
                        }
                        Err(error) => break error,
                    }
                },
                Err(error) => error,
            };
            let delay = retry.next_delay();
            if failures.fail() {
                tracing::warn!(%namespace, %error, retry_ms = delay.as_millis(), "frontend load publishing failed; reconnecting");
            } else {
                tracing::debug!(%namespace, %error, failures = failures.failures(), retry_ms = delay.as_millis(), "frontend load publishing still failing");
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn session_hints_require_both_publishing_flags() {
        use dynamo_runtime::config::environment_names::llm::{
            DYN_SERVING_LOAD_PUBLISH, DYN_SESSION_HINTS_PUBLISH,
        };

        for load in [None, Some("1")] {
            for hints in [None, Some("1")] {
                temp_env::with_vars(
                    [
                        (DYN_SERVING_LOAD_PUBLISH, load),
                        (DYN_SESSION_HINTS_PUBLISH, hints),
                    ],
                    || {
                        assert_eq!(
                            session_hints_publishing_enabled(),
                            load.is_some() && hints.is_some()
                        )
                    },
                );
            }
        }
    }

    #[test]
    fn disabled_session_snapshot_does_not_lock_or_emit_session_fields() {
        let sessions = FrontendSessions::default();
        let _guard = sessions.entries.lock();
        assert!(sessions.snapshot().is_empty());
        let model = FrontendModelLoad::default();
        let json = serde_json::to_value(&model).unwrap();
        assert!(json.get("sessions").is_none());
    }

    #[test]
    fn session_observations_expire_from_last_monotonic_observation() {
        let sessions = FrontendSessions::default();
        sessions.enable();
        let start = Instant::now();
        sessions.observe_at("a", "session", start, 1_000);
        sessions.observe_at("a", "session", start + Duration::from_secs(30), 900);
        let before = sessions.snapshot_at(start + FRONTEND_SESSION_RETENTION);
        assert_eq!(before["a"][0].last_seen_ms, 1_000);
        assert!(
            sessions
                .snapshot_at(start + FRONTEND_SESSION_RETENTION + Duration::from_secs(30))
                .is_empty()
        );
    }

    #[test]
    fn session_observations_are_bounded_and_isolated_by_model() {
        let sessions = FrontendSessions::default();
        sessions.enable();
        let now = Instant::now();
        sessions.observe_at("a", "shared", now, 100);
        sessions.observe_at("b", "shared", now, 200);
        for id in 2..MAX_FRONTEND_SESSIONS {
            sessions.observe_at("a", &id.to_string(), now, 100);
        }
        sessions.observe_at("a", "overflow", now, 100);
        sessions.observe_at("b", "shared", now, 300);
        let snapshot = sessions.snapshot_at(now);
        assert_eq!(
            snapshot.values().map(Vec::len).sum::<usize>(),
            MAX_FRONTEND_SESSIONS
        );
        assert_eq!(snapshot["b"][0].last_seen_ms, 300);
        assert!(
            !snapshot["a"]
                .iter()
                .any(|seen| seen.session_id == "overflow")
        );
        assert!(
            sessions
                .snapshot_at(now + FRONTEND_SESSION_RETENTION)
                .is_empty()
        );
        sessions.observe_at("a", "overflow", now + FRONTEND_SESSION_RETENTION, 400);
        assert_eq!(
            sessions.snapshot_at(now + FRONTEND_SESSION_RETENTION)["a"][0].session_id,
            "overflow"
        );
    }

    #[test]
    fn session_observations_skip_a_busy_tracker_and_resume_after_release() {
        let sessions = FrontendSessions::default();
        sessions.enable();
        let now = Instant::now();
        let snapshot_guard = sessions.entries.lock();
        sessions.observe_at("a", "busy", now, 100);
        drop(snapshot_guard);
        assert!(sessions.snapshot_at(now).is_empty());
        sessions.observe_at("a", "available", now, 200);
        assert_eq!(sessions.snapshot_at(now)["a"][0].session_id, "available");
    }

    #[test]
    fn session_observations_require_publishing_and_valid_ids() {
        let sessions = FrontendSessions::default();
        let now = Instant::now();
        sessions.observe_at("a", "disabled", now, 1);
        sessions.enable();
        for id in [
            "".to_string(),
            "a\nb".to_string(),
            " padded ".to_string(),
            "x".repeat(MAX_SESSION_ID_BYTES + 1),
        ] {
            sessions.observe_at("a", &id, now, 1);
        }
        assert!(sessions.snapshot_at(now).is_empty());
        sessions.observe_at("a", &"x".repeat(MAX_SESSION_ID_BYTES), now, 1);
        assert_eq!(sessions.snapshot_at(now)["a"].len(), 1);
    }

    /// A Relay decodes frames from newer frontends that carry fields it does not
    /// know, but never invents identity or zero load for a frame missing them.
    #[test]
    fn frames_decode_across_versions() {
        use dynamo_runtime::transports::event_plane::Codec;
        use serde_json::json;

        let codec = Codec::default();
        let decode = |frame: serde_json::Value| {
            codec.decode_payload::<FrontendLoadFrame>(&codec.encode_payload(&frame).unwrap())
        };
        let gauges = json!({
            "requests_awaiting_first_token": 1, "requests_generating": 3,
            "awaiting_first_token_input_tokens": 5, "inflight_input_tokens": 7,
            "added_gauge": 1,
        });
        let totals = json!({
            "requests_started": 4, "requests_completed": 1, "requests_failed": 0,
            "requests_cancelled": 0, "input_tokens": 9, "output_tokens": 2,
            "added_total": 1,
        });

        let newer = decode(json!({
            "frontend_instance_id": 10, "incarnation": 1, "sequence": 2,
            "is_serving_ready": true, "added_frame_field": 1,
            "models": [{
                "model": "a", "is_ready": true, "aliases": [], "added_model_field": "x",
                "gauges": gauges, "totals": totals,
            }],
        }))
        .unwrap();
        assert!(newer.is_serving_ready);
        assert_eq!(newer.models[0].gauges.requests_generating, 3);
        assert_eq!(newer.models[0].totals.requests_started, 4);
        assert!(newer.models[0].sessions.is_empty());

        assert!(
            decode(json!({
                "incarnation": 1, "sequence": 2, "is_serving_ready": true, "models": [],
            }))
            .is_err()
        );
        assert!(
            decode(json!({
                "frontend_instance_id": 10, "incarnation": 1, "sequence": 2,
                "is_serving_ready": true,
                "models": [{"model": "a", "is_ready": true, "aliases": [], "totals": totals}],
            }))
            .is_err()
        );
    }
}
