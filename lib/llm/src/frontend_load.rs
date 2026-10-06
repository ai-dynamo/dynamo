// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned load facts published to the KV DC Relay over Dynamo's event plane.
//!
//! Every request is a [`TrackedRequest`] from admission
//! ([`FrontendLoadTracker::start_request`]) to [`TrackedRequest::finish`]. Once per
//! [`FRONTEND_LOAD_PUBLISH_INTERVAL`] the frontend publishes a [`FrontendLoadFrame`]
//! with in-flight gauges and cumulative counters for each committed model, in the
//! namespace chosen by [`frontend_load_namespace`].

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use std::time::Duration;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::Context;
use dynamo_runtime::transports::event_plane::EventPublisher;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::discovery::CommittedModelView;
use crate::discovery::ModelManager;
use crate::utils::retry::{Backoff, FailureStreak};

pub(crate) const FRONTEND_LOAD_TOPIC: &str = "frontend-load";
pub(crate) const FRONTEND_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const PUBLISHER_RETRY_MAX: Duration = Duration::from_secs(30);

/// Namespace a frontend publishes its load in: its configured exact namespace,
/// else `DYN_NAMESPACE`, else `dynamo`.
///
/// This is the rendezvous with the KV DC Relay, which subscribes in every
/// namespace that holds one of its pools and in its own namespace
/// (`DYN_NAMESPACE`, else `dynamo`). A frontend without an exact namespace is
/// therefore seen by a Relay that shares its `DYN_NAMESPACE`.
pub(crate) fn frontend_load_namespace(configured: Option<String>) -> String {
    configured
        .or_else(|| std::env::var("DYN_NAMESPACE").ok())
        .unwrap_or_else(|| "dynamo".to_string())
}

/// One frontend's complete load report. Each frame replaces the previous frame
/// from the same frontend incarnation; the Relay never merges partial frames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendLoadFrame {
    /// Discovery instance id of the frontend process. It survives event-plane
    /// publisher reconnects (which change the envelope `publisher_id`) and, on
    /// Kubernetes, container restarts (the id is derived from the pod name).
    pub(crate) frontend_instance_id: u64,
    /// Random value chosen once per process. A new incarnation for the same
    /// `frontend_instance_id` means the process restarted: its `sequence`
    /// restarts at 1 and its cumulative counters restart at 0.
    pub(crate) incarnation: u64,
    /// Starts at 1 and strictly increases within one incarnation.
    pub(crate) sequence: u64,
    pub(crate) serving_ready: bool,
    pub(crate) models: Vec<FrontendModelLoad>,
}

/// Load for one canonical model on one frontend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendModelLoad {
    pub(crate) model: String,
    pub(crate) aliases: Vec<String>,
    pub(crate) gauges: RequestGauges,
    pub(crate) totals: RequestTotals,
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

    /// Growth since `previous`. A counter that went backwards restarted, so its
    /// whole current value is growth.
    pub(crate) fn growth_since(&self, previous: &Self) -> Self {
        let grow = |current: u64, previous: u64| current.checked_sub(previous).unwrap_or(current);
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

/// Per-process request tracker, shared by the request path and the publisher.
pub(crate) struct FrontendLoadTracker {
    /// See [`FrontendLoadFrame::incarnation`].
    incarnation: u64,
    last_sequence: AtomicU64,
    /// Keyed by metric model name. Bounded by the models this process has
    /// served plus the unknown-model sentinel.
    models: RwLock<HashMap<String, Arc<ModelCounters>>>,
}

impl Default for FrontendLoadTracker {
    fn default() -> Self {
        Self {
            incarnation: rand::random(),
            last_sequence: AtomicU64::new(0),
            models: RwLock::default(),
        }
    }
}

/// Live counters for one model. Gauges are signed because clones of one
/// request racing through different transitions may apply their gauge updates
/// out of order; a gauge can dip below zero for an instant and frames clamp it.
#[derive(Default)]
struct ModelCounters {
    requests_awaiting_first_token: AtomicI64,
    requests_generating: AtomicI64,
    awaiting_first_token_input_tokens: AtomicI64,
    inflight_input_tokens: AtomicI64,
    requests_started: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    requests_cancelled: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
}

impl ModelCounters {
    fn gauges(&self) -> RequestGauges {
        let read = |gauge: &AtomicI64| u64::try_from(gauge.load(Relaxed)).unwrap_or(0);
        RequestGauges {
            requests_awaiting_first_token: read(&self.requests_awaiting_first_token),
            requests_generating: read(&self.requests_generating),
            awaiting_first_token_input_tokens: read(&self.awaiting_first_token_input_tokens),
            inflight_input_tokens: read(&self.inflight_input_tokens),
        }
    }

    fn totals(&self) -> RequestTotals {
        RequestTotals {
            requests_started: self.requests_started.load(Relaxed),
            requests_completed: self.requests_completed.load(Relaxed),
            requests_failed: self.requests_failed.load(Relaxed),
            requests_cancelled: self.requests_cancelled.load(Relaxed),
            input_tokens: self.input_tokens.load(Relaxed),
            output_tokens: self.output_tokens.load(Relaxed),
        }
    }
}

impl FrontendLoadTracker {
    /// Start tracking a request for `model`, its canonical metric model name.
    pub(crate) fn start_request(&self, model: &str) -> TrackedRequest {
        let counters = self.counters(model);
        counters.requests_started.fetch_add(1, Relaxed);
        counters.requests_awaiting_first_token.fetch_add(1, Relaxed);
        TrackedRequest(Arc::new(RequestProgress {
            model: counters,
            state: AtomicU64::new(0),
        }))
    }

    fn counters(&self, model: &str) -> Arc<ModelCounters> {
        if let Some(counters) = self.models.read().get(model) {
            return counters.clone();
        }
        self.models
            .write()
            .entry(model.to_owned())
            .or_default()
            .clone()
    }

    /// Build the next frame: one entry per committed primary model, including
    /// requests tracked under any of its aliases. Requests for any other name
    /// (for example the unknown-model metrics sentinel) are tracked but not
    /// published.
    pub(crate) fn next_frame(
        &self,
        frontend_instance_id: u64,
        serving_ready: bool,
        models: Vec<CommittedModelView>,
    ) -> FrontendLoadFrame {
        let counters = self.models.read();
        let models = models
            .into_iter()
            .map(|view| {
                let mut load = FrontendModelLoad {
                    model: view.name,
                    aliases: view.aliases,
                    ..Default::default()
                };
                for name in std::iter::once(&load.model).chain(&load.aliases) {
                    if let Some(model) = counters.get(name) {
                        load.gauges.add(&model.gauges());
                        load.totals.add(&model.totals());
                    }
                }
                load
            })
            .collect();
        FrontendLoadFrame {
            frontend_instance_id,
            incarnation: self.incarnation,
            sequence: self.last_sequence.fetch_add(1, Relaxed) + 1,
            serving_ready,
            models,
        }
    }
}

/// One in-flight request, from [`FrontendLoadTracker::start_request`] to
/// [`finish`](Self::finish). Clones refer to the same request.
///
/// This is everything the HTTP layer and preprocessor tell the tracker:
/// prompt tokens (`add_input_tokens` / `observe_input_tokens`), output tokens
/// (`add_output_tokens`), and the outcome (`finish`).
#[derive(Clone)]
pub(crate) struct TrackedRequest(Arc<RequestProgress>);

struct RequestProgress {
    model: Arc<ModelCounters>,
    /// Known prompt tokens plus the [`GENERATING`] and [`FINISHED`] flags,
    /// packed so each transition is a single atomic update.
    state: AtomicU64,
}

const FINISHED: u64 = 1 << 63;
const GENERATING: u64 = 1 << 62;
const INPUT_TOKENS: u64 = GENERATING - 1;

const CONTEXT_KEY: &str = "frontend_load.tracked_request";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestOutcome {
    Completed,
    Failed,
    Cancelled,
}

impl TrackedRequest {
    /// Make this request reachable from pipeline stages that receive `context`.
    pub(crate) fn attach_to<T: Data>(&self, context: &mut Context<T>) {
        context.insert(CONTEXT_KEY, self.clone());
    }

    /// The request attached to `context` by [`attach_to`](Self::attach_to).
    pub(crate) fn from_context<T: Data>(context: &Context<T>) -> Option<Self> {
        let request: Arc<Self> = context.get_optional(CONTEXT_KEY).ok().flatten()?;
        Some(Self::clone(&request))
    }

    /// Count a prompt of `tokens` tokens as soon as it is tokenized. A batched
    /// completion request reports each of its prompts.
    pub(crate) fn add_input_tokens(&self, tokens: usize) {
        self.update_input_tokens(|known| known.saturating_add(tokens as u64));
    }

    /// Report the request's whole prompt size from a response annotation, which
    /// repeats it on every chunk. Never lowers the known size, so it neither
    /// double counts nor undoes [`add_input_tokens`](Self::add_input_tokens).
    pub(crate) fn observe_input_tokens(&self, tokens: usize) {
        self.update_input_tokens(|known| known.max(tokens as u64));
    }

    fn update_input_tokens(&self, input_tokens: impl Fn(u64) -> u64) {
        let mut added = 0;
        let Ok(previous) = self.0.state.fetch_update(Relaxed, Relaxed, |state| {
            if state & FINISHED != 0 {
                return None;
            }
            let known = state & INPUT_TOKENS;
            added = input_tokens(known).min(INPUT_TOKENS).saturating_sub(known);
            (added > 0).then_some(state + added)
        }) else {
            return;
        };
        let model = &self.0.model;
        model.input_tokens.fetch_add(added, Relaxed);
        let added = added as i64;
        model.inflight_input_tokens.fetch_add(added, Relaxed);
        if previous & GENERATING == 0 {
            model
                .awaiting_first_token_input_tokens
                .fetch_add(added, Relaxed);
        }
    }

    /// Count `tokens` newly generated output tokens (a delta, not a total). The
    /// first output token moves the request from awaiting to generating.
    pub(crate) fn add_output_tokens(&self, tokens: usize) {
        let state = self.0.state.load(Relaxed);
        if tokens == 0 || state & FINISHED != 0 {
            return;
        }
        let model = &self.0.model;
        model.output_tokens.fetch_add(tokens as u64, Relaxed);
        if state & GENERATING != 0 {
            return;
        }
        let previous = self.0.state.fetch_or(GENERATING, Relaxed);
        if previous & (GENERATING | FINISHED) != 0 {
            return;
        }
        let input = (previous & INPUT_TOKENS) as i64;
        model.requests_awaiting_first_token.fetch_sub(1, Relaxed);
        model
            .awaiting_first_token_input_tokens
            .fetch_sub(input, Relaxed);
        model.requests_generating.fetch_add(1, Relaxed);
    }

    /// Stop tracking the request. Only the first call on any clone counts.
    pub(crate) fn finish(&self, outcome: RequestOutcome) {
        let previous = self.0.state.fetch_or(FINISHED, Relaxed);
        if previous & FINISHED != 0 {
            return;
        }
        let model = &self.0.model;
        let input = (previous & INPUT_TOKENS) as i64;
        model.inflight_input_tokens.fetch_sub(input, Relaxed);
        if previous & GENERATING != 0 {
            model.requests_generating.fetch_sub(1, Relaxed);
        } else {
            model.requests_awaiting_first_token.fetch_sub(1, Relaxed);
            model
                .awaiting_first_token_input_tokens
                .fetch_sub(input, Relaxed);
        }
        let total = match outcome {
            RequestOutcome::Completed => &model.requests_completed,
            RequestOutcome::Failed => &model.requests_failed,
            RequestOutcome::Cancelled => &model.requests_cancelled,
        };
        total.fetch_add(1, Relaxed);
    }
}

/// Publish a [`FrontendLoadFrame`] to `namespace` every
/// [`FRONTEND_LOAD_PUBLISH_INTERVAL`] until `cancel` fires. Every frontend with a
/// [`DistributedRuntime`] runs this.
pub(crate) fn start_frontend_load_publisher(
    runtime: Arc<DistributedRuntime>,
    namespace: String,
    manager: Arc<ModelManager>,
    tracker: Arc<FrontendLoadTracker>,
    is_ready: impl Fn() -> bool + Send + 'static,
    cancel: CancellationToken,
) {
    runtime.runtime().secondary().spawn(async move {
        let namespace_handle = match runtime.namespace(namespace.clone()) {
            Ok(namespace) => namespace,
            Err(error) => {
                tracing::error!(%namespace, %error, "cannot create frontend load event namespace");
                return;
            }
        };
        let frontend_instance_id = runtime.connection_id();
        let mut interval = tokio::time::interval_at(
            tokio::time::Instant::now() + FRONTEND_LOAD_PUBLISH_INTERVAL,
            FRONTEND_LOAD_PUBLISH_INTERVAL,
        );
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut retry = Backoff::new(FRONTEND_LOAD_PUBLISH_INTERVAL, PUBLISHER_RETRY_MAX);
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
                    let frame = tracker.next_frame(
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

    const MODEL: &str = "model-a";

    fn load(tracker: &FrontendLoadTracker) -> FrontendModelLoad {
        let view = CommittedModelView {
            name: MODEL.to_string(),
            aliases: Vec::new(),
        };
        let mut frame = tracker.next_frame(7, true, vec![view]);
        frame.models.remove(0)
    }

    #[test]
    fn request_lifecycle_updates_gauges_and_totals() {
        let tracker = FrontendLoadTracker::default();
        let request = tracker.start_request(MODEL);
        let load_before_tokenization = load(&tracker);
        assert_eq!(
            load_before_tokenization
                .gauges
                .requests_awaiting_first_token,
            1
        );
        assert_eq!(load_before_tokenization.gauges.inflight_input_tokens, 0);

        request.add_input_tokens(12);
        let awaiting = load(&tracker);
        assert_eq!(
            awaiting.gauges,
            RequestGauges {
                requests_awaiting_first_token: 1,
                requests_generating: 0,
                awaiting_first_token_input_tokens: 12,
                inflight_input_tokens: 12,
            }
        );

        request.add_output_tokens(3);
        request.add_output_tokens(2);
        let generating = load(&tracker);
        assert_eq!(
            generating.gauges,
            RequestGauges {
                requests_awaiting_first_token: 0,
                requests_generating: 1,
                awaiting_first_token_input_tokens: 0,
                inflight_input_tokens: 12,
            }
        );

        request.finish(RequestOutcome::Completed);
        let finished = load(&tracker);
        assert_eq!(finished.gauges, RequestGauges::default());
        assert_eq!(
            finished.totals,
            RequestTotals {
                requests_started: 1,
                requests_completed: 1,
                input_tokens: 12,
                output_tokens: 5,
                ..Default::default()
            }
        );
    }

    #[test]
    fn frames_carry_one_incarnation_and_increasing_sequences() {
        let tracker = FrontendLoadTracker::default();
        let first = tracker.next_frame(7, true, Vec::new());
        let second = tracker.next_frame(7, false, Vec::new());
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(first.incarnation, second.incarnation);
    }

    #[test]
    fn annotation_reports_do_not_double_count_prompt_tokens() {
        let tracker = FrontendLoadTracker::default();
        let request = tracker.start_request(MODEL);
        // A two-prompt batch: the preprocessor reports each prompt...
        request.add_input_tokens(3);
        request.add_input_tokens(5);
        // ...and every chunk annotation repeats one prompt's size.
        request.observe_input_tokens(5);
        request.observe_input_tokens(3);
        assert_eq!(load(&tracker).gauges.inflight_input_tokens, 8);

        // Without a preprocessor report, the annotation alone sets the size.
        let annotated = tracker.start_request(MODEL);
        annotated.observe_input_tokens(10);
        annotated.observe_input_tokens(10);
        let load = load(&tracker);
        assert_eq!(load.gauges.awaiting_first_token_input_tokens, 18);
        assert_eq!(load.totals.input_tokens, 18);
    }

    #[test]
    fn finish_counts_once_across_clones() {
        let tracker = FrontendLoadTracker::default();
        let request = tracker.start_request(MODEL);
        request.add_input_tokens(4);
        let clone = request.clone();

        clone.finish(RequestOutcome::Cancelled);
        request.finish(RequestOutcome::Completed);
        request.add_input_tokens(4);
        request.add_output_tokens(1);

        let load = load(&tracker);
        assert_eq!(load.gauges, RequestGauges::default());
        assert_eq!(
            load.totals,
            RequestTotals {
                requests_started: 1,
                requests_cancelled: 1,
                input_tokens: 4,
                ..Default::default()
            }
        );
    }

    #[test]
    fn uncommitted_models_are_tracked_but_not_published() {
        let tracker = FrontendLoadTracker::default();
        let _request = tracker.start_request("unknown");

        let frame = tracker.next_frame(
            7,
            true,
            vec![CommittedModelView {
                name: MODEL.to_string(),
                aliases: Vec::new(),
            }],
        );
        assert_eq!(frame.models.len(), 1);
        assert_eq!(frame.models[0].model, MODEL);
        assert_eq!(frame.models[0].totals, RequestTotals::default());
    }

    #[test]
    fn concurrent_requests_keep_exact_counts() {
        const THREADS: u64 = 8;
        const REQUESTS: u64 = 500;
        let tracker = FrontendLoadTracker::default();
        let held = tracker.start_request(MODEL);
        held.add_input_tokens(7);

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..REQUESTS {
                        let request = tracker.start_request(MODEL);
                        request.add_input_tokens(2);
                        request.observe_input_tokens(2);
                        request.add_output_tokens(1);
                        request.add_output_tokens(1);
                        request.finish(RequestOutcome::Completed);
                    }
                });
            }
        });

        let load = load(&tracker);
        assert_eq!(
            load.gauges,
            RequestGauges {
                requests_awaiting_first_token: 1,
                requests_generating: 0,
                awaiting_first_token_input_tokens: 7,
                inflight_input_tokens: 7,
            }
        );
        let finished = THREADS * REQUESTS;
        assert_eq!(
            load.totals,
            RequestTotals {
                requests_started: finished + 1,
                requests_completed: finished,
                input_tokens: 2 * finished + 7,
                output_tokens: 2 * finished,
                ..Default::default()
            }
        );
    }
}
