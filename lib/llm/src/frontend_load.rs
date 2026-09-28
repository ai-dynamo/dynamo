// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned load facts published to the Relay over Dynamo's event plane.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::transports::event_plane::EventPublisher;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;
use crate::http::service::service_v2::ServiceObserver;

pub(crate) const FRONTEND_LOAD_TOPIC: &str = "frontend-load";
pub(crate) const FRONTEND_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendLoadFrame {
    pub(crate) frontend_instance_id: u64,
    pub(crate) sequence: u64,
    pub(crate) serving_ready: bool,
    pub(crate) models: Vec<FrontendModelLoad>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FrontendModelLoad {
    pub(crate) model: String,
    pub(crate) aliases: Vec<String>,
    pub(crate) pending_first_output_requests: u64,
    pub(crate) pending_first_output_input_tokens: Option<u64>,
    pub(crate) live_input_tokens: Option<u64>,
    pub(crate) input_processing_requests: u64,
    pub(crate) output_generation_requests: u64,
    pub(crate) requests_started_total: Option<u64>,
    pub(crate) requests_completed_total: Option<u64>,
    pub(crate) requests_failed_total: Option<u64>,
    pub(crate) requests_cancelled_total: Option<u64>,
    pub(crate) input_tokens_total: Option<u64>,
    pub(crate) output_tokens_total: Option<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct FrontendLoadMetrics {
    state: Arc<Mutex<FrontendLoadState>>,
}

#[derive(Default)]
struct FrontendLoadState {
    next_request: u64,
    frame_sequence: u64,
    live: HashMap<u64, LiveRequest>,
    counters: HashMap<String, ModelCounters>,
}

struct LiveRequest {
    model: String,
    input_tokens: Option<u64>,
    output_started: bool,
}

#[derive(Default)]
struct ModelCounters {
    requests_started: u64,
    requests_completed: u64,
    requests_failed: u64,
    requests_cancelled: u64,
    input_tokens: u64,
    output_tokens: u64,
    input_incomplete: bool,
    overflowed: bool,
}

impl ModelCounters {
    fn write_totals(&self, load: &mut FrontendModelLoad) {
        if self.overflowed {
            load.requests_started_total = None;
            load.requests_completed_total = None;
            load.requests_failed_total = None;
            load.requests_cancelled_total = None;
            load.input_tokens_total = None;
            load.output_tokens_total = None;
            return;
        }
        load.requests_started_total = Some(self.requests_started);
        load.requests_completed_total = Some(self.requests_completed);
        load.requests_failed_total = Some(self.requests_failed);
        load.requests_cancelled_total = Some(self.requests_cancelled);
        load.input_tokens_total = (!self.input_incomplete).then_some(self.input_tokens);
        load.output_tokens_total = Some(self.output_tokens);
    }
}

#[derive(Clone)]
pub(crate) struct FrontendLoadRequest {
    metrics: FrontendLoadMetrics,
    id: u64,
}

#[derive(Clone, Copy)]
pub(crate) enum RequestOutcome {
    Completed,
    Failed,
    Cancelled,
}

impl FrontendLoadMetrics {
    pub(crate) fn start_request(&self, model: &str) -> FrontendLoadRequest {
        let mut state = self.state.lock();
        let id = state.next_request;
        state.next_request = state.next_request.wrapping_add(1);
        if state
            .live
            .insert(
                id,
                LiveRequest {
                    model: model.to_string(),
                    input_tokens: None,
                    output_started: false,
                },
            )
            .is_some()
        {
            // A complete u64 wrap requires more concurrent requests than the process can hold.
            tracing::error!(request = id, "frontend load request handle wrapped");
        }
        let counters = state.counters.entry(model.to_string()).or_default();
        add_counter(&mut counters.requests_started, 1, &mut counters.overflowed);
        FrontendLoadRequest {
            metrics: self.clone(),
            id,
        }
    }

    fn observe(&self, id: u64, input_tokens: usize, output_tokens: usize) {
        let input_tokens = u64::try_from(input_tokens).ok();
        let output_tokens = u64::try_from(output_tokens).ok();
        let mut state = self.state.lock();
        let Some(request) = state.live.get_mut(&id) else {
            return;
        };
        let model = request.model.clone();
        let newly_observed_input = match (request.input_tokens, input_tokens) {
            (None, Some(tokens)) => {
                request.input_tokens = Some(tokens);
                Some(tokens)
            }
            (Some(previous), Some(tokens)) if tokens > previous => {
                request.input_tokens = Some(tokens);
                Some(tokens - previous)
            }
            _ => None,
        };
        if output_tokens.is_some_and(|tokens| tokens > 0) {
            request.output_started = true;
        }
        let counters = state.counters.entry(model).or_default();
        if let Some(tokens) = newly_observed_input {
            add_counter(&mut counters.input_tokens, tokens, &mut counters.overflowed);
        } else if input_tokens.is_none() {
            counters.input_incomplete = true;
        }
        if let Some(tokens) = output_tokens {
            add_counter(
                &mut counters.output_tokens,
                tokens,
                &mut counters.overflowed,
            );
        } else {
            counters.overflowed = true;
        }
    }

    fn finish(&self, id: u64, outcome: RequestOutcome) {
        let mut state = self.state.lock();
        let Some(request) = state.live.remove(&id) else {
            return;
        };
        let counters = state.counters.entry(request.model).or_default();
        if request.input_tokens.is_none() {
            counters.input_incomplete = true;
        }
        let counter = match outcome {
            RequestOutcome::Completed => &mut counters.requests_completed,
            RequestOutcome::Failed => &mut counters.requests_failed,
            RequestOutcome::Cancelled => &mut counters.requests_cancelled,
        };
        add_counter(counter, 1, &mut counters.overflowed);
    }

    fn snapshot(
        &self,
        frontend_instance_id: u64,
        serving_ready: bool,
        manager: &ModelManager,
    ) -> Option<FrontendLoadFrame> {
        let registrations = manager.committed_model_views();
        let mut state = self.state.lock();
        let Some(sequence) = state.frame_sequence.checked_add(1) else {
            tracing::error!("frontend load frame sequence overflowed");
            return None;
        };
        state.frame_sequence = sequence;
        let mut gauges = BTreeMap::<String, FrontendModelLoad>::new();
        for registration in registrations {
            gauges.insert(
                registration.name.clone(),
                FrontendModelLoad {
                    model: registration.name,
                    aliases: registration.aliases,
                    pending_first_output_input_tokens: Some(0),
                    live_input_tokens: Some(0),
                    requests_started_total: Some(0),
                    requests_completed_total: Some(0),
                    requests_failed_total: Some(0),
                    requests_cancelled_total: Some(0),
                    input_tokens_total: Some(0),
                    output_tokens_total: Some(0),
                    ..Default::default()
                },
            );
        }

        for request in state.live.values() {
            let Some(load) = gauges.get_mut(&request.model) else {
                continue;
            };
            let input_tokens = request.input_tokens;
            if let (Some(total), Some(input_tokens)) = (&mut load.live_input_tokens, input_tokens) {
                if !checked_increment(total, input_tokens) {
                    return None;
                }
            } else {
                load.live_input_tokens = None;
            }
            if request.output_started {
                if !checked_increment(&mut load.output_generation_requests, 1) {
                    return None;
                }
            } else {
                if !checked_increment(&mut load.pending_first_output_requests, 1)
                    || !checked_increment(&mut load.input_processing_requests, 1)
                {
                    return None;
                }
                if let (Some(total), Some(input_tokens)) =
                    (&mut load.pending_first_output_input_tokens, input_tokens)
                {
                    if !checked_increment(total, input_tokens) {
                        return None;
                    }
                } else {
                    load.pending_first_output_input_tokens = None;
                }
            }
        }

        for (model, counters) in &state.counters {
            let Some(load) = gauges.get_mut(model) else {
                continue;
            };
            counters.write_totals(load);
        }

        Some(FrontendLoadFrame {
            frontend_instance_id,
            sequence,
            serving_ready,
            models: gauges.into_values().collect(),
        })
    }
}

impl FrontendLoadRequest {
    pub(crate) fn observe(&self, input_tokens: usize, output_tokens: usize) {
        self.metrics.observe(self.id, input_tokens, output_tokens);
    }

    pub(crate) fn finish(&self, outcome: RequestOutcome) {
        self.metrics.finish(self.id, outcome);
    }
}

fn add_counter(counter: &mut u64, value: u64, overflowed: &mut bool) {
    if let Some(total) = counter.checked_add(value) {
        *counter = total;
    } else {
        *overflowed = true;
    }
}

fn checked_increment(counter: &mut u64, value: u64) -> bool {
    let Some(total) = counter.checked_add(value) else {
        return false;
    };
    *counter = total;
    true
}

pub(crate) fn start_frontend_load_publisher(
    runtime: Arc<DistributedRuntime>,
    manager: Arc<ModelManager>,
    service: Arc<ServiceObserver>,
    metrics: FrontendLoadMetrics,
    cancel: CancellationToken,
) -> tokio::task::AbortHandle {
    runtime.runtime().secondary().spawn(async move {
        let namespace_name = std::env::var("DYN_NAMESPACE").unwrap_or_else(|_| "dynamo".into());
        let namespace = match runtime.namespace(namespace_name.clone()) {
            Ok(namespace) => namespace,
            Err(error) => {
                tracing::error!(namespace = namespace_name, %error, "cannot create frontend load event namespace");
                return;
            }
        };
        let frontend_instance_id = runtime.connection_id();
        let mut interval = tokio::time::interval_at(
            tokio::time::Instant::now() + FRONTEND_LOAD_PUBLISH_INTERVAL,
            FRONTEND_LOAD_PUBLISH_INTERVAL,
        );
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            let publisher = tokio::select! {
                _ = cancel.cancelled() => return,
                result = EventPublisher::for_namespace(&namespace, FRONTEND_LOAD_TOPIC) => match result {
                    Ok(publisher) => publisher,
                    Err(error) => {
                        tracing::warn!(%error, "frontend load publisher initialization failed");
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = tokio::time::sleep(FRONTEND_LOAD_PUBLISH_INTERVAL) => continue,
                        }
                    }
                },
            };

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = interval.tick() => {
                        let Some(frame) = metrics.snapshot(
                            frontend_instance_id,
                            service.is_ready(),
                            &manager,
                        ) else {
                            continue;
                        };
                        let result = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => return,
                            result = publisher.publish(&frame) => result,
                        };
                        if let Err(error) = result {
                            tracing::warn!(%error, "frontend load frame publish failed; reconnecting publisher");
                            break;
                        }
                    }
                }
            }
        }
    }).abort_handle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_phases_update_cumulative_counters() {
        let metrics = FrontendLoadMetrics::default();
        let request = metrics.start_request("model-a");
        request.observe(12, 0);
        request.observe(12, 3);
        request.finish(RequestOutcome::Completed);

        let state = metrics.state.lock();
        assert!(state.live.is_empty());
        let counters = &state.counters["model-a"];
        assert_eq!(counters.requests_started, 1);
        assert_eq!(counters.requests_completed, 1);
        assert_eq!(counters.input_tokens, 12);
        assert_eq!(counters.output_tokens, 3);
        assert!(!counters.input_incomplete);

        drop(state);
        let request = metrics.start_request("model-a");
        request.observe(12, 2);
        request.finish(RequestOutcome::Completed);
        let state = metrics.state.lock();
        let counters = &state.counters["model-a"];
        assert_eq!(counters.requests_started, 2);
        assert_eq!(counters.requests_completed, 2);
        assert_eq!(counters.input_tokens, 24);
        assert_eq!(counters.output_tokens, 5);
    }

    #[test]
    fn failed_request_without_exact_input_marks_the_counter_incomplete() {
        let metrics = FrontendLoadMetrics::default();
        let request = metrics.start_request("model-a");
        request.finish(RequestOutcome::Failed);

        let state = metrics.state.lock();
        let counters = &state.counters["model-a"];
        assert_eq!(counters.requests_failed, 1);
        assert!(counters.input_incomplete);
    }

    #[test]
    fn overflowed_counters_are_published_as_unavailable() {
        let counters = ModelCounters {
            overflowed: true,
            ..Default::default()
        };
        let mut load = FrontendModelLoad {
            requests_started_total: Some(0),
            requests_completed_total: Some(0),
            requests_failed_total: Some(0),
            requests_cancelled_total: Some(0),
            input_tokens_total: Some(0),
            output_tokens_total: Some(0),
            ..Default::default()
        };

        counters.write_totals(&mut load);

        assert_eq!(load.requests_started_total, None);
        assert_eq!(load.requests_completed_total, None);
        assert_eq!(load.requests_failed_total, None);
        assert_eq!(load.requests_cancelled_total, None);
        assert_eq!(load.input_tokens_total, None);
        assert_eq!(load.output_tokens_total, None);
    }
}
