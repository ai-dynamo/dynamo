// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ForwardPassMetrics → selector subscriber.
//!
//! When a [`WorkerSelector`] wants engine-authoritative counters (reports
//! `wants_fpm() == true`), this module starts a background task that
//! subscribes to the event-plane FPM topic, decodes each message, and
//! pushes `(waiting, running)` into the selector via
//! [`WorkerSelector::update_from_fpm`].
//!
//! Closing the gap vs. router-side projection: lifecycle hooks
//! (`on_admit`/`on_running`/`on_finish`) maintain the same per-worker
//! counts from router-observable events. FPM updates overwrite those
//! counts with engine-side reality whenever a new message arrives
//! (typically every forward pass — tens of Hz at the configured cadence).
//! The latest signal wins; FPM messages are authoritative for any worker
//! whose engine has the InstrumentedScheduler enabled.
//!
//! Operator-side requirement: each backend worker must publish FPM. For
//! vLLM that means setting `DYN_FORWARDPASS_METRIC_PORT` on the worker
//! (auto-injects `InstrumentedScheduler` and starts the relay — see
//! `components/src/dynamo/vllm/args.py` and `components/src/dynamo/vllm/main.py`).

use std::sync::Arc;

use dynamo_kv_router::WorkerSelector;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::LocalScheduler;
use dynamo_kv_router::scheduling::policy::SchedulingPolicy;
use dynamo_kv_router::sequences::SequencePublisher;
use dynamo_runtime::component::Component;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::fpm_publisher::FPM_TOPIC;

/// Mirror of Python `ScheduledRequestMetrics`. Only the request-count
/// fields are read here; extra fields on the wire are ignored by serde's
/// default behavior so publisher-side additions don't break us.
#[derive(Debug, Default, Deserialize)]
struct ScheduledRequestMetricsDe {
    #[serde(default)]
    num_prefill_requests: i32,
    #[serde(default)]
    num_decode_requests: i32,
}

#[derive(Debug, Default, Deserialize)]
struct QueuedRequestMetricsDe {
    #[serde(default)]
    num_prefill_requests: i32,
    #[serde(default)]
    num_decode_requests: i32,
}

#[derive(Debug, Deserialize)]
struct ForwardPassMetricsDe {
    #[serde(default)]
    version: i32,
    #[serde(default)]
    worker_id: String,
    #[serde(default)]
    dp_rank: i64,
    #[serde(default)]
    scheduled_requests: ScheduledRequestMetricsDe,
    #[serde(default)]
    queued_requests: QueuedRequestMetricsDe,
}

/// Schema version we know how to parse. Must match Python `FPM_VERSION`.
const FPM_VERSION: i32 = 1;

/// Spawn a background task that subscribes to FPM events and forwards
/// engine-side counters into `scheduler.selector()`. Returns immediately if
/// the active selector does not opt in via [`WorkerSelector::wants_fpm`].
///
/// The task lives until `cancel` is fired (typically tied to the
/// component's DRT child token).
pub fn spawn<P, C, S, Sel>(
    component: Component,
    scheduler: Arc<LocalScheduler<P, C, S, Sel>>,
    cancel: CancellationToken,
    worker_type: &'static str,
) where
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Clone + PartialEq + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + Sync + 'static,
{
    if !scheduler.selector().wants_fpm() {
        return;
    }

    tokio::spawn(async move {
        let mut subscriber = match EventSubscriber::for_component(&component, FPM_TOPIC).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    worker_type,
                    error = %e,
                    "FPM-selector subscriber: failed to create EventSubscriber; falling back to router-side projection"
                );
                return;
            }
        };

        tracing::info!(
            worker_type,
            topic = FPM_TOPIC,
            "FPM-selector subscriber: listening for engine-authoritative (waiting, running) updates"
        );

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::info!(worker_type, "FPM-selector subscriber: shutting down");
                    break;
                }
                event = subscriber.next() => {
                    match event {
                        Some(Ok(envelope)) => {
                            handle_message(&envelope.payload, scheduler.as_ref(), worker_type);
                        }
                        Some(Err(e)) => {
                            tracing::warn!(worker_type, error = %e, "FPM-selector subscriber: event error");
                        }
                        None => {
                            tracing::info!(worker_type, "FPM-selector subscriber: stream ended");
                            break;
                        }
                    }
                }
            }
        }
    });
}

fn handle_message<P, C, S, Sel>(
    payload: &[u8],
    scheduler: &LocalScheduler<P, C, S, Sel>,
    worker_type: &'static str,
) where
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Clone + PartialEq + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + Sync + 'static,
{
    let metrics: ForwardPassMetricsDe = match rmp_serde::from_slice(payload) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(worker_type, error = %e, "FPM-selector subscriber: decode failed");
            return;
        }
    };
    if metrics.version != FPM_VERSION {
        tracing::warn!(
            worker_type,
            seen = metrics.version,
            expected = FPM_VERSION,
            "FPM-selector subscriber: skipping unsupported FPM version"
        );
        return;
    }

    let worker_id = match metrics.worker_id.parse::<u64>() {
        Ok(id) => id,
        Err(_) => {
            // FPM `worker_id` is always the stringified DRT connection_id.
            // A non-numeric value is a misconfigured worker — skip.
            tracing::warn!(
                worker_type,
                worker_id = %metrics.worker_id,
                "FPM-selector subscriber: worker_id is not a u64; cannot map to dynamo WorkerId"
            );
            return;
        }
    };

    let dp_rank = if metrics.dp_rank < 0 {
        0
    } else {
        metrics.dp_rank as u32
    };
    let worker = WorkerWithDpRank::new(worker_id, dp_rank);

    let waiting = nonneg_sum_u32(
        metrics.queued_requests.num_prefill_requests,
        metrics.queued_requests.num_decode_requests,
    );
    let running = nonneg_sum_u32(
        metrics.scheduled_requests.num_prefill_requests,
        metrics.scheduled_requests.num_decode_requests,
    );

    scheduler
        .selector()
        .update_from_fpm(worker, waiting, running);
}

fn nonneg_sum_u32(a: i32, b: i32) -> u32 {
    a.max(0).saturating_add(b.max(0)) as u32
}
