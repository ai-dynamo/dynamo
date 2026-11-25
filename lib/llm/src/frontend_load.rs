// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend metric snapshots published on the event plane.
//!
//! The frame types and their wire rule are in [`dynamo_http_server::load`].

use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::transports::event_plane::EventPublisher;
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;
use crate::utils::retry::{Backoff, FailureStreak};
use dynamo_http_server::metrics::Metrics;

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

/// Start periodic publishing after the caller checks `DYN_SERVING_LOAD_PUBLISH`.
pub(crate) fn start_frontend_load_publisher(
    runtime: DistributedRuntime,
    namespace: String,
    manager: Arc<ModelManager>,
    metrics: Arc<Metrics>,
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
