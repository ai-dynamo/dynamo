// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Once;

pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::KV_METRICS_ENDPOINT;

use crate::kv_router::scheduler::Endpoint;
use crate::kv_router::ProcessedEndpoints;
use dynamo_runtime::component::Component;
use dynamo_runtime::{service::EndpointInfo, utils::Duration, Result};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

static METRICS_WAITING_MESSAGE: Once = Once::new();
static METRICS_FOUND_MESSAGE: Once = Once::new();

pub struct KvMetricsAggregator {
    pub service_name: String,
    pub endpoints_rx: watch::Receiver<ProcessedEndpoints>,
}

impl KvMetricsAggregator {
    pub async fn new(component: Component, cancellation_token: CancellationToken) -> Self {
        let (watch_tx, watch_rx) = watch::channel(ProcessedEndpoints::default());

        tokio::spawn(collect_endpoints_task(
            component.clone(),
            watch_tx,
            cancellation_token.clone(),
        ));

        Self {
            service_name: component.service_name(),
            endpoints_rx: watch_rx,
        }
    }

    pub fn get_endpoints(&self) -> ProcessedEndpoints {
        self.endpoints_rx.borrow().clone()
    }

    pub fn endpoints_watcher(&self) -> watch::Receiver<ProcessedEndpoints> {
        self.endpoints_rx.clone()
    }
}

/// [gluo TODO] 'collect_endpoints' is from component/metrics,
/// should consolidate these functions into generic metrics aggregator
/// functions and shared by KvMetricsAggregator and component/metrics.
/// Collect endpoints from a component
pub async fn collect_endpoints(
    component: &Component,
    subject: &str,
    timeout: Duration,
) -> Result<Vec<EndpointInfo>> {
    // Collect stats from each backend
    let stream = component.scrape_stats(timeout).await?;

    // Filter the stats by the service subject
    let endpoints = stream
        .into_endpoints()
        .filter(|e| e.subject.starts_with(subject))
        .collect::<Vec<_>>();
    if endpoints.is_empty() {
        // Only print it once, we poll while the worker starts
        METRICS_WAITING_MESSAGE.call_once(|| {
            tracing::debug!("Waiting for metrics endpoint..");
        });
    } else {
        METRICS_FOUND_MESSAGE.call_once(|| {
            tracing::debug!("Found metrics endpoint");
        });
    }

    Ok(endpoints)
}

pub async fn collect_endpoints_task(
    component: Component,
    watch_tx: watch::Sender<ProcessedEndpoints>,
    cancel: CancellationToken,
) {
    let backoff_delay = Duration::from_millis(100);
    let scrape_timeout = Duration::from_millis(300);
    let endpoint = component.endpoint(KV_METRICS_ENDPOINT);
    let service_subject = endpoint.subject();

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }
            _ = tokio::time::sleep(backoff_delay) => {
                tracing::trace!("collecting endpoints for service: {}", service_subject);
                let unfiltered_endpoints =
                    match collect_endpoints(&component, &service_subject, scrape_timeout).await
                    {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("Failed to retrieve endpoints for {}: {:?}", service_subject, e);
                            continue;
                        }
                    };
                let endpoints: Vec<Endpoint> = unfiltered_endpoints
                    .into_iter()
                    .filter(|s| s.data.is_some())
                    .filter_map(|s|
                        match s.data.unwrap().decode::<ForwardPassMetrics>() {
                            Ok(data) => Some(Endpoint {
                                name: s.name,
                                subject: s.subject,
                                data,
                            }),
                            Err(e) => {
                                tracing::debug!("skip endpoint data that can't be parsed as ForwardPassMetrics: {:?}", e);
                                None
                            }
                        }
                    )
                    .collect();
                tracing::trace!("Found {} endpoints for service: {service_subject}", endpoints.len());

                let processed = ProcessedEndpoints::new(endpoints);

                if watch_tx.send(processed).is_err() {
                    tracing::trace!("failed to send processed endpoints; shutting down");
                    break;
                }
            }
        }
    }
}
