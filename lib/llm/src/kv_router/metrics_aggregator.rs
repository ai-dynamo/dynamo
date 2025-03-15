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

use std::sync::{Arc, Mutex};

pub use crate::kv_router::protocols::ForwardPassMetrics;
use crate::kv_router::KV_METRICS_ENDPOINT;

use crate::kv_router::scheduler::Endpoint;
use crate::kv_router::ProcessedEndpoints;
use dynamo_runtime::component::Component;
use dynamo_runtime::{service::EndpointInfo, utils::Duration, Result};
use tokio_util::sync::CancellationToken;

pub struct KvMetricsAggregator {
    pub service_name: String,
    pub endpoints: Arc<Mutex<ProcessedEndpoints>>,
}

impl KvMetricsAggregator {
    pub async fn new(component: Component, cancellation_token: CancellationToken) -> Self {
        let (ep_tx, mut ep_rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(collect_endpoints_task(
            component.clone(),
            ep_tx,
            cancellation_token.clone(),
        ));

        tracing::trace!("awaiting the start of the background endpoint subscriber");
        let endpoints = Arc::new(Mutex::new(ProcessedEndpoints::default()));
        let endpoints_clone = endpoints.clone();
        tokio::spawn(async move {
            tracing::debug!("scheduler background task started");
            loop {
                match ep_rx.recv().await {
                    Some(endpoints) => match endpoints_clone.lock() {
                        Ok(mut shared_endpoint) => {
                            *shared_endpoint = endpoints;
                        }
                        Err(e) => {
                            tracing::error!("Failed to acquire lock on endpoints: {:?}", e);
                        }
                    },
                    None => {
                        tracing::warn!("endpoint subscriber shutdown");
                        break;
                    }
                };
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });
        Self {
            service_name: component.service_name(),
            endpoints,
        }
    }

    pub fn get_endpoints(&self) -> ProcessedEndpoints {
        match self.endpoints.lock() {
            Ok(endpoints) => endpoints.clone(),
            Err(e) => {
                tracing::error!("Failed to acquire lock on endpoints: {:?}", e);
                ProcessedEndpoints::default()
            }
        }
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
    tracing::debug!("Endpoints: {endpoints:?}");

    if endpoints.is_empty() {
        tracing::warn!("No endpoints found matching subject {subject}");
    }

    Ok(endpoints)
}

pub async fn collect_endpoints_task(
    component: Component,
    ep_tx: tokio::sync::mpsc::Sender<ProcessedEndpoints>,
    cancel: CancellationToken,
) {
    let backoff_delay = Duration::from_millis(100);
    let scrape_timeout = Duration::from_millis(300);
    let endpoint = component.endpoint(KV_METRICS_ENDPOINT);
    let service_subject = endpoint.subject();

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::debug!("cancellation token triggered");
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
                tracing::debug!("unfiltered endpoints: {:?}", unfiltered_endpoints);

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
                tracing::debug!("endpoints: {:?}", endpoints);

                tracing::trace!(
                    "found {} endpoints for service: {}",
                    endpoints.len(),
                    service_subject
                );

                let processed = ProcessedEndpoints::new(endpoints);
                if ep_tx.send(processed).await.is_err() {
                    tracing::trace!("failed to send processed endpoints; shutting down");
                    break;
                }
            }
        }
    }
}
