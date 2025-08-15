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

pub use crate::kv_router::protocols::{ForwardPassMetrics, LoadMetrics, PredictiveLoadMetrics};
use crate::kv_router::KV_METRICS_ENDPOINT;

use crate::discovery::{ModelEntry, MODEL_ROOT_PATH};
use crate::kv_router::scoring::Endpoint;
use crate::kv_router::ProcessedEndpoints;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::component::Component;
use dynamo_runtime::transports::etcd::{Client as EtcdClient, WatchEvent};
use dynamo_runtime::{service::EndpointInfo, utils::Duration, Result};
use std::collections::HashMap;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

static METRICS_WAITING_MESSAGE: Once = Once::new();
static METRICS_FOUND_MESSAGE: Once = Once::new();

pub struct EndpointCollector {
    pub service_name: String,
    pub endpoints_rx: watch::Receiver<ProcessedEndpoints>,
}

impl EndpointCollector {
    pub async fn new(component: Component, cancellation_token: CancellationToken) -> Self {
        let (watch_tx, watch_rx) = watch::channel(ProcessedEndpoints::default());

        tokio::spawn(collect_endpoints_task(
            component.clone(),
            watch_tx,
            cancellation_token.clone(),
            "generate".to_string(),
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
            KV_METRICS_ENDPOINT.to_string(),
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
    subject: String,
) {
    let backoff_delay = Duration::from_millis(100);
    let scrape_timeout = Duration::from_millis(300);
    let endpoint = component.endpoint(&subject);
    let service_subject = endpoint.subject();

    // Keep track of the last sent value to avoid unnecessary updates
    let mut last_sent: Option<ProcessedEndpoints> = None;

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

                let endpoints: Vec<Endpoint> = if subject == KV_METRICS_ENDPOINT {
                    // Original filtering behavior
                    unfiltered_endpoints
                        .into_iter()
                        .filter_map(|s| {
                            s.data?
                                .decode::<ForwardPassMetrics>()
                                .map(|data| Endpoint {
                                    name: s.name,
                                    subject: s.subject,
                                    data: LoadMetrics::EngineLoadMetrics(data),
                                })
                                .inspect_err(|e| {
                                    tracing::warn!("skip endpoint data that can't be parsed as ForwardPassMetrics: {:?}", e);
                                })
                                .ok()
                        })
                        .collect()
                } else {
                    // No filtering - just use default LoadMetrics
                    unfiltered_endpoints
                        .into_iter()
                        .map(|s| Endpoint {
                            name: s.name,
                            subject: s.subject,
                            data: LoadMetrics::default(),
                        })
                        .collect()
                };

                tracing::trace!("Found {} endpoints for service: {service_subject}", endpoints.len());

                let processed = ProcessedEndpoints::new(endpoints);

                // Only send if different from last sent value
                // This is necessary because the watch channel does not track changes
                // https://docs.rs/tokio/latest/tokio/sync/watch/struct.Receiver.html#method.has_changed
                let should_send = match &last_sent {
                    Some(last) => last != &processed,
                    None => true,
                };

                if should_send {
                    tracing::trace!("Endpoints changed, sending update for service: {service_subject}");
                    if watch_tx.send(processed.clone()).is_err() {
                        tracing::error!("failed to send processed endpoints; shutting down");
                        break;
                    }
                    last_sent = Some(processed);
                } else {
                    tracing::trace!("Endpoints unchanged, skipping update for service: {service_subject}");
                }
            }
        }
    }
}

pub async fn watch_model_runtime_configs(
    etcd_client: EtcdClient,
    cancellation_token: CancellationToken,
) -> Result<watch::Receiver<HashMap<i64, ModelRuntimeConfig>>> {
    let (watch_tx, watch_rx) = watch::channel(HashMap::new());

    let prefix_watcher = etcd_client.kv_get_and_watch_prefix(MODEL_ROOT_PATH).await?;
    let (_prefix, _watcher, mut events_rx) = prefix_watcher.dissolve();

    tokio::spawn(async move {
        let mut runtime_configs: HashMap<i64, ModelRuntimeConfig> = HashMap::new();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("Runtime config watcher cancelled");
                    break;
                }
                event = events_rx.recv() => {
                    let Some(event) = event else {
                        tracing::debug!("Runtime config watch stream closed");
                        break;
                    };

                    match event {
                        WatchEvent::Put(kv) => {
                            let Ok(model_entry) = serde_json::from_slice::<ModelEntry>(kv.value()) else {
                                tracing::warn!(
                                    "Failed to parse ModelEntry from etcd. Key: {}",
                                    kv.key_str().unwrap_or("<invalid>")
                                );
                                continue;
                            };

                            let lease_id = kv.lease();

                            if let Some(runtime_config) = model_entry.runtime_config {
                                runtime_configs.insert(lease_id, runtime_config);
                                tracing::trace!("Updated runtime config for lease_id: {}", lease_id);
                            } else {
                                runtime_configs.remove(&lease_id);
                                tracing::trace!("Removed runtime config (no config in ModelEntry)");
                            }

                            if watch_tx.send(runtime_configs.clone()).is_err() {
                                tracing::error!("Failed to send runtime configs update; receiver dropped");
                                break;
                            }
                        }
                        WatchEvent::Delete(kv) => {
                            let lease_id = kv.lease();
                            runtime_configs.remove(&lease_id);
                            tracing::trace!("Removed runtime config for deleted entry");

                            if watch_tx.send(runtime_configs.clone()).is_err() {
                                tracing::error!("Failed to send runtime configs update; receiver dropped");
                                break;
                            }
                        }
                    }
                }
            }
        }
    });

    Ok(watch_rx)
}
