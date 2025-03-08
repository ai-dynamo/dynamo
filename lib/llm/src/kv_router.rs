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

use anyhow::Result;
use dynamo_runtime::{component::Component, component::Namespace, DistributedRuntime};
use futures::stream::StreamExt;
use std::{sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing;

pub mod indexer;
pub mod metrics_aggregator;
pub mod protocols;
pub mod publisher;
pub mod scheduler;
pub mod scoring;

use crate::kv_router::{
    indexer::{KvIndexer, KvIndexerInterface, RouterEvent},
    scheduler::{Endpoint, KvScheduler, Service},
    scoring::ProcessedEndpoints,
};

// this should be discovered from the backend
pub const KV_EVENT_SUBJECT: &str = "kv_events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";

pub struct KvRouter {
    // properties of request plane
    // maybe rolled up into the generic object or not
    service_name: String,

    cancellation_token: CancellationToken,

    #[allow(dead_code)]
    scheduler: KvScheduler,

    indexer: KvIndexer,
}

impl KvRouter {
    pub async fn from_runtime(
        runtime: DistributedRuntime,
        backend: Component,
    ) -> Result<Arc<Self>> {
        let nats_client = runtime.nats_client();
        let service_name = backend.service_name();
        let kv_subject = backend.event_subject(KV_EVENT_SUBJECT);
        let namespace = runtime.namespace(backend.namespace())?;

        tracing::info!("Component Namespace {}", backend.namespace());
        tracing::info!("Component Service Name {}", service_name);
        tracing::info!("KV Subject {}", kv_subject);
        Self::new(nats_client, service_name, kv_subject, namespace).await
    }

    pub async fn new(
        nats_client: dynamo_runtime::transports::nats::Client,
        service_name: String,
        kv_subject: String,
        namespace: Namespace,
    ) -> Result<Arc<Self>> {
        let cancellation_token = CancellationToken::new();
        let (ep_tx, ep_rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(collect_endpoints(
            nats_client.clone(),
            service_name.clone(),
            ep_tx,
            cancellation_token.clone(),
        ));

        let indexer = KvIndexer::new(cancellation_token.clone());
        let scheduler = KvScheduler::start(ep_rx, namespace).await?;

        tracing::debug!("subscribing to kv events: {}", kv_subject);
        let mut kv_events_rx = nats_client.client().subscribe(kv_subject).await?;
        let kv_events_tx = indexer.event_sender();

        tokio::spawn(async move {
            while let Some(event) = kv_events_rx.next().await {
                let event: RouterEvent = match serde_json::from_slice(&event.payload) {
                    Ok(event) => {
                        tracing::debug!("received kv event: {:?}", event);
                        event
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize RouterEvent: {:?}", e);
                        // Choosing warn and continue to process other events from other workers
                        // A bad event likely signals a problem with a worker, but potentially other workers are still healthy
                        continue;
                    }
                };
                if let Err(e) = kv_events_tx.send(event).await {
                    tracing::trace!("failed to send kv event to indexer; shutting down: {:?}", e);
                }
            }
        });

        Ok(Arc::new(Self {
            service_name,
            cancellation_token,
            scheduler,
            indexer,
        }))
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    // [TODO] indexer needs to take 'lora_id' as parameter
    pub async fn schedule(&self, token_ids: &Vec<u32>, _lora_id: u64) -> Result<i64> {
        // Extracting part of the code in KvRouter::generate() for only
        // the decision making part, routing is done by the caller
        let isl_tokens = token_ids.len();
        let overlap_scores = self
            .indexer
            .find_matches_for_request(token_ids.as_slice())
            .await?;
        tracing::debug!("KV router overlap_scores: {:?}", overlap_scores);
        let worker_id = self.scheduler.schedule(overlap_scores, isl_tokens).await?;
        Ok(worker_id)
    }
}

async fn collect_endpoints(
    nats_client: dynamo_runtime::transports::nats::Client,
    service_name: String,
    ep_tx: tokio::sync::mpsc::Sender<ProcessedEndpoints>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::debug!("cancellation token triggered");
                break;
            }
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                tracing::trace!("collecting endpoints for service: {}", service_name);
            }
        }

        let values = match nats_client
            .get_endpoints(&service_name, Duration::from_secs(1))
            .await
        {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to retrieve endpoints for {}: {:?}", service_name, e);
                continue;
            }
        };

        tracing::debug!("values: {:?}", values);
        let services: Vec<Service> = values
            .into_iter()
            .filter(|v| !v.is_empty())
            .filter_map(|v| match serde_json::from_slice::<Service>(&v) {
                Ok(service) => Some(service),
                Err(e) => {
                    tracing::warn!("For value: {:?} \nFailed to parse service: {:?}", v, e);
                    None
                }
            })
            .collect();
        tracing::debug!("services: {:?}", services);

        let endpoints: Vec<Endpoint> = services
            .into_iter()
            .flat_map(|s| s.endpoints)
            .filter(|s| s.data.is_some())
            .map(|s| Endpoint {
                name: s.name,
                subject: s.subject,
                data: s.data.unwrap(),
            })
            .collect();
        tracing::debug!("endpoints: {:?}", endpoints);

        tracing::trace!(
            "found {} endpoints for service: {}",
            endpoints.len(),
            service_name
        );

        let processed = ProcessedEndpoints::new(endpoints);

        // process endpoints into
        if ep_tx.send(processed).await.is_err() {
            tracing::trace!("failed to send processed endpoints; shutting down");
            break;
        }
    }
}
