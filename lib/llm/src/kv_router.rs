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
use std::sync::Arc;
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
    metrics_aggregator::collect_endpoints_task,
    scheduler::KvScheduler,
    scoring::ProcessedEndpoints,
};

use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component
pub const KV_EVENT_SUBJECT: &str = "kv_events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

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
        component: Component,
        kv_block_size: usize,
    ) -> Result<Arc<Self>> {
        let namespace = runtime.namespace(component.namespace().name())?;

        tracing::info!("Component Namespace {}", component.namespace());
        tracing::info!("Component Service Name {}", component.service_name());
        tracing::info!("KV Subject {}.{}", component.subject(), KV_EVENT_SUBJECT);
        Self::new(component, namespace, kv_block_size).await
    }

    pub async fn new(
        component: Component,
        namespace: Namespace,
        kv_block_size: usize,
    ) -> Result<Arc<Self>> {
        let cancellation_token = CancellationToken::new();
        let (ep_tx, ep_rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(collect_endpoints_task(
            component.clone(),
            ep_tx,
            cancellation_token.clone(),
        ));

        let indexer = KvIndexer::new(cancellation_token.clone(), kv_block_size);
        let scheduler = KvScheduler::start(ep_rx, namespace, kv_block_size).await?;

        // [gluo TODO] try subscribe_with_type::<RouterEvent>,
        // error checking below will be different.
        let mut kv_events_rx = component.subscribe(KV_EVENT_SUBJECT).await?;
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
            service_name: component.service_name(),
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
