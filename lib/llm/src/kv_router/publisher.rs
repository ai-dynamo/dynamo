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

use crate::kv_router::{indexer::RouterEvent, protocols::*, KV_EVENT_SUBJECT, KV_METRICS_ENDPOINT};
use async_trait::async_trait;
use dynamo_runtime::traits::{events::EventPublisher, DistributedRuntimeProvider};
use dynamo_runtime::{
    component::Component,
    pipeline::{
        network::Ingress, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream,
        SingleIn,
    },
    protocols::annotated::Annotated,
    Error, Result,
};
use futures::stream;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing as log;

pub struct KvEventPublisher {
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    kv_block_size: usize,
}

impl KvEventPublisher {
    pub fn new(component: Component, worker_id: i64, kv_block_size: usize) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let p = KvEventPublisher { tx, kv_block_size };

        start_publish_task(component, worker_id, rx);
        Ok(p)
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        log::debug!("Publish event: {:?}", event);
        self.tx.send(event)
    }

    pub fn kv_block_size(&self) -> usize {
        self.kv_block_size
    }
}

fn start_publish_task(
    component: Component,
    worker_id: i64,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    let component_clone = component.clone();
    log::info!("Publishing KV Events to subject: {}", KV_EVENT_SUBJECT);

    _ = component.drt().runtime().secondary().spawn(async move {
        while let Some(event) = rx.recv().await {
            let router_event = RouterEvent::new(worker_id, event);
            component_clone
                .publish(KV_EVENT_SUBJECT, &router_event)
                .await
                .unwrap();
        }
    });
}

pub struct KvMetricsPublisher {
    tx: tokio::sync::watch::Sender<Arc<ForwardPassMetrics>>,
    rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl KvMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Arc::new(ForwardPassMetrics::default()));
        Ok(KvMetricsPublisher { tx, rx })
    }

    pub fn publish(
        &self,
        metrics: Arc<ForwardPassMetrics>,
    ) -> Result<(), tokio::sync::watch::error::SendError<Arc<ForwardPassMetrics>>> {
        log::debug!("Publish metrics: {:?}", metrics);
        self.tx.send(metrics)
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let mut metrics_rx = self.rx.clone();
        let handler = Arc::new(KvLoadEndpoingHander::new(metrics_rx.clone()));
        let handler = Ingress::for_engine(handler)?;

        component
            .endpoint(KV_METRICS_ENDPOINT)
            .endpoint_builder()
            .stats_handler(move |_| {
                let metrics = metrics_rx.borrow_and_update().clone();
                serde_json::to_value(&*metrics).unwrap()
            })
            .handler(handler)
            .start()
            .await
    }
}

struct KvLoadEndpoingHander {
    metrics_rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl KvLoadEndpoingHander {
    pub fn new(metrics_rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>) -> Self {
        Self { metrics_rx }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<()>, ManyOut<Annotated<ForwardPassMetrics>>, Error>
    for KvLoadEndpoingHander
{
    async fn generate(
        &self,
        request: SingleIn<()>,
    ) -> Result<ManyOut<Annotated<ForwardPassMetrics>>> {
        let context = request.context();
        let metrics = self.metrics_rx.borrow().clone();
        let metrics = (*metrics).clone();
        let stream = stream::iter(vec![Annotated::from_data(metrics)]);
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}
