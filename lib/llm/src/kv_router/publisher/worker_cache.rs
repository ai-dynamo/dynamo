// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};

use dynamo_runtime::component::{Component, Namespace};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::MULTIMODAL_EMBEDDING_CACHE_SUBJECT;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalEmbeddingCacheEvent {
    pub worker_id: u64,
    pub cache_keys: Vec<String>,
}

pub struct MultimodalEmbeddingCachePublisher {
    tx: tokio::sync::watch::Sender<Vec<String>>,
    rx: tokio::sync::watch::Receiver<Vec<String>>,
}

impl MultimodalEmbeddingCachePublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Vec::new());
        Ok(Self { tx, rx })
    }

    pub fn publish(&self, mut cache_keys: Vec<String>) -> Result<()> {
        cache_keys.sort();
        cache_keys.dedup();
        self.tx
            .send(cache_keys)
            .map_err(|_| anyhow::anyhow!("multimodal embedding cache channel closed"))
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let worker_id = component.drt().connection_id();
        self.start_nats_cache_state_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    fn start_nats_cache_state_publishing(&self, namespace: Namespace, worker_id: u64) {
        let cache_rx = self.rx.clone();

        tokio::spawn(async move {
            let event_publisher = match EventPublisher::for_namespace(
                &namespace,
                MULTIMODAL_EMBEDDING_CACHE_SUBJECT,
            )
            .await
            {
                Ok(publisher) => publisher,
                Err(e) => {
                    tracing::error!(
                        "Failed to create multimodal embedding cache publisher: {}",
                        e
                    );
                    return;
                }
            };

            let mut rx = cache_rx;
            let mut last_cache_keys: Option<Vec<String>> = None;
            let mut pending_publish: Option<Vec<String>> = None;
            let mut publish_timer =
                Box::pin(tokio::time::sleep(tokio::time::Duration::from_secs(0)));
            publish_timer.as_mut().reset(tokio::time::Instant::now());

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Multimodal embedding cache publisher sender dropped, stopping background task"
                            );
                            break;
                        }

                        let cache_keys = rx.borrow_and_update().clone();
                        let has_changed = last_cache_keys.as_ref() != Some(&cache_keys);

                        if has_changed {
                            pending_publish = Some(cache_keys.clone());
                            last_cache_keys = Some(cache_keys);
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now()
                                    + tokio::time::Duration::from_millis(1)
                            );
                        }
                    }
                    _ = &mut publish_timer => {
                        if let Some(cache_keys) = pending_publish.take() {
                            let event = MultimodalEmbeddingCacheEvent {
                                worker_id,
                                cache_keys,
                            };

                            if let Err(e) = event_publisher.publish(&event).await {
                                tracing::warn!("Failed to publish multimodal embedding cache state: {}", e);
                            }
                        }

                        publish_timer.as_mut().reset(
                            tokio::time::Instant::now()
                                + tokio::time::Duration::from_secs(3600)
                        );
                    }
                }
            }
        });
    }
}