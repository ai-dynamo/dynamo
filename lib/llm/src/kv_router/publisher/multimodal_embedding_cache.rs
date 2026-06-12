// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, sync::OnceLock};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::MULTIMODAL_EMBEDDING_CACHE_SUBJECT;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultimodalEmbeddingCacheUpdate {
    Delta {
        added_keys: Vec<String>,
        removed_keys: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalEmbeddingCacheEvent {
    pub worker_id: u64,
    pub update: MultimodalEmbeddingCacheUpdate,
}

pub struct MultimodalEmbeddingCachePublisher {
    tx: OnceLock<mpsc::UnboundedSender<MultimodalEmbeddingCacheUpdate>>,
    cancellation_token: CancellationToken,
}

impl MultimodalEmbeddingCachePublisher {
    pub fn new() -> Self {
        Self {
            tx: OnceLock::new(),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn publish_delta(&self, added_keys: Vec<String>, removed_keys: Vec<String>) -> Result<()> {
        if added_keys.is_empty() && removed_keys.is_empty() {
            return Ok(());
        }

        self.publish_update(MultimodalEmbeddingCacheUpdate::Delta {
            added_keys,
            removed_keys,
        })
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        if self.tx.get().is_some() {
            return Ok(());
        }

        let worker_id = component.drt().connection_id();
        let publisher =
            EventPublisher::for_component(&component, MULTIMODAL_EMBEDDING_CACHE_SUBJECT).await?;
        let (tx, rx) = mpsc::unbounded_channel();
        let cancellation_token = self.cancellation_token.clone();

        if self.tx.set(tx).is_err() {
            return Ok(());
        }

        component.drt().runtime().secondary().spawn(async move {
            run_multimodal_embedding_cache_processor(publisher, worker_id, cancellation_token, rx)
                .await;
        });

        Ok(())
    }

    fn publish_update(&self, update: MultimodalEmbeddingCacheUpdate) -> Result<()> {
        let tx = self.tx.get().ok_or_else(|| {
            anyhow::anyhow!("multimodal embedding cache publisher not initialized")
        })?;
        tx.send(update)
            .map_err(|_| anyhow::anyhow!("multimodal embedding cache publisher channel closed"))
    }
}

impl Drop for MultimodalEmbeddingCachePublisher {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

async fn run_multimodal_embedding_cache_processor(
    publisher: EventPublisher,
    worker_id: u64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<MultimodalEmbeddingCacheUpdate>,
) {
    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::debug!("Multimodal embedding cache publisher received cancellation signal");
                break;
            }
            update = rx.recv() => {
                let Some(update) = update else {
                    tracing::debug!("Multimodal embedding cache publisher channel closed");
                    break;
                };

                let event = MultimodalEmbeddingCacheEvent {
                    worker_id,
                    update: coalesce_delta_backlog(update, &mut rx),
                };

                if let Err(error) = publisher.publish(&event).await {
                    tracing::warn!(
                        error = %error,
                        "failed to publish embedding cache state"
                    );
                }
            }
        }
    }
}

fn coalesce_delta_backlog(
    first_update: MultimodalEmbeddingCacheUpdate,
    rx: &mut mpsc::UnboundedReceiver<MultimodalEmbeddingCacheUpdate>,
) -> MultimodalEmbeddingCacheUpdate {
    let mut net_delta = BTreeMap::new();
    merge_delta(&mut net_delta, first_update);

    while let Ok(update) = rx.try_recv() {
        merge_delta(&mut net_delta, update);
    }

    let mut added_keys = Vec::new();
    let mut removed_keys = Vec::new();
    for (key, added) in net_delta {
        if added {
            added_keys.push(key);
        } else {
            removed_keys.push(key);
        }
    }

    MultimodalEmbeddingCacheUpdate::Delta {
        added_keys,
        removed_keys,
    }
}

fn merge_delta(net_delta: &mut BTreeMap<String, bool>, update: MultimodalEmbeddingCacheUpdate) {
    match update {
        MultimodalEmbeddingCacheUpdate::Delta {
            added_keys,
            removed_keys,
        } => {
            for key in added_keys {
                net_delta.insert(key, true);
            }
            for key in removed_keys {
                net_delta.insert(key, false);
            }
        }
    }
}
