// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::OnceLock};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::MULTIMODAL_EMBEDDING_CACHE_SUBJECT;

// Cache deltas are routing hints. Bound the queue so a slow event plane cannot
// grow memory without limit; if it fills, routing cache state may become stale.
const CACHE_UPDATE_CHANNEL_CAPACITY: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalEmbeddingCacheUpdate {
    pub added_keys: Vec<String>,
    pub removed_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalEmbeddingCacheEvent {
    pub worker_id: u64,
    pub update: MultimodalEmbeddingCacheUpdate,
}

enum CacheDeltaAction {
    Added,
    Removed,
}

pub struct MultimodalEmbeddingCachePublisher {
    tx: OnceLock<mpsc::Sender<MultimodalEmbeddingCacheUpdate>>,
    cancellation_token: CancellationToken,
}

impl MultimodalEmbeddingCachePublisher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish_delta(&self, added_keys: Vec<String>, removed_keys: Vec<String>) -> Result<()> {
        if added_keys.is_empty() && removed_keys.is_empty() {
            return Ok(());
        }

        self.publish_update(MultimodalEmbeddingCacheUpdate {
            added_keys,
            removed_keys,
        })
    }

    pub async fn create_endpoint(&self, endpoint: Endpoint) -> Result<()> {
        if self.tx.get().is_some() {
            return Ok(());
        }

        let worker_id = endpoint.drt().connection_id();
        let publisher =
            EventPublisher::for_endpoint(&endpoint, MULTIMODAL_EMBEDDING_CACHE_SUBJECT).await?;
        let (tx, rx) = mpsc::channel(CACHE_UPDATE_CHANNEL_CAPACITY);
        let cancellation_token = self.cancellation_token.clone();

        if self.tx.set(tx).is_err() {
            return Ok(());
        }

        endpoint.drt().runtime().secondary().spawn(async move {
            run_multimodal_embedding_cache_processor(publisher, worker_id, cancellation_token, rx)
                .await;
        });

        Ok(())
    }

    fn publish_update(&self, update: MultimodalEmbeddingCacheUpdate) -> Result<()> {
        let tx = self.tx.get().ok_or_else(|| {
            anyhow::anyhow!("multimodal embedding cache publisher not initialized")
        })?;
        tx.try_send(update).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                anyhow::anyhow!("multimodal embedding cache publisher channel full")
            }
            mpsc::error::TrySendError::Closed(_) => {
                anyhow::anyhow!("multimodal embedding cache publisher channel closed")
            }
        })
    }
}

impl Default for MultimodalEmbeddingCachePublisher {
    fn default() -> Self {
        Self {
            tx: OnceLock::new(),
            cancellation_token: CancellationToken::new(),
        }
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
    mut rx: mpsc::Receiver<MultimodalEmbeddingCacheUpdate>,
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
    rx: &mut mpsc::Receiver<MultimodalEmbeddingCacheUpdate>,
) -> MultimodalEmbeddingCacheUpdate {
    let mut net_delta = HashMap::new();
    merge_delta(&mut net_delta, first_update);

    while let Ok(update) = rx.try_recv() {
        merge_delta(&mut net_delta, update);
    }

    let mut added_keys = Vec::new();
    let mut removed_keys = Vec::new();
    for (key, action) in net_delta {
        match action {
            CacheDeltaAction::Added => added_keys.push(key),
            CacheDeltaAction::Removed => removed_keys.push(key),
        }
    }

    MultimodalEmbeddingCacheUpdate {
        added_keys,
        removed_keys,
    }
}

fn merge_delta(
    net_delta: &mut HashMap<String, CacheDeltaAction>,
    update: MultimodalEmbeddingCacheUpdate,
) {
    for key in update.added_keys {
        net_delta.insert(key, CacheDeltaAction::Added);
    }
    for key in update.removed_keys {
        net_delta.insert(key, CacheDeltaAction::Removed);
    }
}
