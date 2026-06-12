// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use serde::{Deserialize, Serialize};

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
    worker_id: OnceLock<u64>,
    publisher: OnceLock<Arc<EventPublisher>>,
    runtime_handle: OnceLock<tokio::runtime::Handle>,
}

impl MultimodalEmbeddingCachePublisher {
    pub fn new() -> Result<Self> {
        Ok(Self {
            worker_id: OnceLock::new(),
            publisher: OnceLock::new(),
            runtime_handle: OnceLock::new(),
        })
    }

    pub fn publish_delta(
        &self,
        mut added_keys: Vec<String>,
        mut removed_keys: Vec<String>,
    ) -> Result<()> {
        added_keys.sort();
        added_keys.dedup();
        removed_keys.sort();
        removed_keys.dedup();

        if added_keys.is_empty() && removed_keys.is_empty() {
            return Ok(());
        }

        self.publish_update(MultimodalEmbeddingCacheUpdate::Delta {
            added_keys,
            removed_keys,
        })
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let runtime_handle = tokio::runtime::Handle::try_current().map_err(|e| {
            anyhow::anyhow!(
                "multimodal embedding cache publisher create_endpoint requires a Tokio runtime: {e}"
            )
        })?;
        let worker_id = component.drt().connection_id();
        let publisher = Arc::new(
            EventPublisher::for_component(&component, MULTIMODAL_EMBEDDING_CACHE_SUBJECT).await?,
        );

        let _ = self.runtime_handle.set(runtime_handle);
        let _ = self.worker_id.set(worker_id);
        let _ = self.publisher.set(publisher);
        Ok(())
    }

    fn publish_update(&self, update: MultimodalEmbeddingCacheUpdate) -> Result<()> {
        let worker_id = *self.worker_id.get().ok_or_else(|| {
            anyhow::anyhow!("multimodal embedding cache publisher not initialized")
        })?;
        let publisher = self.publisher.get().cloned().ok_or_else(|| {
            anyhow::anyhow!("multimodal embedding cache publisher not initialized")
        })?;
        let runtime_handle = self.runtime_handle.get().cloned().ok_or_else(|| {
            anyhow::anyhow!("multimodal embedding cache publisher runtime not initialized")
        })?;
        let event = MultimodalEmbeddingCacheEvent { worker_id, update };

        runtime_handle.spawn(async move {
            if let Err(error) = publisher.publish(&event).await {
                tracing::warn!(
                    "Failed to publish multimodal embedding cache state: {}",
                    error
                );
            }
        });

        Ok(())
    }
}
