// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Event Consolidator
//!
//! This module consolidates kv events from multiple sources (vLLM's G1 events
//! and KVBM's G2/G3 events) before publishing them to the router.
pub mod config;
pub mod publisher;
pub mod subscriber;
pub mod tracker;

pub use config::KvEventConsolidatorConfig;
pub use publisher::KvEventConsolidatorPublisher;
pub use tracker::{CacheStatusTracker, EventSource, StorageTier};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use subscriber::start_simple_zmq_listener;

/// Handle for KVBM to send G2/G3 events directly to the KV Event Consolidator
#[derive(Clone, Debug)]
pub struct KvEventConsolidatorHandle {
    pub(crate) tracker: Arc<RwLock<CacheStatusTracker>>,
}

impl KvEventConsolidatorHandle {
    /// Send a block store event to the KV Event Consolidator
    ///
    /// This is called by KVBM when a block is stored in G2 or G3.
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_store(
        &self,
        block_hash: String,
        source: EventSource,
        token_ids: Vec<u32>,
        parent_hash: Option<String>,
        block_size: usize,
        lora_id: Option<u64>,
        tier: Option<StorageTier>,
        data_parallel_rank: Option<i32>,
    ) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_store(
            block_hash,
            source,
            token_ids,
            parent_hash,
            block_size,
            lora_id.map(|id| id as i32),
            tier,
            data_parallel_rank,
        );
    }

    /// Send a block remove event to the KV Event Consolidator
    ///
    /// This is called by KVBM when a block is removed from G2 or G3.
    pub async fn handle_remove(&self, block_hash: &str, source: EventSource) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_remove(block_hash, source);
    }

    /// Clear all blocks from the KV Event Consolidator
    ///
    /// This is called by KVBM when all blocks should be evicted.
    pub async fn handle_clear_all(&self) {
        let mut tracker = self.tracker.write().await;
        tracker.handle_clear_all();
    }
}

/// The main KV Event Consolidator that manages the event flow
pub struct KvEventConsolidator {
    config: KvEventConsolidatorConfig,
    tracker: Arc<RwLock<CacheStatusTracker>>,
    subscriber_handle: Option<JoinHandle<()>>,
    cancellation_token: CancellationToken,
    publisher: Option<KvEventConsolidatorPublisher>,
}

impl KvEventConsolidator {
    /// Create a new KV Event Consolidator
    pub fn new(config: KvEventConsolidatorConfig) -> Result<Self> {
        let tracker = Arc::new(RwLock::new(CacheStatusTracker::new()));
        let cancellation_token = CancellationToken::new();

        Ok(Self {
            config,
            tracker,
            subscriber_handle: None,
            cancellation_token,
            publisher: None,
        })
    }

    /// Start the KV Event Consolidator
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!(
            "Starting KV Event Consolidator: subscribe from {}, publish to ZMQ at {}",
            self.config.engine_event_endpoint,
            self.config.consolidated_event_endpoint
        );

        // Always publish to ZMQ (worker-side publishers will add worker_id and forward to NATS)
        let publisher = KvEventConsolidatorPublisher::new(
            &self.config.consolidated_event_endpoint,
            self.tracker.clone(),
        )?;
        self.publisher = Some(publisher);
        tracing::info!("Waiting for downstream ZMQ subscribers to connect...");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Start the subscriber (connects to engine's publisher - vLLM or TensorRT-LLM)
        let handle = start_simple_zmq_listener(
            self.config.engine_event_endpoint.clone(),
            self.tracker.clone(),
            self.cancellation_token.clone(),
            self.config.engine_source,
        )
        .await?;

        self.subscriber_handle = Some(handle);

        tracing::info!("KV Event Consolidator fully started and ready");

        Ok(())
    }

    /// Shutdown the KV Event Consolidator
    pub async fn shutdown(self) -> Result<()> {
        tracing::info!("Shutting down KV Event Consolidator");

        // Cancel the ZMQ listener
        self.cancellation_token.cancel();

        // Wait for adapter task to finish
        if let Some(handle) = self.subscriber_handle {
            handle.abort();
            let _ = handle.await;
        }

        if let Some(publisher) = self.publisher {
            publisher.shutdown().await?;
        }

        Ok(())
    }

    /// Get a reference to the cache status tracker (for debugging/metrics)
    pub fn tracker(&self) -> Arc<RwLock<CacheStatusTracker>> {
        self.tracker.clone()
    }

    /// Get a handle that KVBM can use to send G2/G3 kv events directly
    pub fn get_handle(&self) -> KvEventConsolidatorHandle {
        KvEventConsolidatorHandle {
            tracker: self.tracker.clone(),
        }
    }
}
