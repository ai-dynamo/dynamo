// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local Endpoint Registry
//!
//! Provides a registry for locally registered endpoints that can be called in-process
//! without going through the network stack.

use crate::engine::AsyncEngine;
use dashmap::DashMap;
use std::sync::Arc;

/// Type alias for a boxed async engine that can handle generic requests and responses
pub type LocalAsyncEngine = Arc<
    dyn AsyncEngine<
            crate::pipeline::SingleIn<serde_json::Value>,
            crate::pipeline::ManyOut<crate::protocols::annotated::Annotated<serde_json::Value>>,
            anyhow::Error,
        > + Send
        + Sync,
>;

/// Registry for locally registered endpoints
///
/// This registry stores endpoints that are registered locally (in the same process)
/// and allows them to be called directly without going through the network transport layer.
#[derive(Clone)]
pub struct LocalEndpointRegistry {
    /// Map of endpoint name to async engine
    engines: Arc<DashMap<String, LocalAsyncEngine>>,
    /// Number of registrations so far, for callers waiting on the first one
    registrations: Arc<tokio::sync::watch::Sender<usize>>,
}

impl Default for LocalEndpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalEndpointRegistry {
    /// Create a new local endpoint registry
    pub fn new() -> Self {
        Self {
            engines: Arc::new(DashMap::new()),
            registrations: Arc::new(tokio::sync::watch::channel(0).0),
        }
    }

    /// Watch the number of registrations made so far.
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<usize> {
        self.registrations.subscribe()
    }

    /// Register a local endpoint
    ///
    /// # Arguments
    ///
    /// * `endpoint_name` - Name of the endpoint (e.g., "load_lora", "generate")
    /// * `engine` - The async engine that handles requests for this endpoint
    pub fn register(&self, endpoint_name: String, engine: LocalAsyncEngine) {
        tracing::debug!("Registering local endpoint: {endpoint_name}");
        self.engines.insert(endpoint_name, engine);
        self.registrations.send_modify(|count| *count += 1);
    }

    /// Get a registered local endpoint
    ///
    /// The async engine if found, None otherwise
    pub fn get(&self, endpoint_name: &str) -> Option<LocalAsyncEngine> {
        self.engines.get(endpoint_name).map(|e| e.clone())
    }
}
