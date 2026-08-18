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
#[derive(Clone, Default)]
pub struct LocalEndpointRegistry {
    /// Map of endpoint name to async engine
    engines: Arc<DashMap<String, LocalAsyncEngine>>,
}

impl LocalEndpointRegistry {
    /// Create a new local endpoint registry
    pub fn new() -> Self {
        Self {
            engines: Arc::new(DashMap::new()),
        }
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
    }

    /// Get a registered local endpoint
    ///
    /// The async engine if found, None otherwise
    pub fn get(&self, endpoint_name: &str) -> Option<LocalAsyncEngine> {
        self.engines.get(endpoint_name).map(|e| e.clone())
    }

    /// Remove a registered local endpoint
    ///
    /// Returns the engine that was registered under `endpoint_name`, if any. An endpoint
    /// that has stopped must be removed: the canary health check dispatches through this
    /// registry, so an engine left behind stays callable for an endpoint that no longer
    /// has a request-plane or discovery presence.
    pub fn remove(&self, endpoint_name: &str) -> Option<LocalAsyncEngine> {
        tracing::debug!("Removing local endpoint: {endpoint_name}");
        self.engines.remove(endpoint_name).map(|(_, engine)| engine)
    }

    /// Remove a registered local endpoint only while `engine` is the registered one
    ///
    /// Registration is last-writer-wins, so an endpoint that restarts under the same name
    /// replaces the entry. Cleanup for the previous incarnation must not evict the
    /// replacement, so removal is conditional on the stored engine still being the one
    /// this caller registered.
    pub fn remove_if_current(
        &self,
        endpoint_name: &str,
        engine: &LocalAsyncEngine,
    ) -> Option<LocalAsyncEngine> {
        self.engines
            .remove_if(endpoint_name, |_, current| Arc::ptr_eq(current, engine))
            .map(|(_, engine)| engine)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! An engine for tests that only care about which registration is in place.

    use super::LocalAsyncEngine;
    use crate::engine::{AsyncEngine, AsyncEngineContextProvider};
    use crate::pipeline::{ManyOut, ResponseStream, SingleIn};
    use crate::protocols::annotated::Annotated;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct StubEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, anyhow::Error>
        for StubEngine
    {
        async fn generate(
            &self,
            input: SingleIn<serde_json::Value>,
        ) -> anyhow::Result<ManyOut<Annotated<serde_json::Value>>> {
            let (data, ctx) = input.into_parts();
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(vec![Annotated::from_data(data)])),
                ctx.context(),
            ))
        }
    }

    /// A fresh engine. Each call returns a distinct `Arc`, so identity comparison
    /// distinguishes one registration from another.
    pub(crate) fn stub_engine() -> LocalAsyncEngine {
        Arc::new(StubEngine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::stub_engine;

    const ENDPOINT: &str = "generate";

    #[test]
    fn a_removed_endpoint_is_no_longer_callable() {
        let registry = LocalEndpointRegistry::new();
        let engine = stub_engine();
        registry.register(ENDPOINT.to_string(), engine.clone());
        assert!(registry.get(ENDPOINT).is_some());

        let removed = registry
            .remove(ENDPOINT)
            .expect("removal hands back the registered engine");
        assert!(Arc::ptr_eq(&removed, &engine));
        assert!(
            registry.get(ENDPOINT).is_none(),
            "a removed endpoint must not stay dispatchable"
        );
        assert!(registry.remove(ENDPOINT).is_none());
    }

    #[test]
    fn conditional_removal_keeps_the_engine_a_restart_installed() {
        let registry = LocalEndpointRegistry::new();
        let first = stub_engine();
        let second = stub_engine();
        registry.register(ENDPOINT.to_string(), first.clone());
        registry.register(ENDPOINT.to_string(), second.clone());

        assert!(
            registry.remove_if_current(ENDPOINT, &first).is_none(),
            "cleanup for the replaced engine must not evict the replacement"
        );
        let live = registry
            .get(ENDPOINT)
            .expect("the restart's engine is still registered");
        assert!(Arc::ptr_eq(&live, &second));

        assert!(registry.remove_if_current(ENDPOINT, &second).is_some());
        assert!(registry.get(ENDPOINT).is_none());
    }
}
