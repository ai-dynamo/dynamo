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
    /// Endpoint registrations in insertion order; the last entry serves.
    engines: Arc<DashMap<String, Vec<LocalAsyncEngine>>>,
}

impl LocalEndpointRegistry {
    /// Create a new local endpoint registry
    pub fn new() -> Self {
        Self {
            engines: Arc::new(DashMap::new()),
        }
    }

    /// Register a local endpoint.
    /// # Arguments
    ///
    /// * `endpoint_name` - Name of the endpoint (e.g., "load_lora", "generate")
    /// * `engine` - The async engine that handles requests for this endpoint
    pub fn register(&self, endpoint_name: String, engine: LocalAsyncEngine) {
        tracing::debug!("Registering local endpoint: {endpoint_name}");
        self.engines.entry(endpoint_name).or_default().push(engine);
    }

    /// Get the most recently registered endpoint under this name.
    pub fn get(&self, endpoint_name: &str) -> Option<LocalAsyncEngine> {
        self.engines
            .get(endpoint_name)
            .and_then(|outstanding| outstanding.last().cloned())
    }

    /// Remove every registration under this name and return the serving engine.
    pub fn remove(&self, endpoint_name: &str) -> Option<LocalAsyncEngine> {
        tracing::debug!("Removing local endpoint: {endpoint_name}");
        self.engines
            .remove(endpoint_name)
            .and_then(|(_, mut outstanding)| outstanding.pop())
    }

    /// Withdraw one engine registration by identity.
    pub fn remove_registration(
        &self,
        endpoint_name: &str,
        engine: &LocalAsyncEngine,
    ) -> Option<LocalAsyncEngine> {
        let mut outstanding = self.engines.get_mut(endpoint_name)?;
        let position = outstanding
            .iter()
            .rposition(|registered| Arc::ptr_eq(registered, engine))?;
        let removed = outstanding.remove(position);
        let vacated = outstanding.is_empty();
        drop(outstanding);
        if vacated {
            // Only while still empty: another start may have registered in between.
            self.engines
                .remove_if(endpoint_name, |_, outstanding| outstanding.is_empty());
        }
        Some(removed)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::LocalAsyncEngine;
    use crate::engine::{AsyncEngine, AsyncEngineContextProvider};
    use crate::pipeline::{ManyOut, ResponseStream, SingleIn};
    use crate::protocols::annotated::Annotated;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct StubEngine;

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<serde_json::Value>,
            ManyOut<Annotated<serde_json::Value>>,
            anyhow::Error,
        > for StubEngine
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
    fn withdrawing_a_replaced_registration_keeps_the_engine_a_restart_installed() {
        let registry = LocalEndpointRegistry::new();
        let first = stub_engine();
        let second = stub_engine();
        registry.register(ENDPOINT.to_string(), first.clone());
        registry.register(ENDPOINT.to_string(), second.clone());

        let withdrawn = registry
            .remove_registration(ENDPOINT, &first)
            .expect("the replaced registration was still outstanding");
        assert!(Arc::ptr_eq(&withdrawn, &first));
        let live = registry
            .get(ENDPOINT)
            .expect("the restart's engine is still registered");
        assert!(
            Arc::ptr_eq(&live, &second),
            "cleanup for the replaced engine must not evict the replacement"
        );

        assert!(registry.remove_registration(ENDPOINT, &second).is_some());
        assert!(registry.get(ENDPOINT).is_none());
    }

    #[test]
    fn withdrawing_the_serving_registration_re_exposes_the_one_it_displaced() {
        let registry = LocalEndpointRegistry::new();
        let displaced = stub_engine();
        let serving = stub_engine();
        registry.register(ENDPOINT.to_string(), displaced.clone());
        registry.register(ENDPOINT.to_string(), serving.clone());

        registry
            .remove_registration(ENDPOINT, &serving)
            .expect("the serving registration was outstanding");

        let live = registry
            .get(ENDPOINT)
            .expect("the displaced endpoint is still running and must be reachable again");
        assert!(Arc::ptr_eq(&live, &displaced));
    }

    #[test]
    fn withdrawing_a_registration_twice_is_a_no_op() {
        let registry = LocalEndpointRegistry::new();
        let engine = stub_engine();
        registry.register(ENDPOINT.to_string(), engine.clone());

        assert!(registry.remove_registration(ENDPOINT, &engine).is_some());
        assert!(
            registry.remove_registration(ENDPOINT, &engine).is_none(),
            "a second release must not disturb whatever holds the name now"
        );

        let replacement = stub_engine();
        registry.register(ENDPOINT.to_string(), replacement.clone());
        assert!(registry.remove_registration(ENDPOINT, &engine).is_none());
        let live = registry
            .get(ENDPOINT)
            .expect("the replacement still serves");
        assert!(Arc::ptr_eq(&live, &replacement));
    }

    #[test]
    fn removing_the_endpoint_drops_every_outstanding_registration() {
        let registry = LocalEndpointRegistry::new();
        let first = stub_engine();
        let second = stub_engine();
        registry.register(ENDPOINT.to_string(), first.clone());
        registry.register(ENDPOINT.to_string(), second.clone());

        let removed = registry
            .remove(ENDPOINT)
            .expect("removal hands back the engine that was serving");
        assert!(Arc::ptr_eq(&removed, &second));
        assert!(registry.get(ENDPOINT).is_none());
    }
}
