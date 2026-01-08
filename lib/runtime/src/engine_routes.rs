// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

/// Callback type for engine routes (async)
/// Takes JSON body, returns JSON response (or error) wrapped in a Future
pub type EngineRouteCallback = Arc<
    dyn Fn(
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// Registry for engine route callbacks
///
/// This registry stores callbacks that handle requests to `/engine/*` routes.
/// Routes are registered from Python via `runtime.register_engine_route()`.
#[derive(Clone, Default)]
pub struct EngineRouteRegistry {
    routes: Arc<RwLock<HashMap<String, EngineRouteCallback>>>,
}

impl EngineRouteRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a callback for a route (e.g., "start_profile" for /engine/start_profile)
    pub fn register(&self, route: &str, callback: EngineRouteCallback) {
        let mut routes = self.routes.write().unwrap();
        routes.insert(route.to_string(), callback);
        tracing::debug!("Registered engine route: /engine/{}", route);
    }

    /// Get callback for a route
    pub fn get(&self, route: &str) -> Option<EngineRouteCallback> {
        let routes = self.routes.read().unwrap();
        routes.get(route).cloned()
    }

    /// List all registered routes
    pub fn routes(&self) -> Vec<String> {
        let routes = self.routes.read().unwrap();
        routes.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_basic() {
        let registry = EngineRouteRegistry::new();

        // Register a simple callback
        let callback: EngineRouteCallback =
            Arc::new(|body| Box::pin(async move { Ok(serde_json::json!({"echo": body})) }));

        registry.register("test", callback);

        // Verify it's registered
        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());

        // Verify routes list
        let routes = registry.routes();
        assert_eq!(routes.len(), 1);
        assert!(routes.contains(&"test".to_string()));
    }

    #[tokio::test]
    async fn test_callback_execution() {
        let registry = EngineRouteRegistry::new();

        let callback: EngineRouteCallback = Arc::new(|body| {
            Box::pin(async move {
                let input = body.get("input").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({
                    "output": format!("processed: {}", input)
                }))
            })
        });

        registry.register("process", callback);

        // Get and execute callback
        let cb = registry.get("process").unwrap();
        let result = cb(serde_json::json!({"input": "test"})).await.unwrap();

        assert_eq!(result["output"], "processed: test");
    }

    #[tokio::test]
    async fn test_clone_shares_routes() {
        let registry = EngineRouteRegistry::new();

        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }));
        registry.register("test", callback);

        // Clone the registry
        let cloned = registry.clone();

        // Both should see the same route
        assert!(registry.get("test").is_some());
        assert!(cloned.get("test").is_some());

        // Register on clone
        let callback2: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": false})) }));
        cloned.register("test2", callback2);

        // Original should also see it (they share the Arc)
        assert!(registry.get("test2").is_some());
    }
}
