// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use thiserror::Error;

/// Callback type for engine routes (async)
/// Takes JSON body, returns JSON response (or error) wrapped in a Future
pub type EngineRouteCallback = Arc<
    dyn Fn(
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// HTTP method accepted by an engine route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineRouteMethod {
    Get,
    Post,
}

/// Registered engine route, including an optional method restriction.
#[derive(Clone)]
pub struct EngineRoute {
    callback: EngineRouteCallback,
    method: Option<EngineRouteMethod>,
    registration_id: u64,
}

impl EngineRoute {
    pub fn callback(&self) -> EngineRouteCallback {
        Arc::clone(&self.callback)
    }

    pub fn method(&self) -> Option<EngineRouteMethod> {
        self.method
    }
}

/// Registry for engine route callbacks
///
/// This registry stores callbacks that handle requests to `/engine/*` routes.
/// Routes are registered from Python via `runtime.register_engine_route()`.
#[derive(Clone)]
pub struct EngineRouteRegistry {
    routes: Arc<RwLock<HashMap<String, EngineRoute>>>,
    next_registration_id: Arc<AtomicU64>,
}

impl Default for EngineRouteRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Removes a scoped route when its owner goes away without deleting a newer
/// registration that reused the same path.
#[must_use = "dropping the registration removes the engine route"]
pub struct EngineRouteRegistration {
    registry: EngineRouteRegistry,
    route: String,
    registration_id: u64,
}

/// Returned when a set of scoped routes cannot be registered atomically.
#[derive(Debug, Error)]
#[error("engine route is already registered: /engine/{route}")]
pub struct EngineRouteConflict {
    route: String,
}

impl Drop for EngineRouteRegistration {
    fn drop(&mut self) {
        self.registry
            .remove_if_current(&self.route, self.registration_id);
    }
}

impl EngineRouteRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
            next_registration_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Register a callback for a route (e.g., "control/start_profile" for /engine/control/start_profile)
    ///
    /// A route name is expected to be registered exactly once. Re-registering an
    /// existing name overwrites the previous callback and emits a warning, since
    /// it usually signals two registration mechanisms colliding rather than an
    /// intentional replacement.
    pub fn register(&self, route: &str, callback: EngineRouteCallback) {
        self.register_inner(route, None, callback);
    }

    /// Register a callback that accepts only `method`.
    pub fn register_method(
        &self,
        route: &str,
        method: EngineRouteMethod,
        callback: EngineRouteCallback,
    ) {
        self.register_inner(route, Some(method), callback);
    }

    /// Register a method-restricted route whose lifetime is owned by the
    /// returned guard.
    pub fn register_scoped_method(
        &self,
        route: &str,
        method: EngineRouteMethod,
        callback: EngineRouteCallback,
    ) -> EngineRouteRegistration {
        let registration_id = self.register_inner(route, Some(method), callback);
        EngineRouteRegistration {
            registry: self.clone(),
            route: route.to_string(),
            registration_id,
        }
    }

    /// Register method-restricted routes as one all-or-nothing operation.
    ///
    /// This is intended for route groups owned by one component. If any name
    /// is already registered, no route in the group is changed.
    pub fn try_register_scoped_methods(
        &self,
        registrations: Vec<(&str, EngineRouteMethod, EngineRouteCallback)>,
    ) -> Result<Vec<EngineRouteRegistration>, EngineRouteConflict> {
        let mut routes = self.routes.write();
        let mut names = HashSet::with_capacity(registrations.len());
        for (route, _, _) in &registrations {
            if routes.contains_key(*route) || !names.insert((*route).to_string()) {
                return Err(EngineRouteConflict {
                    route: (*route).to_string(),
                });
            }
        }

        let mut guards = Vec::with_capacity(registrations.len());
        for (route, method, callback) in registrations {
            let registration_id = self.next_registration_id.fetch_add(1, Ordering::Relaxed);
            routes.insert(
                route.to_string(),
                EngineRoute {
                    callback,
                    method: Some(method),
                    registration_id,
                },
            );
            tracing::debug!("Registered engine route: /engine/{route}");
            guards.push(EngineRouteRegistration {
                registry: self.clone(),
                route: route.to_string(),
                registration_id,
            });
        }
        Ok(guards)
    }

    fn register_inner(
        &self,
        route: &str,
        method: Option<EngineRouteMethod>,
        callback: EngineRouteCallback,
    ) -> u64 {
        let registration_id = self.next_registration_id.fetch_add(1, Ordering::Relaxed);
        let mut routes = self.routes.write();
        let entry = EngineRoute {
            callback,
            method,
            registration_id,
        };
        if routes.insert(route.to_string(), entry).is_some() {
            tracing::warn!("Overwriting already-registered engine route: /engine/{route}");
        } else {
            tracing::debug!("Registered engine route: /engine/{route}");
        }
        registration_id
    }

    fn remove_if_current(&self, route: &str, registration_id: u64) {
        let mut routes = self.routes.write();
        if routes
            .get(route)
            .is_some_and(|entry| entry.registration_id == registration_id)
        {
            routes.remove(route);
            tracing::debug!("Unregistered engine route: /engine/{route}");
        }
    }

    /// Get callback for a route
    pub fn get(&self, route: &str) -> Option<EngineRouteCallback> {
        let routes = self.routes.read();
        routes.get(route).map(EngineRoute::callback)
    }

    /// Get a route together with its method restriction.
    pub fn get_route(&self, route: &str) -> Option<EngineRoute> {
        let routes = self.routes.read();
        routes.get(route).cloned()
    }

    /// List all registered routes
    pub fn routes(&self) -> Vec<String> {
        let routes = self.routes.read();
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

    #[test]
    fn method_registration_preserves_method_metadata() {
        let registry = EngineRouteRegistry::new();
        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({})) }));

        registry.register_method("drain", EngineRouteMethod::Post, callback);

        let route = registry.get_route("drain").unwrap();
        assert_eq!(route.method(), Some(EngineRouteMethod::Post));
        assert!(registry.get("drain").is_some());
    }

    #[test]
    fn scoped_registration_is_removed_on_drop() {
        let registry = EngineRouteRegistry::new();
        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({})) }));

        let registration =
            registry.register_scoped_method("drain", EngineRouteMethod::Post, callback);
        assert!(registry.get("drain").is_some());

        drop(registration);
        assert!(registry.get("drain").is_none());
    }

    #[test]
    fn stale_scoped_registration_does_not_remove_replacement() {
        let registry = EngineRouteRegistry::new();
        let first: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"version": 1})) }));
        let second: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"version": 2})) }));

        let stale = registry.register_scoped_method("status", EngineRouteMethod::Get, first);
        let current = registry.register_scoped_method("status", EngineRouteMethod::Get, second);
        drop(stale);

        assert!(registry.get("status").is_some());
        drop(current);
        assert!(registry.get("status").is_none());
    }

    #[test]
    fn scoped_method_group_rejects_conflicts_without_partial_registration() {
        let registry = EngineRouteRegistry::new();
        let existing: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"owner": "first"})) }));
        let _existing = registry.register_scoped_method("status", EngineRouteMethod::Get, existing);

        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"owner": "second"})) }));
        let result = registry.try_register_scoped_methods(vec![
            ("drain", EngineRouteMethod::Post, callback.clone()),
            ("status", EngineRouteMethod::Get, callback),
        ]);

        assert!(result.is_err());
        assert!(registry.get("drain").is_none());
        assert!(registry.get("status").is_some());
    }
}
