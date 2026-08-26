// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use crate::config::environment_names::runtime::engine_routes as env_engine_routes;

/// Operator policy governing which `/engine/*` control routes are served.
///
/// Resolved once from the environment when the [`EngineRouteRegistry`] is constructed and
/// enforced at the single `/engine/*` dispatch point, so it applies uniformly across every
/// backend. The default is [`AllowAll`](EngineRoutePolicy::AllowAll), so behavior is
/// unchanged unless an operator opts in.
///
/// Matching is on the **full route string** (the path after `/engine/`, e.g.
/// `control/start_profile` or `update/model_taints`) — some routes have no `control/`
/// prefix, so a prefix-based policy would miss them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum EngineRoutePolicy {
    /// Serve every registered route (default — backward compatible).
    #[default]
    AllowAll,
    /// Serve no `/engine/*` route.
    DisableAll,
    /// Serve only the listed routes.
    Allowlist(HashSet<String>),
    /// Serve every route except the listed ones.
    Denylist(HashSet<String>),
}

impl EngineRoutePolicy {
    /// Resolve the policy from the environment.
    ///
    /// Precedence when more than one variable is set: `DisableAll` > `Allowlist` >
    /// `Denylist`. When more than one is set, the highest-precedence one takes effect and a
    /// warning names which variables were seen and which one won.
    pub fn from_env() -> Self {
        let disable_all = std::env::var(env_engine_routes::DYN_DISABLE_ENGINE_ROUTES)
            .ok()
            .is_some_and(|v| is_truthy(&v));
        // A *present* allow var — even empty/whitespace — is a deliberate opt-in to
        // restrictive mode: an empty allowlist serves nothing (fail closed). Only a
        // truly-unset var leaves the default `AllowAll`. An empty deny var is a no-op,
        // so treat it as unset.
        let allow = std::env::var(env_engine_routes::DYN_ENGINE_ROUTES_ALLOW)
            .ok()
            .map(|v| parse_route_set(&v));
        let deny = std::env::var(env_engine_routes::DYN_ENGINE_ROUTES_DENY)
            .ok()
            .map(|v| parse_route_set(&v))
            .filter(|s| !s.is_empty());

        // Warn if more than one control is set — only the highest-precedence one applies.
        // `set` is built in precedence order, so its first entry is the one that wins.
        let set: Vec<&str> = [
            (disable_all, env_engine_routes::DYN_DISABLE_ENGINE_ROUTES),
            (allow.is_some(), env_engine_routes::DYN_ENGINE_ROUTES_ALLOW),
            (deny.is_some(), env_engine_routes::DYN_ENGINE_ROUTES_DENY),
        ]
        .into_iter()
        .filter_map(|(is_set, name)| is_set.then_some(name))
        .collect();
        if set.len() > 1 {
            tracing::warn!(
                "Multiple engine-route policy variables set ({}); applying {} (precedence: \
                 DYN_DISABLE_ENGINE_ROUTES > DYN_ENGINE_ROUTES_ALLOW > DYN_ENGINE_ROUTES_DENY)",
                set.join(", "),
                set[0]
            );
        }

        let policy = if disable_all {
            EngineRoutePolicy::DisableAll
        } else if let Some(allow) = allow {
            EngineRoutePolicy::Allowlist(allow)
        } else if let Some(deny) = deny {
            EngineRoutePolicy::Denylist(deny)
        } else {
            EngineRoutePolicy::AllowAll
        };
        // Log a restrictive policy so operators can eyeball the parsed set (and spot typos —
        // an unrecognized name simply never matches a registered route).
        if policy != EngineRoutePolicy::AllowAll {
            tracing::info!(?policy, "Engine-route policy in effect");
        }
        policy
    }

    /// Whether `route` (the full route string after `/engine/`) is permitted by this policy.
    pub fn is_allowed(&self, route: &str) -> bool {
        match self {
            EngineRoutePolicy::AllowAll => true,
            EngineRoutePolicy::DisableAll => false,
            EngineRoutePolicy::Allowlist(set) => set.contains(route),
            EngineRoutePolicy::Denylist(set) => !set.contains(route),
        }
    }
}

/// Parse a comma-separated route list: trim each entry, drop empties.
fn parse_route_set(value: &str) -> HashSet<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Truthy env parsing consistent with other runtime flags (`1`/`true`/`yes`).
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

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
#[derive(Clone)]
pub struct EngineRouteRegistry {
    routes: Arc<RwLock<HashMap<String, EngineRouteCallback>>>,
    /// Operator policy resolved once from the environment at construction. Behind an `Arc`
    /// so cloning the registry (it travels with the cheaply-`Clone` `DistributedRuntime`)
    /// doesn't deep-copy the policy's route set.
    policy: Arc<EngineRoutePolicy>,
}

impl Default for EngineRouteRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineRouteRegistry {
    /// Create a new empty registry, resolving the [`EngineRoutePolicy`] from the environment.
    pub fn new() -> Self {
        Self::with_policy(EngineRoutePolicy::from_env())
    }

    /// Create a new empty registry with an explicit policy (primarily for tests).
    pub fn with_policy(policy: EngineRoutePolicy) -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
            policy: Arc::new(policy),
        }
    }

    /// The resolved policy governing which routes are served.
    pub fn policy(&self) -> &EngineRoutePolicy {
        &self.policy
    }

    /// Whether `route` is permitted by the resolved policy.
    pub fn is_allowed(&self, route: &str) -> bool {
        self.policy.is_allowed(route)
    }

    /// Register a callback for a route (e.g., "control/start_profile" for /engine/control/start_profile)
    ///
    /// A route name is expected to be registered exactly once. Re-registering an
    /// existing name overwrites the previous callback and emits a warning, since
    /// it usually signals two registration mechanisms colliding rather than an
    /// intentional replacement.
    pub fn register(&self, route: &str, callback: EngineRouteCallback) {
        let mut routes = self.routes.write().unwrap();
        if routes.insert(route.to_string(), callback).is_some() {
            tracing::warn!("Overwriting already-registered engine route: /engine/{route}");
        } else {
            tracing::debug!("Registered engine route: /engine/{route}");
        }
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

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    // ---- Policy: is_allowed truth table for each variant ----

    #[test]
    fn test_policy_is_allowed() {
        assert!(EngineRoutePolicy::AllowAll.is_allowed("anything"));
        assert!(!EngineRoutePolicy::DisableAll.is_allowed("control/start_profile"));

        let allow = EngineRoutePolicy::Allowlist(set(&["control/start_profile"]));
        assert!(allow.is_allowed("control/start_profile"));
        assert!(!allow.is_allowed("control/update_weights_from_disk"));

        let deny = EngineRoutePolicy::Denylist(set(&["control/update_weights_from_disk"]));
        assert!(!deny.is_allowed("control/update_weights_from_disk"));
        assert!(deny.is_allowed("control/start_profile"));
    }

    // ---- Policy: from_env resolution & precedence ----
    // (also exercises truthy parsing and comma-list trim/empty-set handling)
    //
    // These mutate process env, so they share one serialized test to avoid cross-test
    // interference under the default multi-threaded test runner.

    #[test]
    fn test_from_env_resolution_and_precedence() {
        use env_engine_routes::{
            DYN_DISABLE_ENGINE_ROUTES, DYN_ENGINE_ROUTES_ALLOW, DYN_ENGINE_ROUTES_DENY,
        };

        let all = [
            DYN_DISABLE_ENGINE_ROUTES,
            DYN_ENGINE_ROUTES_ALLOW,
            DYN_ENGINE_ROUTES_DENY,
        ];

        // Default: nothing set -> AllowAll.
        temp_env::with_vars(all.map(|k| (k, None::<&str>)), || {
            assert_eq!(EngineRoutePolicy::from_env(), EngineRoutePolicy::AllowAll);
        });

        // DisableAll truthy variants.
        for v in ["1", "true", "yes"] {
            temp_env::with_vars(
                [
                    (DYN_DISABLE_ENGINE_ROUTES, Some(v)),
                    (DYN_ENGINE_ROUTES_ALLOW, None),
                    (DYN_ENGINE_ROUTES_DENY, None),
                ],
                || {
                    assert_eq!(EngineRoutePolicy::from_env(), EngineRoutePolicy::DisableAll);
                },
            );
        }

        // DisableAll non-truthy -> treated as unset -> AllowAll.
        for v in ["0", "false", ""] {
            temp_env::with_vars(
                [
                    (DYN_DISABLE_ENGINE_ROUTES, Some(v)),
                    (DYN_ENGINE_ROUTES_ALLOW, None),
                    (DYN_ENGINE_ROUTES_DENY, None),
                ],
                || {
                    assert_eq!(EngineRoutePolicy::from_env(), EngineRoutePolicy::AllowAll);
                },
            );
        }

        // Allowlist.
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, None),
                (DYN_ENGINE_ROUTES_ALLOW, Some("a,b, c ")),
                (DYN_ENGINE_ROUTES_DENY, None),
            ],
            || {
                assert_eq!(
                    EngineRoutePolicy::from_env(),
                    EngineRoutePolicy::Allowlist(set(&["a", "b", "c"]))
                );
            },
        );

        // Denylist.
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, None),
                (DYN_ENGINE_ROUTES_ALLOW, None),
                (DYN_ENGINE_ROUTES_DENY, Some("x,y")),
            ],
            || {
                assert_eq!(
                    EngineRoutePolicy::from_env(),
                    EngineRoutePolicy::Denylist(set(&["x", "y"]))
                );
            },
        );

        // Present-but-empty allow env -> Allowlist(empty) = serve nothing (fail closed).
        // (Unset -> AllowAll is covered by the "nothing set" case above.)
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, None),
                (DYN_ENGINE_ROUTES_ALLOW, Some("  , ,")),
                (DYN_ENGINE_ROUTES_DENY, None),
            ],
            || {
                assert_eq!(
                    EngineRoutePolicy::from_env(),
                    EngineRoutePolicy::Allowlist(set(&[]))
                );
            },
        );

        // Precedence: all three set -> DisableAll wins.
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, Some("1")),
                (DYN_ENGINE_ROUTES_ALLOW, Some("a")),
                (DYN_ENGINE_ROUTES_DENY, Some("b")),
            ],
            || {
                assert_eq!(EngineRoutePolicy::from_env(), EngineRoutePolicy::DisableAll);
            },
        );

        // Precedence: allow + deny set -> Allowlist wins.
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, None),
                (DYN_ENGINE_ROUTES_ALLOW, Some("a")),
                (DYN_ENGINE_ROUTES_DENY, Some("b")),
            ],
            || {
                assert_eq!(
                    EngineRoutePolicy::from_env(),
                    EngineRoutePolicy::Allowlist(set(&["a"]))
                );
            },
        );
    }

    // ---- Registry: consults policy (not just route presence), defaults from env ----

    #[test]
    fn test_registry_enforces_policy_and_defaults_from_env() {
        // A registered route is still blocked when the policy denies it.
        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }));
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::DisableAll);
        registry.register("control/start_profile", callback);
        assert!(registry.get("control/start_profile").is_some());
        assert!(!registry.is_allowed("control/start_profile"));

        // new()/default() resolve the policy from env; with no vars set that is AllowAll.
        temp_env::with_vars(
            [
                (env_engine_routes::DYN_DISABLE_ENGINE_ROUTES, None::<&str>),
                (env_engine_routes::DYN_ENGINE_ROUTES_ALLOW, None),
                (env_engine_routes::DYN_ENGINE_ROUTES_DENY, None),
            ],
            || {
                assert_eq!(
                    EngineRouteRegistry::new().policy(),
                    &EngineRoutePolicy::AllowAll
                );
                assert_eq!(
                    EngineRouteRegistry::default().policy(),
                    &EngineRoutePolicy::AllowAll
                );
            },
        );
    }

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
