// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Once, RwLock};

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
        let allow = std::env::var(env_engine_routes::DYN_ENGINE_ROUTES_ALLOW)
            .ok()
            .map(|v| parse_route_set(&v))
            .filter(|s| !s.is_empty());
        let deny = std::env::var(env_engine_routes::DYN_ENGINE_ROUTES_DENY)
            .ok()
            .map(|v| parse_route_set(&v))
            .filter(|s| !s.is_empty());

        // Warn if more than one control is set — only the highest-precedence one applies.
        let set_count = [disable_all, allow.is_some(), deny.is_some()]
            .iter()
            .filter(|b| **b)
            .count();
        if set_count > 1 {
            let winner = if disable_all {
                env_engine_routes::DYN_DISABLE_ENGINE_ROUTES
            } else if allow.is_some() {
                env_engine_routes::DYN_ENGINE_ROUTES_ALLOW
            } else {
                env_engine_routes::DYN_ENGINE_ROUTES_DENY
            };
            let mut set = Vec::new();
            if disable_all {
                set.push(env_engine_routes::DYN_DISABLE_ENGINE_ROUTES);
            }
            if allow.is_some() {
                set.push(env_engine_routes::DYN_ENGINE_ROUTES_ALLOW);
            }
            if deny.is_some() {
                set.push(env_engine_routes::DYN_ENGINE_ROUTES_DENY);
            }
            tracing::warn!(
                "Multiple engine-route policy variables set ({}); applying {} (precedence: \
                 DYN_DISABLE_ENGINE_ROUTES > DYN_ENGINE_ROUTES_ALLOW > DYN_ENGINE_ROUTES_DENY)",
                set.join(", "),
                winner
            );
        }

        if disable_all {
            EngineRoutePolicy::DisableAll
        } else if let Some(allow) = allow {
            EngineRoutePolicy::Allowlist(allow)
        } else if let Some(deny) = deny {
            EngineRoutePolicy::Denylist(deny)
        } else {
            EngineRoutePolicy::AllowAll
        }
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

    /// The set of route names this policy explicitly references (allow/deny entries), if any.
    /// Used to warn about names that match no registered route.
    fn configured_routes(&self) -> Option<&HashSet<String>> {
        match self {
            EngineRoutePolicy::Allowlist(set) | EngineRoutePolicy::Denylist(set) => Some(set),
            EngineRoutePolicy::AllowAll | EngineRoutePolicy::DisableAll => None,
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
    /// Operator policy resolved once from the environment at construction.
    policy: Arc<EngineRoutePolicy>,
    /// Guards the one-time "unrecognized configured route" warning, which is emitted lazily
    /// on the first policy check (by which point the registry is populated).
    warned_unrecognized: Arc<Once>,
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
            warned_unrecognized: Arc::new(Once::new()),
        }
    }

    /// The resolved policy governing which routes are served.
    pub fn policy(&self) -> &EngineRoutePolicy {
        &self.policy
    }

    /// Whether `route` is permitted by the resolved policy.
    ///
    /// On the first call, emits a one-time warning naming any allow/deny entries that match
    /// no registered route (usually a typo, and a false sense of hardening). The registry is
    /// populated by the time the first `/engine/*` request arrives, so the check is accurate.
    pub fn is_allowed(&self, route: &str) -> bool {
        self.warn_unrecognized_once();
        self.policy.is_allowed(route)
    }

    /// Emit, at most once, a warning for configured allow/deny names that no registered
    /// route matches.
    fn warn_unrecognized_once(&self) {
        self.warned_unrecognized.call_once(|| {
            let Some(configured) = self.policy.configured_routes() else {
                return;
            };
            let registered: HashSet<String> = {
                let routes = self.routes.read().unwrap();
                routes.keys().cloned().collect()
            };
            let mut unrecognized: Vec<&str> = configured
                .iter()
                .filter(|r| !registered.contains(*r))
                .map(String::as_str)
                .collect();
            if !unrecognized.is_empty() {
                unrecognized.sort_unstable();
                tracing::warn!(
                    "Engine-route policy references route(s) that are not registered: {} \
                     (check for typos; these entries have no effect)",
                    unrecognized.join(", ")
                );
            }
        });
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

    // ---- Policy: pure is_allowed truth table ----

    #[test]
    fn test_policy_allow_all() {
        let p = EngineRoutePolicy::AllowAll;
        assert!(p.is_allowed("control/start_profile"));
        assert!(p.is_allowed("update/model_taints"));
        assert!(p.is_allowed("anything"));
    }

    #[test]
    fn test_policy_disable_all() {
        let p = EngineRoutePolicy::DisableAll;
        assert!(!p.is_allowed("control/start_profile"));
        assert!(!p.is_allowed("update/model_taints"));
    }

    #[test]
    fn test_policy_allowlist() {
        let p =
            EngineRoutePolicy::Allowlist(set(&["control/start_profile", "update/model_taints"]));
        assert!(p.is_allowed("control/start_profile"));
        assert!(p.is_allowed("update/model_taints"));
        assert!(!p.is_allowed("control/update_weights_from_disk"));
    }

    #[test]
    fn test_policy_denylist() {
        let p = EngineRoutePolicy::Denylist(set(&["control/update_weights_from_disk"]));
        assert!(!p.is_allowed("control/update_weights_from_disk"));
        assert!(p.is_allowed("control/start_profile"));
        assert!(p.is_allowed("update/model_taints"));
    }

    // ---- parse helpers ----

    #[test]
    fn test_parse_route_set_trims_and_drops_empties() {
        assert_eq!(parse_route_set("a,b, c "), set(&["a", "b", "c"]));
        assert_eq!(parse_route_set(" a ,, ,b,"), set(&["a", "b"]));
        assert!(parse_route_set("").is_empty());
        assert!(parse_route_set("  , ,").is_empty());
    }

    #[test]
    fn test_is_truthy() {
        for v in ["1", "true", "TRUE", "Yes", " yes "] {
            assert!(is_truthy(v), "{v:?} should be truthy");
        }
        for v in ["0", "false", "no", "", "2", "on"] {
            assert!(!is_truthy(v), "{v:?} should not be truthy");
        }
    }

    // ---- Policy: from_env resolution & precedence ----
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

        // Empty-set allow env -> treated as unset -> AllowAll (does not disable everything).
        temp_env::with_vars(
            [
                (DYN_DISABLE_ENGINE_ROUTES, None),
                (DYN_ENGINE_ROUTES_ALLOW, Some("  , ,")),
                (DYN_ENGINE_ROUTES_DENY, None),
            ],
            || {
                assert_eq!(EngineRoutePolicy::from_env(), EngineRoutePolicy::AllowAll);
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

    // ---- Registry: policy enforcement ----

    fn dummy_callback() -> EngineRouteCallback {
        Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }))
    }

    #[test]
    fn test_registry_default_policy_allows() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);
        assert!(registry.is_allowed("control/start_profile"));
        assert_eq!(registry.policy(), &EngineRoutePolicy::AllowAll);
    }

    #[test]
    fn test_registry_disable_all_blocks() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::DisableAll);
        registry.register("control/start_profile", dummy_callback());
        // Route is registered, but policy blocks it.
        assert!(!registry.is_allowed("control/start_profile"));
        assert!(registry.get("control/start_profile").is_some());
    }

    #[test]
    fn test_registry_allowlist_enforced() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::Allowlist(set(&[
            "update/model_taints",
        ])));
        assert!(registry.is_allowed("update/model_taints"));
        assert!(!registry.is_allowed("control/update_weights_from_disk"));
    }

    #[test]
    fn test_registry_default_is_from_env() {
        // Default/new resolve from env; with no vars set that is AllowAll.
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
