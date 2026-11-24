// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use axum::{
    Json,
    body::Bytes,
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::environment_names::runtime::engine_routes as env_engine_routes;

/// Operator policy governing which `/engine/*` control routes are served.
///
/// Resolved once from the environment when the [`EngineRouteRegistry`] is constructed and
/// enforced at **registration time** (a denied route is never wired), so it applies
/// uniformly across every backend. The default is [`AllowAll`](EngineRoutePolicy::AllowAll),
/// so behavior is unchanged unless an operator opts in.
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
        let disable_all =
            crate::config::env_is_truthy(env_engine_routes::DYN_DISABLE_ENGINE_ROUTES);
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
                vars_set = %set.join(", "),
                applying = set[0],
                "multiple engine-route policy variables set; applying by precedence \
                 (DYN_DISABLE_ENGINE_ROUTES > DYN_ENGINE_ROUTES_ALLOW > DYN_ENGINE_ROUTES_DENY)"
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

    /// The operator's *explicit* decision for `route`, or `None` if the operator
    /// set no rule that names it (so the route's [`RouteDefault`] governs).
    ///
    /// - `AllowAll` — no explicit rule for any route (`None`).
    /// - `DisableAll` — explicitly denies every route.
    /// - `Allowlist` — explicitly decides every route: allowed iff listed.
    /// - `Denylist` — explicitly denies the listed routes; others fall through.
    fn explicit_decision(&self, route: &str) -> Option<bool> {
        match self {
            EngineRoutePolicy::AllowAll => None,
            EngineRoutePolicy::DisableAll => Some(false),
            EngineRoutePolicy::Allowlist(set) => Some(set.contains(route)),
            EngineRoutePolicy::Denylist(set) => {
                if set.contains(route) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }
}

/// Baseline disposition of an `/engine` route before any operator policy — the
/// secure default a backend declares at registration. The registry consults it
/// together with the [`EngineRoutePolicy`] and only registers permitted routes,
/// so a denied route is never wired (a request to it 404s, and it never appears
/// in discovery).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteDefault {
    /// Served by default (normal routes).
    Enabled,
    /// Off unless an operator explicitly allows it.
    Disabled,
    /// Off unless the caller-supplied flag is set (e.g. `enable_rl` for the
    /// RCE-capable weight-update routes). RL never appears here — the backend
    /// resolves its flag to a bool.
    Gated(bool),
}

impl RouteDefault {
    fn served_by_default(self) -> bool {
        matches!(self, RouteDefault::Enabled | RouteDefault::Gated(true))
    }

    /// Non-`Enabled` routes ship restricted; overriding them warrants a warning.
    fn is_sensitive(self) -> bool {
        !matches!(self, RouteDefault::Enabled)
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

/// Legacy body-only callbacks preserve their JSON response and 200/500 status contract.
pub type EngineRouteCallback =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send>> + Send + Sync>;

#[derive(Serialize)]
pub struct EngineHttpRequest {
    pub method: String,
    /// Path relative to the registered handler name.
    pub path: String,
    pub body: Value,
}

#[derive(Deserialize)]
pub struct EngineHttpResponse {
    pub status: u16,
    pub body: Value,
}

pub type EngineHttpCallback = Arc<
    dyn Fn(
            EngineHttpRequest,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<EngineHttpResponse>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
enum Handler {
    Legacy(EngineRouteCallback),
    Http(EngineHttpCallback),
}

/// Registry for both legacy callbacks and method-aware handlers on `/engine/*`.
///
/// Every registration path applies the operator [`EngineRoutePolicy`] at registration
/// time, so a denied route is never wired: a request to it 404s and it never appears
/// in discovery. Backends need no policy logic of their own.
#[derive(Clone)]
pub struct EngineRouteRegistry {
    routes: Arc<RwLock<HashMap<String, Handler>>>,
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

    /// Whether the policy, combined with the route's default, permits serving `route`.
    /// An operator rule that overrides a restricted route's default is honored, with a
    /// warning.
    fn permits(&self, route: &str, default: RouteDefault) -> bool {
        match self.policy.explicit_decision(route) {
            Some(allowed) => {
                if default.is_sensitive() && allowed != default.served_by_default() {
                    tracing::warn!(
                        route,
                        serving = allowed,
                        "engine-route policy overrides a restricted route's default"
                    );
                }
                allowed
            }
            None => default.served_by_default(),
        }
    }

    /// Register a callback for a route (e.g., "control/start_profile" for /engine/control/start_profile)
    ///
    /// A route name is expected to be registered exactly once. Re-registering an
    /// existing name overwrites the previous callback and emits a warning, since
    /// it usually signals two registration mechanisms colliding rather than an
    /// intentional replacement.
    pub fn register(&self, route: &str, callback: EngineRouteCallback) {
        self.register_with_default(route, callback, RouteDefault::Enabled);
    }

    /// Register a callback with an explicit [`RouteDefault`], applying the operator
    /// policy **at registration time**. A route the policy+default deny is *not
    /// registered* — so a request to it 404s and it never appears in discovery.
    pub fn register_with_default(
        &self,
        route: &str,
        callback: EngineRouteCallback,
        default: RouteDefault,
    ) {
        if !self.permits(route, default) {
            // Defense-in-depth: a denied route must always end up absent. If a
            // prior registration already wired this name, drop it so the policy
            // denial can't leave a stale callback reachable via the dispatch path.
            self.routes.write().remove(route);
            tracing::debug!(
                route,
                "engine route not registered (disabled by policy/default)"
            );
            return;
        }
        if self
            .routes
            .write()
            .insert(route.to_owned(), Handler::Legacy(callback))
            .is_some()
        {
            tracing::warn!(route, "Overwriting already-registered engine route");
        } else {
            tracing::debug!(route, "registered engine route");
        }
    }

    pub fn get(&self, route: &str) -> Option<EngineRouteCallback> {
        match self.routes.read().get(route) {
            Some(Handler::Legacy(callback)) => Some(Arc::clone(callback)),
            _ => None,
        }
    }

    pub fn routes(&self) -> Vec<String> {
        self.routes.read().keys().cloned().collect()
    }

    /// Register a handler for `/engine/{name}/...` without changing legacy callbacks.
    pub fn register_http(&self, name: String, callback: EngineHttpCallback) -> anyhow::Result<()> {
        anyhow::ensure!(
            name.split('/').all(|segment| {
                !segment.is_empty()
                    && segment != "."
                    && segment != ".."
                    && segment
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
            }),
            "handler name must contain nonempty URL segments"
        );
        // The policy names an HTTP handler by its registered name, which covers every
        // path under `/engine/{name}/`. A denied handler is not wired and returns Ok, the
        // same as a denied legacy route, so callers need no policy logic.
        if !self.permits(&name, RouteDefault::Enabled) {
            tracing::debug!(
                route = %name,
                "engine HTTP route not registered (disabled by policy)"
            );
            return Ok(());
        }
        let mut routes = self.routes.write();
        let prefix = format!("{name}/");
        anyhow::ensure!(
            !routes
                .keys()
                .any(|key| key == &name || key.starts_with(&prefix)),
            "engine route namespace already registered"
        );
        routes.insert(name, Handler::Http(callback));
        Ok(())
    }

    pub fn unregister_http(&self, name: &str) {
        let mut routes = self.routes.write();
        if matches!(routes.get(name), Some(Handler::Http(_))) {
            routes.remove(name);
        }
    }

    fn get_http(&self, path: &str) -> Option<(EngineHttpCallback, String)> {
        let (name, suffix) = path.split_once('/').unwrap_or((path, ""));
        let routes = self.routes.read();
        if let Some(Handler::Http(callback)) = routes.get(path) {
            return Some((Arc::clone(callback), String::new()));
        }
        match routes.get(name) {
            Some(Handler::Http(callback)) => Some((Arc::clone(callback), suffix.to_owned())),
            _ => None,
        }
    }
}

fn parse_json(body: &[u8], empty: Value) -> Result<Value, serde_json::Error> {
    if body.is_empty() {
        return Ok(empty);
    }
    serde_json::from_slice(body)
}

fn invalid_json(error: serde_json::Error) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error":"Invalid JSON", "message":error.to_string()})),
    )
        .into_response()
}

pub(crate) async fn dispatch(
    registry: &EngineRouteRegistry,
    path: String,
    method: Method,
    body: Bytes,
) -> Response {
    // Exact legacy routes keep their precedence and body-only callback contract.
    if let Some(callback) = registry.get(&path) {
        let payload = match parse_json(&body, json!({})) {
            Ok(value) => value,
            Err(error) => return invalid_json(error),
        };
        return match callback(payload).await {
            Ok(response) => (StatusCode::OK, response.to_string()).into_response(),
            Err(error) => {
                tracing::error!(path, %error, "Engine route handler failed");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error":"Handler error", "message":error.to_string()}).to_string(),
                )
                    .into_response()
            }
        };
    }
    let Some((callback, relative_path)) = registry.get_http(&path) else {
        if let Err(error) = parse_json(&body, Value::Null) {
            return invalid_json(error);
        }
        return (
            StatusCode::NOT_FOUND,
            json!({"error":"Route not found", "message":format!("Route /engine/{path} not found")})
                .to_string(),
        )
            .into_response();
    };
    if body.len() > 64 * 1024 {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let payload = match parse_json(&body, Value::Null) {
        Ok(value) => value,
        Err(error) => return invalid_json(error),
    };
    match callback(EngineHttpRequest {
        method: method.to_string(),
        path: relative_path,
        body: payload,
    })
    .await
    {
        Ok(response) => match StatusCode::from_u16(response.status) {
            Ok(status) if (200..600).contains(&response.status) => {
                (status, Json(response.body)).into_response()
            }
            _ => {
                tracing::error!(path, "Engine HTTP handler returned an invalid status");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        },
        Err(error) => {
            tracing::error!(path, %error, "Engine HTTP handler failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::{Path, State},
        http::Request,
        routing::any,
    };
    use tower::ServiceExt;

    fn router(registry: EngineRouteRegistry) -> Router {
        Router::new()
            .route(
                "/engine/{*path}",
                any(
                    |State(registry): State<EngineRouteRegistry>,
                     Path(path): Path<String>,
                     method: Method,
                     body: Bytes| async move {
                        dispatch(&registry, path, method, body).await
                    },
                ),
            )
            .with_state(registry)
    }

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    // ---- Policy: explicit_decision truth table for each variant ----

    #[test]
    fn test_policy_explicit_decision() {
        // None = no explicit rule, so the route's RouteDefault governs.
        assert_eq!(
            EngineRoutePolicy::AllowAll.explicit_decision("anything"),
            None
        );
        assert_eq!(
            EngineRoutePolicy::DisableAll.explicit_decision("control/start_profile"),
            Some(false)
        );

        let allow = EngineRoutePolicy::Allowlist(set(&["control/start_profile"]));
        assert_eq!(allow.explicit_decision("control/start_profile"), Some(true));
        assert_eq!(
            allow.explicit_decision("control/update_weights_from_disk"),
            Some(false)
        );

        let deny = EngineRoutePolicy::Denylist(set(&["control/update_weights_from_disk"]));
        assert_eq!(
            deny.explicit_decision("control/update_weights_from_disk"),
            Some(false)
        );
        assert_eq!(deny.explicit_decision("control/start_profile"), None);
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

    // ---- Registry: enforces policy at registration time, defaults from env ----

    #[test]
    fn test_registry_enforces_policy_and_defaults_from_env() {
        let callback = || -> EngineRouteCallback {
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }))
        };

        // Under DisableAll a policy-denied route is dropped at registration: never wired,
        // so `get` returns None and a request to it would 404.
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::DisableAll);
        registry.register("control/start_profile", callback());
        assert!(registry.get("control/start_profile").is_none());

        // Allowlist: a listed route registers; an unlisted one is dropped.
        let registry =
            EngineRouteRegistry::with_policy(EngineRoutePolicy::Allowlist(set(&["control/keep"])));
        registry.register("control/keep", callback());
        registry.register("control/drop", callback());
        assert!(registry.get("control/keep").is_some());
        assert!(registry.get("control/drop").is_none());

        // RouteDefault governs when the policy sets no explicit rule (AllowAll here):
        //  - Enabled   -> served
        //  - Disabled  -> dropped unless the operator explicitly allows it
        //  - Gated(b)  -> served iff b
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);
        registry.register_with_default("r/enabled", callback(), RouteDefault::Enabled);
        registry.register_with_default("r/disabled", callback(), RouteDefault::Disabled);
        registry.register_with_default("r/gated_on", callback(), RouteDefault::Gated(true));
        registry.register_with_default("r/gated_off", callback(), RouteDefault::Gated(false));
        assert!(registry.get("r/enabled").is_some());
        assert!(registry.get("r/disabled").is_none());
        assert!(registry.get("r/gated_on").is_some());
        assert!(registry.get("r/gated_off").is_none());

        // An explicit allowlist entry overrides a Gated(false)/Disabled default (operator wins).
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::Allowlist(set(&[
            "r/gated_off",
            "r/disabled",
        ])));
        registry.register_with_default("r/gated_off", callback(), RouteDefault::Gated(false));
        registry.register_with_default("r/disabled", callback(), RouteDefault::Disabled);
        assert!(registry.get("r/gated_off").is_some());
        assert!(registry.get("r/disabled").is_some());

        // A denylist entry overrides an Enabled/Gated(true) default the other way.
        let registry =
            EngineRouteRegistry::with_policy(EngineRoutePolicy::Denylist(set(&["r/gated_on"])));
        registry.register_with_default("r/gated_on", callback(), RouteDefault::Gated(true));
        assert!(registry.get("r/gated_on").is_none());

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

    #[test]
    fn test_denied_reregistration_removes_prior_callback() {
        // Defense-in-depth: a route registered while permitted, then re-registered
        // under a policy/default that denies it, must end up absent — not leave the
        // earlier callback reachable.
        let cb = || -> EngineRouteCallback {
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }))
        };
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);
        registry.register("control/foo", cb());
        assert!(registry.get("control/foo").is_some());
        registry.register_with_default("control/foo", cb(), RouteDefault::Gated(false));
        assert!(registry.get("control/foo").is_none());
    }

    #[tokio::test]
    async fn test_registry_enforces_policy_on_http_handlers() {
        // HTTP handlers go through the same registration-time policy as legacy routes,
        // keyed on the registered name, so a backend needs no policy logic of its own.
        let handler = || -> EngineHttpCallback {
            Arc::new(|_| {
                Box::pin(async {
                    Ok(EngineHttpResponse {
                        status: 200,
                        body: Value::Null,
                    })
                })
            })
        };
        let status = |registry: &EngineRouteRegistry, path: &str| {
            let app = router(registry.clone());
            let uri = format!("/engine/{path}");
            async move {
                app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status()
            }
        };

        // Denied: registration succeeds but wires nothing, so every path under it 404s.
        let registry =
            EngineRouteRegistry::with_policy(EngineRoutePolicy::Denylist(set(&["drain"])));
        registry.register_http("drain".into(), handler()).unwrap();
        registry.register_http("status".into(), handler()).unwrap();
        assert!(!registry.routes().contains(&"drain".to_string()));
        assert_eq!(status(&registry, "drain").await, StatusCode::NOT_FOUND);
        assert_eq!(status(&registry, "drain/sub").await, StatusCode::NOT_FOUND);
        assert_eq!(status(&registry, "status").await, StatusCode::OK);

        // DisableAll covers HTTP handlers too.
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::DisableAll);
        registry.register_http("planner".into(), handler()).unwrap();
        assert_eq!(
            status(&registry, "planner/config").await,
            StatusCode::NOT_FOUND
        );

        // Allowlist: only listed handler names are wired.
        let registry =
            EngineRouteRegistry::with_policy(EngineRoutePolicy::Allowlist(set(&["planner"])));
        registry.register_http("planner".into(), handler()).unwrap();
        registry.register_http("drain".into(), handler()).unwrap();
        assert_eq!(status(&registry, "planner/config").await, StatusCode::OK);
        assert_eq!(status(&registry, "drain").await, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_registry_basic() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);

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
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);

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
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);

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
    #[tokio::test]
    async fn http_handlers_preserve_methods_status_and_registration_lifetime() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);
        registry
            .register_http(
                "planner".into(),
                Arc::new(|request| {
                    Box::pin(async move {
                        assert_eq!(request.path, "config");
                        let status = match request.method.as_str() {
                            "GET" => 200,
                            "PATCH" if request.body["expected_revision"] == 3 => 202,
                            "PATCH" => 409,
                            _ => 405,
                        };
                        Ok(EngineHttpResponse {
                            status,
                            body: request.body,
                        })
                    })
                }),
            )
            .unwrap();
        let app = router(registry.clone());
        for (method, path, body, expected) in [
            ("GET", "/engine/planner/config", "", 200),
            (
                "PATCH",
                "/engine/planner/config",
                r#"{"expected_revision":3}"#,
                202,
            ),
            ("PATCH", "/engine/planner/config", "{}", 409),
            ("DELETE", "/engine/planner/config", "", 405),
            ("GET", "/engine/missing/config", "", 404),
            ("PATCH", "/engine/planner/config", "{", 400),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
            if expected == 400 {
                assert_eq!(response.headers()["content-type"], "application/json");
                let body = to_bytes(response.into_body(), 1024).await.unwrap();
                let error: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(error["error"], "Invalid JSON");
                assert!(error["message"].as_str().is_some());
            }
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/engine/planner/config")
                    .body(Body::from("x".repeat(65537)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        registry.unregister_http("planner");
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/engine/planner/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
    #[tokio::test]
    async fn status_aware_engine_route_can_use_a_nested_path() {
        let registry = EngineRouteRegistry::new();
        registry
            .register_http(
                "control/pause".into(),
                Arc::new(|request| {
                    Box::pin(async move {
                        assert!(request.path.is_empty());
                        Ok(EngineHttpResponse {
                            status: 409,
                            body: Value::Null,
                        })
                    })
                }),
            )
            .unwrap();
        let response = dispatch(
            &registry,
            "control/pause".into(),
            Method::POST,
            Bytes::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn legacy_payloads_are_not_interpreted_as_http_envelopes() {
        let registry = EngineRouteRegistry::with_policy(EngineRoutePolicy::AllowAll);
        registry.register("legacy", Arc::new(|body| Box::pin(async move { Ok(body) })));
        let app = router(registry);
        let payload = json!({"status":418, "body":{"value":7}});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/engine/legacy")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), payload);
    }
}
