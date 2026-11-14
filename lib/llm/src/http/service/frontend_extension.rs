// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for the HTTP frontend (static-path routes).
//! The context is a narrow, read-only view of live frontend state, so
//! extensions have the same capability in any language. Grow it with typed
//! read-only accessors as needed.

use std::collections::HashSet;
use std::sync::Arc;

use axum::Router;
use axum::handler::Handler;
use axum::http::Method;
use axum::routing::{MethodFilter, on};

use super::RouteDoc;
use super::service_v2::State;

/// Live, read-only view of frontend state exposed to extensions (typed
/// accessors only, not the internal `State`/`ModelManager`).
#[derive(Clone)]
pub struct FrontendExtensionContext {
    state: Arc<State>,
}

impl FrontendExtensionContext {
    pub(crate) fn new(state: Arc<State>) -> Self {
        Self { state }
    }

    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }

    pub fn has_any_ready_model(&self) -> bool {
        self.state.manager().has_any_ready_model()
    }

    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.state.manager().is_model_ready_to_serve(model)
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.state.manager().model_display_names()
    }

    pub fn serving_ready_display_names(&self) -> HashSet<String> {
        self.state.manager().serving_ready_display_names()
    }
}

/// Routes returned by an extension. Built only via [`FrontendRouteSet::builder`],
/// which records each route in both the router and its [`RouteDoc`] so the two
/// can't drift.
pub struct FrontendRouteSet {
    route_docs: Vec<RouteDoc>,
    router: Router,
}

impl FrontendRouteSet {
    /// Start building a route set.
    pub fn builder() -> FrontendRouteSetBuilder {
        FrontendRouteSetBuilder::default()
    }

    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }

    pub(crate) fn into_parts(self) -> (Vec<RouteDoc>, Router) {
        (self.route_docs, self.router)
    }
}

/// Reject paths that would panic axum's `Router::route` (params, wildcards,
/// `:`/`*` segments, whitespace). Single source of truth, reused by Python.
pub fn validate_extension_route_path(path: &str) -> anyhow::Result<()> {
    if !path.starts_with('/') {
        anyhow::bail!("frontend route path must start with '/': {path:?}");
    }
    // matchit (axum's router) reserves segments starting with ':' or '*'.
    if path.split('/').any(|seg| seg.starts_with([':', '*'])) {
        anyhow::bail!("frontend route path segment must not start with ':' or '*': {path:?}");
    }
    if path
        .chars()
        .any(|c| matches!(c, '{' | '}' | '*') || c.is_whitespace() || c.is_control())
    {
        anyhow::bail!(
            "frontend route path must be static (no '{{...}}', '*', or whitespace): {path:?}"
        );
    }
    Ok(())
}

/// Extension routes serve these five methods; `HEAD` is served by a `GET`
/// route. Single source of truth, reused by Python.
pub fn validate_extension_route_method(method: &Method) -> anyhow::Result<()> {
    if !matches!(
        *method,
        Method::GET | Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    ) {
        anyhow::bail!(
            "unsupported frontend route method {method}; expected GET, POST, PUT, PATCH or DELETE"
        );
    }
    Ok(())
}

/// Registers each route into both the router and its docs atomically; `route`
/// validates the path and rejects duplicates instead of panicking axum.
#[derive(Default)]
pub struct FrontendRouteSetBuilder {
    route_docs: Vec<RouteDoc>,
    router: Router,
    seen_routes: HashSet<(Method, String)>,
}

impl FrontendRouteSetBuilder {
    /// Register a `GET` route, recording it in both the router and the docs.
    pub fn get<H, T>(self, path: impl Into<String>, handler: H) -> anyhow::Result<Self>
    where
        H: Handler<T, ()>,
        T: 'static,
    {
        self.route(Method::GET, path, handler)
    }

    /// Register a route, recording it in both the router and the docs. Errors
    /// on an unsupported method, an invalid path, or a duplicate method and
    /// path within this set.
    pub fn route<H, T>(
        mut self,
        method: Method,
        path: impl Into<String>,
        handler: H,
    ) -> anyhow::Result<Self>
    where
        H: Handler<T, ()>,
        T: 'static,
    {
        let path = path.into();
        validate_extension_route_method(&method)?;
        validate_extension_route_path(&path)?;
        if !self.seen_routes.insert((method.clone(), path.clone())) {
            anyhow::bail!("duplicate frontend route registered: {method} {path}");
        }
        let filter = MethodFilter::try_from(method.clone())?;
        self.route_docs.push(RouteDoc::new(method, path.clone()));
        self.router = self.router.route(&path, on(filter, handler));
        Ok(self)
    }

    pub fn build(self) -> FrontendRouteSet {
        FrontendRouteSet {
            route_docs: self.route_docs,
            router: self.router,
        }
    }
}

/// Callback that attaches additional frontend routes during HTTP service build,
/// given a read-only [`FrontendExtensionContext`]. Returns an error so invalid
/// extension configuration becomes a clean startup failure.
pub type FrontendRouteExtension = Arc<
    dyn Fn(FrontendExtensionContext) -> anyhow::Result<FrontendRouteSet> + Send + Sync + 'static,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn validate_route_path_rejects_non_static() {
        assert!(validate_extension_route_path("/ok").is_ok());
        assert!(validate_extension_route_path("/a/b_c").is_ok());
        for bad in [
            "no-slash",
            "/:id",
            "/{id}",
            "/x/{*rest}",
            "/*rest",
            "/has space",
        ] {
            assert!(
                validate_extension_route_path(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn builder_rejects_duplicate_path_within_set() {
        let dup = FrontendRouteSet::builder()
            .get("/a", || async { StatusCode::OK })
            .unwrap()
            .get("/a", || async { StatusCode::OK });
        assert!(dup.is_err(), "duplicate GET path within a set must error");
    }

    #[test]
    fn builder_keys_routes_by_method_and_path() {
        let set = FrontendRouteSet::builder()
            .get("/a", || async { StatusCode::OK })
            .unwrap()
            .route(Method::POST, "/a", || async { StatusCode::OK })
            .expect("GET and POST on one path must both register");
        let dup = set.route(Method::POST, "/a", || async { StatusCode::OK });
        assert!(dup.is_err(), "duplicate POST path within a set must error");
        let trace =
            FrontendRouteSet::builder().route(Method::TRACE, "/b", || async { StatusCode::OK });
        assert!(trace.is_err(), "TRACE must be rejected");
    }

    #[test]
    fn builder_rejects_colon_path() {
        let bad = FrontendRouteSet::builder().get("/:id", || async { StatusCode::OK });
        assert!(bad.is_err(), "colon-prefixed segment must error, not panic");
    }
}
