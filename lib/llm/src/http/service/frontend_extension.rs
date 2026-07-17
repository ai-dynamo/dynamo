// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for the HTTP frontend (currently: static GET routes).
//! The context is a narrow, read-only view of live frontend state, so
//! extensions have the same capability in any language. Grow it with typed
//! read-only accessors as needed.

use std::collections::HashSet;
use std::sync::Arc;

use axum::Router;
use axum::handler::Handler;
use axum::http::Method;
use axum::routing::get;

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
    /// Start building a route set. Currently only `GET` routes are supported.
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

/// Registers each route into both the router and its docs atomically.
#[derive(Default)]
pub struct FrontendRouteSetBuilder {
    route_docs: Vec<RouteDoc>,
    router: Router,
}

impl FrontendRouteSetBuilder {
    /// Register a `GET` route, recording it in both the router and the docs.
    pub fn get<H, T>(mut self, path: impl Into<String>, handler: H) -> Self
    where
        H: Handler<T, ()>,
        T: 'static,
    {
        let path = path.into();
        self.route_docs
            .push(RouteDoc::new(Method::GET, path.clone()));
        self.router = self.router.route(&path, get(handler));
        self
    }

    pub fn build(self) -> FrontendRouteSet {
        FrontendRouteSet {
            route_docs: self.route_docs,
            router: self.router,
        }
    }
}

/// Callback that attaches additional frontend routes during HTTP service build,
/// given a read-only [`FrontendExtensionContext`].
pub type FrontendRouteExtension =
    Arc<dyn Fn(FrontendExtensionContext) -> FrontendRouteSet + Send + Sync + 'static>;
