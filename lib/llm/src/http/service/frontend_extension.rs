// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for the HTTP frontend.
//!
//! Routes are the first extension kind; the context is deliberately a narrow,
//! read-only view of live frontend state (not the internal `State` that
//! built-in routes use), so extensions have the same capability regardless of
//! the language they are written in. Grow the surface with typed read-only
//! accessors as concrete extensions need them.

use std::collections::HashSet;
use std::sync::Arc;

use axum::Router;

use super::RouteDoc;
use super::service_v2::State;

/// Live, read-only frontend state exposed to extensions.
///
/// This is intentionally a narrowed view: it exposes typed read-only accessors
/// rather than the internal `State`/`ModelManager`, so a custom route can answer
/// from current frontend state without reaching into actor internals. Add
/// accessors here as the extension surface matures.
#[derive(Clone)]
pub struct FrontendExtensionContext {
    state: Arc<State>,
}

impl FrontendExtensionContext {
    pub(crate) fn new(state: Arc<State>) -> Self {
        Self { state }
    }

    /// Whether the HTTP service has finished startup and is ready to serve.
    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    /// Whether the frontend is shutting down (draining).
    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }

    /// Whether at least one model is registered and ready to serve.
    pub fn has_any_ready_model(&self) -> bool {
        self.state.manager().has_any_ready_model()
    }

    /// Whether the named model is registered and ready to serve.
    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.state.manager().is_model_ready_to_serve(model)
    }

    /// Display names of all registered models.
    pub fn model_display_names(&self) -> HashSet<String> {
        self.state.manager().model_display_names()
    }

    /// Display names of models that are ready to serve.
    pub fn serving_ready_display_names(&self) -> HashSet<String> {
        self.state.manager().serving_ready_display_names()
    }
}

/// Routes and route documentation returned by a frontend route extension.
pub struct FrontendRouteSet {
    route_docs: Vec<RouteDoc>,
    router: Router,
}

impl FrontendRouteSet {
    pub fn new(route_docs: Vec<RouteDoc>, router: Router) -> Self {
        Self { route_docs, router }
    }

    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }

    pub(crate) fn into_parts(self) -> (Vec<RouteDoc>, Router) {
        (self.route_docs, self.router)
    }
}

impl From<(Vec<RouteDoc>, Router)> for FrontendRouteSet {
    fn from((route_docs, router): (Vec<RouteDoc>, Router)) -> Self {
        Self::new(route_docs, router)
    }
}

/// Callback used to attach additional frontend routes during HTTP service build.
///
/// Extensions receive a narrow, read-only [`FrontendExtensionContext`], so custom
/// routes can answer from current frontend state instead of precomputed startup
/// metadata.
pub type FrontendRouteExtension =
    Arc<dyn Fn(FrontendExtensionContext) -> FrontendRouteSet + Send + Sync + 'static>;
