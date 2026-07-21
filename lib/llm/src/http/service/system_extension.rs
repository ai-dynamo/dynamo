// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for system routes hosted by the HTTP frontend.

use std::sync::Arc;

use axum::Router;
use dynamo_runtime::discovery::Discovery;

use super::RouteDoc;
use super::service_v2::State;
use crate::discovery::ModelManager;

/// Live frontend context available to system route extensions.
///
/// The wrapper keeps the extension API focused on system-route needs. Additional
/// accessors can be added here as the extension surface matures.
#[derive(Clone)]
pub struct SystemRouteContext {
    state: Arc<State>,
}

impl SystemRouteContext {
    pub(crate) fn new(state: Arc<State>) -> Self {
        Self { state }
    }

    /// Return the shared Axum state for routes that need to install handlers
    /// with Router::with_state.
    pub fn state_clone(&self) -> Arc<State> {
        self.state.clone()
    }

    pub fn manager(&self) -> &ModelManager {
        self.state.manager()
    }

    pub fn manager_clone(&self) -> Arc<ModelManager> {
        self.state.manager_clone()
    }

    pub fn discovery(&self) -> Arc<dyn Discovery> {
        self.state.discovery()
    }

    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }
}

/// Routes and route documentation returned by a system route extension.
pub struct SystemRouteSet {
    route_docs: Vec<RouteDoc>,
    router: Router,
}

impl SystemRouteSet {
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

impl From<(Vec<RouteDoc>, Router)> for SystemRouteSet {
    fn from((route_docs, router): (Vec<RouteDoc>, Router)) -> Self {
        Self::new(route_docs, router)
    }
}

/// Callback used to attach additional system routes during HTTP service build.
///
/// Extensions receive live frontend context used by built-in system handlers,
/// so custom routes can answer from current model manager and discovery state
/// instead of precomputed startup metadata.
pub type SystemRouteExtension =
    Arc<dyn Fn(SystemRouteContext) -> SystemRouteSet + Send + Sync + 'static>;
