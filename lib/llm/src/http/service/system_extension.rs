// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for system routes hosted by the HTTP frontend.

use std::sync::Arc;

use axum::Router;

use super::RouteDoc;
use super::service_v2::State;

/// Callback used to attach additional system routes during HTTP service build.
///
/// Extensions receive the shared frontend state used by built-in system
/// handlers, so custom routes can answer from live model manager and discovery
/// state instead of precomputed startup metadata.
pub type SystemRouteExtension =
    Arc<dyn Fn(Arc<State>) -> (Vec<RouteDoc>, Router) + Send + Sync + 'static>;
