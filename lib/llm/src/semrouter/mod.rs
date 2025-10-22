// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public surface for semantic routing (multi-class).
//! Keep this module as the single entrypoint; avoid a parallel hook.rs.

mod layer;
mod decision;
mod config;
mod ctx;
pub mod classifier;
pub mod metrics;

pub use layer::{SemRouterHandle, SemRouterLayer, SemRouterService};
pub use decision::{Decision, Mode, OverridePolicy, RouteAction};
pub use config::{ClassRoute, ClassifierConfig, SemRouterConfig};
pub use ctx::{RequestCtx, RouteMode};

use std::sync::Arc;
use classifier::{build_classifier, Classifier};

/// Build a ready-to-use handle from config (init classifiers, queues, etc.).
pub fn build_router_handle(cfg: SemRouterConfig) -> anyhow::Result<SemRouterHandle> {
    let clf: Arc<dyn Classifier> = build_classifier(&cfg)?;
    Ok(SemRouterHandle::new(cfg, clf))
}
