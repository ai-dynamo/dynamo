// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic Router - Tower middleware for content-based request routing
//!
//! This crate provides a Tower/Axum layer that classifies incoming requests
//! and routes them to appropriate models based on content analysis.
//!
//! # Features
//! - `clf-fasttext`: Enable fastText classifier (recommended)
//!
//! # Example
//! ```no_run
//! use sem_router::maybe_semrouter_layer_from_env;
//! use axum::Router;
//!
//! let mut router = Router::new();
//! // ... add your routes ...
//!
//! #[cfg(feature = "clf-fasttext")]
//! if let Some(layer) = maybe_semrouter_layer_from_env() {
//!     router = router.layer(layer);
//! }
//! ```

mod layer;
mod decision;
mod config;
mod ctx;
pub mod classifier;

pub use layer::{SemRouterLayer, SemRouterService, SemRouterHandle};
pub use decision::{Decision, RouteAction, OverridePolicy, Mode};
pub use config::{SemRouterConfig, ClassRoute, ClassifierConfig};
pub use ctx::{RequestCtx, RouteMode};

use std::sync::Arc;
use classifier::{build_classifier, Classifier};

/// Build a ready-to-use handle from config (init classifiers, queues, etc.).
pub fn build_router_handle(cfg: SemRouterConfig) -> anyhow::Result<SemRouterHandle> {
    let clf: Arc<dyn Classifier> = build_classifier(&cfg)?;
    Ok(SemRouterHandle::new(cfg, clf))
}

/// Helper for zero-intrusion integration: load config from env and return layer if enabled.
///
/// Reads `DYN_SEMROUTER_CONFIG` environment variable for YAML config path.
/// Returns `None` if config is missing, disabled, or fails to load.
pub fn maybe_semrouter_layer_from_env() -> Option<SemRouterLayer> {
    let cfg = SemRouterConfig::load_from_env_and_defaults().ok()?;
    if !cfg.enabled { return None; }
    let handle = build_router_handle(cfg).ok()?;
    Some(SemRouterLayer::new(handle))
}

