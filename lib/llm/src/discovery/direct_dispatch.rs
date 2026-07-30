// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition-root registry for direct-gRPC transport dispatch providers.
//!
//! When a worker registers with `runtime_data["direct_backend"] = "<name>"`, the
//! frontend dispatches inference straight to that worker's native engine gRPC
//! server instead of over the Dynamo request plane — while `PushRouter` keeps
//! instance selection, occupancy, fault detection, and migration (only the
//! transport below the seam changes; see [`dynamo_runtime::pipeline::StreamingDispatch`]).
//!
//! The engine-specific translation lives OUTSIDE `dynamo-llm` (e.g. in the
//! TensorRT-LLM sidecar crate). It is injected here via a provider registered at
//! a composition root (a frontend binary / the Python bindings) before the
//! frontend runs. This keeps `dynamo-llm` engine-agnostic — a Rust
//! dependency-inversion boundary, not a dynamic-plugin ABI.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dynamo_runtime::pipeline::StreamingDispatch;
use dynamo_runtime::protocols::annotated::Annotated;
use parking_lot::RwLock;

use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

/// `runtime_data` key a direct-backend worker sets to name its dispatch provider
/// (e.g. `"trtllm"`). The registrar writes it; the watcher reads it to select a
/// [`DirectDispatchProvider`]. Single source shared across the write/read/lookup
/// sites so they can't drift.
pub const DIRECT_BACKEND_KEY: &str = "direct_backend";

/// The transport-seam engine `PushRouter` dispatches through for an LLM model:
/// typed `PreprocessedRequest` in, `Annotated<LLMEngineOutput>` out.
pub type LlmStreamingDispatch =
    Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>;

/// Builds a direct-gRPC transport dispatch for a model whose worker advertises
/// `runtime_data["direct_backend"] == self.backend()`.
#[async_trait]
pub trait DirectDispatchProvider: Send + Sync {
    /// The `direct_backend` name this provider handles (e.g. `"trtllm"`).
    fn backend(&self) -> &str;

    /// Build the transport dispatch for one model. Engine parameters (context
    /// length, etc.) come from the model card; per-instance gRPC addresses are
    /// resolved per request from the routed `AddressedRequest`.
    async fn build(&self, card: &ModelDeploymentCard) -> anyhow::Result<LlmStreamingDispatch>;
}

type Registry = RwLock<HashMap<String, Arc<dyn DirectDispatchProvider>>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a provider at a composition root. Last registration for a given
/// backend name wins. Call before running the frontend.
pub fn register_direct_dispatch_provider(provider: Arc<dyn DirectDispatchProvider>) {
    registry()
        .write()
        .insert(provider.backend().to_string(), provider);
}

/// Look up the provider registered for a `direct_backend` name, if any.
pub fn direct_dispatch_provider(backend: &str) -> Option<Arc<dyn DirectDispatchProvider>> {
    registry().read().get(backend).cloned()
}
