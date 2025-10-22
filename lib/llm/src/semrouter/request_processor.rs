// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-agnostic request processing for semantic routing
//!
//! # Architecture
//!
//! Semantic routing must occur **before engine selection** because it determines
//! which model (and thus which engine) to use. The flow is:
//!
//! ```text
//! Request (model="router")
//!   ↓
//! process_chat_request()  ← Mutates request.model to actual target
//!   ↓
//! get_engine(request.model)  ← Selects engine based on routed model
//!   ↓
//! engine.generate()  ← Engine has its own preprocessor
//! ```
//!
//! This is different from audit/observability which wraps around the entire
//! execution pipeline. Routing is a pre-processing step that affects which
//! pipeline gets selected.
//!
//! # Performance
//!
//! - **Zero allocation** when routing is disabled (`router: None`)
//! - **Inline-friendly** with `#[inline]` for minimal call overhead
//! - **No boxing** or trait objects beyond the classifier itself
//! - **Single code path** for all transports (HTTP, gRPC, etc.)
//!
//! # Transport Decoupling
//!
//! Each transport (HTTP, gRPC) calls `process_chat_request()` before engine
//! selection. The function is transport-agnostic; only header extraction is
//! transport-specific, which is minimal and unavoidable (HTTP uses HeaderMap,
//! gRPC uses MetadataMap, etc.).

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::semrouter::{RouteDecision, RoutingMode, SemRouter};

/// Process a chat completion request with semantic routing
///
/// This is the single entry point for all transports (HTTP, gRPC, etc.)
/// Call this before engine selection to allow routing to determine the target model.
///
/// # Performance
/// - Zero allocation if routing is disabled
/// - Inline-friendly with `#[inline]`
/// - No boxing or dynamic dispatch beyond the classifier
///
/// # Arguments
/// - `request`: Mutable reference to the request (model field will be mutated)
/// - `router`: Optional router from state
/// - `routing_header`: Optional routing control header value
/// - `transport`: Transport identifier for observability ("http", "grpc", etc.)
///
/// # Returns
/// Optional routing decision for observability/metrics
#[inline]
pub fn process_chat_request(
    request: &mut NvCreateChatCompletionRequest,
    router: Option<&SemRouter>,
    routing_header: Option<&str>,
    transport: &str,
) -> Option<RouteDecision> {
    let router = router?;
    let mode = RoutingMode::from_header(routing_header);

    use crate::semrouter::routing::apply_routing_direct;
    apply_routing_direct(request, router, mode, transport)
}

/// Extract routing control header value from transport-specific header map
///
/// Each transport implements this trait to provide header access
pub trait RoutingHeaderProvider {
    fn routing_header(&self) -> Option<&str>;
}

// HTTP implementation (axum is always available)
impl RoutingHeaderProvider for axum::http::HeaderMap {
    #[inline]
    fn routing_header(&self) -> Option<&str> {
        self.get("x-dynamo-routing")?.to_str().ok()
    }
}

// gRPC implementation would go here when needed
// impl RoutingHeaderProvider for tonic::metadata::MetadataMap { ... }

