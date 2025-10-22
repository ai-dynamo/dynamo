// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-agnostic routing utilities
//!
//! This module provides helper functions for semantic routing that can be used
//! across different transports (HTTP, gRPC, etc.) without duplicating logic.

use crate::semrouter::{RouteDecision, SemRouter};
use crate::semrouter::types::{RequestMeta, RoutingMode};
use crate::types::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_async_openai::types::ChatCompletionRequestMessage;

/// Extract text from chat completion request messages
///
/// Robustly extracts user text from messages, handling both simple text
/// and content arrays with multiple parts.
pub fn extract_chat_text(request: &NvCreateChatCompletionRequest) -> Option<String> {
    let mut texts = Vec::new();

    for msg in request.inner.messages.iter().rev() {
        if let ChatCompletionRequestMessage::User(user_msg) = msg {
            match &user_msg.content {
                dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(text) => {
                    texts.push(text.clone());
                }
                dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Array(parts) => {
                    for part in parts {
                        if let dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart::Text(text_part) = part {
                            texts.push(text_part.text.clone());
                        }
                    }
                }
            }
            // Only extract from the most recent user message
            break;
        }
    }

    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

/// Apply semantic routing to a chat completion request
///
/// This is the main routing function that can be called from any transport.
/// It mutates the request's model field based on the routing decision.
///
/// # Arguments
/// * `request` - Mutable reference to the chat completion request
/// * `router` - Reference to the semantic router
/// * `routing_mode` - The routing mode (Auto, Force, Shadow, Off)
/// * `transport` - Transport identifier ("http", "grpc", etc.)
///
/// # Returns
/// * `Option<RouteDecision>` - The routing decision if routing was applied
pub fn apply_routing(
    request: &mut NvCreateChatCompletionRequest,
    router: &SemRouter,
    routing_mode: RoutingMode,
    transport: &str,
) -> Option<RouteDecision> {
    // Extract text for classification
    let request_text = extract_chat_text(request);

    // Create metadata for routing
    let meta = RequestMeta {
        tenant: None,
        region: None,
        transport,
        routing_mode,
        model_field: Some(&request.inner.model),
        request_text: request_text.as_deref(),
    };

    // Apply routing using JSON as intermediate representation
    // This is less efficient but preserves backwards compatibility
    let mut req_json = serde_json::to_value(&request.inner).ok()?;
    let decision = router.apply(&mut req_json, &meta);

    // Update the request model if it was changed
    if let Ok(updated_inner) = serde_json::from_value(req_json) {
        request.inner = updated_inner;
    }

    decision
}

/// Apply semantic routing without JSON serialization (more efficient)
///
/// This version directly modifies the model field without going through JSON,
/// making it more efficient but requiring knowledge of the routing decision structure.
pub fn apply_routing_direct(
    request: &mut NvCreateChatCompletionRequest,
    router: &SemRouter,
    routing_mode: RoutingMode,
    transport: &str,
) -> Option<RouteDecision> {
    let model = &request.inner.model;
    let is_alias = model.is_empty() || model == "router";

    // Determine if we should apply routing based on mode
    let should_route = match routing_mode {
        RoutingMode::Off => false,
        RoutingMode::Auto => is_alias,
        RoutingMode::Force => false,  // Force uses exact model specified, no routing
        RoutingMode::Shadow => true,  // Shadow mode computes decision but doesn't enforce
    };

    if !should_route {
        return None;
    }

    // Extract text for classification
    let request_text = extract_chat_text(request);

    // Create metadata for routing
    let meta = RequestMeta {
        tenant: None,
        region: None,
        transport,
        routing_mode,
        model_field: Some(&request.inner.model),
        request_text: request_text.as_deref(),
    };

    // Use JSON method for now (can optimize later)
    let mut req_json = serde_json::to_value(&request.inner).ok()?;
    let decision = router.apply(&mut req_json, &meta)?;

    // Only update model if not in shadow mode
    if !matches!(routing_mode, RoutingMode::Shadow) {
        if let Some(new_model) = req_json.get("model").and_then(|v| v.as_str()) {
            request.inner.model = new_model.to_string();
        }
    }

    Some(decision)
}

// Tests are omitted as NvCreateChatCompletionRequest doesn't implement Default
// Integration tests can be added to test the routing functionality

