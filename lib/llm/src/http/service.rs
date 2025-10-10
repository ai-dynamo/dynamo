// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP Service for Dynamo LLM
//!
//! The primary purpose of this crate is to service the dynamo-llm protocols via OpenAI compatible HTTP endpoints. This component
//! is meant to be a gateway/ingress into the Dynamo LLM Distributed Runtime.
//!
//! In order to create a common pattern, the HttpService forwards the incoming OAI Chat Request or OAI Completion Request to the
//! to a model-specific engines.  The engines can be attached and detached dynamically using the [`ModelManager`].
//!
//! Note: All requests, whether the client requests `stream=true` or `stream=false`, are propagated downstream as `stream=true`.
//! This enables use to handle only 1 pattern of request-response in the downstream services. Non-streaming user requests are
//! aggregated by the HttpService and returned as a single response.
//!
//! TODO(): Add support for model-specific metadata and status. Status will allow us to return a 503 when the model is supposed
//! to be ready, but there is a problem with the model.
//!
//! The [`service_v2::HttpService`] can be further extended to host any [`axum::Router`] using the [`service_v2::HttpServiceConfigBuilder`].

mod openai;

pub mod disconnect;
pub mod error;
pub mod health;
pub mod metrics;
pub mod openapi_docs;
pub mod service_v2;

pub use axum;
pub use metrics::Metrics;

/// Documentation for a route
#[derive(Debug, Clone)]
pub struct RouteDoc {
    method: axum::http::Method,
    path: String,
}

impl std::fmt::Display for RouteDoc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {}", self.method, self.path)
    }
}

impl RouteDoc {
    pub fn new<T: Into<String>>(method: axum::http::Method, path: T) -> Self {
        RouteDoc {
            method,
            path: path.into(),
        }
    }
}
