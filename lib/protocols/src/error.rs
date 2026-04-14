// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for protocol type operations.
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum OpenAIError {
    /// OpenAI returns error object with details of API call failure
    #[error("{0}")]
    ApiError(ApiError),
    /// Error when a response cannot be deserialized into a Rust type
    #[error("failed to deserialize api response: {0}")]
    JSONDeserialize(serde_json::Error),
    /// Error from client side validation
    /// or when builder fails to build request before making API call
    #[error("invalid args: {0}")]
    InvalidArgument(String),
}

/// OpenAI API returns error object on failure
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiError {
    pub message: String,
    pub r#type: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();

        if let Some(r#type) = &self.r#type {
            parts.push(format!("{}:", r#type));
        }

        parts.push(self.message.clone());

        if let Some(param) = &self.param {
            parts.push(format!("(param: {param})"));
        }

        if let Some(code) = &self.code {
            parts.push(format!("(code: {code})"));
        }

        write!(f, "{}", parts.join(" "))
    }
}

/// Wrapper to deserialize the error object nested in "error" JSON key
#[derive(Debug, Deserialize, Serialize)]
pub struct WrappedError {
    pub error: ApiError,
}
