// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request preprocessing before the gateway chooses an inference pool.

use std::time::Duration;

use bytes::Bytes;

use crate::picker::RequestInfo;

/// Mutations applied after the complete request has been inspected.
#[derive(Debug)]
pub struct RequestMutation {
    pub body: Bytes,
    pub headers: Vec<(String, String)>,
    pub remove_headers: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PreprocessError {
    pub status_code: u16,
    pub message: String,
}

impl PreprocessError {
    pub fn new(status_code: u16, message: impl Into<String>) -> Self {
        Self {
            status_code,
            message: message.into(),
        }
    }
}

/// A decision hook independent of worker discovery and reservation bookkeeping.
#[tonic::async_trait]
pub trait RequestPreprocessor: Send + Sync + 'static {
    fn validate_headers(
        &self,
        _headers: &[(String, String)],
        _end_of_stream: bool,
    ) -> Result<(), PreprocessError> {
        Ok(())
    }

    async fn preprocess(&self, request: &RequestInfo) -> Result<RequestMutation, PreprocessError>;
}

/// Bounds for preprocessing; endpoint-picker mode retains its existing policy.
#[derive(Debug, Clone)]
pub struct PreprocessLimits {
    pub max_body_bytes: usize,
    pub max_concurrent_requests: usize,
    pub max_in_flight_streams: usize,
    pub request_timeout: Duration,
    pub response_timeout: Duration,
}

impl Default for PreprocessLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: 2 * 1024 * 1024,
            max_concurrent_requests: 8,
            max_in_flight_streams: 16,
            request_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(120),
        }
    }
}
