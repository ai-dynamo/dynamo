// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::metadata::InvalidEppMetadata;

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error(transparent)]
    InvalidEppMetadata(#[from] InvalidEppMetadata),
    #[error("decode engine request failed: {0}")]
    DecodeUpstream(#[source] reqwest::Error),
    #[error("request was cancelled")]
    Cancelled,
    #[error("no P/D adapter is configured")]
    AdapterUnavailable,
    #[error("P/D adapter failed: {message}")]
    Adapter {
        status: StatusCode,
        code: &'static str,
        message: String,
    },
}

impl SidecarError {
    pub fn adapter(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self::Adapter {
            status,
            code,
            message: message.into(),
        }
    }

    fn into_response_fields(self) -> (StatusCode, &'static str, String) {
        match self {
            Self::InvalidEppMetadata(_) => (
                StatusCode::BAD_GATEWAY,
                "invalid_epp_metadata",
                "Invalid EPP routing metadata".to_string(),
            ),
            Self::DecodeUpstream(error) if error.is_timeout() => (
                StatusCode::GATEWAY_TIMEOUT,
                "decode_upstream_timeout",
                "The local decode engine timed out".to_string(),
            ),
            Self::DecodeUpstream(_) => (
                StatusCode::BAD_GATEWAY,
                "decode_upstream_error",
                "The local decode engine request failed".to_string(),
            ),
            Self::Cancelled => (
                StatusCode::BAD_GATEWAY,
                "request_cancelled",
                "The upstream request was cancelled".to_string(),
            ),
            Self::AdapterUnavailable => (
                StatusCode::NOT_IMPLEMENTED,
                "pd_adapter_unavailable",
                "No P/D adapter is configured".to_string(),
            ),
            Self::Adapter {
                status,
                code,
                message,
            } => (status, code, message),
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
    param: Option<String>,
    code: &'static str,
}

impl IntoResponse for SidecarError {
    fn into_response(self) -> Response {
        let error = self.to_string();
        let (status, code, message) = self.into_response_fields();
        tracing::warn!(%error, %status, code, "Sidecar request failed");
        (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    message,
                    r#type: "server_error",
                    param: None,
                    code,
                },
            }),
        )
            .into_response()
    }
}
