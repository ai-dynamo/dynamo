// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{Encoder, TextEncoder};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing;

/// Start HTTP server, bind DRT's child token to shut down
pub async fn start_http_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler));

    let address = format!("{}:{}", host, port);
    tracing::debug!("Starting HTTP server on: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::debug!("HTTP server bound to: {}", actual_address);
            listener
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };

    let observer = cancel_token.child_token();
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(observer.cancelled_owned())
        .await
    {
        tracing::error!("HTTP server error: {}", e);
    }
    Ok(())
}

/// Health handler
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Metrics handler, using prometheus to collect metrics
async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();

    match encoder.encode(&prometheus::gather(), &mut buffer) {
        Ok(()) => match String::from_utf8(buffer) {
            Ok(response) => (StatusCode::OK, response),
            Err(e) => {
                tracing::error!("Failed to encode metrics as UTF-8: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to encode metrics as UTF-8".to_string(),
                )
            }
        },
        Err(e) => {
            tracing::error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let _ = start_http_server("127.0.0.1", 0, cancel_token_for_server).await;
        });

        // wait for a while to let the server start
        sleep(Duration::from_millis(100)).await;

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        let response = response.into_response();
        let (parts, _body) = response.into_parts();

        assert_eq!(parts.status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_handler() {
        let response = metrics_handler().await;
        let response = response.into_response();
        let (parts, _body) = response.into_parts();

        assert_eq!(parts.status, StatusCode::OK);
    }
}
