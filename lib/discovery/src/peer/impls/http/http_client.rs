// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client backend for peer discovery.
//!
//! This backend communicates with a discovery service over HTTP/REST API.

use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::{
    DiscoveryError, InstanceId, PeerDiscovery, PeerInfo, WorkerAddress, WorkerId,
    http::{
        ErrorResponse, PeerInfoResponse, PeerListResponse, RegisterRequest, RegisterResponse,
        endpoints,
    },
};

/// HTTP client for discovery service.
///
/// Communicates with the discovery service REST API to register and lookup peers.
///
/// # Example
///
/// ```no_run
/// use dynamo_am_discovery::http_client::{HttpClientConfig, HttpClientDiscovery};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HttpClientConfig::builder()
///     .base_url("http://discovery-service:8080")
///     .build()?;
///
/// let discovery = config.build().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HttpClientDiscovery {
    client: reqwest::Client,
    base_url: String,
}

impl HttpClientDiscovery {
    /// Create a new HTTP client discovery instance.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client, base_url })
    }

    /// Build the full URL for an endpoint.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Handle HTTP error responses.
    async fn handle_error_response(&self, response: reqwest::Response) -> anyhow::Error {
        let status = response.status();
        if let Ok(error_body) = response.json::<ErrorResponse>().await {
            anyhow::anyhow!(
                "HTTP {}: {} - {:?}",
                status,
                error_body.error,
                error_body.details
            )
        } else {
            anyhow::anyhow!("HTTP {}: Request failed", status)
        }
    }
}

#[async_trait]
impl PeerDiscovery for HttpClientDiscovery {
    async fn discover_by_worker_id(&self, worker_id: WorkerId) -> Result<PeerInfo> {
        let url = format!("{}/{}", self.url(endpoints::DISCOVER_WORKER), worker_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send discover_by_worker_id request")?;

        if !response.status().is_success() {
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Err(DiscoveryError::NotFound(format!("worker_id {}", worker_id)).into());
            }
            return Err(self.handle_error_response(response).await);
        }

        let peer_response: PeerInfoResponse = response
            .json()
            .await
            .context("Failed to deserialize PeerInfoResponse")?;

        Ok(peer_response.peer_info)
    }

    async fn discover_by_instance_id(&self, instance_id: InstanceId) -> Result<PeerInfo> {
        let url = format!("{}/{}", self.url(endpoints::DISCOVER_INSTANCE), instance_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send discover_by_instance_id request")?;

        if !response.status().is_success() {
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Err(
                    DiscoveryError::NotFound(format!("instance_id {}", instance_id)).into(),
                );
            }
            return Err(self.handle_error_response(response).await);
        }

        let peer_response: PeerInfoResponse = response
            .json()
            .await
            .context("Failed to deserialize PeerInfoResponse")?;

        Ok(peer_response.peer_info)
    }

    async fn register(&self, instance_id: InstanceId, worker_address: WorkerAddress) -> Result<()> {
        let url = self.url(endpoints::REGISTER);
        let request = RegisterRequest {
            instance_id,
            worker_address,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send register request")?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let register_response: RegisterResponse = response
            .json()
            .await
            .context("Failed to deserialize RegisterResponse")?;

        if !register_response.success {
            if let Some(error) = register_response.error {
                return Err(anyhow::anyhow!("Registration failed: {}", error));
            }
            return Err(anyhow::anyhow!("Registration failed with no error message"));
        }

        Ok(())
    }

    async fn unregister(&self, instance_id: InstanceId) -> Result<()> {
        let url = format!("{}/{}", self.url(endpoints::UNREGISTER), instance_id);

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .context("Failed to send unregister request")?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        Ok(())
    }

    async fn discover_all(&self) -> Result<Vec<PeerInfo>> {
        let url = self.url(endpoints::DISCOVER_ALL);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send discover_all request")?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let peer_list: PeerListResponse = response
            .json()
            .await
            .context("Failed to deserialize PeerListResponse")?;

        Ok(peer_list.peers)
    }
}

/// Configuration for HTTP client discovery.
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Base URL of the discovery service (e.g., "http://discovery-service:8080")
    pub base_url: String,
}

impl HttpClientConfig {
    /// Create a new builder for HttpClientConfig.
    pub fn builder() -> HttpClientConfigBuilder {
        HttpClientConfigBuilder::default()
    }

    /// Build an HTTP client discovery instance from this configuration.
    pub async fn build(self) -> Result<std::sync::Arc<dyn PeerDiscovery>> {
        let discovery = HttpClientDiscovery::new(self.base_url)?;
        Ok(std::sync::Arc::new(discovery))
    }
}

/// Builder for HttpClientConfig.
#[derive(Debug, Clone, Default)]
pub struct HttpClientConfigBuilder {
    base_url: Option<String>,
}

impl HttpClientConfigBuilder {
    /// Set the base URL of the discovery service (required).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the HttpClientConfig.
    ///
    /// # Errors
    ///
    /// Returns an error if the required `base_url` field was not set.
    pub fn build(self) -> Result<HttpClientConfig> {
        let base_url = self
            .base_url
            .ok_or_else(|| anyhow::anyhow!("base_url is required for HttpClientConfig"))?;

        Ok(HttpClientConfig { base_url })
    }
}
