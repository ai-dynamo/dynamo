// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use thiserror::Error;
use url::Url;

use crate::replicas::ReplicaEndpoint;
use crate::watch::{Completeness, Snapshot};
use crate::wire::{PoolListing, PoolRegistration, PoolsResponse, RoutersResponse};

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiscoveryClientError {
    /// The base URL cannot have a path appended, for example `mailto:`.
    #[error("{0} cannot be a base URL")]
    InvalidBaseUrl(Url),
    #[error("building the HTTP client failed")]
    Build(#[source] reqwest::Error),
    #[error("request to {url} failed")]
    Request {
        url: Url,
        #[source]
        source: reqwest::Error,
    },
    #[error("{url} returned {status}: {body}")]
    Status {
        url: Url,
        status: StatusCode,
        body: String,
    },
    /// The body is not in the shape this client expects, for example a
    /// renamed field or a router URL that does not parse.
    #[error("{url} returned a body this client cannot parse")]
    Decode {
        url: Url,
        #[source]
        source: serde_json::Error,
    },
}

/// Cloning is cheap: clones share one connection pool, so give each task its
/// own clone instead of wrapping the client in an `Arc`.
#[derive(Clone, Debug)]
pub struct GlobalDiscoveryClient {
    http: reqwest::Client,
    routers_url: Url,
    pools_url: Url,
}

impl GlobalDiscoveryClient {
    /// `base_url` is the root of the service, for example
    /// `http://global-discovery:8080/`. The client keeps any path in it as a
    /// prefix. `request_timeout` bounds each call.
    pub fn new(base_url: Url, request_timeout: Duration) -> Result<Self, DiscoveryClientError> {
        let http = reqwest::Client::builder()
            .timeout(request_timeout)
            // A followed redirect can turn the registration POST into a GET,
            // so a 3xx must fail instead of looking like a success.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(DiscoveryClientError::Build)?;
        Ok(Self {
            http,
            routers_url: endpoint(&base_url, "routers")?,
            pools_url: endpoint(&base_url, "pools")?,
        })
    }

    /// Calls `GET /routers` and returns every Global Router replica the
    /// service lists.
    ///
    /// Each call returns the full current set. If any entry does not parse,
    /// the whole call fails, so a caller never acts on a partial list.
    pub async fn routers(&self) -> Result<Vec<ReplicaEndpoint>, DiscoveryClientError> {
        let body: RoutersResponse = self.get_json(&self.routers_url).await?;
        Ok(body.routers)
    }

    /// Calls `GET /pools` and returns every pool the service lists. A Global
    /// Router calls this to learn the pools and their frontends.
    ///
    /// Like [`Self::routers`], one entry that does not parse fails the whole
    /// call.
    pub async fn pools(&self) -> Result<Snapshot<PoolListing>, DiscoveryClientError> {
        let body: PoolsResponse = self.get_json(&self.pools_url).await?;
        Ok(Snapshot {
            items: body.pools,
            completeness: if body.complete {
                Completeness::Exact
            } else {
                Completeness::Estimated
            },
        })
    }

    async fn get_json<T: DeserializeOwned>(&self, url: &Url) -> Result<T, DiscoveryClientError> {
        let response = self.send(self.http.get(url.clone()), url).await?;
        let body = response
            .bytes()
            .await
            .map_err(|source| DiscoveryClientError::Request {
                url: url.clone(),
                source,
            })?;
        serde_json::from_slice(&body).map_err(|source| DiscoveryClientError::Decode {
            url: url.clone(),
            source,
        })
    }

    /// Calls `POST /pools` to register a pool or refresh its registration.
    /// The PoolRelay calls this on a schedule, and each call is also the
    /// pool's heartbeat.
    pub async fn register_pool(
        &self,
        registration: &PoolRegistration,
    ) -> Result<(), DiscoveryClientError> {
        let url = &self.pools_url;
        self.send(self.http.post(url.clone()).json(registration), url)
            .await?;
        Ok(())
    }

    async fn send(
        &self,
        request: reqwest::RequestBuilder,
        url: &Url,
    ) -> Result<reqwest::Response, DiscoveryClientError> {
        let response = request
            .send()
            .await
            .map_err(|source| DiscoveryClientError::Request {
                url: url.clone(),
                source,
            })?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        Err(DiscoveryClientError::Status {
            url: url.clone(),
            status,
            body: response.text().await.unwrap_or_default(),
        })
    }
}

/// `base_url` with `segment` appended, keeping any path it already has.
fn endpoint(base_url: &Url, segment: &str) -> Result<Url, DiscoveryClientError> {
    let mut url = base_url.clone();
    url.path_segments_mut()
        .map_err(|()| DiscoveryClientError::InvalidBaseUrl(base_url.clone()))?
        .pop_if_empty()
        .push(segment);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_keeps_the_base_path() {
        for (base, expected) in [
            ("http://hub:8080", "http://hub:8080/pools"),
            (
                "http://hub:8080/discovery",
                "http://hub:8080/discovery/pools",
            ),
            (
                "http://hub:8080/discovery/",
                "http://hub:8080/discovery/pools",
            ),
        ] {
            let url = endpoint(&base.parse().unwrap(), "pools").unwrap();
            assert_eq!(url.as_str(), expected, "{base}");
        }
        let error = endpoint(&"mailto:ops@example.com".parse().unwrap(), "pools").unwrap_err();
        assert!(matches!(error, DiscoveryClientError::InvalidBaseUrl(_)));
    }
}
