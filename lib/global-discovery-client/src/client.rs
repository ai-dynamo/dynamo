// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_global_common::{Lease, PoolRegistration, RouterReplica, RoutersResponse};
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use thiserror::Error;
use url::Url;

/// The most of an error response's body a [`DiscoveryClientError::Status`]
/// keeps.
const MAX_ERROR_BODY: usize = 1024;

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
    /// `body` keeps at most the first 1 KiB of the response, so a proxy's
    /// error page does not fill logs or health reports.
    #[error("{url} returned {status}: {body}")]
    Status {
        url: Url,
        status: StatusCode,
        body: String,
    },
    /// The body is not in the shape this client expects, for example a
    /// renamed field, a router URL that does not parse, or an invalid
    /// [`Lease`].
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
    pub async fn routers(&self) -> Result<Vec<RouterReplica>, DiscoveryClientError> {
        let url = &self.routers_url;
        let response = self.send(self.http.get(url.clone()), url).await?;
        let body: RoutersResponse = read_json(response, url).await?;
        Ok(body.routers)
    }

    /// Calls `POST /pools` to register a pool or refresh its registration,
    /// and returns the service's [`Lease`]. Each call is also the pool's
    /// heartbeat, so the PoolRelay calls this again after
    /// `Lease::heartbeat_interval`.
    pub async fn register_pool(
        &self,
        registration: &PoolRegistration,
    ) -> Result<Lease, DiscoveryClientError> {
        let url = &self.pools_url;
        let response = self
            .send(self.http.post(url.clone()).json(registration), url)
            .await?;
        read_json(response, url).await
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
        let mut body = response.text().await.unwrap_or_default();
        body.truncate(body.floor_char_boundary(MAX_ERROR_BODY));
        Err(DiscoveryClientError::Status {
            url: url.clone(),
            status,
            body,
        })
    }
}

/// A body cut off mid-transfer is a `Request` error. Only JSON that does not
/// match `T` is a `Decode` error.
async fn read_json<T: DeserializeOwned>(
    response: reqwest::Response,
    url: &Url,
) -> Result<T, DiscoveryClientError> {
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
