// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The other service instances, read over HTTP: `GET /registrations`.

use async_trait::async_trait;

use crate::pool_discovery::{
    Asker, DiscoveryError, RegistrationSource, RegistrationsRequest, RouterReplica,
    SharedRegistrations,
};

/// Asks other instances over plain HTTP, never through a proxy, and without
/// following redirects. The caller bounds how long it waits.
#[derive(Clone, Debug)]
pub struct HttpRegistrationSource {
    client: reqwest::Client,
}

impl HttpRegistrationSource {
    pub fn new() -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?;
        Ok(Self { client })
    }
}

#[async_trait]
impl RegistrationSource for HttpRegistrationSource {
    async fn shared_registrations(
        &self,
        instance: &RouterReplica,
        asker: Asker,
    ) -> Result<SharedRegistrations, DiscoveryError> {
        let unavailable = |error: reqwest::Error| DiscoveryError::Unavailable(error.into());
        let url = instance
            .url
            .join("registrations")
            .map_err(|error| DiscoveryError::Unavailable(error.into()))?;
        self.client
            .get(url)
            .query(&RegistrationsRequest::from(asker))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(unavailable)?
            .json()
            .await
            .map_err(unavailable)
    }
}
