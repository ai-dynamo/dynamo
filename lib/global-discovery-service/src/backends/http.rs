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
        let response = self
            .client
            .get(url)
            .query(&RegistrationsRequest::from(asker))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(unavailable)?;
        // A redirect is not followed, so it is no answer.
        if !response.status().is_success() {
            return Err(DiscoveryError::Unavailable(anyhow::anyhow!(
                "{} answered {} instead of its registrations",
                response.url(),
                response.status()
            )));
        }
        response.json().await.map_err(unavailable)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::response::Redirect;
    use axum::routing::get;
    use tokio::net::TcpListener;
    use url::Url;
    use uuid::Uuid;

    use super::*;
    use crate::pool_discovery::RouterReplicaId;

    /// Serves `routes` on a free port, and returns its URL.
    async fn serve(routes: Router) -> Url {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        tokio::spawn(async move { axum::serve(listener, routes).await });
        url
    }

    #[tokio::test]
    async fn a_redirect_is_an_error_and_is_not_followed() {
        let reached = Arc::new(AtomicUsize::new(0));
        let counter = reached.clone();
        let target = serve(Router::new().route(
            "/registrations",
            get(move || async move {
                counter.fetch_add(1, Ordering::SeqCst);
                "{}"
            }),
        ))
        .await;
        let location = target.join("registrations").unwrap().to_string();
        let redirecting = serve(Router::new().route(
            "/registrations",
            get(move || async move { Redirect::temporary(&location) }),
        ))
        .await;
        let instance = RouterReplica {
            replica: RouterReplicaId("redirecting".to_owned()),
            url: redirecting,
        };
        let asker = Asker {
            instance: Uuid::new_v4(),
            holding: true,
        };

        let error = HttpRegistrationSource::new()
            .unwrap()
            .shared_registrations(&instance, asker)
            .await
            .unwrap_err();
        let DiscoveryError::Unavailable(cause) = error else {
            panic!("{error:?}");
        };
        assert!(format!("{cause:#}").contains("307"), "{cause:#}");
        assert_eq!(reached.load(Ordering::SeqCst), 0);
    }
}
