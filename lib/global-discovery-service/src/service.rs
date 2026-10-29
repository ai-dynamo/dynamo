// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Discovery Service's HTTP API. Its JSON types are in
//! `dynamo_global_common`, shared with the service's clients.
//!
//! - `POST /pools`: a PoolRelay registers its pool. It calls this on a timer,
//!   and each call is also the pool's heartbeat.
//! - `GET /routers`: the Global Router replicas a PoolRelay connects to.
//! - `GET /pools`: the pools a Global Router routes to.
//! - `GET /registrations`: the registrations this instance's PoolRelays
//!   made, for the other instances ([`crate::peers`]).
//! - `GET /healthz`: OK while the service can answer from its store. Use it
//!   for readiness; liveness needs only the TCP port. With mTLS, a probe
//!   needs a client certificate too, so probe with TCP or a command.
//!
//! The service reads and writes only through the [`crate::pool_discovery`]
//! traits. It runs on `dynamo-http-server`, so it gets TLS and mTLS the way
//! the Dynamo frontend does.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, DiscoveryError, PoolDirectory, PoolRegistrar, PoolRegistration,
    RegistrationExchange, ReplicaDirectory,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use dynamo_global_common::{PoolsResponse, RoutersResponse};
use dynamo_http_server::lifecycle::ServiceObserver;
use dynamo_http_server::server::{HttpServer, HttpServerConfig};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// What the service reads and writes.
pub struct DiscoveryStore {
    pub pools: Arc<PoolDirectory>,
    pub routers: Arc<ReplicaDirectory>,
    /// `None` for a store that takes no registrations.
    pub registrar: Option<Arc<dyn PoolRegistrar>>,
    /// `Some` to serve `GET /registrations` to the other instances.
    pub sharing: Option<Sharing>,
}

/// The registrations this instance shares, from
/// [`crate::peers::PeerSync::new`].
pub struct Sharing {
    /// Names this process in `GET /registrations`, so it can skip its own
    /// reply.
    pub(crate) instance: String,
    pub(crate) registrations: Arc<dyn RegistrationExchange>,
    /// Set once this instance has asked every other one. Until then
    /// `GET /healthz` fails, so a new replica does not answer before it has
    /// heard from the others.
    pub(crate) synced: Arc<AtomicBool>,
}

impl fmt::Debug for DiscoveryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscoveryStore")
            .field("takes_registrations", &self.registrar.is_some())
            .field("sharing", &self.sharing)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Sharing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sharing")
            .field("instance", &self.instance)
            .field("synced", &self.synced.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// TLS for the listener. With `client_ca`, callers must also present a
/// certificate that it signed (mTLS).
#[derive(Clone, Debug)]
pub struct Tls {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub client_ca: Option<PathBuf>,
}

/// The service's HTTP routes.
pub fn routes(store: DiscoveryStore) -> Router {
    Router::new()
        .route("/pools", get(list_pools).post(register_pool))
        .route("/routers", get(list_routers))
        .route("/registrations", get(shared_registrations))
        .route("/healthz", get(healthz))
        .with_state(Arc::new(store))
}

/// Serves until `shutdown` is cancelled. Then it stops taking connections and
/// gives requests in flight up to `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`
/// (5 s by default) to finish.
pub async fn serve(
    listener: TcpListener,
    store: DiscoveryStore,
    tls: Option<Tls>,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let address = listener.local_addr()?;
    let config = HttpServerConfig {
        host: address.ip().to_string(),
        port: address.port(),
        enable_tls: tls.is_some(),
        tls_cert_path: tls.as_ref().map(|tls| tls.cert.clone()),
        tls_key_path: tls.as_ref().map(|tls| tls.key.clone()),
        tls_client_ca_cert_path: tls.and_then(|tls| tls.client_ca),
    };
    // The observer counts inference requests, and this service has none, so
    // shutdown goes straight to letting open requests finish.
    let observer = Arc::new(ServiceObserver::default());
    HttpServer::prepare(routes(store), config, Some(listener))
        .await?
        .serve(observer, CancellationToken::new(), shutdown)
        .await
}

async fn register_pool(
    State(store): State<Arc<DiscoveryStore>>,
    Json(registration): Json<PoolRegistration>,
) -> Response {
    let Some(registrar) = &store.registrar else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this store takes no registrations",
        )
            .into_response();
    };
    match registrar.register(registration).await {
        Ok(lease) => Json(lease).into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_pools(State(store): State<Arc<DiscoveryStore>>) -> Response {
    match store.pools.list().await {
        Ok(snapshot) => Json(PoolsResponse {
            complete: snapshot.completeness == Completeness::Exact,
            pools: snapshot.items,
        })
        .into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_routers(State(store): State<Arc<DiscoveryStore>>) -> Response {
    match store.routers.list().await {
        Ok(snapshot) => Json(RoutersResponse {
            routers: snapshot.items,
        })
        .into_response(),
        Err(error) => error_response(error),
    }
}

async fn shared_registrations(State(store): State<Arc<DiscoveryStore>>) -> Response {
    let Some(sharing) = &store.sharing else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this instance does not share registrations",
        )
            .into_response();
    };
    Json(SharedRegistrations {
        instance: sharing.instance.clone(),
        registrations: sharing
            .registrations
            .local_registrations()
            .into_iter()
            .map(|(registration, remaining)| SharedRegistration {
                registration,
                expires_in_ms: u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX),
            })
            .collect(),
    })
    .into_response()
}

/// Lists what `GET /pools` and `GET /routers` would serve. A Kubernetes store
/// answers from its cache once that has synced, so an API server outage does
/// not turn every replica unready at once. The first probe also starts the
/// store's watch. An instance that shares registrations is ready only once
/// it has asked the others; an unreachable one later does not make it
/// unready.
async fn healthz(State(store): State<Arc<DiscoveryStore>>) -> StatusCode {
    if let Some(sharing) = &store.sharing
        && !sharing.synced.load(Ordering::Acquire)
    {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    let (pools, routers) = tokio::join!(store.pools.list(), store.routers.list());
    match pools.and(routers) {
        Ok(_) => StatusCode::OK,
        Err(error) => {
            tracing::warn!(?error, "discovery store is unreachable");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

/// Maps a store error to a status. The cause is logged, not returned.
fn error_response(error: DiscoveryError) -> Response {
    let status = match &error {
        DiscoveryError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    tracing::warn!(?error, %status, "discovery request failed");
    (status, error.to_string()).into_response()
}

/// `GET /registrations` reply. Only service instances read it, so it is not
/// in `dynamo_global_common`.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct SharedRegistrations {
    /// The instance that answered.
    pub(crate) instance: String,
    pub(crate) registrations: Vec<SharedRegistration>,
}

/// A registration as its PoolRelay sent it, and how long it has left.
/// Durations, not times, so the instances' clocks need not agree.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct SharedRegistration {
    registration: PoolRegistration,
    expires_in_ms: u64,
}

impl SharedRegistration {
    pub(crate) fn into_registration(self) -> (PoolRegistration, Duration) {
        (self.registration, Duration::from_millis(self.expires_in_ms))
    }
}

#[cfg(test)]
mod tests {
    use crate::pool_discovery::{Directory, RouterReplica, RouterReplicaId, Snapshot, WatchStream};
    use async_trait::async_trait;
    use dynamo_global_common::Lease;
    use serde_json::{Value, json};
    use url::Url;

    use super::*;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};

    /// The body `dynamo-global-discovery-client` sends.
    fn registration() -> Value {
        json!({
            "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {
                "region": "us-west-2",
                "availability_zone": null,
                "datacenter": null
            },
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"],
            "capacity": [{"sku": "H100", "workers": 8}]
        })
    }

    fn router_replica() -> RouterReplica {
        RouterReplica {
            replica: RouterReplicaId("aws-ohio/global-router-0".to_owned()),
            url: Url::parse("http://10.42.0.5:50051").unwrap(),
        }
    }

    fn memory_store() -> DiscoveryStore {
        let pools = Arc::new(MemoryPoolStore::new(Lease::new(5, 15).unwrap()));
        DiscoveryStore {
            pools: pools.clone(),
            routers: Arc::new(StaticReplicaDirectory::new(vec![router_replica()])),
            registrar: Some(pools),
            sharing: None,
        }
    }

    /// Serves `store` on a free local port and returns its base URL.
    async fn start(store: DiscoveryStore) -> (String, CancellationToken) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let shutdown = CancellationToken::new();
        tokio::spawn(serve(listener, store, None, shutdown.clone()));
        (base, shutdown)
    }

    async fn get_json(base: &str, path: &str) -> (reqwest::StatusCode, Value) {
        let response = reqwest::get(format!("{base}{path}")).await.unwrap();
        let status = response.status();
        let body = response.bytes().await.unwrap();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    async fn post_pool(base: &str, body: &Value) -> (reqwest::StatusCode, Value) {
        let response = reqwest::Client::new()
            .post(format!("{base}/pools"))
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.bytes().await.unwrap();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn register_then_list_pools() {
        let (base, shutdown) = start(memory_store()).await;
        let (status, lease) = post_pool(&base, &registration()).await;
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(
            lease,
            json!({"heartbeat_interval_s": 5, "lease_duration_s": 15})
        );

        let (status, pools) = get_json(&base, "/pools").await;
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(
            pools,
            json!({
                "pools": [{
                    "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
                    "location": {
                        "region": "us-west-2",
                        "availability_zone": null,
                        "datacenter": null
                    },
                    "model": "Qwen/Qwen3-0.6B",
                    "capabilities": ["aggregated"],
                    "frontends": ["http://10.43.1.2:8000/"],
                    "capacity": [{"sku": "H100", "workers": 8}],
                    "relay_identities": []
                }],
                "complete": false
            })
        );
        shutdown.cancel();
    }

    #[tokio::test]
    async fn routers_use_the_client_shape() {
        let (base, shutdown) = start(memory_store()).await;
        let (status, routers) = get_json(&base, "/routers").await;
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(
            routers,
            json!({"routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/"}]})
        );
        shutdown.cancel();
    }

    #[tokio::test]
    async fn invalid_key_is_rejected() {
        let (base, shutdown) = start(memory_store()).await;
        let mut body = registration();
        body["key"]["cluster_id"] = json!(" ");
        let (status, _) = post_pool(&base, &body).await;
        assert_eq!(status, reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        let (_, pools) = get_json(&base, "/pools").await;
        assert_eq!(pools["pools"], json!([]));
        shutdown.cancel();
    }

    #[tokio::test]
    async fn store_without_registrar_refuses_registrations() {
        let mut store = memory_store();
        store.registrar = None;
        let (base, shutdown) = start(store).await;
        let (status, _) = post_pool(&base, &registration()).await;
        assert_eq!(status, reqwest::StatusCode::NOT_IMPLEMENTED);
        shutdown.cancel();
    }

    /// A store whose backend is down.
    struct Down;

    #[async_trait]
    impl<T: Send + 'static, K: Send + Sync + 'static> Directory<T, K> for Down {
        async fn list(&self) -> Result<Snapshot<T>, DiscoveryError> {
            Err(DiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<T, K>, DiscoveryError> {
            Err(DiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }
    }

    #[tokio::test]
    async fn unreachable_store_is_unavailable() {
        let (base, shutdown) = start(DiscoveryStore {
            pools: Arc::new(Down),
            routers: Arc::new(Down),
            registrar: None,
            sharing: None,
        })
        .await;
        for path in ["/pools", "/routers", "/healthz"] {
            let response = reqwest::get(format!("{base}{path}")).await.unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                "{path}"
            );
            // The cause stays in the log.
            assert!(!response.text().await.unwrap().contains("down"), "{path}");
        }
        shutdown.cancel();
    }

    #[tokio::test]
    async fn stops_when_shut_down() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let healthz = format!("http://{}/healthz", listener.local_addr().unwrap());
        let shutdown = CancellationToken::new();
        let server = tokio::spawn(serve(listener, memory_store(), None, shutdown.clone()));
        let response = reqwest::get(&healthz).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        shutdown.cancel();
        server.await.unwrap().unwrap();
        assert!(reqwest::get(&healthz).await.is_err());
    }
}
