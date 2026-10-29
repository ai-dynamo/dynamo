// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Discovery Service's HTTP API. Its JSON types are in
//! `dynamo_global_common`, shared with the service's clients.
//!
//! - `POST /pools`: a PoolRelay registers its pool. It calls this on a timer,
//!   and each call is also the pool's heartbeat.
//! - `GET /routers`: the Global Router replicas a PoolRelay connects to.
//! - `GET /pools`: the pools a Global Router routes to.
//! - `GET /healthz`: OK while the service can answer from its backend. Use it
//!   for readiness; liveness needs only the TCP port. With mTLS, a probe
//!   needs a client certificate too, so probe with TCP or a command.
//!
//! The service reads and writes only through the [`crate::pool_discovery`]
//! traits. It runs on `dynamo-http-server`, so it gets TLS and mTLS the way
//! the Dynamo frontend does.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::pool_discovery::{
    Completeness, DiscoveryError, PoolDirectory, PoolRegistrar, PoolRegistration, ReplicaDirectory,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use dynamo_global_common::{PoolsResponse, RoutersResponse};
use dynamo_http_server::lifecycle::ServiceObserver;
use dynamo_http_server::server::{HttpServer, HttpServerConfig};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// The backend behind the API: where the pool and router lists come from,
/// and where registrations go.
pub struct DiscoveryBackend {
    pub pools: Arc<PoolDirectory>,
    pub routers: Arc<ReplicaDirectory>,
    /// `None` when the service takes no registrations, as with pools from
    /// MCS.
    pub registrar: Option<Arc<dyn PoolRegistrar>>,
}

impl fmt::Debug for DiscoveryBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscoveryBackend")
            .field("takes_registrations", &self.registrar.is_some())
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
pub fn routes(backend: DiscoveryBackend) -> Router {
    Router::new()
        .route("/pools", get(list_pools).post(register_pool))
        .route("/routers", get(list_routers))
        .route("/healthz", get(healthz))
        .with_state(Arc::new(backend))
}

/// Serves until `shutdown` is cancelled. Then it stops taking connections and
/// gives requests in flight up to `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`
/// (5 s by default) to finish.
pub async fn serve(
    listener: TcpListener,
    backend: DiscoveryBackend,
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
    HttpServer::prepare(routes(backend), config, Some(listener))
        .await?
        .serve(observer, CancellationToken::new(), shutdown)
        .await
}

async fn register_pool(
    State(backend): State<Arc<DiscoveryBackend>>,
    Json(registration): Json<PoolRegistration>,
) -> Response {
    let Some(registrar) = &backend.registrar else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this service takes no registrations: its pools come from MCS",
        )
            .into_response();
    };
    match registrar.register(registration).await {
        Ok(lease) => Json(lease).into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_pools(State(backend): State<Arc<DiscoveryBackend>>) -> Response {
    match backend.pools.list().await {
        Ok(snapshot) => Json(PoolsResponse {
            complete: snapshot.completeness == Completeness::Exact,
            pools: snapshot.items,
        })
        .into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_routers(State(backend): State<Arc<DiscoveryBackend>>) -> Response {
    match backend.routers.list().await {
        Ok(snapshot) => Json(RoutersResponse {
            routers: snapshot.items,
        })
        .into_response(),
        Err(error) => error_response(error),
    }
}

/// Lists what `GET /pools` and `GET /routers` would serve. With pools from
/// MCS, the backend answers from its cache once that has synced, so an API
/// server outage does not turn every replica unready at once. The first probe
/// also starts its watch.
async fn healthz(State(backend): State<Arc<DiscoveryBackend>>) -> StatusCode {
    let (pools, routers) = tokio::join!(backend.pools.list(), backend.routers.list());
    match pools.and(routers) {
        Ok(_) => StatusCode::OK,
        Err(error) => {
            tracing::warn!(?error, "discovery backend is unreachable");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

/// Maps a backend error to a status. The cause is logged, not returned.
fn error_response(error: DiscoveryError) -> Response {
    let status = match &error {
        DiscoveryError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    tracing::warn!(?error, %status, "discovery request failed");
    (status, error.to_string()).into_response()
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

    fn registrations_backend() -> DiscoveryBackend {
        let pools = Arc::new(MemoryPoolStore::new(Lease::new(5, 15).unwrap()));
        DiscoveryBackend {
            pools: pools.clone(),
            routers: Arc::new(StaticReplicaDirectory::new(vec![router_replica()])),
            registrar: Some(pools),
        }
    }

    /// Serves `backend` on a free local port and returns its base URL.
    async fn start(backend: DiscoveryBackend) -> (String, CancellationToken) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let shutdown = CancellationToken::new();
        tokio::spawn(serve(listener, backend, None, shutdown.clone()));
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
        let (base, shutdown) = start(registrations_backend()).await;
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
        let (base, shutdown) = start(registrations_backend()).await;
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
        let (base, shutdown) = start(registrations_backend()).await;
        let mut body = registration();
        body["key"]["cluster_id"] = json!(" ");
        let (status, _) = post_pool(&base, &body).await;
        assert_eq!(status, reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        let (_, pools) = get_json(&base, "/pools").await;
        assert_eq!(pools["pools"], json!([]));
        shutdown.cancel();
    }

    #[tokio::test]
    async fn backend_without_registrar_refuses_registrations() {
        let mut backend = registrations_backend();
        backend.registrar = None;
        let (base, shutdown) = start(backend).await;
        let (status, _) = post_pool(&base, &registration()).await;
        assert_eq!(status, reqwest::StatusCode::NOT_IMPLEMENTED);
        shutdown.cancel();
    }

    /// A directory that is down.
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
    async fn unreachable_backend_is_unavailable() {
        let (base, shutdown) = start(DiscoveryBackend {
            pools: Arc::new(Down),
            routers: Arc::new(Down),
            registrar: None,
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
        let server = tokio::spawn(serve(
            listener,
            registrations_backend(),
            None,
            shutdown.clone(),
        ));
        let response = reqwest::get(&healthz).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        shutdown.cancel();
        server.await.unwrap().unwrap();
        assert!(reqwest::get(&healthz).await.is_err());
    }
}
