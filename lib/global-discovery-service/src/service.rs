// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service: pool discovery for the Global Router, over HTTP.
//!
//! - `POST /pools`: a PoolRelay registers its pool. It calls this on a timer,
//!   and each call is also the pool's heartbeat. The MCS store does not use
//!   the body's `frontends`: MCS gives addresses other clusters can reach.
//! - `GET /routers`: the Global Router replicas a PoolRelay connects to.
//! - `GET /pools`: the pools a Global Router routes to.
//! - `GET /registrations`: the registrations this instance's PoolRelays
//!   made, for the other instances ([`crate::peers`]).
//! - `GET /healthz`: OK while the service can answer from its store. Use it
//!   for readiness; liveness needs only the TCP port.
//!
//! The service reads and writes through the `crate::pool_discovery`
//! traits, so the store can change without changing PoolRelays or routers.
//! [`MemoryPoolStore`] keeps registrations in this process. The MCS store
//! ([`RegisteredPoolDirectory`](crate::backends::registered::RegisteredPoolDirectory))
//! lists a pool while MCS imports a Ready PoolRelay endpoint for it and the
//! pool has a live registration, and reads router replicas from the
//! EndpointSlices that MCS imports
//! ([`kubernetes`](crate::backends::kubernetes)). PoolRelays call
//! `POST /pools` the same way with either store.
//!
//! The JSON matches the `dynamo-global-discovery-client` crate.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::backends::memory::MemoryPoolStore;
use crate::pool_discovery::{
    Capacity, Completeness, LeasePolicy, PeerCredentials, PoolDirectory, PoolDiscoveryError,
    PoolRecord, PoolRegistrar, RelayAuthenticator, ReplicaDirectory, ReplicaEndpoint, Revision,
    verify,
};
use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use dynamo_global_router_shared::PoolLocation;
use dynamo_global_router_shared::{PoolKey, PoolKeyError};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

/// What the service reads and writes.
pub struct DiscoveryStore {
    pub pools: Arc<PoolDirectory>,
    pub routers: Arc<ReplicaDirectory>,
    /// `None` for a store that takes no registrations.
    pub registrar: Option<Arc<dyn PoolRegistrar>>,
    /// Names the caller of `POST /pools` from its connection.
    pub authenticator: Arc<dyn RelayAuthenticator>,
    /// `Some` to serve `GET /registrations` to the other instances.
    pub sharing: Option<Sharing>,
}

/// The registrations this instance shares (see [`crate::peers`]).
pub struct Sharing {
    /// Names this process in `GET /registrations`, so it can skip its own
    /// reply.
    pub instance: String,
    pub registrations: Arc<MemoryPoolStore>,
    /// Set once this instance has asked every other one; until then
    /// `GET /healthz` fails, so a new replica does not serve a partial pool
    /// list. See [`crate::peers::PeerSync::synced`].
    pub synced: Arc<AtomicBool>,
}

pub fn router(store: DiscoveryStore) -> Router {
    Router::new()
        .route("/pools", get(list_pools).post(register_pool))
        .route("/routers", get(list_routers))
        .route("/registrations", get(shared_registrations))
        .route("/healthz", get(healthz))
        .with_state(Arc::new(store))
}

/// Serves until `shutdown` is cancelled.
pub async fn serve(
    listener: TcpListener,
    store: DiscoveryStore,
    shutdown: CancellationToken,
) -> std::io::Result<()> {
    axum::serve(
        listener,
        router(store).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown.cancelled_owned())
    .await
}

async fn register_pool(
    State(store): State<Arc<DiscoveryStore>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(registration): Json<PoolRegistration>,
) -> Response {
    let Some(registrar) = &store.registrar else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this store takes no registrations",
        )
            .into_response();
    };
    let caller = match verify(
        &*store.authenticator,
        &PeerCredentials::PeerAddress(peer.ip()),
    )
    .await
    {
        Ok(caller) => caller,
        Err(error) => return error_response(error),
    };
    let record = match registration.into_record() {
        Ok(record) => record,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    match registrar.register(record, &caller).await {
        Ok(registration) => Json(LeaseView::from(registration.lease)).into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_pools(State(store): State<Arc<DiscoveryStore>>) -> Response {
    match store.pools.list().await {
        Ok(snapshot) => Json(PoolsResponse {
            complete: snapshot.completeness == Completeness::Exact,
            pools: snapshot.items.into_iter().map(PoolView::from).collect(),
        })
        .into_response(),
        Err(error) => error_response(error),
    }
}

async fn list_routers(State(store): State<Arc<DiscoveryStore>>) -> Response {
    match store.routers.list().await {
        Ok(snapshot) => Json(RoutersResponse {
            routers: snapshot.items.into_iter().map(RouterView::from).collect(),
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
            .filter_map(|(record, remaining)| {
                Some(SharedRegistration {
                    registration: PoolRegistration::from_record(record)?,
                    expires_in_ms: u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX),
                })
            })
            .collect(),
    })
    .into_response()
}

/// Lists what `GET /pools` and `GET /routers` would serve. A Kubernetes store
/// answers from its cache once that has synced, so an API server outage does
/// not turn every replica unready at once. The first probe also starts the
/// store's watch. An instance that shares registrations is ready only once it
/// has asked the others; an unreachable one later does not make it unready.
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
fn error_response(error: PoolDiscoveryError) -> Response {
    let status = match &error {
        PoolDiscoveryError::Unauthenticated => StatusCode::UNAUTHORIZED,
        PoolDiscoveryError::PermissionDenied(_) => StatusCode::FORBIDDEN,
        PoolDiscoveryError::UnknownPool(_) => StatusCode::NOT_FOUND,
        PoolDiscoveryError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    tracing::warn!(?error, %status, "discovery request failed");
    (status, error.to_string()).into_response()
}

/// `POST /pools` body.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PoolRegistration {
    key: WirePoolKey,
    location: WirePoolLocation,
    /// The model or checkpoint the pool serves.
    model: String,
    #[serde(default)]
    capabilities: Vec<String>,
    /// Local frontends a Global Router sends requests to.
    #[serde(default)]
    frontends: Vec<Url>,
    #[serde(default)]
    capacity: Vec<WireCapacity>,
}

impl PoolRegistration {
    /// The body a PoolRelay sent for `record`. `None` without a model, which
    /// every registration has.
    fn from_record(record: PoolRecord) -> Option<Self> {
        Some(Self {
            key: WirePoolKey::from(&record.key),
            location: record.location.into(),
            model: record.model?,
            capabilities: record.capabilities,
            frontends: record.frontends,
            capacity: record
                .capacity
                .into_iter()
                .map(WireCapacity::from)
                .collect(),
        })
    }

    /// Sorts the lists, so that a heartbeat listing them in another order, for
    /// example from another PoolRelay replica, is not a change.
    fn into_record(mut self) -> Result<PoolRecord, PoolKeyError> {
        self.capabilities.sort_unstable();
        self.capabilities.dedup();
        self.frontends.sort_unstable();
        self.frontends.dedup();
        self.capacity.sort_unstable();
        self.capacity.dedup();
        Ok(PoolRecord {
            key: PoolKey::new(self.key.site_id, self.key.namespace, self.key.dgd_name)?,
            location: self.location.into(),
            runtime_namespace: None,
            frontend_endpoint: None,
            model: Some(self.model),
            capabilities: self.capabilities,
            frontends: self.frontends,
            capacity: self.capacity.into_iter().map(Capacity::from).collect(),
            relay_identity: None,
            revision: Revision::default(),
        })
    }
}

/// `namespace` is the Kubernetes namespace containing the DGD.
#[derive(Debug, Deserialize, Serialize)]
struct WirePoolKey {
    site_id: String,
    namespace: String,
    dgd_name: String,
}

impl From<&PoolKey> for WirePoolKey {
    fn from(key: &PoolKey) -> Self {
        Self {
            site_id: key.site_id().to_owned(),
            namespace: key.namespace().to_owned(),
            dgd_name: key.dgd_name().to_owned(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct WirePoolLocation {
    region: String,
    availability_zone: Option<String>,
    cluster: Option<String>,
    datacenter: Option<String>,
}

impl From<WirePoolLocation> for PoolLocation {
    fn from(location: WirePoolLocation) -> Self {
        Self {
            region: location.region,
            availability_zone: location.availability_zone,
            cluster: location.cluster,
            datacenter: location.datacenter,
        }
    }
}

impl From<PoolLocation> for WirePoolLocation {
    fn from(location: PoolLocation) -> Self {
        Self {
            region: location.region,
            availability_zone: location.availability_zone,
            cluster: location.cluster,
            datacenter: location.datacenter,
        }
    }
}

/// The workers of one hardware SKU.
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct WireCapacity {
    sku: String,
    count: u32,
}

impl From<WireCapacity> for Capacity {
    fn from(capacity: WireCapacity) -> Self {
        Self {
            sku: capacity.sku,
            count: capacity.count,
        }
    }
}

impl From<Capacity> for WireCapacity {
    fn from(capacity: Capacity) -> Self {
        Self {
            sku: capacity.sku,
            count: capacity.count,
        }
    }
}

/// `GET /registrations` reply.
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
    pub(crate) fn into_record(self) -> Result<(PoolRecord, Duration), PoolKeyError> {
        Ok((
            self.registration.into_record()?,
            Duration::from_millis(self.expires_in_ms),
        ))
    }
}

/// `POST /pools` reply: when to register again, and how long a registration
/// lasts.
#[derive(Serialize)]
struct LeaseView {
    heartbeat_interval_s: u64,
    lease_duration_s: u64,
}

impl From<LeasePolicy> for LeaseView {
    fn from(lease: LeasePolicy) -> Self {
        Self {
            heartbeat_interval_s: lease.heartbeat_interval.as_secs(),
            lease_duration_s: lease.lease_duration.as_secs(),
        }
    }
}

#[derive(Serialize)]
struct PoolsResponse {
    pools: Vec<PoolView>,
    /// True when the store lists every pool, so a new router can tell a full
    /// list from a partial one.
    complete: bool,
}

/// One pool in `GET /pools`. It leaves out the record's revision, which
/// orders updates only within one service process and restarts at 1.
#[derive(Serialize)]
struct PoolView {
    key: WirePoolKey,
    location: WirePoolLocation,
    model: Option<String>,
    capabilities: Vec<String>,
    frontends: Vec<Url>,
    capacity: Vec<WireCapacity>,
    /// The only Relay allowed to stream this pool's state, when the store
    /// knows it.
    relay_identity: Option<String>,
}

impl From<PoolRecord> for PoolView {
    fn from(record: PoolRecord) -> Self {
        Self {
            key: WirePoolKey::from(&record.key),
            location: record.location.into(),
            model: record.model,
            capabilities: record.capabilities,
            frontends: record.frontends,
            capacity: record
                .capacity
                .into_iter()
                .map(WireCapacity::from)
                .collect(),
            relay_identity: record.relay_identity.map(|identity| identity.0),
        }
    }
}

#[derive(Serialize)]
struct RoutersResponse {
    routers: Vec<RouterView>,
}

/// A `ReplicaEndpoint` as `{"replica": "<id>", "url": "<url>"}`.
#[derive(Serialize)]
struct RouterView {
    replica: String,
    url: Url,
}

impl From<ReplicaEndpoint> for RouterView {
    fn from(endpoint: ReplicaEndpoint) -> Self {
        Self {
            replica: endpoint.replica.0,
            url: endpoint.url,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};
    use crate::pool_discovery::{RelayIdentity, ReplicaId};
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::backends::kubernetes::TrustedNetworkAuthenticator;
    use crate::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};

    const LEASE: LeasePolicy = LeasePolicy {
        heartbeat_interval: Duration::from_secs(5),
        lease_duration: Duration::from_secs(15),
    };

    /// The body `dynamo-global-discovery-client` sends.
    fn registration() -> Value {
        json!({
            "key": {"site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {
                "region": "us-west-2",
                "availability_zone": null,
                "cluster": "aws-west",
                "datacenter": null
            },
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"],
            "capacity": [{"sku": "H100", "count": 8}]
        })
    }

    fn router_replica() -> ReplicaEndpoint {
        ReplicaEndpoint {
            replica: ReplicaId("aws-ohio/global-router-0".to_owned()),
            url: Url::parse("http://10.42.0.5:50051").unwrap(),
        }
    }

    fn memory_store() -> DiscoveryStore {
        let pools = Arc::new(MemoryPoolStore::new(LEASE).unwrap());
        DiscoveryStore {
            pools: pools.clone(),
            routers: Arc::new(StaticReplicaDirectory::new(vec![router_replica()])),
            registrar: Some(pools),
            authenticator: Arc::new(TrustedNetworkAuthenticator),
            sharing: None,
        }
    }

    /// Serves `store` on a free local port and returns its base URL.
    async fn start(store: DiscoveryStore) -> (String, CancellationToken) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let shutdown = CancellationToken::new();
        tokio::spawn(serve(listener, store, shutdown.clone()));
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
                    "key": {"site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
                    "location": {
                        "region": "us-west-2",
                        "availability_zone": null,
                        "cluster": "aws-west",
                        "datacenter": null
                    },
                    "model": "Qwen/Qwen3-0.6B",
                    "capabilities": ["aggregated"],
                    "frontends": ["http://10.43.1.2:8000/"],
                    "capacity": [{"sku": "H100", "count": 8}],
                    "relay_identity": null
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
    async fn invalid_key_is_a_bad_request() {
        let (base, shutdown) = start(memory_store()).await;
        let mut body = registration();
        body["key"]["site_id"] = json!(" ");
        let (status, _) = post_pool(&base, &body).await;
        assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
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
        async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
            Err(PoolDiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }

        async fn list(&self) -> Result<Snapshot<T>, PoolDiscoveryError> {
            Err(PoolDiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }

        async fn get(&self, _key: &K) -> Result<Option<T>, PoolDiscoveryError> {
            Err(PoolDiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<T, K>, PoolDiscoveryError> {
            Err(PoolDiscoveryError::Unavailable(anyhow::anyhow!("down")))
        }
    }

    #[tokio::test]
    async fn unreachable_store_is_unavailable() {
        let (base, shutdown) = start(DiscoveryStore {
            pools: Arc::new(Down),
            routers: Arc::new(Down),
            registrar: None,
            authenticator: Arc::new(TrustedNetworkAuthenticator),
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
    async fn registrations_are_shared_only_when_enabled() {
        let (base, shutdown) = start(memory_store()).await;
        let response = reqwest::get(format!("{base}/registrations")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
        shutdown.cancel();

        let pools = Arc::new(MemoryPoolStore::new(LEASE).unwrap());
        let synced = Arc::new(AtomicBool::new(false));
        let mut store = memory_store();
        store.pools = pools.clone();
        store.registrar = Some(pools.clone());
        store.sharing = Some(Sharing {
            instance: "a".to_owned(),
            registrations: pools,
            synced: synced.clone(),
        });
        let (base, shutdown) = start(store).await;
        // Not ready until it has asked the other instances once.
        let response = reqwest::get(format!("{base}/healthz")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        synced.store(true, Ordering::Release);
        let response = reqwest::get(format!("{base}/healthz")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        post_pool(&base, &registration()).await;
        let (status, shared) = get_json(&base, "/registrations").await;
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(shared["instance"], json!("a"));
        let expires_in_ms = shared["registrations"][0]["expires_in_ms"]
            .as_u64()
            .unwrap();
        assert!(
            expires_in_ms > 0 && expires_in_ms <= 15_000,
            "{expires_in_ms}"
        );
        // Shared as the PoolRelay sent it.
        assert_eq!(shared["registrations"][0]["registration"], registration());
        shutdown.cancel();
    }

    #[tokio::test]
    async fn healthy_store_is_ok() {
        let (base, shutdown) = start(memory_store()).await;
        let response = reqwest::get(format!("{base}/healthz")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        shutdown.cancel();
    }

    #[test]
    fn relay_identity_is_reported_as_text() {
        let mut record = PoolRegistration::deserialize(registration())
            .unwrap()
            .into_record()
            .unwrap();
        record.relay_identity = Some(RelayIdentity("ip:10.43.66.175".to_owned()));
        let view = serde_json::to_value(PoolView::from(record)).unwrap();
        assert_eq!(view["relay_identity"], json!("ip:10.43.66.175"));
    }

    #[test]
    fn list_order_does_not_change_the_record() {
        let record = |capabilities: Value, frontends: Value, capacity: Value| {
            let mut body = registration();
            body["capabilities"] = capabilities;
            body["frontends"] = frontends;
            body["capacity"] = capacity;
            PoolRegistration::deserialize(body)
                .unwrap()
                .into_record()
                .unwrap()
        };
        let one = record(
            json!(["prefill", "decode"]),
            json!(["http://10.43.1.3:8000/", "http://10.43.1.2:8000/"]),
            json!([{"sku": "H100", "count": 8}, {"sku": "B200", "count": 4}]),
        );
        let other = record(
            json!(["decode", "prefill", "decode"]),
            json!(["http://10.43.1.2:8000/", "http://10.43.1.3:8000/"]),
            json!([{"sku": "B200", "count": 4}, {"sku": "H100", "count": 8}]),
        );
        assert_eq!(one, other);
        assert_eq!(one.capabilities, ["decode", "prefill"]);
    }
}
