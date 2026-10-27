// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use dynamo_global_discovery_client::{Completeness, GlobalDiscoveryClient, PoolListing};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::catalog::{Pool, PoolCatalog, PoolSnapshot};

/// The pools the Global Discovery Service lists, polled on an interval.
///
/// A failed poll keeps the last snapshot, so a discovery outage does not stop
/// routing to the pools already known.
pub struct DiscoveryPoolCatalog {
    snapshot: watch::Sender<Arc<PoolSnapshot>>,
}

impl DiscoveryPoolCatalog {
    /// Polls once before returning, so the catalog starts with the service's
    /// pools when it is reachable, then polls every `interval` until `cancel`
    /// fires.
    pub async fn start(
        client: GlobalDiscoveryClient,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Self {
        let (snapshot, _) = watch::channel(Arc::new(PoolSnapshot::default()));
        refresh(&client, &snapshot).await;
        let publisher = snapshot.clone();
        tokio::spawn(async move {
            let mut ticks = tokio::time::interval(interval);
            ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
            // The first tick fires at once, and `start` has just polled.
            ticks.tick().await;
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = ticks.tick() => refresh(&client, &publisher).await,
                }
            }
        });
        Self { snapshot }
    }
}

impl PoolCatalog for DiscoveryPoolCatalog {
    fn subscribe(&self) -> watch::Receiver<Arc<PoolSnapshot>> {
        self.snapshot.subscribe()
    }
}

async fn refresh(client: &GlobalDiscoveryClient, snapshot: &watch::Sender<Arc<PoolSnapshot>>) {
    let listing = match client.pools().await {
        Ok(listing) => listing,
        Err(error) => {
            tracing::warn!(%error, "cannot list pools; keeping the last known pools");
            return;
        }
    };
    if listing.completeness == Completeness::Estimated {
        tracing::debug!("the discovery service's pool list may be partial");
    }
    let pools = PoolSnapshot::new(routable_pools(listing.items));
    snapshot.send_if_modified(|current| {
        if **current == pools {
            return false;
        }
        tracing::info!(pools = pools.pools().len(), "pool catalog changed");
        *current = Arc::new(pools);
        true
    });
}

/// The listed pools a router can route to. They are sorted by ID, so an
/// unchanged list compares equal and round robin keeps its place.
fn routable_pools(listings: Vec<PoolListing>) -> Vec<Pool> {
    let mut pools = listings
        .into_iter()
        .filter_map(|listing| {
            let key = &listing.key;
            let Some(model) = listing.model else {
                tracing::debug!(?key, "skipping a pool with no model");
                return None;
            };
            let key = match PoolKey::new(&key.site_id, &key.namespace, &key.dgd_name) {
                Ok(key) => key,
                Err(error) => {
                    tracing::warn!(?key, %error, "skipping a pool with an invalid key");
                    return None;
                }
            };
            let location = listing.location;
            Some(Pool {
                id: V1PoolIdDeriver.derive(&key),
                name: format!("{}/{}/{}", key.site_id(), key.namespace(), key.dgd_name()),
                model,
                location: PoolLocation {
                    region: location.region,
                    availability_zone: location.availability_zone,
                    cluster: location.cluster,
                    datacenter: location.datacenter,
                },
                frontends: listing.frontends,
                relay: listing.relay_identity.as_deref().and_then(relay_address),
            })
        })
        .collect::<Vec<_>>();
    pools.sort_by(|left, right| left.id.as_str().cmp(right.id.as_str()));
    pools
}

/// The address in a Relay identity of the form `ip:<address>`, which is the
/// only form the service lists today.
fn relay_address(identity: &str) -> Option<IpAddr> {
    let address = identity.strip_prefix("ip:")?.parse();
    if address.is_err() {
        tracing::debug!(
            identity,
            "ignoring a Relay identity that is not an IP address"
        );
    }
    address.ok()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::http::StatusCode;
    use axum::routing::get;
    use dynamo_global_discovery_client as wire;
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;

    fn listing(dgd: &str, model: Option<&str>) -> PoolListing {
        PoolListing {
            key: wire::PoolKey {
                site_id: "aws-west".into(),
                namespace: "dynamo".into(),
                dgd_name: dgd.into(),
            },
            location: wire::PoolLocation {
                region: "us-west-2".into(),
                availability_zone: None,
                cluster: Some("aws-west".into()),
                datacenter: None,
            },
            model: model.map(str::to_owned),
            capabilities: Vec::new(),
            frontends: vec![format!("http://{dgd}:8000/").parse().unwrap()],
            capacity: Vec::new(),
            relay_identity: None,
        }
    }

    #[test]
    fn keeps_pools_a_router_can_route_to_in_id_order() {
        let mut bad_key = listing("blank", Some("qwen"));
        bad_key.key.namespace = String::new();
        let mut green = listing("green", Some("qwen"));
        green.relay_identity = Some("ip:10.0.0.7".into());
        let pools = routable_pools(vec![
            green,
            listing("unnamed", None),
            bad_key,
            listing("blue", Some("qwen")),
        ]);
        let green = pools
            .iter()
            .find(|pool| pool.name == "aws-west/dynamo/green");
        assert_eq!(green.unwrap().relay, Some("10.0.0.7".parse().unwrap()));
        assert_eq!(relay_address("10.0.0.7"), None);

        let hosts = pools
            .iter()
            .map(|pool| pool.frontends[0].host_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(hosts.len(), 2);
        assert!(hosts.contains(&"blue") && hosts.contains(&"green"));
        assert!(pools[0].id.as_str() < pools[1].id.as_str());
        assert_eq!(pools[0].location.cluster.as_deref(), Some("aws-west"));
        let mut green = listing("green", Some("qwen"));
        green.relay_identity = Some("ip:10.0.0.7".into());
        let reordered = routable_pools(vec![listing("blue", Some("qwen")), green]);
        assert_eq!(pools, reordered);
    }

    /// A discovery service whose `GET /pools` answer the test can change.
    struct FakeService {
        response: Arc<Mutex<(StatusCode, String)>>,
        client: GlobalDiscoveryClient,
    }

    impl FakeService {
        async fn start() -> Self {
            let response = Arc::new(Mutex::new((StatusCode::OK, pools_body(&[]))));
            let shared = response.clone();
            let app = axum::Router::new().route(
                "/pools",
                get(move || {
                    let (status, body) = shared.lock().unwrap().clone();
                    async move { (status, body) }
                }),
            );
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let base_url = Url::parse(&format!("http://{address}/")).unwrap();
            let client = GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap();
            Self { response, client }
        }

        fn answer(&self, status: StatusCode, body: String) {
            *self.response.lock().unwrap() = (status, body);
        }
    }

    fn pools_body(dgds: &[&str]) -> String {
        let pools = dgds
            .iter()
            .map(|dgd| {
                serde_json::json!({
                    "key": {"site_id": "aws-west", "namespace": "dynamo", "dgd_name": dgd},
                    "location": {"region": "us-west-2", "availability_zone": null, "cluster": null, "datacenter": null},
                    "model": "qwen",
                    "frontends": [format!("http://{dgd}:8000/")],
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({"complete": true, "pools": pools}).to_string()
    }

    async fn next_snapshot(pools: &mut watch::Receiver<Arc<PoolSnapshot>>) -> Arc<PoolSnapshot> {
        tokio::time::timeout(Duration::from_secs(5), pools.changed())
            .await
            .expect("the catalog did not change")
            .unwrap();
        pools.borrow_and_update().clone()
    }

    #[tokio::test]
    async fn follows_the_service_and_keeps_pools_through_an_outage() {
        let service = FakeService::start().await;
        service.answer(StatusCode::OK, pools_body(&["blue"]));
        let cancel = CancellationToken::new();
        let catalog = DiscoveryPoolCatalog::start(
            service.client.clone(),
            Duration::from_millis(20),
            cancel.clone(),
        )
        .await;
        let mut pools = catalog.subscribe();
        assert_eq!(pools.borrow_and_update().pools().len(), 1);

        service.answer(StatusCode::OK, pools_body(&["blue", "green"]));
        assert_eq!(next_snapshot(&mut pools).await.pools().len(), 2);

        // An outage keeps the pools, and an unchanged list publishes nothing.
        service.answer(StatusCode::SERVICE_UNAVAILABLE, String::new());
        tokio::time::sleep(Duration::from_millis(100)).await;
        service.answer(StatusCode::OK, pools_body(&["green", "blue"]));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!pools.has_changed().unwrap());
        assert_eq!(pools.borrow().pools().len(), 2);

        service.answer(StatusCode::OK, pools_body(&["green"]));
        assert_eq!(next_snapshot(&mut pools).await.pools().len(), 1);
        cancel.cancel();
    }
}
