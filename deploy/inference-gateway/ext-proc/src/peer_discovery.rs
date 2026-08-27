// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovers replica-sync peers for embedded mode.
//!
//! Watches the EPP's Kubernetes `Service` EndpointSlices and updates its
//! in-process [`SelectionService`] as sibling EPP replicas join or leave.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use k8s_openapi::api::core::v1::Service;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

/// Label Kubernetes sets on every EndpointSlice pointing back to its Service.
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

type Store = kube::runtime::reflector::Store<EndpointSlice>;

/// Verifies the peer Service exists before enabling replica synchronization.
pub(crate) async fn ensure_peer_service_exists(
    client: Client,
    namespace: &str,
    service_name: &str,
) -> Result<()> {
    let services: Api<Service> = Api::namespaced(client, namespace);
    services
        .get(service_name)
        .await
        .with_context(|| format!("getting EPP peer Service {namespace}/{service_name}"))?;
    Ok(())
}

/// Starts peer discovery for the EPP's own Kubernetes Service, keeping
/// replica-sync peers registered on `service` and excluding `self_ip`.
pub async fn spawn(
    client: Client,
    service: Arc<SelectionService>,
    namespace: &str,
    service_name: &str,
    sync_port: u16,
    self_ip: String,
    cancel: CancellationToken,
) -> Result<()> {
    use futures::StreamExt;
    use kube::runtime::{WatchStreamExt, reflector, watcher};
    let slices: Api<EndpointSlice> = Api::namespaced(client, namespace);
    let cfg_watch =
        watcher::Config::default().labels(&format!("{SERVICE_NAME_LABEL}={service_name}"));

    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let reflect = reflector::reflector(writer, watcher(slices, cfg_watch).default_backoff());

    tracing::info!(
        %namespace,
        service = %service_name,
        sync_port,
        %self_ip,
        "Starting EPP peer EndpointSlice watch (embedded replication)"
    );

    tokio::spawn(async move {
        tokio::pin!(reflect);
        let mut known = BTreeSet::new();
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                item = reflect.next() => match item {
                    // Store state during a relist is incomplete until InitDone.
                    Some(Ok(watcher::Event::Init | watcher::Event::InitApply(_))) => {}
                    Some(Ok(_)) => {
                        reconcile_once(&service, &store, sync_port, &self_ip, &mut known).await;
                    }
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "EPP peer EndpointSlice watch error");
                    }
                    None => {
                        tracing::warn!("EPP peer EndpointSlice reflector stream ended");
                        return;
                    }
                },
            }
        }
    });

    Ok(())
}

async fn reconcile_once(
    service: &SelectionService,
    store: &Store,
    sync_port: u16,
    self_ip: &str,
    known: &mut BTreeSet<String>,
) {
    let live = live_peer_ips(store, self_ip);
    let added: Vec<String> = live.difference(known).cloned().collect();
    let removed: Vec<String> = known.difference(&live).cloned().collect();

    for ip in &added {
        let endpoint = format!("tcp://{}", authority(ip, sync_port));
        if let Err(e) = service.register_replica_peer(endpoint.clone()).await {
            tracing::debug!(%endpoint, error = %e, "register_replica_peer failed");
        }
    }
    for ip in &removed {
        let endpoint = format!("tcp://{}", authority(ip, sync_port));
        if let Err(e) = service.deregister_replica_peer(endpoint.clone()).await {
            tracing::debug!(%endpoint, error = %e, "deregister_replica_peer failed");
        }
    }
    if !added.is_empty() || !removed.is_empty() {
        tracing::info!(added = ?added, removed = ?removed, total = live.len(), "EPP peer set changed");
    }
    *known = live;
}

/// Returns sibling EPP IPs matching `self_ip`'s address family, excluding
/// `self_ip`. Using one address family prevents duplicate dual-stack peers.
fn live_peer_ips(store: &Store, self_ip: &str) -> BTreeSet<String> {
    let want_ipv6 = is_ipv6(self_ip);
    let mut ips = peer_ips(store.state().iter().map(|s| s.as_ref()), want_ipv6);
    ips.remove(self_ip);
    ips
}

/// Format `host:port`, bracketing IPv6 literals (`fd00::1` -> `[fd00::1]`) so the
/// resulting `tcp://` endpoint stays valid on dual-stack clusters.
fn authority(ip: &str, port: u16) -> String {
    if ip.contains(':') {
        format!("[{ip}]:{port}")
    } else {
        format!("{ip}:{port}")
    }
}

fn is_ipv6(ip: &str) -> bool {
    ip.contains(':')
}

/// Collects peer IPs for the requested address family. Includes not-ready peers
/// and terminating peers that are still serving to preserve synchronization
/// while they start or drain.
fn peer_ips<'a>(
    slices: impl Iterator<Item = &'a EndpointSlice>,
    want_ipv6: bool,
) -> BTreeSet<String> {
    let mut ips = BTreeSet::new();
    for slice in slices {
        if slice.address_type.eq_ignore_ascii_case("IPv6") != want_ipv6 {
            continue;
        }
        // Replica-sync membership follows EndpointSlice membership, not traffic
        // readiness. Connecting early is safe because ZMQ retries asynchronously;
        // retaining endpoints until removal allows final lifecycle events through.
        for endpoint in &slice.endpoints {
            for addr in &endpoint.addresses {
                if !addr.is_empty() {
                    ips.insert(addr.clone());
                }
            }
        }
    }
    ips
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};

    fn slice_with(
        ips: &[&str],
        conditions: Option<EndpointConditions>,
        address_type: &str,
    ) -> EndpointSlice {
        EndpointSlice {
            address_type: address_type.to_string(),
            endpoints: ips
                .iter()
                .map(|ip| Endpoint {
                    addresses: vec![ip.to_string()],
                    conditions: conditions.clone(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn peer_ips_follows_membership_regardless_of_endpoint_conditions() {
        for (name, conditions) in [
            ("no conditions", None),
            (
                "not ready",
                Some(EndpointConditions {
                    ready: Some(false),
                    ..Default::default()
                }),
            ),
            (
                "terminating but serving",
                Some(EndpointConditions {
                    terminating: Some(true),
                    serving: Some(true),
                    ..Default::default()
                }),
            ),
            (
                "terminating and not serving",
                Some(EndpointConditions {
                    terminating: Some(true),
                    serving: Some(false),
                    ..Default::default()
                }),
            ),
        ] {
            let slices = [slice_with(&["10.0.0.2"], conditions, "IPv4")];
            assert_eq!(
                peer_ips(slices.iter(), false),
                BTreeSet::from(["10.0.0.2".to_string()]),
                "{name}"
            );
        }
    }

    #[test]
    fn peer_ips_filters_by_address_family() {
        // A dual-stack sibling is present in both an IPv4 and an IPv6 slice; only
        // the family matching our own IP is kept, so it is registered once.
        let slices = [
            slice_with(&["10.0.0.1"], None, "IPv4"),
            slice_with(&["fd00::1"], None, "IPv6"),
        ];
        let v4 = peer_ips(slices.iter(), false);
        assert_eq!(v4, BTreeSet::from(["10.0.0.1".to_string()]));

        let v6 = peer_ips(slices.iter(), true);
        assert_eq!(v6, BTreeSet::from(["fd00::1".to_string()]));
    }

    #[test]
    fn authority_brackets_ipv6_only() {
        assert_eq!(authority("10.0.0.1", 9092), "10.0.0.1:9092");
        assert_eq!(authority("fd00::1", 9092), "[fd00::1]:9092");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_reconciles_initial_list_and_watch_update() {
        use axum::extract::Request;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::routing::get;
        use axum::{Json, Router};
        use dynamo_kv_router::WorkerType;
        use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
        use dynamo_kv_router::services::selection::{
            SelectionServiceBuilder, WorkerSelectionPolicyRegistry,
        };
        use kube::Config;
        use tokio::net::TcpListener;
        use tokio::sync::Notify;
        use tokio::time::{Duration, sleep, timeout};

        let listener_port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("reserve a replica-sync listener port")
            .local_addr()
            .expect("read listener address")
            .port();
        let service = Arc::new(
            SelectionServiceBuilder::new(
                kv_router_config_from_dynamo_env(),
                WorkerType::Aggregated,
                WorkerSelectionPolicyRegistry::default(),
            )
            .indexer_threads(1)
            .replica_sync(listener_port, Vec::new())
            .build()
            .await
            .expect("build replica-sync selection service"),
        );

        let self_ip = "10.0.0.1";
        let peer_a = "10.0.0.2";
        let peer_b = "10.0.0.3";
        let sync_port = 9192;
        let mut initial = slice_with(&[self_ip, peer_a], None, "IPv4");
        initial.metadata.name = Some("epp-peers".to_string());
        let mut updated = slice_with(&[self_ip, peer_b], None, "IPv4");
        updated.metadata.name = Some("epp-peers".to_string());
        updated.metadata.resource_version = Some("2".to_string());
        let list = serde_json::json!({
            "apiVersion": "discovery.k8s.io/v1",
            "kind": "EndpointSliceList",
            "metadata": {"resourceVersion": "1"},
            "items": [initial],
        });
        let release_watch_update = Arc::new(Notify::new());
        let watch_started = Arc::new(Notify::new());
        let router = Router::new().fallback(get({
            let release_watch_update = Arc::clone(&release_watch_update);
            let watch_started = Arc::clone(&watch_started);
            move |request: Request| {
                let list = list.clone();
                let updated = updated.clone();
                let release_watch_update = Arc::clone(&release_watch_update);
                let watch_started = Arc::clone(&watch_started);
                async move {
                    if request
                        .uri()
                        .query()
                        .is_some_and(|query| query.contains("watch=true"))
                    {
                        watch_started.notify_one();
                        release_watch_update.notified().await;
                        (
                            StatusCode::OK,
                            format!(
                                "{}\n",
                                serde_json::json!({
                                    "type": "MODIFIED",
                                    "object": updated,
                                })
                            ),
                        )
                            .into_response()
                    } else {
                        Json(list).into_response()
                    }
                }
            }
        }));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind Kubernetes API test server");
        let address = listener.local_addr().expect("read test server address");
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .await
                .expect("serve Kubernetes API test server");
        });
        let client = Client::try_from(Config::new(
            format!("http://{address}")
                .parse()
                .expect("parse Kubernetes API URL"),
        ))
        .expect("build Kubernetes API test client");
        let cancel = CancellationToken::new();

        spawn(
            client,
            Arc::clone(&service),
            "test-ns",
            "epp-peers",
            sync_port,
            self_ip.to_string(),
            cancel.clone(),
        )
        .await
        .expect("start peer discovery");

        let peer_a_endpoint = format!("tcp://{}", authority(peer_a, sync_port));
        timeout(Duration::from_secs(2), async {
            while service.list_replica_peers() != [peer_a_endpoint.clone()] {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("initial EndpointSlice list should register the peer");

        timeout(Duration::from_secs(2), watch_started.notified())
            .await
            .expect("EndpointSlice watch should begin after the initial list");
        release_watch_update.notify_one();
        let peer_b_endpoint = format!("tcp://{}", authority(peer_b, sync_port));
        timeout(Duration::from_secs(2), async {
            while service.list_replica_peers() != [peer_b_endpoint.clone()] {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("EndpointSlice watch update should replace the peer");

        cancel.cancel();
        service.shutdown().await;
        server.abort();
    }
}
