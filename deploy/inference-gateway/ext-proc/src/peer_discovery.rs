// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovers replica-sync peers for embedded mode.
//!
//! Watches the EPP's Kubernetes `Service` EndpointSlices and updates its
//! in-process [`SelectionService`] as sibling EPP replicas join or leave.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

/// Label Kubernetes sets on every EndpointSlice pointing back to its Service.
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

/// Named Service/EndpointSlice port used for aggregated replica synchronization.
pub const REPLICA_AGG_PORT_NAME: &str = "replica-agg";

type Store = kube::runtime::reflector::Store<EndpointSlice>;

/// Resolve the required aggregated replica-sync port from the peer Service's
/// EndpointSlices. Every slice must expose the same named `replica-agg` port;
/// missing or inconsistent ports fail EPP startup before replica sync is built.
pub async fn resolve_replica_sync_port(namespace: &str, service_name: &str) -> Result<u16> {
    use kube::{Api, Client, api::ListParams};

    let client = Client::try_default()
        .await
        .context("building Kubernetes client for EPP peer port resolution")?;
    let slices: Api<EndpointSlice> = Api::namespaced(client, namespace);
    let list = slices
        .list(&ListParams::default().labels(&format!("{SERVICE_NAME_LABEL}={service_name}")))
        .await
        .with_context(|| {
            format!("listing EndpointSlices for EPP peer Service {namespace}/{service_name}")
        })?;

    replica_sync_port(list.items.iter()).with_context(|| {
        format!(
            "resolving named port {REPLICA_AGG_PORT_NAME:?} for EPP peer Service \
             {namespace}/{service_name}"
        )
    })
}

fn replica_sync_port<'a>(slices: impl Iterator<Item = &'a EndpointSlice>) -> Result<u16> {
    let mut resolved = BTreeSet::new();
    let mut slice_count = 0usize;

    for slice in slices {
        slice_count += 1;
        let slice_name = slice.metadata.name.as_deref().unwrap_or("<unnamed>");
        let mut matches = slice
            .ports
            .as_deref()
            .unwrap_or_default()
            .iter()
            // Only a TCP `replica-agg` port satisfies the contract: the replica
            // plane binds and dials `tcp://`. Kubernetes defaults `protocol` to
            // TCP when absent, so treat `None` as TCP and reject explicit
            // UDP/SCTP rather than let a mismatched port through.
            .filter(|port| {
                port.name.as_deref() == Some(REPLICA_AGG_PORT_NAME)
                    && port
                        .protocol
                        .as_deref()
                        .is_none_or(|protocol| protocol.eq_ignore_ascii_case("TCP"))
            });
        let endpoint_port = matches.next().with_context(|| {
            format!(
                "EndpointSlice {slice_name} does not expose named port \
                 {REPLICA_AGG_PORT_NAME:?}"
            )
        })?;
        anyhow::ensure!(
            matches.next().is_none(),
            "EndpointSlice {slice_name} exposes named port {REPLICA_AGG_PORT_NAME:?} more than once"
        );
        let raw_port = endpoint_port.port.with_context(|| {
            format!(
                "EndpointSlice {slice_name} named port {REPLICA_AGG_PORT_NAME:?} has no port number"
            )
        })?;
        let port = u16::try_from(raw_port).with_context(|| {
            format!(
                "EndpointSlice {slice_name} named port {REPLICA_AGG_PORT_NAME:?} has invalid port {raw_port}"
            )
        })?;
        anyhow::ensure!(
            port > 0,
            "named port {REPLICA_AGG_PORT_NAME:?} must be greater than zero"
        );
        resolved.insert(port);
    }

    anyhow::ensure!(slice_count > 0, "peer Service has no EndpointSlices");
    anyhow::ensure!(
        resolved.len() == 1,
        "named port {REPLICA_AGG_PORT_NAME:?} resolves to inconsistent ports {resolved:?}"
    );
    Ok(*resolved.first().expect("validated one resolved port"))
}

/// Starts peer discovery for the EPP's own Kubernetes Service, keeping
/// replica-sync peers registered on `service` and excluding `self_ip`.
///
/// Returns a readiness flag that becomes `true` after the initial reconciliation.
pub async fn spawn(
    service: Arc<SelectionService>,
    namespace: &str,
    service_name: &str,
    sync_port: u16,
    self_ip: String,
    cancel: CancellationToken,
) -> Result<Arc<AtomicBool>> {
    use futures::StreamExt;
    use kube::{Api, Client, runtime::WatchStreamExt, runtime::reflector, runtime::watcher};

    let client = Client::try_default()
        .await
        .context("building Kubernetes client for EPP peer discovery")?;
    let slices: Api<EndpointSlice> = Api::namespaced(client, namespace);
    let cfg_watch =
        watcher::Config::default().labels(&format!("{SERVICE_NAME_LABEL}={service_name}"));

    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let reflect = reflector::reflector(writer, watcher(slices, cfg_watch).default_backoff());
    let (changes_tx, changes_rx) = watch::channel(0u64);

    tracing::info!(
        %namespace,
        service = %service_name,
        sync_port,
        %self_ip,
        "Starting EPP peer EndpointSlice watch (embedded replication)"
    );

    // EndpointSlice reflector stream -> bump the change generation. The watcher
    // retries transient errors internally; the stream ends only on writer drop.
    let cancel_watch = cancel.clone();
    tokio::spawn(async move {
        tokio::pin!(reflect);
        let mut generation = 0u64;
        loop {
            tokio::select! {
                _ = cancel_watch.cancelled() => return,
                item = reflect.next() => match item {
                    // Skip the per-object relist events (Init/InitApply) and errors:
                    // the store is consistent at InitDone, and Apply/Delete are
                    // single-object deltas. Reconcile reads the store, so bumping on
                    // partial relist state only triggers redundant reconciles.
                    Some(Ok(watcher::Event::Init | watcher::Event::InitApply(_))) => {}
                    Some(Ok(_)) => {
                        generation = generation.wrapping_add(1);
                        let _ = changes_tx.send(generation);
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

    let peer_ready = Arc::new(AtomicBool::new(false));

    tokio::spawn(reconcile_loop(
        service,
        store,
        sync_port,
        self_ip,
        changes_rx,
        cancel,
        peer_ready.clone(),
    ));
    Ok(peer_ready)
}

/// React to EndpointSlice changes: diff the live sibling set against the peers
/// currently registered and apply the delta. Exits when `cancel` fires or the
/// change channel closes.
async fn reconcile_loop(
    service: Arc<SelectionService>,
    store: Store,
    sync_port: u16,
    self_ip: String,
    mut changes_rx: watch::Receiver<u64>,
    cancel: CancellationToken,
    peer_ready: Arc<AtomicBool>,
) {
    // Block on the first authoritative LIST before the initial reconcile so we
    // never latch readiness on an empty snapshot. The reflector retries watch
    // errors with backoff, so this resolves once the LIST lands; a writer drop
    // (watch task gone) means we can't sync, so bail without latching.
    tokio::select! {
        _ = cancel.cancelled() => return,
        result = store.wait_until_ready() => {
            if result.is_err() {
                tracing::warn!(
                    "EPP peer EndpointSlice writer dropped before initial LIST; \
                     peer discovery never became ready"
                );
                return;
            }
        }
    }

    let mut known: BTreeSet<String> = BTreeSet::new();
    reconcile_once(&service, &store, sync_port, &self_ip, &mut known).await;
    // Set readiness to true after the initial reconciliation.
    // Subsequent transient watch failures keep the last-known peers and must not clear it.
    peer_ready.store(true, Ordering::Release);
    tracing::info!("EPP peer discovery initial sync complete");

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            changed = changes_rx.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }
        reconcile_once(&service, &store, sync_port, &self_ip, &mut known).await;
    }
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
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};

    fn slice_with(ips: &[&str], terminating: bool, address_type: &str) -> EndpointSlice {
        EndpointSlice {
            address_type: address_type.to_string(),
            endpoints: ips
                .iter()
                .map(|ip| Endpoint {
                    addresses: vec![ip.to_string()],
                    conditions: Some(EndpointConditions {
                        terminating: Some(terminating),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn slice_with_replica_port(port: Option<i32>) -> EndpointSlice {
        let mut slice = slice_with(&["10.0.0.1"], false, "IPv4");
        slice.metadata.name = Some("epp-peers-abc".to_string());
        slice.ports = Some(vec![EndpointPort {
            name: Some(REPLICA_AGG_PORT_NAME.to_string()),
            port,
            ..Default::default()
        }]);
        slice
    }

    #[test]
    fn peer_ips_keeps_non_terminating() {
        let slices = [slice_with(&["10.0.0.1", "10.0.0.2"], false, "IPv4")];
        let ips = peer_ips(slices.iter(), false);
        assert!(ips.contains("10.0.0.1"));
        assert!(ips.contains("10.0.0.2"));
    }

    #[test]
    fn peer_ips_preserves_terminating() {
        let slices = [slice_with(&["10.0.0.9"], true, "IPv4")];
        assert!(peer_ips(slices.iter(), false).contains("10.0.0.9"));
    }

    fn slice_with_serving(ip: &str, terminating: bool, serving: bool) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints: vec![Endpoint {
                addresses: vec![ip.to_string()],
                conditions: Some(EndpointConditions {
                    terminating: Some(terminating),
                    serving: Some(serving),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn peer_ips_keeps_terminating_but_serving() {
        // A terminating sibling that is still serving is draining in-flight
        // requests and will emit final PrefillComplete/Free events; keep it so
        // that load is not stranded in the local aggregate.
        let slices = [slice_with_serving("10.0.0.5", true, true)];
        assert!(peer_ips(slices.iter(), false).contains("10.0.0.5"));
    }

    #[test]
    fn peer_ips_preserves_terminating_not_serving() {
        // Once a terminating sibling stops serving it is truly done; drop it.
        let slices = [slice_with_serving("10.0.0.6", true, false)];
        assert!(peer_ips(slices.iter(), false).contains("10.0.0.6"));
    }

    #[test]
    fn peer_ips_filters_by_address_family() {
        // A dual-stack sibling is present in both an IPv4 and an IPv6 slice; only
        // the family matching our own IP is kept, so it is registered once.
        let slices = [
            slice_with(&["10.0.0.1"], false, "IPv4"),
            slice_with(&["fd00::1"], false, "IPv6"),
        ];
        let v4 = peer_ips(slices.iter(), false);
        assert_eq!(v4.len(), 1);
        assert!(v4.contains("10.0.0.1"));

        let v6 = peer_ips(slices.iter(), true);
        assert_eq!(v6.len(), 1);
        assert!(v6.contains("fd00::1"));
    }

    #[test]
    fn authority_brackets_ipv6_only() {
        assert_eq!(authority("10.0.0.1", 9092), "10.0.0.1:9092");
        assert_eq!(authority("fd00::1", 9092), "[fd00::1]:9092");
    }

    #[test]
    fn resolves_replica_agg_named_port() {
        let slices = [
            slice_with_replica_port(Some(9092)),
            slice_with_replica_port(Some(9092)),
        ];
        assert_eq!(replica_sync_port(slices.iter()).unwrap(), 9092);
    }

    #[test]
    fn rejects_missing_replica_agg_named_port() {
        let slices = [slice_with(&["10.0.0.1"], false, "IPv4")];
        let error = replica_sync_port(slices.iter()).unwrap_err().to_string();
        assert!(error.contains(REPLICA_AGG_PORT_NAME));
    }

    #[test]
    fn rejects_inconsistent_replica_agg_named_ports() {
        let slices = [
            slice_with_replica_port(Some(9092)),
            slice_with_replica_port(Some(9093)),
        ];
        let error = replica_sync_port(slices.iter()).unwrap_err().to_string();
        assert!(error.contains("inconsistent ports"));
    }

    fn slice_with_replica_port_protocol(protocol: Option<&str>) -> EndpointSlice {
        let mut slice = slice_with(&["10.0.0.1"], false, "IPv4");
        slice.metadata.name = Some("epp-peers-proto".to_string());
        slice.ports = Some(vec![EndpointPort {
            name: Some(REPLICA_AGG_PORT_NAME.to_string()),
            port: Some(9092),
            protocol: protocol.map(str::to_string),
            ..Default::default()
        }]);
        slice
    }

    #[test]
    fn accepts_absent_or_tcp_replica_agg_protocol() {
        // Absent protocol defaults to TCP in Kubernetes; explicit TCP is fine.
        assert_eq!(
            replica_sync_port([slice_with_replica_port_protocol(None)].iter()).unwrap(),
            9092
        );
        assert_eq!(
            replica_sync_port([slice_with_replica_port_protocol(Some("TCP"))].iter()).unwrap(),
            9092
        );
    }

    #[test]
    fn rejects_non_tcp_replica_agg_port() {
        // A UDP `replica-agg` port must not resolve: the replica plane dials
        // tcp://, so treating it as valid would be a silent transport mismatch.
        // With no TCP match left, resolution fails with the "does not expose"
        // error naming the port.
        let error = replica_sync_port([slice_with_replica_port_protocol(Some("UDP"))].iter())
            .unwrap_err()
            .to_string();
        assert!(error.contains(REPLICA_AGG_PORT_NAME));
    }

    fn free_tcp_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    /// Build a reflector `Store<EndpointSlice>` from a fixed slice set (no
    /// cluster), so `reconcile_once` can be driven over scripted transitions.
    fn store_from_slices(slices: Vec<EndpointSlice>) -> Store {
        use kube::runtime::watcher;
        let mut writer = kube::runtime::reflector::store::Writer::<EndpointSlice>::default();
        let store = writer.as_reader();
        writer.apply_watcher_event(&watcher::Event::Init);
        for (i, mut slice) in slices.into_iter().enumerate() {
            // The reflector keys by name; give each slice a distinct one.
            slice.metadata.name = Some(format!("epp-peers-{i}"));
            writer.apply_watcher_event(&watcher::Event::InitApply(slice));
        }
        writer.apply_watcher_event(&watcher::Event::InitDone);
        store
    }

    /// End-to-end at the reconcile boundary: a sibling that enters termination
    /// while still `serving` is draining in-flight ext-proc streams and will emit
    /// final `PrefillComplete`/`Free` events over replica sync. `reconcile_once`
    /// must therefore keep it *registered* on the `SelectionService` (so those
    /// events still arrive and release its load — see kv-router's
    /// `selector_replica_sync_propagates_request_lifecycle` for the release path)
    /// and only deregister it once it stops serving. This covers the reconcile
    /// wiring that consumes `peer_ips`, which the predicate tests above do not.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_retains_draining_peer_until_it_stops_serving() {
        use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
        use dynamo_kv_router::services::selection::SelectionServiceBuilder;

        // A real replica-sync-enabled service. `register_replica_peer` is a lazy
        // ZMQ connect, so no live sibling is needed — we assert only the peer set
        // that reconcile maintains via `list_replica_peers`.
        let service = Arc::new(
            SelectionServiceBuilder::new(kv_router_config_from_dynamo_env())
                .indexer_threads(1)
                .replica_sync(free_tcp_port(), Vec::new())
                .build()
                .await
                .expect("build replica-sync selection service"),
        );

        let self_ip = "10.0.0.1";
        let peer = "10.0.0.2";
        let sync_port = 9092; // dial port; only used to format the peer endpoint
        let peer_endpoint = format!("tcp://{}", authority(peer, sync_port));
        let mut known = BTreeSet::new();

        // 1) Sibling serving normally -> registered.
        let store = store_from_slices(vec![slice_with_serving(peer, false, true)]);
        reconcile_once(&service, &store, sync_port, self_ip, &mut known).await;
        assert!(
            service.list_replica_peers().contains(&peer_endpoint),
            "a live sibling must be registered"
        );

        // 2) Sibling enters termination but is still serving -> RETAINED, so its
        //    final PrefillComplete/Free events can still be delivered.
        let store = store_from_slices(vec![slice_with_serving(peer, true, true)]);
        reconcile_once(&service, &store, sync_port, self_ip, &mut known).await;
        assert!(
            service.list_replica_peers().contains(&peer_endpoint),
            "a draining (terminating+serving) sibling must stay registered"
        );

        // 3) Sibling stops serving -> still RETAINED
        let store = store_from_slices(vec![slice_with_serving(peer, true, false)]);
        reconcile_once(&service, &store, sync_port, self_ip, &mut known).await;
        assert!(
            service.list_replica_peers().contains(&peer_endpoint),
            "a sibling that stopped serving must stay registered"
        );

        // 4) Sibling disappears -> truly done, deregistered.
        let store = store_from_slices(vec![]);
        reconcile_once(&service, &store, sync_port, self_ip, &mut known).await;
        assert!(
            !service.list_replica_peers().contains(&peer_endpoint),
            "a sibling that disappears must be deregistered"
        );

        service.shutdown().await;
    }
}
