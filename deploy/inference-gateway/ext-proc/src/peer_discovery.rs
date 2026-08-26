// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovers replica-sync peers for embedded mode.
//!
//! Watches the EPP's Kubernetes `Service` EndpointSlices and updates its
//! in-process [`SelectionService`] as sibling EPP replicas join or leave.

use std::collections::BTreeSet;
use std::collections::hash_map::RandomState;
use std::future::Future;
use std::hash::BuildHasher;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

/// Label Kubernetes sets on every EndpointSlice pointing back to its Service.
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

/// Named Service/EndpointSlice port used for aggregated replica synchronization.
pub const REPLICA_AGG_PORT_NAME: &str = "replica-agg";

/// Named Service/EndpointSlice port used for startup KV-index recovery.
pub const SELECTION_HTTP_PORT_NAME: &str = "selection-http";

const INITIAL_RECOVERY_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
const MAX_RECOVERY_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PeerPorts {
    pub(crate) replica_sync: u16,
    /// `None` when the Service does not declare the `selection-http` port (an
    /// image-only upgrade from a deployment that predates the dump endpoint).
    /// Peer KV-index recovery is then disabled — the replica bootstraps empty
    /// — rather than failing startup.
    pub(crate) selection_http: Option<u16>,
}

type Store = kube::runtime::reflector::Store<EndpointSlice>;
type RecoveryAttempt<'a> = Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

/// Resolve both peer-plane ports from one authoritative EndpointSlice snapshot.
pub(crate) async fn resolve_peer_ports(namespace: &str, service_name: &str) -> Result<PeerPorts> {
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

    peer_ports(list.items.iter())
        .with_context(|| format!("resolving peer ports for EPP Service {namespace}/{service_name}"))
}

fn peer_ports<'a>(slices: impl Iterator<Item = &'a EndpointSlice>) -> Result<PeerPorts> {
    let slices: Vec<_> = slices.collect();
    Ok(PeerPorts {
        replica_sync: named_tcp_port(&slices, REPLICA_AGG_PORT_NAME)?,
        // `selection-http` is optional: a deployment upgraded before the dump
        // endpoint existed declares only `replica-agg`. A missing dump port
        // degrades to "no recovery" (bootstrap empty) instead of failing.
        selection_http: optional_named_tcp_port(&slices, SELECTION_HTTP_PORT_NAME)?,
    })
}

/// Like [`named_tcp_port`], but returns `Ok(None)` when no EndpointSlice
/// exposes the named port. Slices that omit the port are skipped; conflicting
/// values or invalid ports still error.
fn optional_named_tcp_port(slices: &[&EndpointSlice], port_name: &str) -> Result<Option<u16>> {
    let mut resolved = BTreeSet::new();

    for slice in slices {
        let slice_name = slice.metadata.name.as_deref().unwrap_or("<unnamed>");
        let mut matches = slice
            .ports
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|port| port.name.as_deref() == Some(port_name));
        let Some(endpoint_port) = matches.next() else {
            continue;
        };
        anyhow::ensure!(
            matches.next().is_none(),
            "EndpointSlice {slice_name} exposes named port {port_name:?} more than once"
        );
        anyhow::ensure!(
            endpoint_port
                .protocol
                .as_deref()
                .is_none_or(|protocol| protocol.eq_ignore_ascii_case("TCP")),
            "EndpointSlice {slice_name} named port {port_name:?} must use TCP"
        );
        let raw_port = endpoint_port.port.with_context(|| {
            format!("EndpointSlice {slice_name} named port {port_name:?} has no port number")
        })?;
        let port = u16::try_from(raw_port).with_context(|| {
            format!(
                "EndpointSlice {slice_name} named port {port_name:?} has invalid port {raw_port}"
            )
        })?;
        anyhow::ensure!(
            port > 0,
            "named port {port_name:?} must be greater than zero"
        );
        resolved.insert(port);
    }

    anyhow::ensure!(!slices.is_empty(), "peer Service has no EndpointSlices");
    if resolved.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        resolved.len() == 1,
        "named port {port_name:?} resolves to inconsistent ports {resolved:?}"
    );
    Ok(resolved.first().copied())
}

fn named_tcp_port(slices: &[&EndpointSlice], port_name: &str) -> Result<u16> {
    let mut resolved = BTreeSet::new();

    for slice in slices {
        let slice_name = slice.metadata.name.as_deref().unwrap_or("<unnamed>");
        let mut matches = slice
            .ports
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|port| port.name.as_deref() == Some(port_name));
        let endpoint_port = matches.next().with_context(|| {
            format!(
                "EndpointSlice {slice_name} does not expose named port \
                 {port_name:?}"
            )
        })?;
        anyhow::ensure!(
            matches.next().is_none(),
            "EndpointSlice {slice_name} exposes named port {port_name:?} more than once"
        );
        anyhow::ensure!(
            endpoint_port
                .protocol
                .as_deref()
                .is_none_or(|protocol| protocol.eq_ignore_ascii_case("TCP")),
            "EndpointSlice {slice_name} named port {port_name:?} must use TCP"
        );
        let raw_port = endpoint_port.port.with_context(|| {
            format!("EndpointSlice {slice_name} named port {port_name:?} has no port number")
        })?;
        let port = u16::try_from(raw_port).with_context(|| {
            format!(
                "EndpointSlice {slice_name} named port {port_name:?} has invalid port {raw_port}"
            )
        })?;
        anyhow::ensure!(
            port > 0,
            "named port {port_name:?} must be greater than zero"
        );
        resolved.insert(port);
    }

    anyhow::ensure!(!slices.is_empty(), "peer Service has no EndpointSlices");
    anyhow::ensure!(
        resolved.len() == 1,
        "named port {port_name:?} resolves to inconsistent ports {resolved:?}"
    );
    resolved
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("named port {port_name:?} did not resolve"))
}

/// Starts peer discovery for the EPP's own Kubernetes Service, keeping
/// replica-sync peers registered on `service` and excluding `self_ip`.
///
/// This call does not return until initial KV-index recovery succeeds or the
/// authoritative sibling set is empty. The dump server must already be bound.
#[allow(clippy::too_many_arguments)]
pub async fn spawn(
    service: Arc<SelectionService>,
    namespace: &str,
    service_name: &str,
    sync_port: u16,
    selection_http_port: u16,
    self_ip: String,
    cancel: CancellationToken,
) -> Result<()> {
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
    let (changes_tx, mut changes_rx) = watch::channel(0u64);

    tracing::info!(
        %namespace,
        service = %service_name,
        sync_port,
        selection_http_port,
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

    // Block on the first authoritative LIST before the initial reconcile so we
    // never latch readiness on an empty snapshot. The reflector retries watch
    // errors with backoff, so this resolves once the LIST lands; a writer drop
    // (watch task gone) means we can't sync, so bail without latching.
    tokio::select! {
        _ = cancel.cancelled() => anyhow::bail!("EPP peer discovery cancelled before initial LIST"),
        result = store.wait_until_ready() => {
            result.context("EPP peer EndpointSlice writer dropped before initial LIST")?;
        }
    }

    let mut known: BTreeSet<String> = BTreeSet::new();
    // InitDone generated the snapshot we just consumed; do not mistake it for a
    // peer change after the first failed recovery attempt.
    changes_rx.borrow_and_update();
    recover_initial_index(
        &service,
        &store,
        sync_port,
        selection_http_port,
        &self_ip,
        &mut known,
        &mut changes_rx,
        &cancel,
        INITIAL_RECOVERY_BACKOFF,
        MAX_RECOVERY_BACKOFF,
    )
    .await?;

    tracing::info!("EPP peer discovery and KV-index bootstrap complete");

    tokio::spawn(async move {
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
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn recover_initial_index(
    service: &SelectionService,
    store: &Store,
    sync_port: u16,
    selection_http_port: u16,
    self_ip: &str,
    known: &mut BTreeSet<String>,
    changes_rx: &mut watch::Receiver<u64>,
    cancel: &CancellationToken,
    initial_backoff: std::time::Duration,
    max_backoff: std::time::Duration,
) -> Result<()> {
    recover_initial_index_with_attempt(
        service,
        store,
        sync_port,
        selection_http_port,
        self_ip,
        known,
        changes_rx,
        cancel,
        initial_backoff,
        max_backoff,
        |service, peers| Box::pin(service.recover_indexer_from_peers(peers)),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn recover_initial_index_with_attempt<F>(
    service: &SelectionService,
    store: &Store,
    sync_port: u16,
    selection_http_port: u16,
    self_ip: &str,
    known: &mut BTreeSet<String>,
    changes_rx: &mut watch::Receiver<u64>,
    cancel: &CancellationToken,
    initial_backoff: std::time::Duration,
    max_backoff: std::time::Duration,
    mut recover: F,
) -> Result<()>
where
    F: for<'a> FnMut(&'a SelectionService, &'a [String]) -> RecoveryAttempt<'a>,
{
    let mut backoff = initial_backoff;

    loop {
        reconcile_once(service, store, sync_port, self_ip, known).await;

        // Deterministic eligible set for change detection; the shuffled order is
        // derived once per cycle (attempt-scoped priority), never compared, so
        // an unrelated EndpointSlice update cannot look like a membership change.
        let eligible = recovery_peer_set(store, self_ip, selection_http_port);
        if eligible.is_empty() {
            crate::metrics::set_kv_recovery_state(crate::metrics::KV_RECOVERY_EMPTY_BOOTSTRAP);
            tracing::warn!(
                "No serving sibling EPP peer found; bootstrapping an EMPTY KV index \
                 (normal on first deploy, but a full-index loss if the cluster was \
                 already serving — check expected replica count / recovery history)"
            );
            return Ok(());
        }

        let mut priority = shuffled_peer_urls(&eligible);
        let mut tried: BTreeSet<String> = BTreeSet::new();

        // Try one peer per attempt. EndpointSlice churn never cancels an
        // in-flight request — the dump may still complete after its source
        // leaves the slice — so churn only reconciles the *pending* candidates:
        // drop unattempted peers that are no longer eligible, add newly eligible
        // ones. Peers that failed stay in `tried` until the cycle is exhausted.
        'attempt: loop {
            let Some(peer) = priority.iter().find(|p| !tried.contains(*p)).cloned() else {
                break 'attempt;
            };

            let peers = [peer.clone()];
            let attempt = recover(service, &peers);
            tokio::pin!(attempt);

            let result = loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        anyhow::bail!("EPP peer discovery cancelled during KV-index recovery")
                    }
                    changed = changes_rx.changed() => {
                        changed.context("EPP peer EndpointSlice watch ended during KV-index recovery")?;
                        let current = recovery_peer_set(store, self_ip, selection_http_port);
                        // Keep the active request (even if its own source left
                        // the slice); reconcile only the unattempted pending set.
                        priority.retain(|p| tried.contains(p) || current.contains(p));
                        for newly in current.difference(&tried) {
                            if !priority.contains(newly) {
                                priority.push(newly.clone());
                            }
                        }
                    }
                    result = &mut attempt => break result,
                }
            };

            match result {
                Ok(true) => {
                    crate::metrics::set_kv_recovery_state(crate::metrics::KV_RECOVERY_RECOVERED);
                    return Ok(());
                }
                Ok(false) => tracing::warn!(
                    peer = %peer,
                    "No reachable EPP peer dump; trying next recovery candidate"
                ),
                Err(error) => tracing::warn!(
                    peer = %peer,
                    %error,
                    "EPP peer KV-index recovery failed; trying next recovery candidate"
                ),
            }
            tried.insert(peer);
        }

        // The current eligible set is exhausted (or emptied by churn): back off,
        // then start a fresh cycle with a fresh shuffle.
        tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("EPP peer discovery cancelled during KV-index recovery")
            }
            changed = changes_rx.changed() => {
                changed.context("EPP peer EndpointSlice watch ended during KV-index recovery")?;
                // Only a genuine candidate-set change earns an immediate retry
                // with the initial backoff. Unrelated churn (readiness flips,
                // metadata/zone updates) must not reset the backoff either:
                // under a rolling update it would keep the retry loop hot at
                // the initial backoff instead of letting it grow.
                if recovery_peer_set(store, self_ip, selection_http_port) != eligible {
                    backoff = initial_backoff;
                }
            }
            _ = tokio::time::sleep(backoff) => {
                backoff = backoff.saturating_mul(2).min(max_backoff);
            }
        }
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

/// Recovery targets only siblings that are actively serving and not
/// terminating. Not-ready siblings cannot contribute a meaningful index: this
/// replica only advertises readiness after recovery and the buffered listener
/// handoff completes, so a not-ready sibling's index is not an authoritative
/// recovery source. Treating those as recovery candidates turns a cold start
/// (all replicas empty, none serving) into a mutual-recovery deadlock. An empty
/// candidate set means "no eligible peer" and bootstraps an empty index
/// immediately.
/// Deterministic eligible peer URL set. Used for change detection and as the
/// input to an attempt-scoped shuffle. Unlike a shuffled vector, the BTreeSet
/// order is stable, so equality across calls detects only real membership
/// changes — never a re-randomization.
fn recovery_peer_set(store: &Store, self_ip: &str, port: u16) -> BTreeSet<String> {
    let want_ipv6 = is_ipv6(self_ip);
    let mut peers = BTreeSet::new();

    for slice in store.state() {
        if !matches_address_family(&slice.address_type, want_ipv6) {
            continue;
        }
        for endpoint in &slice.endpoints {
            if !is_eligible_recovery_endpoint(endpoint) {
                continue;
            }
            for address in &endpoint.addresses {
                if !address.is_empty() && address != self_ip {
                    peers.insert(format!("http://{}", authority(address, port)));
                }
            }
        }
    }
    peers
}

/// One attempt-scoped random priority order over the eligible set. Shuffling
/// happens only when a new recovery cycle starts, never inside a comparison:
/// N simultaneously-joining replicas must not all deterministically pick the
/// lowest-IP peer (BTreeSet order), concentrating dump work on one serving EPP.
/// `RandomState` is seeded per process, so each bootstrap gets a different
/// order; serial fallback through the shuffled order is retained.
fn shuffled_peer_urls(eligible: &BTreeSet<String>) -> Vec<String> {
    let hasher = RandomState::new();
    let mut urls: Vec<String> = eligible.iter().cloned().collect();
    urls.sort_by_cached_key(|url| hasher.hash_one(url));
    urls
}

/// A peer is an eligible recovery source only when it is actively serving and
/// not terminating. `ready` alone is insufficient in both directions: with
/// `publishNotReadyAddresses` an endpoint can be `ready=true, serving=false`
/// (a live-but-cold replica that would return an empty dump), while a draining
/// pod is `serving=true, terminating=true` and can vanish mid-transfer. When
/// the `serving` condition is absent (legacy slices), fall back to `ready`.
fn is_eligible_recovery_endpoint(endpoint: &Endpoint) -> bool {
    let Some(conditions) = endpoint.conditions.as_ref() else {
        return false;
    };
    if conditions.terminating == Some(true) {
        return false;
    }
    match conditions.serving {
        Some(serving) => serving,
        None => conditions.ready == Some(true),
    }
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

fn matches_address_family(address_type: &str, want_ipv6: bool) -> bool {
    match address_type {
        address_type if address_type.eq_ignore_ascii_case("IPv4") => !want_ipv6,
        address_type if address_type.eq_ignore_ascii_case("IPv6") => want_ipv6,
        _ => false,
    }
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
        if !matches_address_family(&slice.address_type, want_ipv6) {
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
    use std::sync::atomic::Ordering;

    use super::*;
    use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::sync::Notify;

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

    fn slice_with_peer_ports(replica_sync: i32, selection_http: i32) -> EndpointSlice {
        let mut slice = slice_with(&["10.0.0.1"], false, "IPv4");
        slice.metadata.name = Some("epp-peers-abc".to_string());
        slice.ports = Some(vec![
            EndpointPort {
                name: Some(REPLICA_AGG_PORT_NAME.to_string()),
                port: Some(replica_sync),
                ..Default::default()
            },
            EndpointPort {
                name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                port: Some(selection_http),
                ..Default::default()
            },
        ]);
        slice
    }

    fn parse_named_port(slices: &[EndpointSlice], port_name: &str) -> Result<u16> {
        let slices: Vec<_> = slices.iter().collect();
        named_tcp_port(&slices, port_name)
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
    fn peer_ips_rejects_fqdn_addresses() {
        let slices = [slice_with(&["epp.example.test"], false, "FQDN")];
        assert!(peer_ips(slices.iter(), false).is_empty());
        assert!(
            recovery_peer_set(&store_from_slices(slices.to_vec()), "10.0.0.9", 9093).is_empty()
        );
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
        assert_eq!(
            parse_named_port(&slices, REPLICA_AGG_PORT_NAME).unwrap(),
            9092
        );
    }

    #[test]
    fn resolves_selection_http_named_port() {
        let slices = [
            slice_with_peer_ports(9092, 9093),
            slice_with_peer_ports(9092, 9093),
        ];
        assert_eq!(
            peer_ports(slices.iter()).unwrap(),
            PeerPorts {
                replica_sync: 9092,
                selection_http: Some(9093),
            }
        );
    }

    #[test]
    fn rejects_missing_replica_agg_named_port() {
        let slices = [slice_with(&["10.0.0.1"], false, "IPv4")];
        let error = parse_named_port(&slices, REPLICA_AGG_PORT_NAME)
            .unwrap_err()
            .to_string();
        assert!(error.contains(REPLICA_AGG_PORT_NAME));
    }

    #[test]
    fn rejects_inconsistent_replica_agg_named_ports() {
        let slices = [
            slice_with_replica_port(Some(9092)),
            slice_with_replica_port(Some(9093)),
        ];
        let error = parse_named_port(&slices, REPLICA_AGG_PORT_NAME)
            .unwrap_err()
            .to_string();
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
            parse_named_port(
                &[slice_with_replica_port_protocol(None)],
                REPLICA_AGG_PORT_NAME
            )
            .unwrap(),
            9092
        );
        assert_eq!(
            parse_named_port(
                &[slice_with_replica_port_protocol(Some("TCP"))],
                REPLICA_AGG_PORT_NAME
            )
            .unwrap(),
            9092
        );
    }

    #[test]
    fn rejects_non_tcp_replica_agg_port() {
        // A UDP `replica-agg` port must not resolve: the replica plane dials
        // tcp://, so treating it as valid would be a silent transport mismatch.
        // With no TCP match left, resolution fails with the "does not expose"
        // error naming the port.
        let error = parse_named_port(
            &[slice_with_replica_port_protocol(Some("UDP"))],
            REPLICA_AGG_PORT_NAME,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(REPLICA_AGG_PORT_NAME));
    }

    #[test]
    fn rejects_invalid_named_tcp_ports() {
        let cases = [
            (
                "missing number",
                vec![EndpointPort {
                    name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                    port: None,
                    ..Default::default()
                }],
            ),
            (
                "zero",
                vec![EndpointPort {
                    name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                    port: Some(0),
                    ..Default::default()
                }],
            ),
            (
                "out of range",
                vec![EndpointPort {
                    name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                    port: Some(65_536),
                    ..Default::default()
                }],
            ),
            (
                "udp",
                vec![EndpointPort {
                    name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                    port: Some(9093),
                    protocol: Some("UDP".to_string()),
                    ..Default::default()
                }],
            ),
            (
                "duplicate",
                vec![
                    EndpointPort {
                        name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                        port: Some(9093),
                        ..Default::default()
                    },
                    EndpointPort {
                        name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                        port: Some(9094),
                        ..Default::default()
                    },
                ],
            ),
        ];

        for (name, ports) in cases {
            let mut slice = slice_with(&["10.0.0.1"], false, "IPv4");
            slice.metadata.name = Some(name.to_string());
            slice.ports = Some(ports);
            assert!(
                parse_named_port(&[slice], SELECTION_HTTP_PORT_NAME).is_err(),
                "case {name} must fail"
            );
        }

        let slices = [
            slice_with_peer_ports(9092, 9093),
            slice_with_peer_ports(9092, 9094),
        ];
        assert!(
            parse_named_port(&slices, SELECTION_HTTP_PORT_NAME)
                .unwrap_err()
                .to_string()
                .contains("inconsistent ports")
        );
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

    fn recovery_slice(ip: &str, ready: Option<bool>, serving: Option<bool>) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints: vec![Endpoint {
                addresses: vec![ip.to_string()],
                conditions: Some(EndpointConditions {
                    ready,
                    serving,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ports: Some(vec![
                EndpointPort {
                    name: Some(REPLICA_AGG_PORT_NAME.to_string()),
                    port: Some(9092),
                    ..Default::default()
                },
                EndpointPort {
                    name: Some(SELECTION_HTTP_PORT_NAME.to_string()),
                    port: Some(9093),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        }
    }

    fn store_and_writer(
        slices: Vec<EndpointSlice>,
    ) -> (
        Store,
        kube::runtime::reflector::store::Writer<EndpointSlice>,
    ) {
        use kube::runtime::watcher;

        let mut writer = kube::runtime::reflector::store::Writer::<EndpointSlice>::default();
        let store = writer.as_reader();
        writer.apply_watcher_event(&watcher::Event::Init);
        for (index, mut slice) in slices.into_iter().enumerate() {
            slice
                .metadata
                .name
                .get_or_insert_with(|| format!("epp-peers-{index}"));
            writer.apply_watcher_event(&watcher::Event::InitApply(slice));
        }
        writer.apply_watcher_event(&watcher::Event::InitDone);
        (store, writer)
    }

    fn start_recovery(
        service: Arc<SelectionService>,
        store: Store,
        selection_http_port: u16,
        changes_rx: watch::Receiver<u64>,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<Result<()>> {
        tokio::spawn(async move {
            let mut known = BTreeSet::new();
            let mut changes_rx = changes_rx;
            recover_initial_index(
                &service,
                &store,
                9092,
                selection_http_port,
                "127.0.0.9",
                &mut known,
                &mut changes_rx,
                &cancel,
                Duration::from_millis(10),
                Duration::from_millis(40),
            )
            .await
        })
    }

    async fn recovery_service() -> Arc<SelectionService> {
        use dynamo_kv_router::config::KvRouterConfig;
        use dynamo_kv_router::services::selection::SelectionServiceBuilder;

        Arc::new(
            SelectionServiceBuilder::new(KvRouterConfig::default())
                .indexer_threads(1)
                .build()
                .await
                .expect("build selection service"),
        )
    }

    #[derive(Clone)]
    struct DumpGate {
        requested: Arc<Notify>,
        release: Arc<Notify>,
    }

    async fn gated_dump(State(gate): State<DumpGate>) -> axum::response::Response {
        gate.requested.notify_one();
        gate.release.notified().await;
        // An empty JSON object = an empty snapshot, a valid recovery.
        axum::response::Response::new(axum::body::Body::from("{}"))
    }

    #[derive(Clone)]
    struct FlakyDump {
        first_failed: Arc<Notify>,
        attempts: Arc<AtomicUsize>,
    }

    async fn flaky_dump(State(state): State<FlakyDump>) -> impl IntoResponse {
        if state.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            state.first_failed.notify_one();
            (StatusCode::SERVICE_UNAVAILABLE, "not ready").into_response()
        } else {
            // An empty JSON object = an empty snapshot, a valid recovery.
            axum::response::Response::new(axum::body::Body::from("{}")).into_response()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initial_recovery_bootstraps_without_peer() {
        let service = recovery_service().await;
        let (store, _writer) = store_and_writer(Vec::new());
        let (_changes_tx, mut changes_rx) = watch::channel(0u64);
        let cancel = CancellationToken::new();
        let mut known = BTreeSet::new();

        recover_initial_index(
            &service,
            &store,
            9092,
            9093,
            "127.0.0.9",
            &mut known,
            &mut changes_rx,
            &cancel,
            Duration::from_millis(10),
            Duration::from_millis(40),
        )
        .await
        .expect("empty peer set must bootstrap immediately");
        assert!(known.is_empty());

        service.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_uses_selection_http_port_from_endpoint_slice() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route(
                    "/dump",
                    get(|| async { axum::response::Response::new(axum::body::Body::from("{}")) }),
                ),
            )
            .await
        });
        let mut slice = recovery_slice("127.0.0.1", Some(true), Some(true));
        slice.ports.as_mut().unwrap()[1].port = Some(i32::from(port));
        let ports = peer_ports([&slice].into_iter()).expect("resolve peer ports");
        let (store, _writer) = store_and_writer(vec![slice]);
        let (_changes_tx, changes_rx) = watch::channel(0u64);
        let service = recovery_service().await;
        let cancel = CancellationToken::new();

        start_recovery(
            service.clone(),
            store,
            ports.selection_http.unwrap(),
            changes_rx,
            cancel.clone(),
        )
        .await
        .expect("recovery task joins")
        .expect("recovery must use the EndpointSlice HTTP port");

        cancel.cancel();
        server.abort();
        service.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_retries_unchanged_peer_until_reachable() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = FlakyDump {
            first_failed: Arc::new(Notify::new()),
            attempts: Arc::new(AtomicUsize::new(0)),
        };
        let server = tokio::spawn({
            let state = state.clone();
            async move {
                axum::serve(
                    listener,
                    Router::new()
                        .route("/dump", get(flaky_dump))
                        .with_state(state),
                )
                .await
            }
        });
        let (store, _writer) =
            store_and_writer(vec![recovery_slice("127.0.0.1", Some(true), Some(true))]);
        let (_changes_tx, changes_rx) = watch::channel(0u64);
        let service = recovery_service().await;
        let cancel = CancellationToken::new();
        let task = start_recovery(service.clone(), store, port, changes_rx, cancel.clone());

        tokio::time::timeout(Duration::from_secs(3), state.first_failed.notified())
            .await
            .expect("first recovery request must fail");
        tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .expect("unchanged peer must be retried")
            .expect("recovery task joins")
            .expect("second recovery succeeds");
        assert_eq!(state.attempts.load(Ordering::SeqCst), 2);

        cancel.cancel();
        server.abort();
        service.shutdown().await;
    }

    /// Shared driver for the churn-behavior tests: drives
    /// `recover_initial_index_with_attempt` with a scripted recover closure.
    struct ChurnHarness {
        task: tokio::task::JoinHandle<anyhow::Result<()>>,
        cancel: CancellationToken,
        service: Arc<SelectionService>,
        changes_tx: watch::Sender<u64>,
        writer: kube::runtime::reflector::store::Writer<EndpointSlice>,
        attempts: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
        first_started: Arc<Notify>,
        release: Arc<Notify>,
        first_dropped: Arc<Notify>,
    }

    impl ChurnHarness {
        /// Start recovery with one slice containing `initial_ips`, all
        /// eligible. The first attempt is held until `release`.
        async fn start(initial_ips: &[&str], port: u16, first_attempt_outcome: bool) -> Self {
            use kube::runtime::watcher;

            let mut slices: Vec<EndpointSlice> = initial_ips
                .iter()
                .map(|ip| recovery_slice(ip, Some(true), Some(true)))
                .collect();
            // The reflector keys slices by name; name them by IP so later
            // Apply/Delete events for the same peer hit the same object.
            for (ip, slice) in initial_ips.iter().zip(slices.iter_mut()) {
                slice.metadata.name = Some(format!("peer-{}", ip.replace('.', "-")));
            }
            let mut writer = kube::runtime::reflector::store::Writer::<EndpointSlice>::default();
            let store = writer.as_reader();
            writer.apply_watcher_event(&watcher::Event::Init);
            for slice in slices {
                writer.apply_watcher_event(&watcher::Event::InitApply(slice));
            }
            writer.apply_watcher_event(&watcher::Event::InitDone);

            let (changes_tx, changes_rx) = watch::channel(0u64);
            let service = recovery_service().await;
            let cancel = CancellationToken::new();

            let attempts = Arc::new(std::sync::Mutex::new(Vec::<Vec<String>>::new()));
            let first_started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let first_dropped = Arc::new(Notify::new());

            struct DropSignal(Arc<Notify>);
            impl Drop for DropSignal {
                fn drop(&mut self) {
                    self.0.notify_one();
                }
            }

            let attempt_no = Arc::new(AtomicUsize::new(0));
            let task = {
                let service = service.clone();
                let cancel = cancel.clone();
                let attempts = attempts.clone();
                let attempt_no = attempt_no.clone();
                let first_started = first_started.clone();
                let release = release.clone();
                let first_dropped = first_dropped.clone();
                tokio::spawn(async move {
                    let mut known = BTreeSet::new();
                    let mut changes_rx = changes_rx;
                    recover_initial_index_with_attempt(
                        &service,
                        &store,
                        9092,
                        port,
                        "192.0.2.99",
                        &mut known,
                        &mut changes_rx,
                        &cancel,
                        Duration::from_millis(10),
                        Duration::from_millis(40),
                        move |_service, peers| {
                            let peers = peers.to_vec();
                            attempts.lock().unwrap().push(peers.clone());
                            let number = attempt_no.fetch_add(1, Ordering::SeqCst);
                            let first_started = first_started.clone();
                            let release = release.clone();
                            let first_dropped = first_dropped.clone();
                            Box::pin(async move {
                                let _drop_signal = DropSignal(first_dropped);
                                if number == 0 {
                                    first_started.notify_one();
                                    // Hold the first attempt until the test
                                    // releases it. The test chooses its outcome:
                                    // `Ok(false)` moves on to the next pending
                                    // peer, `Ok(true)` completes recovery.
                                    release.notified().await;
                                    Ok(first_attempt_outcome)
                                } else {
                                    // Later attempts succeed: recovery completes.
                                    Ok(true)
                                }
                            })
                        },
                    )
                    .await
                })
            };

            Self {
                task,
                cancel,
                service,
                changes_tx,
                writer,
                attempts,
                first_started,
                release,
                first_dropped,
            }
        }

        async fn apply(&mut self, ip: &str, eligible: bool) {
            let mut slice = recovery_slice(ip, Some(true), Some(eligible));
            slice.metadata.name = Some(format!("peer-{}", ip.replace('.', "-")));
            self.writer
                .apply_watcher_event(&kube::runtime::watcher::Event::Apply(slice));
            self.changes_tx.send(1).unwrap();
        }

        async fn remove(&mut self, ip: &str) {
            let mut slice = recovery_slice(ip, Some(true), Some(true));
            slice.metadata.name = Some(format!("peer-{}", ip.replace('.', "-")));
            self.writer
                .apply_watcher_event(&kube::runtime::watcher::Event::Delete(slice));
            self.changes_tx.send(1).unwrap();
        }

        async fn assert_active_not_dropped(&self) {
            // The in-flight attempt must survive the churn: no Drop.
            let dropped = self.first_dropped.clone();
            let result = tokio::time::timeout(Duration::from_millis(200), dropped.notified()).await;
            assert!(
                result.is_err(),
                "in-flight recovery attempt must NOT be cancelled by EndpointSlice churn"
            );
        }

        async fn finish(self) {
            self.release.notify_one();
            tokio::time::timeout(Duration::from_secs(3), self.task)
                .await
                .expect("recovery must complete")
                .expect("recovery task joins")
                .expect("recovery must succeed");
            self.cancel.cancel();
            self.service.shutdown().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unchanged_churn_keeps_inflight_attempt() {
        let port = 9093;
        let mut harness = ChurnHarness::start(&["192.0.2.10"], port, true).await;
        tokio::time::timeout(Duration::from_secs(1), harness.first_started.notified())
            .await
            .expect("first recovery attempt must be in flight");

        // Unrelated churn: same eligible set, metadata-only change (the reflector
        // bumps the change channel on every Apply).
        let mut slice = recovery_slice("192.0.2.10", Some(true), Some(true));
        slice.metadata.name = Some("epp-peers-churn".to_string());
        slice.metadata.annotations = Some(
            [("note".to_string(), "churn".to_string())]
                .into_iter()
                .collect(),
        );
        harness
            .writer
            .apply_watcher_event(&kube::runtime::watcher::Event::Apply(slice));
        harness.changes_tx.send(1).unwrap();

        harness.assert_active_not_dropped().await;
        let attempts = harness.attempts.clone();
        harness.finish().await;
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![vec![format!("http://192.0.2.10:{port}")]],
            "unrelated churn must not restart the attempt"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn join_during_recovery_keeps_active_then_tries_new_peer() {
        let port = 9093;
        let mut harness = ChurnHarness::start(&["192.0.2.10"], port, false).await;
        tokio::time::timeout(Duration::from_secs(1), harness.first_started.notified())
            .await
            .expect("first recovery attempt must be in flight");

        // A new serving peer joins mid-recovery: the active request must not be
        // cancelled; once it fails, the new peer is tried next.
        harness.apply("192.0.2.11", true).await;
        harness.assert_active_not_dropped().await;
        let attempts = harness.attempts.clone();
        harness.finish().await;
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![
                vec![format!("http://192.0.2.10:{port}")],
                vec![format!("http://192.0.2.11:{port}")],
            ],
            "a join must not cancel the active attempt; the new peer is tried after it fails"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn removal_of_unattempted_peer_drops_it_from_pending() {
        let port = 9093;
        let mut harness = ChurnHarness::start(&["192.0.2.10", "192.0.2.11"], port, false).await;
        tokio::time::timeout(Duration::from_secs(1), harness.first_started.notified())
            .await
            .expect("first recovery attempt must be in flight");

        // The first attempt is on whichever peer the attempt-scoped shuffle put
        // first; the OTHER peer is unattempted. Remove that one mid-flight: it
        // must be dropped from pending and never tried.
        let first_peer = {
            let attempts = harness.attempts.lock().unwrap();
            attempts.last().expect("first attempt recorded")[0].clone()
        };
        let unattempted = if first_peer.contains("192.0.2.10") {
            "192.0.2.11"
        } else {
            "192.0.2.10"
        };
        harness.remove(unattempted).await;
        harness.assert_active_not_dropped().await;
        let attempts = harness.attempts.clone();
        harness.finish().await;
        let attempts = attempts.lock().unwrap().clone();
        assert!(
            attempts.iter().all(|a| a == &vec![first_peer.clone()]),
            "the removed peer must never be tried, got: {attempts:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn removal_of_active_peer_does_not_cancel_its_request() {
        let port = 9093;
        let mut harness = ChurnHarness::start(&["192.0.2.10"], port, true).await;
        tokio::time::timeout(Duration::from_secs(1), harness.first_started.notified())
            .await
            .expect("first recovery attempt must be in flight");

        // The active peer leaves the slice mid-transfer: the request is kept
        // and may still complete successfully.
        harness.remove("192.0.2.10").await;
        harness.assert_active_not_dropped().await;
        let attempts = harness.attempts.clone();
        harness.finish().await;
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![vec![format!("http://192.0.2.10:{port}")]],
            "removal of the active peer must not cancel its request"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peers_disappearing_during_recovery_keep_active_request() {
        use kube::runtime::watcher;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let gate = DumpGate {
            requested: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let server = tokio::spawn({
            let gate = gate.clone();
            async move {
                axum::serve(
                    listener,
                    Router::new()
                        .route("/dump", get(gated_dump))
                        .with_state(gate),
                )
                .await
            }
        });
        let mut old_slice = recovery_slice("127.0.0.1", Some(true), Some(true));
        old_slice.metadata.name = Some("epp-peers-0".to_string());
        let (store, mut writer) = store_and_writer(vec![old_slice.clone()]);
        let (changes_tx, changes_rx) = watch::channel(0u64);
        let service = recovery_service().await;
        let cancel = CancellationToken::new();
        let task = start_recovery(service.clone(), store, port, changes_rx, cancel.clone());

        tokio::time::timeout(Duration::from_secs(3), gate.requested.notified())
            .await
            .expect("old recovery request must be in flight");
        writer.apply_watcher_event(&watcher::Event::Delete(old_slice));
        changes_tx.send(1).unwrap();

        // The in-flight request is kept even though every peer is gone; only
        // once it completes (empty snapshot = success) does recovery finish.
        gate.release.notify_one();
        tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .expect("recovery must finish after the kept request completes")
            .expect("recovery task joins")
            .expect("kept request completes recovery");

        cancel.cancel();
        server.abort();
        service.shutdown().await;
    }

    #[test]
    fn recovery_candidate_order_does_not_change_replica_membership() {
        let slice = EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints: vec![
                Endpoint {
                    addresses: vec!["10.0.0.2".to_string()],
                    conditions: Some(EndpointConditions {
                        ready: Some(false),
                        serving: Some(false),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Endpoint {
                    addresses: vec!["10.0.0.3".to_string()],
                    conditions: Some(EndpointConditions {
                        ready: Some(true),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let (store, _writer) = store_and_writer(vec![slice.clone()]);

        assert_eq!(
            peer_ips([&slice].into_iter(), false),
            BTreeSet::from(["10.0.0.2".to_string(), "10.0.0.3".to_string()])
        );
        // Recovery candidates exclude the not-ready sibling (10.0.0.2): a
        // not-ready replica has no KV index yet, so it cannot bootstrap a peer.
        assert_eq!(
            recovery_peer_set(&store, "10.0.0.9", 9093),
            BTreeSet::from(["http://10.0.0.3:9093".to_string()])
        );
    }

    #[test]
    fn recovery_excludes_ready_but_not_serving_peer() {
        // `publishNotReadyAddresses` can yield ready=true, serving=false for a
        // live-but-cold replica: it would return an empty dump, so it must not
        // be a recovery source.
        let slice = EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints: vec![Endpoint {
                addresses: vec!["10.0.0.2".to_string()],
                conditions: Some(EndpointConditions {
                    ready: Some(true),
                    serving: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (store, _writer) = store_and_writer(vec![slice]);
        assert!(recovery_peer_set(&store, "10.0.0.9", 9093).is_empty());
    }

    #[test]
    fn recovery_excludes_terminating_serving_peer() {
        // A draining pod is serving=true, terminating=true and can vanish
        // mid-transfer, so it must not be a recovery source.
        let slice = EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints: vec![Endpoint {
                addresses: vec!["10.0.0.2".to_string()],
                conditions: Some(EndpointConditions {
                    serving: Some(true),
                    terminating: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (store, _writer) = store_and_writer(vec![slice]);
        assert!(recovery_peer_set(&store, "10.0.0.9", 9093).is_empty());
    }

    #[test]
    fn selection_http_port_is_optional() {
        // A Service without `selection-http` (a deployment predating the dump
        // endpoint) resolves with selection_http=None instead of failing.
        let slice = slice_with_replica_port(Some(9092));
        let ports = peer_ports([&slice].into_iter()).expect("resolve ports");
        assert_eq!(ports.replica_sync, 9092);
        assert_eq!(ports.selection_http, None);
    }

    #[test]
    fn recovery_peer_set_brackets_ipv6() {
        let slice = EndpointSlice {
            address_type: "IPv6".to_string(),
            endpoints: vec![Endpoint {
                addresses: vec!["fd00::2".to_string()],
                conditions: Some(EndpointConditions {
                    ready: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (store, _writer) = store_and_writer(vec![slice]);
        assert_eq!(
            recovery_peer_set(&store, "fd00::1", 9093),
            BTreeSet::from(["http://[fd00::2]:9093".to_string()])
        );
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
