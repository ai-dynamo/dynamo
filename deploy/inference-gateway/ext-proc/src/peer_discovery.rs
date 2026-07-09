// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded-mode peer discovery.
//!
//! In replicated embedded mode each EPP pod runs its own in-process
//! [`SelectionService`] with replica-sync enabled. This module watches the EPP's
//! OWN Kubernetes `Service` EndpointSlices and keeps that service's replica-sync
//! peer set in step with the live sibling EPP replicas: it registers peers that
//! appear and deregisters peers that leave, so active-load/admission events flow
//! across the whole EPP fleet over ZMQ.
//!
//! This mirrors the HTTP fleet's EndpointSlice watch, but the target is the EPP's
//! own Service (siblings), not a separate selector Service, and the peers are
//! wired into the in-process service rather than remote replicas.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

/// Label Kubernetes sets on every EndpointSlice pointing back to its Service.
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

/// Best-effort bound on how long startup waits for the initial EndpointSlice
/// LIST before returning; the background task keeps syncing regardless.
const INITIAL_LIST_TIMEOUT: Duration = Duration::from_secs(30);

type Store = kube::runtime::reflector::Store<EndpointSlice>;

/// Start the peer-discovery watch for the EPP's own `service_name` in
/// `namespace`. Registers/deregisters replica-sync peers on `service` as sibling
/// EPP pods come and go, dialing them on `sync_port`. `self_ip` (the pod's own
/// IP) is excluded so a replica never peers with itself. Returns after the
/// initial LIST (bounded); the watch keeps running until `cancel` fires.
pub async fn spawn(
    service: Arc<SelectionService>,
    namespace: &str,
    service_name: &str,
    sync_port: u16,
    self_ip: String,
    cancel: CancellationToken,
) -> Result<()> {
    use futures::StreamExt;
    use kube::{Api, Client, runtime::reflector, runtime::watcher};

    let client = Client::try_default()
        .await
        .context("building Kubernetes client for EPP peer discovery")?;
    let slices: Api<EndpointSlice> = Api::namespaced(client, namespace);
    let cfg_watch =
        watcher::Config::default().labels(&format!("{SERVICE_NAME_LABEL}={service_name}"));

    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let reflect = reflector::reflector(writer, watcher(slices, cfg_watch));
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
                    Some(_) => {
                        generation = generation.wrapping_add(1);
                        let _ = changes_tx.send(generation);
                    }
                    None => {
                        tracing::warn!("EPP peer EndpointSlice reflector stream ended");
                        return;
                    }
                },
            }
        }
    });

    let store_for_wait = store.clone();
    match tokio::time::timeout(INITIAL_LIST_TIMEOUT, store_for_wait.wait_until_ready()).await {
        Ok(Ok(())) => tracing::info!("EPP peer EndpointSlice initial LIST sync complete"),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "EPP peer EndpointSlice writer dropped before initial LIST")
        }
        Err(_) => tracing::warn!(
            "EPP peer EndpointSlice initial LIST timed out after {}s; continuing",
            INITIAL_LIST_TIMEOUT.as_secs()
        ),
    }

    tokio::spawn(reconcile_loop(
        service, store, sync_port, self_ip, changes_rx, cancel,
    ));
    Ok(())
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
) {
    let mut known: BTreeSet<String> = BTreeSet::new();
    loop {
        reconcile_once(&service, &store, sync_port, &self_ip, &mut known).await;
        tokio::select! {
            _ = cancel.cancelled() => break,
            changed = changes_rx.changed() => {
                if changed.is_err() {
                    break;
                }
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

/// Live sibling EPP IPs from the current EndpointSlice snapshot, restricted to
/// `self_ip`'s address family and excluding this pod's own IP. Restricting to a
/// single family means a dual-stack sibling — which appears in both an IPv4 and
/// an IPv6 slice — is registered exactly once. Registering both would open two
/// ZMQ connections to the same peer and double-count its load.
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

/// Collect non-terminating endpoint IPs of the requested address family from a
/// set of EndpointSlices. Not-ready endpoints are kept: registering a peer whose
/// ZMQ socket is not up yet is harmless (the sync connect retries), and it avoids
/// missing a sibling that is still starting. Pure function.
fn peer_ips<'a>(
    slices: impl Iterator<Item = &'a EndpointSlice>,
    want_ipv6: bool,
) -> BTreeSet<String> {
    let mut ips = BTreeSet::new();
    for slice in slices {
        if slice.address_type.eq_ignore_ascii_case("IPv6") != want_ipv6 {
            continue;
        }
        for endpoint in &slice.endpoints {
            if endpoint
                .conditions
                .as_ref()
                .and_then(|c| c.terminating)
                .unwrap_or(false)
            {
                continue;
            }
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

    #[test]
    fn peer_ips_keeps_non_terminating() {
        let slices = [slice_with(&["10.0.0.1", "10.0.0.2"], false, "IPv4")];
        let ips = peer_ips(slices.iter(), false);
        assert!(ips.contains("10.0.0.1"));
        assert!(ips.contains("10.0.0.2"));
    }

    #[test]
    fn peer_ips_drops_terminating() {
        let slices = [slice_with(&["10.0.0.9"], true, "IPv4")];
        assert!(peer_ips(slices.iter(), false).is_empty());
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
}
