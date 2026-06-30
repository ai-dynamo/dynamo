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
//! wired into the in-process service rather than remote replicas. Built only with
//! the `selector-embedded` feature.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex as StdMutex};
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

/// Backoff before re-establishing the EndpointSlice watch if its stream ends.
const WATCH_RESTART_BACKOFF: Duration = Duration::from_secs(3);

/// Shared handle to the current EndpointSlice reflector store. The watch task
/// swaps in a fresh store when it re-establishes the watch, so the reconcile
/// loop always reads the live snapshot.
type SharedStore = Arc<StdMutex<kube::runtime::reflector::Store<EndpointSlice>>>;

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
    self_ip: Option<String>,
    cancel: CancellationToken,
) -> Result<()> {
    use kube::{Api, Client, runtime::reflector, runtime::watcher};

    let client = Client::try_default()
        .await
        .context("building Kubernetes client for EPP peer discovery")?;
    let slices: Api<EndpointSlice> = Api::namespaced(client, namespace);
    let cfg_watch =
        watcher::Config::default().labels(&format!("{SERVICE_NAME_LABEL}={service_name}"));

    let initial_writer = reflector::store::Writer::default();
    let store: SharedStore = Arc::new(StdMutex::new(initial_writer.as_reader()));
    let (changes_tx, changes_rx) = watch::channel(0u64);

    tracing::info!(
        %namespace,
        service = %service_name,
        sync_port,
        self_ip = ?self_ip,
        "Starting EPP peer EndpointSlice watch (embedded replication)"
    );

    tokio::spawn(watch_loop(
        slices,
        cfg_watch,
        initial_writer,
        store.clone(),
        changes_tx,
        cancel.clone(),
    ));

    let store_for_wait = store.lock().expect("store lock poisoned").clone();
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
    store: SharedStore,
    sync_port: u16,
    self_ip: Option<String>,
    mut changes_rx: watch::Receiver<u64>,
    cancel: CancellationToken,
) {
    let mut known: BTreeSet<String> = BTreeSet::new();
    loop {
        reconcile_once(&service, &store, sync_port, self_ip.as_deref(), &mut known).await;
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
    store: &SharedStore,
    sync_port: u16,
    self_ip: Option<&str>,
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

/// Live sibling EPP IPs from the current EndpointSlice snapshot, excluding this
/// pod's own IP so a replica never registers itself as a peer.
fn live_peer_ips(store: &SharedStore, self_ip: Option<&str>) -> BTreeSet<String> {
    let snapshot = store.lock().expect("store lock poisoned").clone();
    let mut ips = ready_ips(snapshot.state().iter().map(|s| s.as_ref()));
    if let Some(me) = self_ip {
        ips.remove(me);
    }
    ips
}

/// Long-lived EndpointSlice watch. If the reflector stream ends (writer dropped
/// / fatal — transient errors are retried inside the watcher), swap in a fresh
/// store and re-establish after a backoff so discovery never gets stuck on a
/// stale snapshot. Stops when `cancel` fires.
async fn watch_loop(
    slices: kube::Api<EndpointSlice>,
    cfg_watch: kube::runtime::watcher::Config,
    initial_writer: kube::runtime::reflector::store::Writer<EndpointSlice>,
    store: SharedStore,
    changes_tx: watch::Sender<u64>,
    cancel: CancellationToken,
) {
    use futures::StreamExt;
    use kube::runtime::{reflector, watcher};

    let mut writer = Some(initial_writer);
    let mut generation = 0u64;
    loop {
        if cancel.is_cancelled() {
            return;
        }
        // First pass reuses the initial writer (already published to `store`);
        // later passes create and publish a fresh one.
        let writer = match writer.take() {
            Some(w) => w,
            None => {
                let w = reflector::store::Writer::default();
                *store.lock().expect("store lock poisoned") = w.as_reader();
                w
            }
        };

        let reflect = reflector::reflector(writer, watcher(slices.clone(), cfg_watch.clone()));
        tokio::pin!(reflect);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                item = reflect.next() => {
                    if item.is_none() {
                        break;
                    }
                    generation = generation.wrapping_add(1);
                    let _ = changes_tx.send(generation);
                }
            }
        }

        tracing::warn!(
            "EPP peer EndpointSlice reflector stream ended; re-establishing the watch in {}s",
            WATCH_RESTART_BACKOFF.as_secs()
        );
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(WATCH_RESTART_BACKOFF) => {}
        }
        generation = generation.wrapping_add(1);
        let _ = changes_tx.send(generation);
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

/// Collect non-terminating endpoint IPs from a set of EndpointSlices. Not-ready
/// endpoints are kept: registering a peer whose ZMQ socket is not up yet is
/// harmless (the sync connect retries), and it avoids missing a sibling that is
/// still starting. Pure function.
fn ready_ips<'a>(slices: impl Iterator<Item = &'a EndpointSlice>) -> BTreeSet<String> {
    let mut ips = BTreeSet::new();
    for slice in slices {
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

    fn slice_with(ips: &[&str], terminating: bool) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_string(),
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
    fn ready_ips_keeps_non_terminating() {
        let slices = [slice_with(&["10.0.0.1", "10.0.0.2"], false)];
        let ips = ready_ips(slices.iter());
        assert!(ips.contains("10.0.0.1"));
        assert!(ips.contains("10.0.0.2"));
    }

    #[test]
    fn ready_ips_drops_terminating() {
        let slices = [slice_with(&["10.0.0.9"], true)];
        assert!(ready_ips(slices.iter()).is_empty());
    }

    #[test]
    fn authority_brackets_ipv6_only() {
        assert_eq!(authority("10.0.0.1", 9092), "10.0.0.1:9092");
        assert_eq!(authority("fd00::1", 9092), "[fd00::1]:9092");
    }
}
