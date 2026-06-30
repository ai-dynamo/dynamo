// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP selector fleet for the standalone selection service
//! (`python -m dynamo.select_service`).
//!
//! Production/replicated path: the EPP delegates worker selection to a fleet of
//! selection-service replicas fronted by a Kubernetes `Service`. Rather than a
//! static list of URLs, the fleet **watches the Service's EndpointSlices** and
//! keeps an in-memory set of replica endpoints in sync as replicas come and go.
//!
//! Because each replica owns its own catalog and KV index, the fleet reconciles
//! every live replica independently ([`SelectionBackend::reconcile`]): it reads
//! each replica's actual catalog (`GET /workers`), applies the diff toward the
//! desired set (`POST`/`DELETE /workers`), and marks the replica routable only
//! once it reports `GET /ready`. New (or restarted) replicas are therefore
//! bootstrapped from scratch before they receive any selection traffic.
//!
//! As replicas appear and disappear the fleet also wires the selectors'
//! replica-sync peer mesh (`POST /replica_sync/(de)register_peer`) so
//! active-load and admission events propagate across the fleet.
//!
//! The wire types live in [`crate::selection_backend`]; this module only adds
//! the HTTP transport, the EndpointSlice watch, and a [`SelectionBackend`]
//! implementation. Built only with the `selector-http` feature.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use serde::Deserialize;
use tokio::sync::{Mutex, watch};

use crate::epp_config::EppConfig;
use crate::selection_backend::{
    SelectRequest, SelectResponse, SelectionBackend, WorkerRegistration,
};

/// Label Kubernetes sets on every EndpointSlice pointing back to its Service.
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

/// Per-request timeout for selector HTTP calls. Keeps a dead/slow replica from
/// stalling reconcile or a selection.
const HTTP_TIMEOUT: Duration = Duration::from_secs(5);

/// Best-effort bound on how long fleet startup waits for the initial
/// EndpointSlice LIST before serving anyway.
const INITIAL_LIST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Deserialize)]
struct ReadyResponse {
    #[serde(default)]
    ready: bool,
}

/// Subset of the selector's `/workers` catalog record we compare against to
/// decide whether a replica already has an up-to-date registration. Unknown
/// fields are ignored by serde.
#[derive(Debug, Clone, Deserialize)]
struct CatalogRecordWire {
    worker_id: u64,
    #[serde(default)]
    model_name: String,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    kv_events_endpoints: HashMap<u32, String>,
    #[serde(default)]
    replay_endpoint: Option<String>,
    #[serde(default)]
    block_size: Option<u32>,
    #[serde(default)]
    data_parallel_size: Option<u32>,
    #[serde(default)]
    max_num_batched_tokens: Option<u64>,
    #[serde(default)]
    total_kv_blocks: Option<u64>,
    #[serde(default)]
    stable_routing_id: Option<String>,
}

/// EPP-side state for one selector replica.
#[derive(Debug, Clone, Default)]
struct ReplicaState {
    /// Last `GET /ready` result; gates selection traffic.
    routable: bool,
}

/// HTTP-backed [`SelectionBackend`] over a dynamically-discovered fleet of
/// selection-service replicas.
pub struct SelectorFleet {
    http: reqwest::Client,
    store: kube::runtime::reflector::Store<EndpointSlice>,
    http_port: u16,
    replica_sync_port: u16,
    /// Tracked replicas keyed by pod IP. `BTreeMap` keeps selection targeting
    /// deterministic.
    replicas: Mutex<BTreeMap<String, ReplicaState>>,
    changes_rx: watch::Receiver<u64>,
}

impl SelectorFleet {
    /// Start the EndpointSlice watch for the configured selector Service and
    /// return a fleet ready to reconcile. Fails if the config was not built for
    /// HTTP mode (no service name).
    pub async fn spawn(cfg: &EppConfig) -> Result<Self> {
        use futures::StreamExt;
        use kube::{Api, Client, runtime::reflector, runtime::watcher};

        if cfg.selector_service.is_empty() {
            bail!("SelectorFleet requires DYN_EPP_SELECTOR_SERVICE (selection-service Service name)");
        }

        let client = Client::try_default()
            .await
            .context("building Kubernetes client for selector discovery")?;
        let namespace = match &cfg.selector_service_namespace {
            Some(ns) => ns.clone(),
            None => std::env::var("POD_NAMESPACE").map_err(|_| {
                anyhow::anyhow!(
                    "DYN_EPP_SELECTOR_SERVICE_NAMESPACE is unset and POD_NAMESPACE is not \
                     available. Ensure the EPP pod spec injects POD_NAMESPACE via the downward \
                     API (fieldRef metadata.namespace), or set DYN_EPP_SELECTOR_SERVICE_NAMESPACE."
                )
            })?,
        };

        let http = reqwest::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .context("building selection-service HTTP client")?;

        // Watch only the EndpointSlices owned by the selector Service.
        let slices: Api<EndpointSlice> = Api::namespaced(client, &namespace);
        let cfg_watch = watcher::Config::default()
            .labels(&format!("{SERVICE_NAME_LABEL}={}", cfg.selector_service));
        let writer = reflector::store::Writer::default();
        let store = writer.as_reader();
        let reflect = reflector::reflector(writer, watcher(slices, cfg_watch));

        let (changes_tx, changes_rx) = watch::channel(0u64);

        tracing::info!(
            namespace = %namespace,
            service = %cfg.selector_service,
            http_port = cfg.selector_http_port,
            replica_sync_port = cfg.selector_replica_sync_port,
            "Starting selector EndpointSlice watch for router-only HTTP mode"
        );

        tokio::spawn(async move {
            tokio::pin!(reflect);
            let mut generation = 0u64;
            while reflect.next().await.is_some() {
                generation = generation.wrapping_add(1);
                let _ = changes_tx.send(generation);
            }
            tracing::warn!("Selector EndpointSlice reflector stream ended unexpectedly");
        });

        let store_for_wait = store.clone();
        match tokio::time::timeout(INITIAL_LIST_TIMEOUT, store_for_wait.wait_until_ready()).await {
            Ok(Ok(())) => tracing::info!("Selector EndpointSlice initial LIST sync complete"),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Selector EndpointSlice writer dropped before initial LIST")
            }
            Err(_) => tracing::warn!(
                "Selector EndpointSlice initial LIST timed out after {}s; continuing",
                INITIAL_LIST_TIMEOUT.as_secs()
            ),
        }

        Ok(Self {
            http,
            store,
            http_port: cfg.selector_http_port,
            replica_sync_port: cfg.selector_replica_sync_port,
            replicas: Mutex::new(BTreeMap::new()),
            changes_rx,
        })
    }

    /// Live selector replica IPs from the current EndpointSlice snapshot.
    fn live_ips(&self) -> BTreeSet<String> {
        ready_ips(self.store.state().iter().map(|s| s.as_ref()))
    }

    fn http_base(&self, ip: &str) -> String {
        format!("http://{ip}:{}", self.http_port)
    }

    fn zmq_endpoint(&self, ip: &str) -> String {
        format!("tcp://{ip}:{}", self.replica_sync_port)
    }

    /// Reconcile the replica set against the live EndpointSlices: track new
    /// replicas, drop gone ones, and wire the replica-sync peer mesh for the
    /// delta. Returns the full current replica IP set.
    async fn sync_replica_set(&self) -> Vec<String> {
        let live = self.live_ips();
        let (added, removed, all): (Vec<String>, Vec<String>, Vec<String>) = {
            let mut replicas = self.replicas.lock().await;
            let added: Vec<String> = live
                .iter()
                .filter(|ip| !replicas.contains_key(*ip))
                .cloned()
                .collect();
            let removed: Vec<String> = replicas
                .keys()
                .filter(|ip| !live.contains(*ip))
                .cloned()
                .collect();
            for ip in &removed {
                replicas.remove(ip);
            }
            for ip in &added {
                replicas.insert(ip.clone(), ReplicaState::default());
            }
            let all: Vec<String> = replicas.keys().cloned().collect();
            (added, removed, all)
        };

        if !added.is_empty() || !removed.is_empty() {
            tracing::info!(
                added = ?added,
                removed = ?removed,
                total = all.len(),
                "Selector replica set changed"
            );
        }

        // Wire the peer mesh for the delta (registration is idempotent).
        for new_ip in &added {
            let new_zmq = self.zmq_endpoint(new_ip);
            for other_ip in &all {
                if other_ip == new_ip {
                    continue;
                }
                self.register_peer(other_ip, &new_zmq).await;
                self.register_peer(new_ip, &self.zmq_endpoint(other_ip)).await;
            }
        }
        for gone_ip in &removed {
            let gone_zmq = self.zmq_endpoint(gone_ip);
            for other_ip in &all {
                self.deregister_peer(other_ip, &gone_zmq).await;
            }
        }

        all
    }

    /// Drive one replica's catalog toward `desired` and return its routability.
    async fn reconcile_replica(
        &self,
        ip: &str,
        desired: &HashMap<u64, WorkerRegistration>,
    ) -> Result<bool> {
        let base = self.http_base(ip);

        let actual = self.get_workers(&base).await?;

        // Upsert new or changed workers (POST is an upsert, so it covers both).
        for (worker_id, reg) in desired {
            let in_sync = actual
                .get(worker_id)
                .map(|rec| record_matches(reg, rec))
                .unwrap_or(false);
            if in_sync {
                continue;
            }
            let resp = self
                .http
                .post(format!("{base}/workers"))
                .json(reg)
                .send()
                .await
                .with_context(|| format!("POST /workers to {base}"))?;
            ensure_success(resp, "POST /workers", &base).await?;
        }

        // Delete workers this replica has that are no longer desired.
        for worker_id in actual.keys() {
            if desired.contains_key(worker_id) {
                continue;
            }
            let resp = self
                .http
                .delete(format!("{base}/workers/{worker_id}"))
                .send()
                .await
                .with_context(|| format!("DELETE /workers/{worker_id} to {base}"))?;
            if resp.status() != reqwest::StatusCode::NOT_FOUND {
                ensure_success(resp, "DELETE /workers/{id}", &base).await?;
            }
        }

        Ok(self.ready(&base).await)
    }

    async fn get_workers(&self, base: &str) -> Result<HashMap<u64, CatalogRecordWire>> {
        let resp = self
            .http
            .get(format!("{base}/workers"))
            .send()
            .await
            .with_context(|| format!("GET /workers from {base}"))?;
        let resp = ensure_success(resp, "GET /workers", base).await?;
        let records: Vec<CatalogRecordWire> =
            resp.json().await.context("decoding /workers response")?;
        Ok(records.into_iter().map(|r| (r.worker_id, r)).collect())
    }

    async fn ready(&self, base: &str) -> bool {
        let Ok(resp) = self.http.get(format!("{base}/ready")).send().await else {
            return false;
        };
        if !resp.status().is_success() {
            return false;
        }
        resp.json::<ReadyResponse>()
            .await
            .map(|r| r.ready)
            .unwrap_or(false)
    }

    async fn register_peer(&self, ip: &str, peer_endpoint: &str) {
        self.peer_op(ip, "register_peer", peer_endpoint).await;
    }

    async fn deregister_peer(&self, ip: &str, peer_endpoint: &str) {
        self.peer_op(ip, "deregister_peer", peer_endpoint).await;
    }

    async fn peer_op(&self, ip: &str, op: &str, peer_endpoint: &str) {
        let base = self.http_base(ip);
        let result = self
            .http
            .post(format!("{base}/replica_sync/{op}"))
            .json(&serde_json::json!({ "endpoint": peer_endpoint }))
            .send()
            .await;
        match result {
            // Replica-sync disabled (single-replica selector) responds 409; not
            // an error we can act on, so stay quiet at debug level.
            Ok(resp) if resp.status() == reqwest::StatusCode::CONFLICT => {}
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => tracing::debug!(
                %base, op, peer_endpoint, status = %resp.status(),
                "replica-sync peer op returned non-success"
            ),
            Err(e) => tracing::debug!(%base, op, peer_endpoint, error = %e, "replica-sync peer op failed"),
        }
    }
}

#[tonic::async_trait]
impl SelectionBackend for SelectorFleet {
    async fn reconcile(&self, desired: &HashMap<u64, WorkerRegistration>) -> Result<()> {
        let all = self.sync_replica_set().await;
        if all.is_empty() {
            tracing::debug!("No selector replicas discovered yet; nothing to reconcile");
            return Ok(());
        }

        let mut replicas = self.replicas.lock().await;
        for ip in &all {
            match self.reconcile_replica(ip, desired).await {
                Ok(routable) => {
                    if let Some(state) = replicas.get_mut(ip) {
                        state.routable = routable;
                    }
                }
                Err(e) => {
                    tracing::warn!(replica = %ip, error = %e, "Failed to reconcile selector replica; marking not routable");
                    if let Some(state) = replicas.get_mut(ip) {
                        state.routable = false;
                    }
                }
            }
        }
        Ok(())
    }

    async fn select(&self, req: &SelectRequest) -> Result<SelectResponse> {
        let base = {
            let replicas = self.replicas.lock().await;
            replicas
                .iter()
                .find(|(_, s)| s.routable)
                .map(|(ip, _)| self.http_base(ip))
        };
        let Some(base) = base else {
            bail!("no routable selector replica available");
        };
        let resp = self
            .http
            .post(format!("{base}/select"))
            .json(req)
            .send()
            .await
            .with_context(|| format!("POST /select to {base}"))?;
        let resp = ensure_success(resp, "POST /select", &base).await?;
        resp.json::<SelectResponse>()
            .await
            .context("decoding /select response")
    }

    async fn any_ready(&self) -> bool {
        self.replicas.lock().await.values().any(|s| s.routable)
    }

    fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes_rx.clone()
    }
}

/// Collect the routable replica IPs from a set of EndpointSlices. Endpoints that
/// are terminating are excluded; not-ready endpoints are kept so the fleet can
/// bootstrap them before they start reporting ready. Pure function.
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

/// Whether a selector's catalog record already matches the desired registration
/// for the fields router-only mode owns. Pure function — unit-testable.
fn record_matches(desired: &WorkerRegistration, actual: &CatalogRecordWire) -> bool {
    actual.model_name == desired.model_name
        && actual.endpoint.as_deref() == Some(desired.endpoint.as_str())
        && actual.block_size == Some(desired.block_size)
        && actual.data_parallel_size == Some(desired.data_parallel_size)
        && actual.kv_events_endpoints == desired.kv_events_endpoints
        && actual.replay_endpoint == desired.replay_endpoint
        && actual.total_kv_blocks == desired.total_kv_blocks
        && actual.max_num_batched_tokens == desired.max_num_batched_tokens
        && actual.stable_routing_id == desired.stable_routing_id
}

async fn ensure_success(resp: reqwest::Response, op: &str, url: &str) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    bail!("{op} to {url} failed with status {status}: {body}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};

    fn reg(ip: &str) -> WorkerRegistration {
        let mut kv = HashMap::new();
        kv.insert(0u32, format!("tcp://{ip}:5557"));
        WorkerRegistration {
            worker_id: 1,
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            endpoint: format!("http://{ip}:8000"),
            block_size: 16,
            data_parallel_size: 1,
            kv_events_endpoints: kv,
            replay_endpoint: None,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: None,
            stable_routing_id: Some("vllm-0".to_string()),
        }
    }

    fn record_from(reg: &WorkerRegistration) -> CatalogRecordWire {
        CatalogRecordWire {
            worker_id: reg.worker_id,
            model_name: reg.model_name.clone(),
            endpoint: Some(reg.endpoint.clone()),
            kv_events_endpoints: reg.kv_events_endpoints.clone(),
            replay_endpoint: reg.replay_endpoint.clone(),
            block_size: Some(reg.block_size),
            data_parallel_size: Some(reg.data_parallel_size),
            max_num_batched_tokens: reg.max_num_batched_tokens,
            total_kv_blocks: reg.total_kv_blocks,
            stable_routing_id: reg.stable_routing_id.clone(),
        }
    }

    #[test]
    fn record_matches_identical() {
        let r = reg("10.0.0.1");
        assert!(record_matches(&r, &record_from(&r)));
    }

    #[test]
    fn record_mismatch_on_endpoint() {
        let r = reg("10.0.0.1");
        let mut actual = record_from(&r);
        actual.endpoint = Some("http://10.0.0.9:8000".to_string());
        assert!(!record_matches(&r, &actual));
    }

    #[test]
    fn record_mismatch_when_missing_metadata() {
        let r = reg("10.0.0.1");
        let mut actual = record_from(&r);
        actual.block_size = None;
        assert!(!record_matches(&r, &actual));
    }

    fn slice(endpoints: Vec<Endpoint>) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_string(),
            endpoints,
            metadata: Default::default(),
            ports: None,
        }
    }

    fn endpoint(addr: &str, terminating: Option<bool>) -> Endpoint {
        Endpoint {
            addresses: vec![addr.to_string()],
            conditions: Some(EndpointConditions {
                ready: Some(true),
                serving: Some(true),
                terminating,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ready_ips_collects_non_terminating() {
        let s = slice(vec![
            endpoint("10.0.0.1", None),
            endpoint("10.0.0.2", Some(false)),
            endpoint("10.0.0.3", Some(true)), // terminating -> excluded
        ]);
        let ips = ready_ips([&s].into_iter());
        assert_eq!(
            ips,
            BTreeSet::from(["10.0.0.1".to_string(), "10.0.0.2".to_string()])
        );
    }
}
