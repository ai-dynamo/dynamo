// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One Relay connection epoch for a configured DGD's catalog, readiness, and CKF.
//! The caller owns reconnect/backoff and the deployed Global Router lifecycle.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::global_view::source::{PlaneLease, PoolObservationAssembler, SourcePlane};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use super::scorer::RelayCkfOverlapStore;
use super::stats::{StatsCatalog, run_stats_view};
use super::stream::run_exact_aggregated_producer;
use crate::global_view::{RelayPoolScope, project_catalog, project_readiness};
use crate::kv_dc_relay::wan::grpc::protocol::{
    self as wire, KvPoolDescriptor, RelayIdentity, SubscribeServingReadinessRequest,
    WatchKvPoolCatalogRequest,
};

struct ActiveCkf {
    descriptor: KvPoolDescriptor,
    generation: u64,
    handle: JoinHandle<()>,
}

struct Epoch {
    assembler: Arc<PoolObservationAssembler>,
    catalog_lease: PlaneLease,
    readiness_lease: PlaneLease,
    active_ckf: Option<ActiveCkf>,
    stats_catalog_tx: Option<watch::Sender<Option<StatsCatalog>>>,
}

impl Epoch {
    fn new(
        assembler: Arc<PoolObservationAssembler>,
        stats_catalog_tx: Option<watch::Sender<Option<StatsCatalog>>>,
    ) -> Result<Self> {
        let catalog_lease = assembler.open(SourcePlane::Catalog)?;
        let readiness_lease = match assembler.open(SourcePlane::Readiness) {
            Ok(lease) => lease,
            Err(error) => {
                assembler.disconnect(catalog_lease);
                return Err(error.into());
            }
        };
        Ok(Self {
            assembler,
            catalog_lease,
            readiness_lease,
            active_ckf: None,
            stats_catalog_tx,
        })
    }

    fn stop_ckf(&mut self) {
        if let Some(active) = self.active_ckf.take() {
            active.handle.abort();
        }
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        self.stop_ckf();
        if let Some(tx) = &self.stats_catalog_tx {
            tx.send_replace(None);
        }
        self.assembler.disconnect(self.catalog_lease);
        self.assembler.disconnect(self.readiness_lease);
    }
}

/// Keep the DGD view connected across stream failures. Each retry starts a
/// fresh catalog/readiness epoch and can only score a newly validated CKF
/// producer. The delay uses bounded equal jitter across router replicas.
// Each source parameter owns a distinct connection or lifecycle boundary.
#[allow(clippy::too_many_arguments)]
async fn run_relay_view_publishing(
    channel: Channel,
    scope: RelayPoolScope,
    model: String,
    subscriber_id: String,
    assembler: Arc<PoolObservationAssembler>,
    store: Arc<RelayCkfOverlapStore>,
    cancel: CancellationToken,
    stats_catalog_tx: Option<watch::Sender<Option<StatsCatalog>>>,
) -> Result<()> {
    if model.trim().is_empty()
        || scope.runtime_namespace.trim().is_empty()
        || scope.frontend_endpoint.trim().is_empty()
    {
        bail!("Global View model and relay scope must be non-empty");
    }
    if subscriber_id.is_empty()
        || subscriber_id.len() > 118
        || subscriber_id.chars().any(char::is_control)
    {
        bail!("Global View subscriber ID is invalid or too long");
    }
    let mut backoff = Duration::from_millis(500);
    loop {
        let started = Instant::now();
        let result = run_relay_view_epoch_publishing(
            channel.clone(),
            scope.clone(),
            model.clone(),
            subscriber_id.clone(),
            Arc::clone(&assembler),
            Arc::clone(&store),
            cancel.child_token(),
            stats_catalog_tx.clone(),
        )
        .await;
        if cancel.is_cancelled() {
            return Ok(());
        }
        match result {
            Ok(()) => tracing::warn!("Relay Global View epoch ended without cancellation"),
            Err(error) => tracing::warn!(%error, "Relay Global View epoch failed; reconnecting"),
        }
        if started.elapsed() >= Duration::from_secs(30) {
            backoff = Duration::from_millis(500);
        }
        let half = backoff / 2;
        let jitter_ms = rand::random::<u64>() % (half.as_millis() as u64 + 1);
        let delay = half + Duration::from_millis(jitter_ms);
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            _ = tokio::time::sleep(delay) => {}
        }
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

/// Consume the catalog/CKF Relay and the separate PR #13187 stats listener
/// against one DGD assembler. Deployment supplies the stats proxy channel.
// Each source parameter owns a distinct connection or lifecycle boundary.
#[allow(clippy::too_many_arguments)]
pub async fn run_relay_view_with_stats(
    relay_channel: Channel,
    stats_channel: Channel,
    scope: RelayPoolScope,
    model: String,
    subscriber_id: String,
    assembler: Arc<PoolObservationAssembler>,
    store: Arc<RelayCkfOverlapStore>,
    cancel: CancellationToken,
) -> Result<()> {
    let (catalog_tx, catalog_rx) = watch::channel(None);
    tokio::try_join!(
        run_relay_view_publishing(
            relay_channel,
            scope,
            model,
            subscriber_id,
            Arc::clone(&assembler),
            store,
            cancel.child_token(),
            Some(catalog_tx),
        ),
        run_stats_view(stats_channel, catalog_rx, assembler, cancel.child_token()),
    )?;
    Ok(())
}

/// Existing Relay-only entry point for deployments without the stats proxy.
pub async fn run_relay_view(
    channel: Channel,
    scope: RelayPoolScope,
    model: String,
    subscriber_id: String,
    assembler: Arc<PoolObservationAssembler>,
    store: Arc<RelayCkfOverlapStore>,
    cancel: CancellationToken,
) -> Result<()> {
    run_relay_view_publishing(
        channel,
        scope,
        model,
        subscriber_id,
        assembler,
        store,
        cancel,
        None,
    )
    .await
}

/// Watch complete Relay snapshots until cancellation or any stream failure.
/// A failed/closed CKF stream also ends this connection epoch so the caller can
/// reconnect and fetch a new catalog before subscribing to another generation.
/// All three observations are invalidated on exit, including task cancellation.
// Each source parameter owns a distinct connection or lifecycle boundary.
#[allow(clippy::too_many_arguments)]
async fn run_relay_view_epoch_publishing(
    channel: Channel,
    scope: RelayPoolScope,
    model: String,
    subscriber_id: String,
    assembler: Arc<PoolObservationAssembler>,
    store: Arc<RelayCkfOverlapStore>,
    cancel: CancellationToken,
    stats_catalog_tx: Option<watch::Sender<Option<StatsCatalog>>>,
) -> Result<()> {
    if model.trim().is_empty() {
        bail!("Global View model must be non-empty");
    }
    let mut epoch = Epoch::new(assembler, stats_catalog_tx)?;
    let mut catalog_client = wire::KvEventRelayClient::new(channel.clone());
    let mut readiness_client = wire::KvEventRelayClient::new(channel.clone());
    let mut catalog_stream = catalog_client
        .watch_kv_pool_catalog(WatchKvPoolCatalogRequest {
            subscriber_id: format!("{subscriber_id}/catalog"),
            contract_marker: wire::RELAY_CONTRACT_MARKER,
        })
        .await?
        .into_inner();
    let mut readiness_stream = readiness_client
        .subscribe_serving_readiness(SubscribeServingReadinessRequest {
            subscriber_id: format!("{subscriber_id}/readiness"),
            contract_marker: wire::RELAY_CONTRACT_MARKER,
        })
        .await?
        .into_inner();
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<(u64, Result<()>)>();
    let mut relay_key = None;
    let mut last_catalog_revision = None;
    let mut last_catalog_snapshot = None;
    let mut last_readiness_revision = None;
    let mut last_readiness_entries = None;
    let mut next_ckf_generation = 0u64;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            update = catalog_stream.message() => {
                let update = update?.context("Relay catalog stream closed")?;
                check_relay(&mut relay_key, update.relay.as_ref())?;
                if last_catalog_revision.is_some_and(|last| update.revision < last) {
                    bail!("Relay catalog revision moved backwards");
                }
                let projection = project_catalog(&update, &scope, now_unix_ms())?;
                if last_catalog_revision == Some(update.revision) {
                    if last_catalog_snapshot.as_ref() != update.snapshot.as_ref() {
                        bail!("Relay catalog changed without advancing its revision");
                    }
                    if !epoch.assembler.apply(epoch.catalog_lease, projection.observation)? {
                        bail!("Relay catalog heartbeat was superseded");
                    }
                    continue;
                }
                last_catalog_revision = Some(update.revision);
                last_catalog_snapshot = update.snapshot.clone();
                let selected = projection.sole_aggregated_overlap_producer(&model).cloned();
                let stats_catalog = relay_key.and_then(|relay| {
                    StatsCatalog::from_projection(&projection, &model, relay)
                });
                if !epoch.assembler.apply(epoch.catalog_lease, projection.observation.clone())? {
                    bail!("Relay catalog observation was superseded");
                }
                if let Some(tx) = &epoch.stats_catalog_tx {
                    tx.send_replace(stats_catalog);
                }
                if epoch.active_ckf.as_ref().map(|active| &active.descriptor) == selected.as_ref() {
                    continue;
                }
                epoch.stop_ckf();
                if let Some(descriptor) = selected {
                    next_ckf_generation = next_ckf_generation
                        .checked_add(1)
                        .context("CKF coordinator generation exhausted")?;
                    let generation = next_ckf_generation;
                    let task_client = wire::KvEventRelayClient::new(channel.clone());
                    let task_model = model.clone();
                    let task_subscriber_id = format!("{subscriber_id}/ckf");
                    let task_store = Arc::clone(&store);
                    let task_assembler = Arc::clone(&epoch.assembler);
                    let task_pool_id = epoch.assembler.pool_id();
                    let task_cancel = cancel.child_token();
                    let task_done_tx = done_tx.clone();
                    let handle = tokio::spawn(async move {
                        let result = run_exact_aggregated_producer(
                            task_client,
                            task_pool_id,
                            task_model,
                            projection,
                            task_subscriber_id,
                            task_store,
                            task_assembler,
                            task_cancel,
                        ).await;
                        let _ = task_done_tx.send((generation, result));
                    });
                    epoch.active_ckf = Some(ActiveCkf { descriptor, generation, handle });
                }
            }
            update = readiness_stream.message() => {
                let update = update?.context("Relay readiness stream closed")?;
                check_relay(&mut relay_key, update.relay.as_ref())?;
                if last_readiness_revision.is_some_and(|last| update.revision < last) {
                    bail!("Relay readiness revision moved backwards");
                }
                let observation = project_readiness(&update, &scope, now_unix_ms())?;
                if last_readiness_revision == Some(update.revision) {
                    if last_readiness_entries.as_ref() != Some(&update.entries) {
                        bail!("Relay readiness changed without advancing its revision");
                    }
                } else {
                    last_readiness_revision = Some(update.revision);
                    last_readiness_entries = Some(update.entries.clone());
                }
                if !epoch.assembler.apply(epoch.readiness_lease, observation)? {
                    bail!("Relay readiness observation was superseded");
                }
            }
            Some((generation, result)) = done_rx.recv() => {
                if epoch.active_ckf.as_ref().is_some_and(|active| active.generation == generation) {
                    result.context("Relay CKF stream failed")?;
                    bail!("Relay CKF stream ended before its connection epoch");
                }
            }
        }
    }
}

pub async fn run_relay_view_epoch(
    channel: Channel,
    scope: RelayPoolScope,
    model: String,
    subscriber_id: String,
    assembler: Arc<PoolObservationAssembler>,
    store: Arc<RelayCkfOverlapStore>,
    cancel: CancellationToken,
) -> Result<()> {
    run_relay_view_epoch_publishing(
        channel,
        scope,
        model,
        subscriber_id,
        assembler,
        store,
        cancel,
        None,
    )
    .await
}

fn check_relay(expected: &mut Option<(u64, u64)>, relay: Option<&RelayIdentity>) -> Result<()> {
    let relay = relay.context("Relay update is missing relay identity")?;
    let actual = (relay.drt_instance_id, relay.relay_incarnation);
    if expected.is_some_and(|value| value != actual) {
        bail!("Relay incarnation changed within a connection epoch");
    }
    *expected = Some(actual);
    Ok(())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
