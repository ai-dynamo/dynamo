// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent PR #13187 stats stream lifecycle for one configured DGD.

use super::projection::{
    StatsCatalog, combined_capacity, project_kvless_frontend, project_load, project_usage,
};
use super::proto;

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::global_view::source::{
    PlaneLease, PoolObservation, PoolObservationAssembler, SourcePlane,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

struct StatsPlanes {
    assembler: Arc<PoolObservationAssembler>,
    capacity: PlaneLease,
    load: PlaneLease,
    usage: PlaneLease,
    catalog: Option<PlaneLease>,
    readiness: Option<PlaneLease>,
}

impl StatsPlanes {
    fn new(assembler: Arc<PoolObservationAssembler>, stats_only: bool) -> Result<Self> {
        let capacity = assembler.open(SourcePlane::Capacity)?;
        let load = match assembler.open(SourcePlane::Load) {
            Ok(lease) => lease,
            Err(error) => {
                assembler.disconnect(capacity);
                return Err(error.into());
            }
        };
        let usage = match assembler.open(SourcePlane::KvUsage) {
            Ok(lease) => lease,
            Err(error) => {
                assembler.disconnect(capacity);
                assembler.disconnect(load);
                return Err(error.into());
            }
        };
        let (catalog, readiness) = if stats_only {
            (
                Some(assembler.open(SourcePlane::Catalog)?),
                Some(assembler.open(SourcePlane::Readiness)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            assembler,
            capacity,
            load,
            usage,
            catalog,
            readiness,
        })
    }

    fn clear(&mut self) -> Result<()> {
        self.capacity = self.assembler.open(SourcePlane::Capacity)?;
        self.load = self.assembler.open(SourcePlane::Load)?;
        self.usage = self.assembler.open(SourcePlane::KvUsage)?;
        if self.catalog.is_some() {
            self.catalog = Some(self.assembler.open(SourcePlane::Catalog)?);
            self.readiness = Some(self.assembler.open(SourcePlane::Readiness)?);
        }
        Ok(())
    }

    fn replace(&mut self, plane: SourcePlane, observation: Option<PoolObservation>) -> Result<()> {
        let lease = self.assembler.open(plane)?;
        let slot = match plane {
            SourcePlane::Capacity => &mut self.capacity,
            SourcePlane::Load => &mut self.load,
            SourcePlane::KvUsage => &mut self.usage,
            SourcePlane::Catalog => self
                .catalog
                .as_mut()
                .context("stats catalog plane is disabled")?,
            SourcePlane::Readiness => self
                .readiness
                .as_mut()
                .context("stats readiness plane is disabled")?,
            SourcePlane::KvOverlap => bail!("invalid stats plane"),
        };
        *slot = lease;
        if let Some(observation) = observation
            && !self.assembler.apply(lease, observation)?
        {
            bail!("stats observation was superseded");
        }
        Ok(())
    }
}

impl Drop for StatsPlanes {
    fn drop(&mut self) {
        self.assembler.disconnect(self.capacity);
        self.assembler.disconnect(self.load);
        self.assembler.disconnect(self.usage);
        if let Some(lease) = self.catalog {
            self.assembler.disconnect(lease);
        }
        if let Some(lease) = self.readiness {
            self.assembler.disconnect(lease);
        }
    }
}

/// Reconnect the separate PR #13187 stats listener. The caller supplies a
/// channel to its deployed proxy; the listener itself binds to loopback.
pub async fn run_stats_view(
    channel: Channel,
    catalog_rx: watch::Receiver<Option<StatsCatalog>>,
    assembler: Arc<PoolObservationAssembler>,
    cancel: CancellationToken,
    stats_only: bool,
) -> Result<()> {
    let mut backoff = Duration::from_millis(500);
    loop {
        let started = Instant::now();
        let result = run_stats_epoch(
            channel.clone(),
            catalog_rx.clone(),
            Arc::clone(&assembler),
            cancel.child_token(),
            stats_only,
        )
        .await;
        if cancel.is_cancelled() {
            return Ok(());
        }
        match result {
            Ok(()) => tracing::warn!("Global View stats epoch ended without cancellation"),
            Err(error) => tracing::warn!(%error, "Global View stats epoch failed; reconnecting"),
        }
        if started.elapsed() >= Duration::from_secs(30) {
            backoff = Duration::from_millis(500);
        }
        let half = backoff / 2;
        let delay =
            half + Duration::from_millis(rand::random::<u64>() % (half.as_millis() as u64 + 1));
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            _ = tokio::time::sleep(delay) => {}
        }
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

/// One pair of stats streams. Opening leases invalidates values left by a
/// previous connection. A catalog revision fences both streams' old values.
pub async fn run_stats_epoch(
    channel: Channel,
    mut catalog_rx: watch::Receiver<Option<StatsCatalog>>,
    assembler: Arc<PoolObservationAssembler>,
    cancel: CancellationToken,
    stats_only: bool,
) -> Result<()> {
    let mut planes = StatsPlanes::new(assembler, stats_only)?;
    let mut usage_client = proto::kv_dc_relay_client::KvDcRelayClient::new(channel.clone());
    let mut load_client = proto::kv_dc_relay_client::KvDcRelayClient::new(channel);
    let mut usage_stream = usage_client.watch_kv_usage(()).await?.into_inner();
    let mut load_stream = load_client.watch_load(()).await?.into_inner();
    let mut catalog = catalog_rx.borrow_and_update().clone();
    let mut usage = None;
    let mut load = None;
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            changed = catalog_rx.changed() => {
                changed.context("Global View catalog publisher closed")?;
                catalog = catalog_rx.borrow_and_update().clone();
                usage = None;
                load = None;
                planes.clear()?;
            }
            update = usage_stream.message() => {
                let snapshot = update?.context("stats KV usage stream closed")?;
                let Some(current) = catalog.as_ref() else { continue; };
                if !current.accepts_metadata(snapshot.metadata.as_ref()) {
                    bail!("stats usage relay identity differs from current catalog");
                }
                usage = project_usage(&snapshot, current, now_unix_ms());
                planes.replace(SourcePlane::KvUsage, usage.as_ref().and_then(|value| value.usage.clone()))?;
                planes.replace(SourcePlane::Capacity, combined_capacity(usage.as_ref(), load.as_ref(), now_unix_ms()))?;
            }
            update = load_stream.message() => {
                let snapshot = update?.context("stats load stream closed")?;
                let Some(current) = catalog.as_ref() else { continue; };
                if !current.accepts_metadata(snapshot.metadata.as_ref()) {
                    bail!("stats load relay identity differs from current catalog");
                }
                load = project_load(&snapshot, current, now_unix_ms());
                planes.replace(SourcePlane::Load, load.as_ref().and_then(|value| value.load.clone()))?;
                planes.replace(SourcePlane::Capacity, combined_capacity(usage.as_ref(), load.as_ref(), now_unix_ms()))?;
                if current.is_kvless() {
                    let projected = project_kvless_frontend(&snapshot, current, now_unix_ms());
                    planes.replace(SourcePlane::Catalog, projected.as_ref().map(|value| value.0.clone()))?;
                    planes.replace(SourcePlane::Readiness, projected.map(|value| value.1))?;
                }
            }
        }
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
