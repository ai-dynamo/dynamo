// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process Global View owner for one Global Router replica.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "global-view-diagnostics")]
use axum::Router;
#[cfg(feature = "global-view-diagnostics")]
use dynamo_kv_router::global_view::http::global_view_diagnostics_router;
use dynamo_kv_router::global_view::overlap::KvOverlapScorer;
use dynamo_kv_router::global_view::source::PoolObservationAssembler;
#[cfg(any(test, feature = "global-view-diagnostics"))]
use dynamo_kv_router::global_view::state::FreshnessPolicy;
use dynamo_kv_router::global_view::state::{
    InMemoryPoolStateRepository, PoolLocation, PoolStateRepository, PoolStateSink,
};
use dynamo_kv_router::global_view::{PoolKey, V1PoolIdDeriver};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use super::coordinator::run_relay_view_with_stats;
use super::scorer::RelayCkfOverlapStore;
use crate::global_view::RelayPoolScope;

/// The deployment resolves channels to the Relay WAN and stats services.
#[derive(Clone)]
pub struct RelayDgdSource {
    pub key: PoolKey,
    pub location: PoolLocation,
    pub scope: RelayPoolScope,
    pub model: String,
    pub subscriber_id: String,
    pub relay_channel: Channel,
    pub stats_channel: Channel,
}

struct DgdTask {
    source: RelayDgdSource,
    assembler: Arc<PoolObservationAssembler>,
}

/// Repository and subscriber lifecycle owned by the Global Router process.
/// Cloned repository/scorer handles are used directly on its request path.
pub struct GlobalViewRuntime {
    repository: Arc<InMemoryPoolStateRepository>,
    overlap: Arc<RelayCkfOverlapStore>,
    dgds: Vec<DgdTask>,
}

impl GlobalViewRuntime {
    pub fn new(sources: Vec<RelayDgdSource>, overlap_max_age: Duration) -> Result<Self> {
        if sources.is_empty() || overlap_max_age.is_zero() {
            bail!("Global View needs at least one DGD and a positive overlap age limit");
        }
        let repository = Arc::new(InMemoryPoolStateRepository::default());
        let sink: Arc<dyn PoolStateSink> = repository.clone();
        let overlap = Arc::new(RelayCkfOverlapStore::new(overlap_max_age));
        let mut seen = HashSet::new();
        let mut dgds = Vec::with_capacity(sources.len());
        for source in sources {
            if source.model.trim().is_empty()
                || source.scope.runtime_namespace.trim().is_empty()
                || source.scope.frontend_endpoint.trim().is_empty()
                || source.subscriber_id.is_empty()
                || source.subscriber_id.len() > 118
                || source.subscriber_id.chars().any(char::is_control)
            {
                bail!("Global View DGD source has an empty model, scope, or subscriber ID");
            }
            let assembler = Arc::new(PoolObservationAssembler::new(
                &source.key,
                source.location.clone(),
                &V1PoolIdDeriver,
                sink.clone(),
            ));
            if !seen.insert(assembler.pool_id()) {
                bail!("Global View has duplicate routing PoolIds");
            }
            dgds.push(DgdTask { source, assembler });
        }
        Ok(Self {
            repository,
            overlap,
            dgds,
        })
    }

    pub fn repository(&self) -> Arc<dyn PoolStateRepository> {
        self.repository.clone()
    }

    pub fn overlap_scorer(&self) -> Arc<dyn KvOverlapScorer> {
        self.overlap.clone()
    }

    /// Mount on the router's diagnostic listener. Request routing uses the
    /// repository and scorer handles without a local HTTP hop.
    #[cfg(feature = "global-view-diagnostics")]
    pub fn diagnostics_router(&self, freshness: FreshnessPolicy) -> Router {
        global_view_diagnostics_router(self.repository(), freshness, self.overlap_scorer())
    }

    /// Run every configured DGD until shutdown. If a source task exits in an
    /// unexpected way, stop the remaining tasks so its view is not partial.
    pub async fn run(&self, cancel: CancellationToken) -> Result<()> {
        let mut tasks = JoinSet::new();
        for dgd in &self.dgds {
            let source = dgd.source.clone();
            let assembler = Arc::clone(&dgd.assembler);
            let overlap = Arc::clone(&self.overlap);
            let child_cancel = cancel.child_token();
            tasks.spawn(async move {
                run_relay_view_with_stats(
                    source.relay_channel,
                    source.stats_channel,
                    source.scope,
                    source.model,
                    source.subscriber_id,
                    assembler,
                    overlap,
                    child_cancel,
                )
                .await
            });
        }
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => Ok(()),
            completed = tasks.join_next() => {
                match completed {
                    Some(Ok(Ok(()))) => Err(anyhow!("Global View source exited before shutdown")),
                    Some(Ok(Err(error))) => Err(error).context("Global View source failed"),
                    Some(Err(error)) => Err(error).context("Global View source task failed"),
                    None => Err(anyhow!("Global View has no running sources")),
                }
            }
        };
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(site: &str) -> RelayDgdSource {
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        RelayDgdSource {
            key: PoolKey::new(site, "dynamo", "mocker").unwrap(),
            location: PoolLocation {
                region: site.into(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            },
            scope: RelayPoolScope {
                runtime_namespace: format!("{site}-mocker"),
                frontend_endpoint: "mocker.frontend.generate".into(),
            },
            model: "model".into(),
            subscriber_id: format!("global-router-{site}"),
            relay_channel: channel.clone(),
            stats_channel: channel,
        }
    }

    #[tokio::test]
    async fn one_router_replica_owns_distinct_dgd_pools() {
        let runtime = GlobalViewRuntime::new(
            vec![source("ohio"), source("west")],
            Duration::from_secs(10),
        )
        .unwrap();
        let pools = runtime.repository().list(
            0,
            &FreshnessPolicy {
                catalog_max_age_ms: 10,
                readiness_max_age_ms: 10,
                capacity_max_age_ms: 10,
                load_max_age_ms: 10,
                kv_usage_max_age_ms: 10,
                kv_overlap_max_age_ms: 10,
            },
        );
        assert_eq!(pools.len(), 2);
        assert_ne!(pools[0].pool_id, pools[1].pool_id);
        assert_eq!(pools[0].descriptors.models.len(), 0);
    }

    #[tokio::test]
    async fn duplicate_dgd_id_is_rejected() {
        assert!(
            GlobalViewRuntime::new(
                vec![source("ohio"), source("ohio")],
                Duration::from_secs(10),
            )
            .is_err()
        );
    }
}
