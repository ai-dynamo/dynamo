// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_runtime::component::Component;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::super::identity::{DcPoolCatalog, DcRelayIdentity};
use super::super::load::PoolLoadSnapshot;
use super::super::pool_registry::PoolRegistry;
use super::super::publication_hub::{PublicationHubError, PublicationHubSubscription};
use super::super::topology::{TopologyPublisher, TopologySnapshot};

#[derive(Clone)]
pub(crate) struct WanPublicationSource {
    component: Component,
    pools: Arc<PoolRegistry>,
    topology: Arc<TopologyPublisher>,
    relay_identity: DcRelayIdentity,
    lifecycle: CancellationToken,
}

impl WanPublicationSource {
    pub(crate) fn new(
        component: Component,
        pools: Arc<PoolRegistry>,
        topology: Arc<TopologyPublisher>,
        relay_identity: DcRelayIdentity,
        lifecycle: CancellationToken,
    ) -> Self {
        Self {
            component,
            pools,
            topology,
            relay_identity,
            lifecycle,
        }
    }

    pub(crate) fn component(&self) -> &Component {
        &self.component
    }

    pub(crate) const fn relay_identity(&self) -> DcRelayIdentity {
        self.relay_identity
    }

    pub(crate) fn lifecycle(&self) -> &CancellationToken {
        &self.lifecycle
    }

    pub(crate) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.pools.watch_catalog()
    }

    pub(crate) fn watch_readiness(&self) -> watch::Receiver<Arc<TopologySnapshot>> {
        self.topology.watch()
    }

    pub(crate) async fn subscribe_pool(
        &self,
        pool_id: PoolId,
        identity_matches: impl Fn(ProducerIdentity) -> bool + Send,
    ) -> Result<PublicationHubSubscription, PublicationHubError> {
        self.pools.subscribe_pool(pool_id, identity_matches).await
    }

    pub(crate) fn load_snapshots(&self) -> Vec<PoolLoadSnapshot> {
        self.pools.load_snapshots()
    }

    pub(crate) fn pools(&self) -> &Arc<PoolRegistry> {
        &self.pools
    }

    pub(crate) fn topology(&self) -> &Arc<TopologyPublisher> {
        &self.topology
    }

    pub(crate) fn fence_publication(&self, producer: ProducerIdentity, reason: &str) -> bool {
        self.pools
            .fence_generation(producer.pool_id(), producer.layout_generation(), reason)
    }
}
