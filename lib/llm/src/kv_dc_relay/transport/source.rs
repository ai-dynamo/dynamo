// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::Status;

use super::super::identity::{DcPoolCatalog, DcRelayIdentity};
use super::super::load::PoolLoadSnapshot;
use super::super::publication::{PoolPublicationStream, RelayPublicationSource};
use super::super::topology::TopologySnapshot;

/// WAN driver facade over the transport-neutral publisher.
/// The lifecycle token is supplied by the host, separately from the read-only source.
#[derive(Clone)]
pub(crate) struct WanPublicationSource {
    publication: Arc<dyn RelayPublicationSource>,
    lifecycle: CancellationToken,
}

impl WanPublicationSource {
    pub(crate) fn new(
        publication: Arc<dyn RelayPublicationSource>,
        lifecycle: CancellationToken,
    ) -> Self {
        Self {
            publication,
            lifecycle,
        }
    }

    pub(crate) fn relay_identity(&self) -> DcRelayIdentity {
        self.publication.relay_identity()
    }

    pub(crate) fn lifecycle(&self) -> &CancellationToken {
        &self.lifecycle
    }

    pub(crate) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.publication.watch_catalog()
    }

    pub(crate) fn watch_readiness(&self) -> watch::Receiver<Arc<TopologySnapshot>> {
        self.publication.watch_readiness()
    }

    pub(crate) async fn subscribe_pool(
        &self,
        pool_id: PoolId,
        identity_matches: impl Fn(ProducerIdentity) -> bool + Send,
    ) -> Result<PoolPublicationStream, Status> {
        let expected = self
            .publication
            .watch_catalog()
            .borrow()
            .pools()
            .iter()
            .find(|pool| pool.pool_id() == pool_id)
            .map(|pool| pool.producer())
            .ok_or_else(|| Status::not_found(format!("unknown pool {pool_id}")))?;
        if !identity_matches(expected) {
            return Err(Status::failed_precondition(
                "requested producer is no longer active",
            ));
        }
        self.publication
            .subscribe_pool(expected)
            .await
            .map_err(super::grpc::publication_status)
    }

    pub(crate) fn load_snapshots(&self) -> Vec<PoolLoadSnapshot> {
        self.publication.watch_load().borrow().clone()
    }
}
