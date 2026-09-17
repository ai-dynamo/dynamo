// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::Status;

use super::protocol::RelayErrorReason;
use crate::kv_dc_relay::identity::{DcPoolCatalog, DcRelayIdentity};
use crate::kv_dc_relay::load::PoolLoadSnapshot;
use crate::kv_dc_relay::publication::{PoolPublicationStream, RelayPublicationSource};
use crate::kv_dc_relay::topology::TopologySnapshot;

/// Private gRPC driver facade over the transport-neutral publisher.
/// The lifecycle token is supplied by the host, separately from the read-only source.
#[derive(Clone)]
pub(super) struct GrpcPublicationSource {
    publication: Arc<dyn RelayPublicationSource>,
    lifecycle: CancellationToken,
}

impl GrpcPublicationSource {
    pub(super) fn new(
        publication: Arc<dyn RelayPublicationSource>,
        lifecycle: CancellationToken,
    ) -> Self {
        Self {
            publication,
            lifecycle,
        }
    }

    pub(super) fn relay_identity(&self) -> DcRelayIdentity {
        self.publication.relay_identity()
    }

    pub(super) fn lifecycle(&self) -> &CancellationToken {
        &self.lifecycle
    }

    pub(super) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.publication.watch_catalog()
    }

    pub(super) fn watch_readiness(&self) -> watch::Receiver<Arc<TopologySnapshot>> {
        self.publication.watch_readiness()
    }

    pub(super) async fn subscribe_pool(
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
            .ok_or_else(|| {
                RelayErrorReason::PoolNotFound.status(format!("unknown pool {pool_id}"))
            })?;
        if !identity_matches(expected) {
            return Err(
                RelayErrorReason::ProducerChanged.status("requested producer is no longer active")
            );
        }
        self.publication
            .subscribe_pool(expected)
            .await
            .map_err(super::service::publication_status)
    }

    pub(super) fn load_snapshots(&self) -> Vec<PoolLoadSnapshot> {
        self.publication.watch_load().borrow().clone()
    }
}
