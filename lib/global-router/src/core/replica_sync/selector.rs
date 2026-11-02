// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The replica-sync side of [`CoreSelector`]: its gRPC transport, the
//! routes it serves, and the peers it follows.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::{ReplicaHub, ReplicaSyncStats};
use crate::core::CoreSelector;

/// The gRPC transport a [`CoreSelector`] replicates over.
pub(in crate::core) struct GrpcTransport {
    pub(in crate::core) hub: Arc<ReplicaHub>,
    pub(in crate::core) cancel: CancellationToken,
}

impl Drop for GrpcTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl CoreSelector {
    /// The gRPC `ReplicaSync` routes to merge into the API router, when
    /// this replica replicates over gRPC.
    pub fn replica_sync_routes(&self) -> Option<axum::Router> {
        self.grpc
            .as_ref()
            .map(|grpc| super::server::routes(grpc.hub.clone()))
    }

    /// What the gRPC transport has dropped and seen, when this replica
    /// replicates over gRPC.
    pub fn replica_sync_stats(&self) -> Option<ReplicaSyncStats> {
        self.grpc.as_ref().map(|grpc| grpc.hub.stats())
    }

    /// Ends the gRPC transport now: open `ReplicaSync` streams close and
    /// peers are no longer followed. The Global Frontend calls it when
    /// shutdown begins, so the long-lived streams do not hold the graceful
    /// drain open until its grace runs out. Dropping the selector does the
    /// same.
    pub fn stop_replication(&self) {
        if let Some(grpc) = &self.grpc {
            grpc.cancel.cancel();
        }
    }

    /// The peers the gRPC transport follows right now, when this replica
    /// replicates over gRPC.
    pub fn replica_peers(&self) -> Option<Vec<Url>> {
        self.grpc.as_ref().map(|grpc| grpc.hub.peers())
    }

    /// Keeps the gRPC transport's peers matching each set `peers`
    /// publishes, such as the replicas Global Discovery lists, until
    /// `cancel` fires or the sender drops. Peers given at build time stay
    /// however the set changes. `None` unless this replica replicates over
    /// gRPC. Run it as a task.
    pub fn reconcile_peers(
        &self,
        peers: watch::Receiver<Vec<Url>>,
        cancel: CancellationToken,
    ) -> Option<impl Future<Output = ()> + Send + 'static> {
        self.grpc
            .as_ref()
            .map(|grpc| super::follow_peer_set(grpc.hub.clone(), peers, cancel))
    }

    /// Subscribes to another replica's events, `endpoint` as its API
    /// address, `http://host:port`. `Ok(false)` if it was already a peer.
    pub fn register_peer(&self, endpoint: &str) -> Result<bool> {
        let grpc = self
            .grpc
            .as_ref()
            .ok_or_else(|| anyhow!("replica sync is off"))?;
        Ok(grpc.hub.add_peer(super::peer_url(endpoint)?))
    }
}
