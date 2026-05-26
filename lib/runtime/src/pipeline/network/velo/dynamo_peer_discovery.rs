// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo peer discovery backed by Dynamo discovery.
//!
//! This adapter is the boundary between Velo's `PeerDiscovery` trait and
//! Dynamo's discovery plane. It does not know whether discovery is KV, file,
//! memory, or Kubernetes-backed.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;

use ::velo::discovery::{PeerDiscovery, PeerRegistrationGuard};
use ::velo::{InstanceId, PeerInfo, WorkerId};

use crate::discovery::{Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec};

/// Velo `PeerDiscovery` adapter over Dynamo's discovery abstraction.
#[derive(Clone)]
pub struct DynamoPeerDiscovery {
    discovery: Arc<dyn Discovery>,
}

impl DynamoPeerDiscovery {
    pub fn new(discovery: Arc<dyn Discovery>) -> Self {
        Self { discovery }
    }

    /// Register this process' Velo peer in Dynamo discovery.
    pub async fn register(&self, peer_info: PeerInfo) -> Result<DynamoPeerRegistrationGuard> {
        let instance = self
            .discovery
            .register(DiscoverySpec::VeloPeer { peer_info })
            .await?;

        Ok(DynamoPeerRegistrationGuard {
            discovery: self.discovery.clone(),
            instance: Some(instance),
        })
    }

    async fn fetch_by_instance(&self, instance_id: InstanceId) -> Result<PeerInfo> {
        self.fetch_one(DiscoveryQuery::VeloPeerByInstance { instance_id })
            .await
    }

    async fn fetch_by_worker(&self, worker_id: WorkerId) -> Result<PeerInfo> {
        self.fetch_one(DiscoveryQuery::VeloPeerByWorker { worker_id })
            .await
    }

    async fn fetch_one(&self, query: DiscoveryQuery) -> Result<PeerInfo> {
        let mut peers = self
            .discovery
            .list(query)
            .await?
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::VeloPeer { peer_info, .. } => Some(peer_info),
                _ => None,
            });

        peers
            .next()
            .ok_or_else(|| anyhow!("Velo peer is not registered in Dynamo discovery"))
    }
}

impl PeerDiscovery for DynamoPeerDiscovery {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(async move { self.fetch_by_worker(worker_id).await })
    }

    fn discover_by_instance_id(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(async move { self.fetch_by_instance(instance_id).await })
    }
}

/// Registration guard for a single Velo peer.
pub struct DynamoPeerRegistrationGuard {
    discovery: Arc<dyn Discovery>,
    instance: Option<DiscoveryInstance>,
}

impl DynamoPeerRegistrationGuard {
    async fn cleanup(&mut self) -> Result<()> {
        let Some(instance) = self.instance.take() else {
            return Ok(());
        };

        if let Err(err) = self.discovery.unregister(instance.clone()).await {
            self.instance = Some(instance);
            return Err(err);
        }

        Ok(())
    }
}

impl PeerRegistrationGuard for DynamoPeerRegistrationGuard {
    fn unregister(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { self.cleanup().await })
    }
}

impl Drop for DynamoPeerRegistrationGuard {
    fn drop(&mut self) {
        let Some(instance) = self.instance.take() else {
            return;
        };

        let discovery = self.discovery.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = discovery.unregister(instance).await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use ::velo::WorkerAddress;
    use bytes::Bytes;

    use crate::discovery::{MockDiscovery, SharedMockRegistry};

    fn dummy_peer() -> PeerInfo {
        let instance_id = InstanceId::new_v4();
        let mut entries = HashMap::<String, Vec<u8>>::new();
        entries.insert("tcp".to_string(), b"127.0.0.1:0".to_vec());
        let address = WorkerAddress::from_encoded(Bytes::from(
            rmp_serde::to_vec(&entries).expect("encode worker address"),
        ));
        PeerInfo::new(instance_id, address)
    }

    fn discovery() -> DynamoPeerDiscovery {
        DynamoPeerDiscovery::new(Arc::new(MockDiscovery::new(
            Some(42),
            SharedMockRegistry::new(),
        )))
    }

    #[tokio::test]
    async fn register_and_discover_round_trip() {
        let discovery = discovery();
        let peer = dummy_peer();

        let _guard = discovery.register(peer.clone()).await.expect("register");

        let by_instance = discovery
            .discover_by_instance_id(peer.instance_id())
            .await
            .expect("discover by instance");
        assert_eq!(by_instance, peer);

        let by_worker = discovery
            .discover_by_worker_id(peer.worker_id())
            .await
            .expect("discover by worker");
        assert_eq!(by_worker, peer);
    }

    #[tokio::test]
    async fn unregister_removes_entry() {
        let discovery = discovery();
        let peer = dummy_peer();

        let mut guard = discovery.register(peer.clone()).await.expect("register");
        PeerRegistrationGuard::unregister(&mut guard)
            .await
            .expect("unregister");

        assert!(
            discovery
                .discover_by_instance_id(peer.instance_id())
                .await
                .is_err()
        );
        assert!(
            discovery
                .discover_by_worker_id(peer.worker_id())
                .await
                .is_err()
        );
    }
}
