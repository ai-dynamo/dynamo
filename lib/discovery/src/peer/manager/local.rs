// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::peer::{
    DiscoveryError, DiscoveryQueryError, InstanceId, PeerInfo, WorkerAddress, WorkerId,
};

#[derive(Debug, Default, Clone)]
pub struct LocalPeerDiscovery {
    inner: Arc<RwLock<LocalPeerDiscoveryInner>>,
}

#[derive(Debug, Default, Clone)]
struct LocalPeerDiscoveryInner {
    by_worker_id: HashMap<WorkerId, InstanceId>,
    by_instance_id: HashMap<InstanceId, PeerInfo>,
}

impl LocalPeerDiscovery {
    pub fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        let state = self.inner.read();
        let by_worker_id = state.by_worker_id.get(&worker_id);
        if let Some(instance_id) = by_worker_id {
            let peer_info = state.by_instance_id.get(instance_id);
            if let Some(peer_info) = peer_info {
                return Ok(peer_info.clone());
            }
        }
        Err(DiscoveryQueryError::NotFound)
    }

    pub fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> Result<PeerInfo, DiscoveryQueryError> {
        let state = self.inner.read();
        let by_instance_id = state.by_instance_id.get(&instance_id);
        if let Some(peer_info) = by_instance_id {
            return Ok(peer_info.clone());
        }
        Err(DiscoveryQueryError::NotFound)
    }

    pub fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<(), DiscoveryError> {
        let mut state = self.inner.write();

        // Validate no worker_id collision
        let worker_id = instance_id.worker_id();
        if let Some(existing_instance) = state.by_worker_id.get(&worker_id)
            && *existing_instance != instance_id
        {
            return Err(DiscoveryError::WorkerIdCollision(
                worker_id,
                *existing_instance,
                instance_id,
            ));
        }

        // Fail-fast for any duplicate registration attempt
        if let Some(existing_peer_info) = state.by_instance_id.get(&instance_id) {
            // Check if it's the same address (idempotent attempt) or different
            if existing_peer_info.address_checksum() == worker_address.checksum() {
                // Duplicate registration with same address - fail to detect bugs
                return Err(DiscoveryError::InstanceAlreadyRegistered(instance_id));
            } else {
                // Re-registration with different address - fail with checksum mismatch
                return Err(DiscoveryError::ChecksumMismatch(
                    instance_id,
                    existing_peer_info.address_checksum(),
                    worker_address.checksum(),
                ));
            }
        }

        // Register peer
        let peer_info = PeerInfo::new(instance_id, worker_address);
        state.by_worker_id.insert(worker_id, instance_id);
        state.by_instance_id.insert(instance_id, peer_info);
        Ok(())
    }

    #[expect(dead_code)]
    pub fn unregister_instance(&self, instance_id: InstanceId) -> Result<(), DiscoveryError> {
        let mut state = self.inner.write();
        state.by_worker_id.remove(&instance_id.worker_id());
        state.by_instance_id.remove(&instance_id);
        Ok(())
    }
}
