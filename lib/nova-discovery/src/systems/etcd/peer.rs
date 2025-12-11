// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::peer::{
    DiscoveryError, DiscoveryQueryError, InstanceId, PeerDiscovery, PeerInfo, WorkerAddress,
    WorkerId,
};
use crate::systems::etcd::lease::LeaseState;
use crate::systems::etcd::operations::OperationExecutor;

use anyhow::{Context, Result};
use etcd_client::{Compare, CompareOp, PutOptions, Txn, TxnOp};
use futures::future::BoxFuture;
use parking_lot::RwLock;

pub(crate) struct EtcdPeerDiscovery {
    executor: OperationExecutor,
    lease_state: Arc<RwLock<LeaseState>>,
    cluster_id: String,
}

impl std::fmt::Debug for EtcdPeerDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdPeerDiscovery")
            .field("cluster_id", &self.cluster_id)
            .finish()
    }
}

impl EtcdPeerDiscovery {
    pub fn new(
        executor: OperationExecutor,
        lease_state: Arc<RwLock<LeaseState>>,
        cluster_id: String,
    ) -> Self {
        Self {
            executor,
            lease_state,
            cluster_id,
        }
    }

    /// Generate etcd key for worker_id lookup.
    fn worker_key(&self, worker_id: WorkerId) -> String {
        format!(
            "discovery://{}/peer-discovery/by-worker-id/{}",
            self.cluster_id,
            worker_id.as_u64()
        )
    }

    /// Generate etcd key for instance_id lookup.
    fn instance_key(&self, instance_id: InstanceId) -> String {
        format!(
            "discovery://{}/peer-discovery/by-instance-id/{}",
            self.cluster_id, instance_id
        )
    }
}

impl PeerDiscovery for EtcdPeerDiscovery {
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let key = self.worker_key(worker_id);
        let executor = self.executor.clone();

        Box::pin(async move {
            executor
                .execute_query(|mut client| {
                    let key = key.clone();
                    Box::pin(async move {
                        let resp = client.get(key, None).await?;

                        let kv = resp.kvs().first().ok_or_else(|| {
                            etcd_client::Error::from(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                "key not found",
                            ))
                        })?;

                        let value = kv.value().to_vec();

                        let peer_info: PeerInfo = serde_json::from_slice(&value).map_err(|e| {
                            etcd_client::Error::from(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Failed to deserialize PeerInfo: {}", e),
                            ))
                        })?;

                        Ok(peer_info)
                    })
                })
                .await
        })
    }

    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let key = self.instance_key(instance_id);
        let executor = self.executor.clone();

        Box::pin(async move {
            executor
                .execute_query(|mut client| {
                    let key = key.clone();
                    Box::pin(async move {
                        let resp = client.get(key, None).await?;

                        let kv = resp.kvs().first().ok_or_else(|| {
                            etcd_client::Error::from(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                "key not found",
                            ))
                        })?;

                        let value = kv.value().to_vec();

                        let peer_info: PeerInfo = serde_json::from_slice(&value).map_err(|e| {
                            etcd_client::Error::from(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Failed to deserialize PeerInfo: {}", e),
                            ))
                        })?;

                        Ok(peer_info)
                    })
                })
                .await
        })
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let executor = self.executor.clone();
        let worker_id = instance_id.worker_id();
        let worker_key = self.worker_key(worker_id);
        let instance_key = self.instance_key(instance_id);
        let lease_state = self.lease_state.clone();

        Box::pin(async move {
            // Get current lease ID
            let lease_id = lease_state
                .read()
                .lease_id()
                .ok_or_else(|| DiscoveryError::Backend(anyhow::anyhow!("No lease ID available")))?;

            // Serialize PeerInfo once
            let value = serde_json::to_vec(&PeerInfo::new(instance_id, worker_address))
                .context("Failed to serialize PeerInfo")?;

            let put_options = PutOptions::new().with_lease(lease_id);

            // Atomic registration: both keys must not exist
            executor
                .execute_write(|mut client| {
                    let worker_key = worker_key.clone();
                    let instance_key = instance_key.clone();
                    let value = value.clone();
                    let put_options = put_options.clone();

                    Box::pin(async move {
                        // Build transaction to ensure atomic registration
                        let txn = Txn::new()
                            .when(vec![
                                // Ensure worker_key doesn't exist (version == 0)
                                Compare::version(worker_key.clone(), CompareOp::Equal, 0),
                                // Ensure instance_key doesn't exist (version == 0)
                                Compare::version(instance_key.clone(), CompareOp::Equal, 0),
                            ])
                            .and_then(vec![
                                // If both keys don't exist, write both
                                TxnOp::put(
                                    worker_key.clone(),
                                    value.clone(),
                                    Some(put_options.clone()),
                                ),
                                TxnOp::put(instance_key.clone(), value.clone(), Some(put_options)),
                            ]);

                        // Execute transaction
                        let result = client.txn(txn).await?;

                        if result.succeeded() {
                            Ok(())
                        } else {
                            // Transaction failed - one or both keys already exist
                            // This could be a collision or checksum mismatch
                            // For now, return a generic error
                            // TODO: Check if existing values match (idempotent registration)
                            Err(etcd_client::Error::from(std::io::Error::new(
                                std::io::ErrorKind::AlreadyExists,
                                "Worker ID or Instance ID already registered",
                            )))
                        }
                    })
                })
                .await
        })
    }

    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let executor = self.executor.clone();
        let worker_id = instance_id.worker_id();
        let worker_key = self.worker_key(worker_id);
        let instance_key = self.instance_key(instance_id);

        Box::pin(async move {
            // Delete both keys (not atomic, but that's okay for unregister)
            executor
                .execute_write(|mut client| {
                    let worker_key = worker_key.clone();
                    let instance_key = instance_key.clone();

                    Box::pin(async move {
                        // Delete worker key
                        client.delete(worker_key, None).await?;

                        // Delete instance key
                        client.delete(instance_key, None).await?;

                        Ok(())
                    })
                })
                .await
        })
    }
}
