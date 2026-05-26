// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo peer discovery backed by Dynamo's KV store.
//!
//! The delegated response stream path needs a way to resolve a Velo peer from
//! the worker ID embedded in a `StreamAnchorHandle`. For this first milestone
//! we keep the mapping in a KV bucket without changing Dynamo's first-class
//! discovery API.
//!
//! Bucket layout:
//! - `by-instance/<uuid>` stores JSON encoded `PeerInfo`
//! - `by-worker/<u64>` stores the corresponding instance UUID as UTF-8

use anyhow::{Context as _, Result, anyhow};
use futures::future::BoxFuture;
use uuid::Uuid;

use ::velo::discovery::{PeerDiscovery, PeerRegistrationGuard};
use ::velo::{InstanceId, PeerInfo, WorkerId};

use crate::storage::kv::{self, Bucket as _, Key};

/// Default bucket name for Velo peer entries.
pub const VELO_PEERS_BUCKET: &str = "velo-peers";

fn instance_key(id: &InstanceId) -> Key {
    Key::new(format!("by-instance/{}", id.as_uuid()))
}

fn worker_key(worker_id: WorkerId) -> Key {
    Key::new(format!("by-worker/{}", worker_id.as_u64()))
}

/// Velo `PeerDiscovery` adapter backed by a Dynamo KV manager.
#[derive(Clone)]
pub struct KvPeerDiscovery {
    kv: kv::Manager,
    bucket: String,
}

impl KvPeerDiscovery {
    /// Create a peer discovery adapter using the default Velo peers bucket.
    pub fn new(kv: kv::Manager) -> Self {
        Self {
            kv,
            bucket: VELO_PEERS_BUCKET.to_string(),
        }
    }

    /// Register a peer and return a guard that removes both index entries.
    pub async fn register(&self, peer_info: PeerInfo) -> Result<KvPeerRegistrationGuard> {
        let bucket = self
            .kv
            .get_or_create_bucket(&self.bucket, None)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?;

        let by_instance = instance_key(&peer_info.instance_id());
        let by_worker = worker_key(peer_info.worker_id());

        let payload = serde_json::to_vec(&peer_info)
            .with_context(|| "serializing PeerInfo for Velo discovery")?;
        bucket
            .insert(&by_instance, payload.into(), 0)
            .await
            .with_context(|| format!("inserting {by_instance} into {}", self.bucket))?;

        let instance_uuid = peer_info.instance_id().as_uuid().to_string();
        if let Err(err) = bucket
            .insert(&by_worker, instance_uuid.into_bytes().into(), 0)
            .await
        {
            if let Err(rollback_err) = bucket.delete(&by_instance).await {
                return Err(err).with_context(|| {
                    format!(
                        "inserting {by_worker} into {} and rolling back {by_instance}: {rollback_err:#}",
                        self.bucket
                    )
                });
            }
            return Err(err).with_context(|| format!("inserting {by_worker} into {}", self.bucket));
        }

        Ok(KvPeerRegistrationGuard {
            kv: self.kv.clone(),
            bucket: self.bucket.clone(),
            by_instance: Some(by_instance),
            by_worker: Some(by_worker),
        })
    }

    async fn fetch_by_instance(&self, instance_id: &InstanceId) -> Result<PeerInfo> {
        let bucket = self
            .kv
            .get_bucket(&self.bucket)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?
            .ok_or_else(|| anyhow!("Velo discovery bucket {} not found", self.bucket))?;

        let key = instance_key(instance_id);
        let bytes = bucket
            .get(&key)
            .await
            .with_context(|| format!("fetching peer {instance_id} from {}", self.bucket))?
            .ok_or_else(|| anyhow!("peer {instance_id} not registered in {}", self.bucket))?;

        serde_json::from_slice::<PeerInfo>(&bytes)
            .with_context(|| format!("decoding PeerInfo for {instance_id}"))
    }

    async fn fetch_by_worker(&self, worker_id: WorkerId) -> Result<PeerInfo> {
        let bucket = self
            .kv
            .get_bucket(&self.bucket)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?
            .ok_or_else(|| anyhow!("Velo discovery bucket {} not found", self.bucket))?;

        let key = worker_key(worker_id);
        let instance_id_bytes = bucket
            .get(&key)
            .await
            .with_context(|| {
                format!(
                    "fetching worker {} index from {}",
                    worker_id.as_u64(),
                    self.bucket
                )
            })?
            .ok_or_else(|| {
                anyhow!(
                    "worker {} not registered in {}",
                    worker_id.as_u64(),
                    self.bucket
                )
            })?;

        let uuid_str = std::str::from_utf8(&instance_id_bytes)
            .with_context(|| format!("worker index for {} is not UTF-8", worker_id.as_u64()))?;
        let uuid = Uuid::parse_str(uuid_str)
            .with_context(|| format!("worker index entry {uuid_str} is not a valid UUID"))?;

        self.fetch_by_instance(&InstanceId::from(uuid)).await
    }
}

impl PeerDiscovery for KvPeerDiscovery {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(async move { self.fetch_by_worker(worker_id).await })
    }

    fn discover_by_instance_id(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        Box::pin(async move { self.fetch_by_instance(&instance_id).await })
    }
}

/// Registration guard for a single Velo peer.
pub struct KvPeerRegistrationGuard {
    kv: kv::Manager,
    bucket: String,
    by_instance: Option<Key>,
    by_worker: Option<Key>,
}

impl KvPeerRegistrationGuard {
    async fn cleanup(&mut self) -> Result<()> {
        let Some(bucket) = self.kv.get_bucket(&self.bucket).await? else {
            self.by_instance.take();
            self.by_worker.take();
            return Ok(());
        };

        let mut first_error: Option<anyhow::Error> = None;

        if let Some(key) = self.by_instance.as_ref() {
            match bucket.delete(key).await {
                Ok(()) => {
                    self.by_instance.take();
                }
                Err(err) => {
                    first_error = Some(anyhow!("deleting {key} from {}: {err:#}", self.bucket));
                }
            }
        }

        if let Some(key) = self.by_worker.as_ref() {
            match bucket.delete(key).await {
                Ok(()) => {
                    self.by_worker.take();
                }
                Err(err) => {
                    first_error
                        .get_or_insert(anyhow!("deleting {key} from {}: {err:#}", self.bucket));
                }
            }
        }

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl PeerRegistrationGuard for KvPeerRegistrationGuard {
    fn unregister(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { self.cleanup().await })
    }
}

impl Drop for KvPeerRegistrationGuard {
    fn drop(&mut self) {
        if self.by_instance.is_none() && self.by_worker.is_none() {
            return;
        }

        let kv = self.kv.clone();
        let bucket_name = self.bucket.clone();
        let by_instance = self.by_instance.take();
        let by_worker = self.by_worker.take();

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Ok(Some(bucket)) = kv.get_bucket(&bucket_name).await {
                    if let Some(key) = by_instance {
                        let _ = bucket.delete(&key).await;
                    }
                    if let Some(key) = by_worker {
                        let _ = bucket.delete(&key).await;
                    }
                }
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

    fn dummy_peer() -> PeerInfo {
        let instance_id = InstanceId::new_v4();
        let mut entries = HashMap::<String, Vec<u8>>::new();
        entries.insert("tcp".to_string(), b"127.0.0.1:0".to_vec());
        let address = WorkerAddress::from_encoded(Bytes::from(
            rmp_serde::to_vec(&entries).expect("encode worker address"),
        ));
        PeerInfo::new(instance_id, address)
    }

    #[tokio::test]
    async fn register_and_discover_round_trip() {
        let discovery = KvPeerDiscovery::new(kv::Manager::memory());
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
    async fn unregister_removes_entries() {
        let discovery = KvPeerDiscovery::new(kv::Manager::memory());
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

    #[tokio::test]
    async fn missing_peer_errors() {
        let discovery = KvPeerDiscovery::new(kv::Manager::memory());
        let result = discovery
            .discover_by_instance_id(InstanceId::new_v4())
            .await;

        assert!(result.is_err());
    }
}
