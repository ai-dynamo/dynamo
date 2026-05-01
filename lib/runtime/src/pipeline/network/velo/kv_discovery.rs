// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo `PeerDiscovery` implementation backed by Dynamo's `kv::Manager`.
//!
//! Storing peer info in the same KV the rest of the runtime already uses keeps a single
//! discovery surface across both layers — a velo node registered here is reachable from
//! any process that can read the same KV (memory / file / etcd).
//!
//! ## Layout
//!
//! Two keys per peer in the `velo-peers` bucket so both `discover_by_instance_id` and
//! `discover_by_worker_id` resolve in one or two reads:
//!
//! - `by-instance/<uuid>` &rarr; JSON-encoded [`PeerInfo`]
//! - `by-worker/<u64>`    &rarr; UUID string (hop &rarr; instance entry)

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use futures::future::BoxFuture;
use uuid::Uuid;

use velo_common::{InstanceId, PeerInfo, WorkerId};
use velo_discovery::{PeerDiscovery, PeerRegistrationGuard};

use crate::storage::kv::{self, Bucket as _, Key};

/// Default bucket name used for velo peer entries.
pub const VELO_PEERS_BUCKET: &str = "velo-peers";

fn instance_key(id: &InstanceId) -> Key {
    Key::new(format!("by-instance/{}", id.as_uuid()))
}

fn worker_key(wid: WorkerId) -> Key {
    Key::new(format!("by-worker/{}", wid.as_u64()))
}

/// Velo [`PeerDiscovery`] adapter backed by a Dynamo [`kv::Manager`].
#[derive(Clone)]
pub struct KvPeerDiscovery {
    kv: Arc<kv::Manager>,
    bucket: String,
}

impl KvPeerDiscovery {
    /// Create a new adapter that stores peers in [`VELO_PEERS_BUCKET`].
    pub fn new(kv: Arc<kv::Manager>) -> Self {
        Self {
            kv,
            bucket: VELO_PEERS_BUCKET.to_string(),
        }
    }

    /// Create a new adapter using a custom bucket name (mostly useful for tests).
    pub fn with_bucket(kv: Arc<kv::Manager>, bucket: impl Into<String>) -> Self {
        Self {
            kv,
            bucket: bucket.into(),
        }
    }

    /// Persist `peer_info` in the KV. Returns a guard that removes both index keys when
    /// dropped (or when [`unregister`](PeerRegistrationGuard::unregister) is awaited).
    pub async fn register(&self, peer_info: PeerInfo) -> Result<KvPeerRegistrationGuard> {
        let bucket = self
            .kv
            .get_or_create_bucket(&self.bucket, None)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?;

        let by_inst = instance_key(&peer_info.instance_id());
        let by_worker = worker_key(peer_info.worker_id());

        let payload = serde_json::to_vec(&peer_info)
            .with_context(|| "serializing PeerInfo for velo discovery")?;
        bucket
            .insert(&by_inst, payload.into(), 0)
            .await
            .with_context(|| format!("inserting {by_inst} into {}", self.bucket))?;

        let uuid_value = peer_info.instance_id().as_uuid().to_string();
        bucket
            .insert(&by_worker, uuid_value.into_bytes().into(), 0)
            .await
            .with_context(|| format!("inserting {by_worker} into {}", self.bucket))?;

        Ok(KvPeerRegistrationGuard {
            kv: self.kv.clone(),
            bucket: self.bucket.clone(),
            by_inst: Some(by_inst),
            by_worker: Some(by_worker),
        })
    }

    async fn fetch_by_instance(&self, id: &InstanceId) -> Result<PeerInfo> {
        let bucket = self
            .kv
            .get_bucket(&self.bucket)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?
            .ok_or_else(|| anyhow!("velo discovery bucket {} not found", self.bucket))?;
        let bytes = bucket
            .get(&instance_key(id))
            .await
            .with_context(|| format!("fetching peer {id} from {}", self.bucket))?
            .ok_or_else(|| anyhow!("peer {id} not registered in {}", self.bucket))?;
        serde_json::from_slice::<PeerInfo>(&bytes)
            .with_context(|| format!("decoding PeerInfo for {id}"))
    }

    async fn fetch_by_worker(&self, wid: WorkerId) -> Result<PeerInfo> {
        let bucket = self
            .kv
            .get_bucket(&self.bucket)
            .await
            .with_context(|| format!("opening bucket {}", self.bucket))?
            .ok_or_else(|| anyhow!("velo discovery bucket {} not found", self.bucket))?;
        let id_bytes = bucket
            .get(&worker_key(wid))
            .await
            .with_context(|| {
                format!("fetching worker {} index from {}", wid.as_u64(), self.bucket)
            })?
            .ok_or_else(|| anyhow!("worker {} not registered in {}", wid.as_u64(), self.bucket))?;
        let uuid_str = std::str::from_utf8(&id_bytes)
            .with_context(|| format!("worker index for {} is not utf-8", wid.as_u64()))?;
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

/// Removes the `by-instance` and `by-worker` keys for a peer when dropped or unregistered.
pub struct KvPeerRegistrationGuard {
    kv: Arc<kv::Manager>,
    bucket: String,
    by_inst: Option<Key>,
    by_worker: Option<Key>,
}

impl KvPeerRegistrationGuard {
    async fn cleanup(&mut self) -> Result<()> {
        let Some(b) = self.kv.get_bucket(&self.bucket).await? else {
            self.by_inst.take();
            self.by_worker.take();
            return Ok(());
        };
        if let Some(k) = self.by_inst.take() {
            let _ = b.delete(&k).await;
        }
        if let Some(k) = self.by_worker.take() {
            let _ = b.delete(&k).await;
        }
        Ok(())
    }
}

impl PeerRegistrationGuard for KvPeerRegistrationGuard {
    fn unregister(&mut self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { self.cleanup().await })
    }
}

impl Drop for KvPeerRegistrationGuard {
    fn drop(&mut self) {
        // Best-effort: spawn a cleanup task if either index entry is still live and a
        // tokio runtime is available. We can't await in Drop, and we don't want to block
        // the executor — silent failure is acceptable because peer entries are coupled
        // to a velo TCP listener that's already torn down by the time we reach here.
        if self.by_inst.is_none() && self.by_worker.is_none() {
            return;
        }
        let kv = self.kv.clone();
        let bucket = self.bucket.clone();
        let by_inst = self.by_inst.take();
        let by_worker = self.by_worker.take();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Ok(Some(b)) = kv.get_bucket(&bucket).await {
                    if let Some(k) = by_inst {
                        let _ = b.delete(&k).await;
                    }
                    if let Some(k) = by_worker {
                        let _ = b.delete(&k).await;
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use velo_common::{PeerInfo, WorkerAddress};

    fn dummy_peer() -> PeerInfo {
        let id = InstanceId::new_v4();
        let addr = WorkerAddress::from_encoded(Bytes::from_static(b"placeholder"));
        PeerInfo::new(id, addr)
    }

    #[tokio::test]
    async fn register_and_discover_round_trip() {
        let kv = Arc::new(kv::Manager::memory());
        let disco = KvPeerDiscovery::new(kv.clone());

        let peer = dummy_peer();
        let _guard = disco.register(peer.clone()).await.expect("register");

        let by_id = disco
            .discover_by_instance_id(peer.instance_id())
            .await
            .expect("by instance");
        assert_eq!(by_id.instance_id(), peer.instance_id());

        let by_worker = disco
            .discover_by_worker_id(peer.worker_id())
            .await
            .expect("by worker");
        assert_eq!(by_worker.instance_id(), peer.instance_id());
    }

    #[tokio::test]
    async fn unregister_removes_entries() {
        let kv = Arc::new(kv::Manager::memory());
        let disco = KvPeerDiscovery::new(kv.clone());

        let peer = dummy_peer();
        let mut guard = disco.register(peer.clone()).await.expect("register");
        PeerRegistrationGuard::unregister(&mut guard)
            .await
            .expect("unregister");

        let res = disco.discover_by_instance_id(peer.instance_id()).await;
        assert!(res.is_err(), "expected error after unregister, got {res:?}");
    }

    #[tokio::test]
    async fn missing_peer_errors() {
        let kv = Arc::new(kv::Manager::memory());
        let disco = KvPeerDiscovery::new(kv);
        let id = InstanceId::new_v4();
        let res = disco.discover_by_instance_id(id).await;
        assert!(res.is_err());
    }
}
