// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic async handle for registry operations.
//!
//! This module provides a non-blocking interface to any `Registry` implementation
//! by using a dedicated async task that processes commands via channels. This avoids
//! `block_in_place` calls that would block Tokio worker threads.
//!
//! # Example
//!
//! ```ignore
//! // Create a handle for distributed registry operations
//! let handle = DistributedRemoteHandle::spawn(registry);
//!
//! // Async lookup - no blocking!
//! let matched = handle.match_prefix(keys).await;
//! ```

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::block_manager::distributed::registry::{
    Registry, RegistryKey, RegistryMetadata, RegistryValue,
};

/// Fallback runtime for sync operations when not in a Tokio context.
/// This is used when `_blocking` methods are called from threads without a runtime
/// (e.g., Python multiprocessing processes via PyO3).
static FALLBACK_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn get_fallback_runtime() -> &'static tokio::runtime::Runtime {
    FALLBACK_RUNTIME.get_or_init(|| {
        tracing::info!(
            "RemoteHandle: creating fallback Tokio runtime (2 worker threads) for sync operations"
        );
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("remote-handle-fallback")
            .enable_all()
            .build()
            .expect("Failed to create fallback runtime for RemoteHandle")
    })
}

/// Execute a future from synchronous code, using the current runtime if available,
/// or a fallback runtime if not.
fn block_on_with_fallback<F, R>(future: F) -> R
where
    F: Future<Output = R> + Send,
    R: Send,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // We're in a Tokio context - use block_in_place to avoid blocking worker threads
            tracing::debug!(
                "RemoteHandle: using current Tokio runtime (thread: {:?})",
                std::thread::current().name()
            );
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        Err(_) => {
            // No runtime on this thread - use the fallback runtime
            tracing::debug!(
                "RemoteHandle: using fallback runtime (thread: {:?})",
                std::thread::current().name()
            );
            get_fallback_runtime().block_on(future)
        }
    }
}

/// Default timeout for registry operations.
const REGISTRY_TIMEOUT: Duration = Duration::from_secs(10);

/// Channel buffer size for registry commands.
const CHANNEL_BUFFER_SIZE: usize = 256;

/// Result of a can_offload check.
#[derive(Debug, Clone)]
pub struct CanOffloadResult<K> {
    /// Keys that can be safely offloaded (not in registry, not leased).
    pub can_offload: Vec<K>,
    /// Keys already stored in the registry (skip these).
    pub already_stored: Vec<K>,
    /// Keys currently leased by another worker.
    pub leased: Vec<K>,
}

impl<K> Default for CanOffloadResult<K> {
    fn default() -> Self {
        Self {
            can_offload: vec![],
            already_stored: vec![],
            leased: vec![],
        }
    }
}

/// Commands for the registry task.
///
/// Generic over:
/// - `K`: Key type (e.g., `PositionalKey`)
/// - `V`: Value type (e.g., `RemoteKey`)
/// - `M`: Metadata type (e.g., `NoMetadata`)
pub enum RemoteOperation<K, V, M> {
    /// Match keys by prefix in the registry.
    MatchPrefix {
        keys: Vec<K>,
        reply: oneshot::Sender<Vec<(K, V, M)>>,
    },
    /// Register entries in the registry.
    Register {
        entries: Vec<(K, V, M)>,
        /// Optional reply for confirmation (None = fire-and-forget).
        reply: Option<oneshot::Sender<Result<(), String>>>,
    },
    /// Check which keys can be offloaded.
    CanOffload {
        keys: Vec<K>,
        reply: oneshot::Sender<CanOffloadResult<K>>,
    },
    /// Flush pending registrations.
    Flush {
        reply: oneshot::Sender<Result<(), String>>,
    },
}

/// Handle for communicating with a registry task.
///
/// This provides an async interface to registry operations without blocking
/// Tokio worker threads. Commands are sent via an mpsc channel to a dedicated
/// task that owns the registry.
///
/// Generic over:
/// - `K`: Key type (implements `RegistryKey`)
/// - `V`: Value type (implements `RegistryValue`)
/// - `M`: Metadata type (implements `RegistryMetadata`)
#[derive(Clone)]
pub struct RemoteHandle<K, V, M>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    tx: mpsc::Sender<RemoteOperation<K, V, M>>,
}

impl<K, V, M> RemoteHandle<K, V, M>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    /// Spawn the registry task and return a handle.
    ///
    /// The spawned task will process registry commands until all handles are dropped.
    pub fn spawn(registry: Arc<dyn Registry<K, V, M> + Send + Sync>) -> Self {
        let (tx, rx) = mpsc::channel::<RemoteOperation<K, V, M>>(CHANNEL_BUFFER_SIZE);

        tokio::spawn(Self::run_task(registry, rx));

        Self { tx }
    }

    /// The main task loop that processes registry commands.
    async fn run_task(
        registry: Arc<dyn Registry<K, V, M> + Send + Sync>,
        mut rx: mpsc::Receiver<RemoteOperation<K, V, M>>,
    ) {
        while let Some(cmd) = rx.recv().await {
            match cmd {
                RemoteOperation::MatchPrefix { keys, reply } => {
                    let result = Self::do_match_prefix(&registry, &keys).await;
                    let _ = reply.send(result);
                }
                RemoteOperation::Register { entries, reply } => {
                    let result = Self::do_register(&registry, entries).await;
                    if let Some(reply) = reply {
                        let _ = reply.send(result);
                    }
                }
                RemoteOperation::CanOffload { keys, reply } => {
                    let result = Self::do_can_offload(&registry, &keys).await;
                    let _ = reply.send(result);
                }
                RemoteOperation::Flush { reply } => {
                    let result = Self::do_flush(&registry).await;
                    let _ = reply.send(result);
                }
            }
        }
        tracing::debug!("RemoteHandle task shutting down");
    }

    /// Match keys by prefix in the registry.
    async fn do_match_prefix(
        registry: &Arc<dyn Registry<K, V, M> + Send + Sync>,
        keys: &[K],
    ) -> Vec<(K, V, M)> {
        match tokio::time::timeout(REGISTRY_TIMEOUT, registry.match_prefix(keys)).await {
            Ok(Ok(entries)) => {
                tracing::debug!(matched = entries.len(), "Registry match_prefix complete");
                entries
            }
            Ok(Err(e)) => {
                tracing::error!(error = %e, "Registry match_prefix failed");
                vec![]
            }
            Err(_) => {
                tracing::error!("Registry match_prefix timed out");
                vec![]
            }
        }
    }

    /// Register entries in the registry.
    async fn do_register(
        registry: &Arc<dyn Registry<K, V, M> + Send + Sync>,
        entries: Vec<(K, V, M)>,
    ) -> Result<(), String> {
        if let Err(e) = registry.register(&entries).await {
            let msg = format!("Failed to register entries: {e}");
            tracing::error!("{}", msg);
            return Err(msg);
        }

        // Flush immediately to ensure entries are sent
        if let Err(e) = registry.flush().await {
            let msg = format!("Failed to flush registry: {e}");
            tracing::error!("{}", msg);
            return Err(msg);
        }

        tracing::debug!(count = entries.len(), "Registered entries");
        Ok(())
    }

    /// Check which keys can be offloaded.
    async fn do_can_offload(
        registry: &Arc<dyn Registry<K, V, M> + Send + Sync>,
        keys: &[K],
    ) -> CanOffloadResult<K> {
        match registry.can_offload(keys).await {
            Ok(result) => {
                tracing::debug!(
                    can_offload = result.can_offload.len(),
                    already_stored = result.already_stored.len(),
                    leased = result.leased.len(),
                    "can_offload check complete"
                );
                CanOffloadResult {
                    can_offload: result.can_offload,
                    already_stored: result.already_stored,
                    leased: result.leased,
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "can_offload query failed");
                CanOffloadResult {
                    can_offload: keys.to_vec(),
                    already_stored: vec![],
                    leased: vec![],
                }
            }
        }
    }

    /// Flush pending registrations.
    async fn do_flush(registry: &Arc<dyn Registry<K, V, M> + Send + Sync>) -> Result<(), String> {
        registry.flush().await.map_err(|e| e.to_string())
    }

    /// Match keys by prefix - returns matching entries.
    pub async fn match_prefix(&self, keys: Vec<K>) -> Vec<(K, V, M)> {
        if keys.is_empty() {
            return vec![];
        }

        let (tx, rx) = oneshot::channel();
        if self
            .tx
            .send(RemoteOperation::MatchPrefix { keys, reply: tx })
            .await
            .is_err()
        {
            tracing::error!("RemoteHandle task has shut down");
            return vec![];
        }

        rx.await.unwrap_or_default()
    }

    /// Register entries - fire and forget.
    pub async fn register(&self, entries: Vec<(K, V, M)>) {
        if entries.is_empty() {
            return;
        }

        if self
            .tx
            .send(RemoteOperation::Register {
                entries,
                reply: None,
            })
            .await
            .is_err()
        {
            tracing::error!("RemoteHandle task has shut down");
        }
    }

    /// Register entries and wait for confirmation.
    pub async fn register_and_wait(&self, entries: Vec<(K, V, M)>) -> Result<(), String> {
        if entries.is_empty() {
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        if self
            .tx
            .send(RemoteOperation::Register {
                entries,
                reply: Some(tx),
            })
            .await
            .is_err()
        {
            return Err("RemoteHandle task has shut down".to_string());
        }

        rx.await
            .unwrap_or_else(|_| Err("Channel closed".to_string()))
    }

    /// Check which keys can be offloaded.
    pub async fn can_offload(&self, keys: Vec<K>) -> CanOffloadResult<K> {
        if keys.is_empty() {
            return CanOffloadResult::default();
        }

        let (tx, rx) = oneshot::channel();
        if self
            .tx
            .send(RemoteOperation::CanOffload {
                keys: keys.clone(),
                reply: tx,
            })
            .await
            .is_err()
        {
            tracing::error!("RemoteHandle task has shut down");
            return CanOffloadResult {
                can_offload: keys,
                already_stored: vec![],
                leased: vec![],
            };
        }

        rx.await.unwrap_or_else(|_| CanOffloadResult {
            can_offload: keys,
            already_stored: vec![],
            leased: vec![],
        })
    }

    /// Flush pending registrations.
    pub async fn flush(&self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        if self
            .tx
            .send(RemoteOperation::Flush { reply: tx })
            .await
            .is_err()
        {
            return Err("RemoteHandle task has shut down".to_string());
        }

        rx.await
            .unwrap_or_else(|_| Err("Channel closed".to_string()))
    }

    /// Check if the handle is still connected to the task.
    pub fn is_connected(&self) -> bool {
        !self.tx.is_closed()
    }
}

use crate::block_manager::block::transfer::remote::RemoteKey;
use crate::block_manager::distributed::registry::{NoMetadata, PositionalKey};

/// Type alias for positional registry handle (used by G4/object storage).
pub type PositionalRemoteHandle = RemoteHandle<PositionalKey, RemoteKey, NoMetadata>;

/// Extension trait for async remote operations using sequence hashes.
///
/// These methods convert sequence hashes to/from PositionalKeys automatically.
#[async_trait::async_trait]
pub trait RemoteHashOperations {
    /// Lookup sequence hashes - returns hashes that are in the registry.
    async fn lookup_hashes(&self, hashes: &[u64], worker_id: u64) -> Vec<u64>;

    /// Register sequence hashes with their positions and remote keys.
    async fn register_hashes(&self, entries: &[(u64, u32, RemoteKey)], worker_id: u64);

    /// Check which sequence hashes can be offloaded.
    /// Returns (can_offload, already_stored, leased) hashes.
    async fn can_offload_hashes(
        &self,
        hashes: &[u64],
        worker_id: u64,
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>);
}

#[async_trait::async_trait]
impl RemoteHashOperations for PositionalRemoteHandle {
    async fn lookup_hashes(&self, hashes: &[u64], worker_id: u64) -> Vec<u64> {
        if hashes.is_empty() {
            return vec![];
        }

        let keys = hashes_to_positional_keys(hashes, worker_id);
        let matches = self.match_prefix(keys).await;

        matches
            .into_iter()
            .map(|(key, _, _)| key.sequence_hash)
            .collect()
    }

    async fn register_hashes(&self, entries: &[(u64, u32, RemoteKey)], worker_id: u64) {
        if entries.is_empty() {
            return;
        }

        let reg_entries: Vec<_> = entries
            .iter()
            .map(|(hash, position, key)| {
                let positional_key = PositionalKey {
                    worker_id,
                    sequence_hash: *hash,
                    position: *position,
                };
                (positional_key, key.clone(), NoMetadata)
            })
            .collect();

        self.register(reg_entries).await;
    }

    async fn can_offload_hashes(
        &self,
        hashes: &[u64],
        worker_id: u64,
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        if hashes.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let keys = hashes_to_positional_keys(hashes, worker_id);
        let result = self.can_offload(keys).await;

        (
            result.can_offload.iter().map(|k| k.sequence_hash).collect(),
            result
                .already_stored
                .iter()
                .map(|k| k.sequence_hash)
                .collect(),
            result.leased.iter().map(|k| k.sequence_hash).collect(),
        )
    }
}

/// Convert sequence hashes to positional keys.
fn hashes_to_positional_keys(hashes: &[u64], worker_id: u64) -> Vec<PositionalKey> {
    hashes
        .iter()
        .enumerate()
        .map(|(pos, &hash)| PositionalKey {
            worker_id,
            sequence_hash: hash,
            position: pos as u32,
        })
        .collect()
}

/// Extension trait for sync remote operations using sequence hashes.
///
/// These methods block the current thread using `block_in_place`.
pub trait RemoteHashOperationsSync {
    /// Sync lookup - blocks using `block_in_place`.
    fn lookup_hashes_blocking(&self, hashes: &[u64], worker_id: u64) -> Vec<u64>;

    /// Sync register - blocks using `block_in_place`.
    fn register_hashes_blocking(&self, entries: &[(u64, u32, RemoteKey)], worker_id: u64);

    /// Sync can_offload - blocks using `block_in_place`.
    fn can_offload_hashes_blocking(
        &self,
        hashes: &[u64],
        worker_id: u64,
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>);
}

impl RemoteHashOperationsSync for PositionalRemoteHandle {
    fn lookup_hashes_blocking(&self, hashes: &[u64], worker_id: u64) -> Vec<u64> {
        if hashes.is_empty() {
            return vec![];
        }

        let keys = hashes_to_positional_keys(hashes, worker_id);
        let handle = self.clone();

        block_on_with_fallback(async move {
            handle
                .match_prefix(keys)
                .await
                .into_iter()
                .map(|(key, _, _)| key.sequence_hash)
                .collect()
        })
    }

    fn register_hashes_blocking(&self, entries: &[(u64, u32, RemoteKey)], worker_id: u64) {
        if entries.is_empty() {
            return;
        }

        let reg_entries: Vec<_> = entries
            .iter()
            .map(|(hash, position, key)| {
                let positional_key = PositionalKey {
                    worker_id,
                    sequence_hash: *hash,
                    position: *position,
                };
                (positional_key, key.clone(), NoMetadata)
            })
            .collect();

        let handle = self.clone();

        block_on_with_fallback(async move {
            handle.register(reg_entries).await;
        });
    }

    fn can_offload_hashes_blocking(
        &self,
        hashes: &[u64],
        worker_id: u64,
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        if hashes.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let keys = hashes_to_positional_keys(hashes, worker_id);
        let handle = self.clone();

        block_on_with_fallback(async move {
            let result = handle.can_offload(keys).await;
            (
                result.can_offload.iter().map(|k| k.sequence_hash).collect(),
                result
                    .already_stored
                    .iter()
                    .map(|k| k.sequence_hash)
                    .collect(),
                result.leased.iter().map(|k| k.sequence_hash).collect(),
            )
        })
    }
}

#[cfg(test)]
mod tests {}
