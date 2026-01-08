// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic registry client.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;

use super::codec::{BinaryCodec, OffloadStatus, QueryType, RegistryCodec, ResponseType};
use super::key::RegistryKey;
use super::metadata::RegistryMetadata;
use super::transport::RegistryTransport;
use super::value::RegistryValue;

#[derive(Debug, Clone)]
pub struct OffloadResult<K> {
    pub can_offload: Vec<K>,
    pub already_stored: Vec<K>,
    pub leased: Vec<K>,
}

impl<K> Default for OffloadResult<K> {
    fn default() -> Self {
        Self {
            can_offload: Vec::new(),
            already_stored: Vec::new(),
            leased: Vec::new(),
        }
    }
}

#[async_trait]
pub trait Registry<K, V, M>: Send + Sync
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    async fn register(&self, entries: &[(K, V, M)]) -> Result<()>;
    async fn can_offload(&self, keys: &[K]) -> Result<OffloadResult<K>>;
    async fn match_prefix(&self, keys: &[K]) -> Result<Vec<(K, V, M)>>;
    async fn flush(&self) -> Result<()>;
}

/// Pending batch state.
struct PendingBatch<K, V, M> {
    entries: Vec<(K, V, M)>,
    /// When the first entry was added to this batch.
    first_entry_time: Option<tokio::time::Instant>,
}

impl<K, V, M> Default for PendingBatch<K, V, M> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            first_entry_time: None,
        }
    }
}

impl<K: Clone, V: Clone, M: Clone> PendingBatch<K, V, M> {
    fn add(&mut self, entries: &[(K, V, M)]) {
        if self.first_entry_time.is_none() && !entries.is_empty() {
            self.first_entry_time = Some(tokio::time::Instant::now());
        }
        self.entries.extend(entries.iter().cloned());
    }

    fn take(&mut self) -> Vec<(K, V, M)> {
        self.first_entry_time = None;
        std::mem::take(&mut self.entries)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if the batch has exceeded the timeout.
    fn is_timed_out(&self, timeout: Duration) -> bool {
        self.first_entry_time
            .map(|t| t.elapsed() >= timeout)
            .unwrap_or(false)
    }
}

pub struct RegistryClient<K, V, M, T, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
    C: RegistryCodec<K, V, M>,
{
    transport: T,
    codec: C,
    pending: Arc<Mutex<PendingBatch<K, V, M>>>,
    batch_size: usize,
    batch_timeout: Duration,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, T, C> RegistryClient<K, V, M, T, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
    C: RegistryCodec<K, V, M>,
{
    pub fn new(transport: T, codec: C) -> Self {
        Self {
            transport,
            codec,
            pending: Arc::new(Mutex::new(PendingBatch::default())),
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
            _phantom: PhantomData,
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn with_batch_timeout(mut self, timeout: Duration) -> Self {
        self.batch_timeout = timeout;
        self
    }

    /// Start a background task that flushes pending entries on timeout.
    ///
    /// Returns a handle that can be used to stop the task.
    pub fn start_batch_flush_task(
        self: &Arc<Self>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()>
    where
        K: 'static,
        V: 'static,
        M: 'static,
        T: 'static,
        C: 'static,
    {
        let client = Arc::clone(self);
        let timeout = self.batch_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(timeout / 2);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        tracing::debug!("Batch flush task shutting down");
                        // Final flush on shutdown
                        let _ = client.flush().await;
                        break;
                    }
                    _ = interval.tick() => {
                        // Check if batch has timed out
                        let should_flush = {
                            let pending = client.pending.lock().await;
                            !pending.is_empty() && pending.is_timed_out(timeout)
                        };

                        if should_flush {
                            if let Err(e) = client.flush().await {
                                tracing::warn!(error = %e, "Batch flush failed");
                            }
                        }
                    }
                }
            }
        })
    }
}

#[async_trait]
impl<K, V, M, T, C> Registry<K, V, M> for RegistryClient<K, V, M, T, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
    C: RegistryCodec<K, V, M>,
{
    async fn register(&self, entries: &[(K, V, M)]) -> Result<()> {
        let should_flush = {
            let mut pending = self.pending.lock().await;
            pending.add(entries);
            pending.len() >= self.batch_size
        };

        if should_flush {
            self.flush().await?;
        }

        Ok(())
    }

    async fn can_offload(&self, keys: &[K]) -> Result<OffloadResult<K>> {
        if keys.is_empty() {
            return Ok(OffloadResult::default());
        }

        let mut buf = Vec::new();
        self.codec
            .encode_query(&QueryType::CanOffload(keys.to_vec()), &mut buf)?;

        let response = self.transport.request(&buf).await?;
        let decoded = self
            .codec
            .decode_response(&response)
            .ok_or_else(|| anyhow::anyhow!("invalid response"))?;

        match decoded {
            ResponseType::CanOffload(statuses) => {
                let mut result = OffloadResult::default();
                for (key, status) in keys.iter().zip(statuses.iter()) {
                    match status {
                        OffloadStatus::Granted => result.can_offload.push(*key),
                        OffloadStatus::AlreadyStored => result.already_stored.push(*key),
                        OffloadStatus::Leased => result.leased.push(*key),
                    }
                }
                Ok(result)
            }
            _ => Err(anyhow::anyhow!("unexpected response type")),
        }
    }

    async fn match_prefix(&self, keys: &[K]) -> Result<Vec<(K, V, M)>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut buf = Vec::new();
        self.codec
            .encode_query(&QueryType::Match(keys.to_vec()), &mut buf)?;

        let response = self.transport.request(&buf).await?;
        let decoded = self
            .codec
            .decode_response(&response)
            .ok_or_else(|| anyhow::anyhow!("invalid response"))?;

        match decoded {
            ResponseType::Match(entries) => Ok(entries),
            _ => Err(anyhow::anyhow!("unexpected response type")),
        }
    }

    async fn flush(&self) -> Result<()> {
        let entries = {
            let mut pending = self.pending.lock().await;
            pending.take()
        };

        if entries.is_empty() {
            return Ok(());
        }

        let mut buf = Vec::new();
        self.codec.encode_register(&entries, &mut buf)?;
        self.transport.publish(&buf).await
    }
}

/// Create a simple registry client with binary codec.
pub fn simple_client<K, V, M, T>(transport: T) -> RegistryClient<K, V, M, T, BinaryCodec<K, V, M>>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
{
    RegistryClient::new(transport, BinaryCodec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::codec::BinaryCodec;
    use crate::block_manager::distributed::registry::core::metadata::NoMetadata;
    use crate::block_manager::distributed::registry::core::transport::InProcessTransport;

    #[tokio::test]
    async fn test_registry_client_can_offload() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let (transport, _rx) = InProcessTransport::new(move |data| {
            let query = codec.decode_query(data);
            match query {
                Some(QueryType::CanOffload(keys)) => {
                    let statuses: Vec<_> = keys
                        .iter()
                        .map(|k| {
                            if *k % 2 == 0 {
                                OffloadStatus::AlreadyStored
                            } else {
                                OffloadStatus::Granted
                            }
                        })
                        .collect();
                    let mut buf = Vec::new();
                    codec
                        .encode_response(&ResponseType::CanOffload(statuses), &mut buf)
                        .unwrap();
                    buf
                }
                _ => Vec::new(),
            }
        });

        let client: RegistryClient<u64, u64, NoMetadata, _, _> =
            RegistryClient::new(transport, BinaryCodec::new());

        let result = client.can_offload(&[1, 2, 3, 4]).await.unwrap();

        assert_eq!(result.can_offload, vec![1, 3]);
        assert_eq!(result.already_stored, vec![2, 4]);
    }

    #[tokio::test]
    async fn test_registry_client_leased_status() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let (transport, _rx) = InProcessTransport::new(move |data| {
            let query = codec.decode_query(data);
            match query {
                Some(QueryType::CanOffload(keys)) => {
                    let statuses: Vec<_> = keys
                        .iter()
                        .map(|k| {
                            if *k == 1 {
                                OffloadStatus::Leased
                            } else {
                                OffloadStatus::Granted
                            }
                        })
                        .collect();
                    let mut buf = Vec::new();
                    codec
                        .encode_response(&ResponseType::CanOffload(statuses), &mut buf)
                        .unwrap();
                    buf
                }
                _ => Vec::new(),
            }
        });

        let client: RegistryClient<u64, u64, NoMetadata, _, _> =
            RegistryClient::new(transport, BinaryCodec::new());

        let result = client.can_offload(&[1, 2, 3]).await.unwrap();

        assert_eq!(result.leased, vec![1]);
        assert_eq!(result.can_offload, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_registry_client_match() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let (transport, _rx) = InProcessTransport::new(move |data| {
            let query = codec.decode_query(data);
            match query {
                Some(QueryType::Match(keys)) => {
                    let entries: Vec<_> = keys
                        .into_iter()
                        .take(2)
                        .map(|k| (k, k * 100, NoMetadata))
                        .collect();
                    let mut buf = Vec::new();
                    codec
                        .encode_response(&ResponseType::Match(entries), &mut buf)
                        .unwrap();
                    buf
                }
                _ => Vec::new(),
            }
        });

        let client: RegistryClient<u64, u64, NoMetadata, _, _> =
            RegistryClient::new(transport, BinaryCodec::new());

        let result = client.match_prefix(&[1, 2, 3, 4]).await.unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 100, NoMetadata));
        assert_eq!(result[1], (2, 200, NoMetadata));
    }

    #[tokio::test]
    async fn test_batch_size_flush() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let flush_count = Arc::new(AtomicUsize::new(0));
        let flush_count_clone = Arc::clone(&flush_count);

        let (transport, mut rx) = InProcessTransport::new(move |_| Vec::new());

        // Count publishes in background
        let flush_counter = flush_count_clone;
        tokio::spawn(async move {
            while let Some(_) = rx.recv().await {
                flush_counter.fetch_add(1, Ordering::SeqCst);
            }
        });

        let client: RegistryClient<u64, u64, NoMetadata, _, _> =
            RegistryClient::new(transport, BinaryCodec::new()).with_batch_size(3);

        // Add 2 entries (below threshold)
        client
            .register(&[(1, 100, NoMetadata), (2, 200, NoMetadata)])
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(flush_count.load(Ordering::SeqCst), 0);

        // Add 1 more (reaches threshold of 3)
        client.register(&[(3, 300, NoMetadata)]).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(flush_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_batch_timeout_flush() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let flush_count = Arc::new(AtomicUsize::new(0));
        let flush_count_clone = Arc::clone(&flush_count);

        let (transport, mut rx) = InProcessTransport::new(move |_| Vec::new());

        // Count publishes in background
        let flush_counter = flush_count_clone;
        tokio::spawn(async move {
            while let Some(_) = rx.recv().await {
                flush_counter.fetch_add(1, Ordering::SeqCst);
            }
        });

        let client = Arc::new(
            RegistryClient::<u64, u64, NoMetadata, _, _>::new(transport, BinaryCodec::new())
                .with_batch_size(100) // High batch size
                .with_batch_timeout(Duration::from_millis(50)), // Short timeout
        );

        let cancel = tokio_util::sync::CancellationToken::new();
        let _task = client.start_batch_flush_task(cancel.clone());

        // Add entries (below batch size threshold)
        client.register(&[(1, 100, NoMetadata)]).await.unwrap();

        // Wait for timeout to trigger flush
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(flush_count.load(Ordering::SeqCst) >= 1);

        cancel.cancel();
    }
}
