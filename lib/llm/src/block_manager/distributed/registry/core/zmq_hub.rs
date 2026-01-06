// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ-based registry hub with proper async handling.
//!
//! Uses separate tasks for queries and registrations to avoid
//! issues with tokio::select! and ZMQ streams.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use tmq::{Context, Message, Multipart, pull, router};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::codec::{OffloadStatus, QueryType, RegistryCodec, ResponseType};
use super::key::RegistryKey;
use super::metadata::RegistryMetadata;
use super::storage::Storage;
use super::value::RegistryValue;

/// Configuration for ZMQ hub.
#[derive(Clone, Debug)]
pub struct ZmqHubConfig {
    pub query_addr: String,
    pub pull_addr: String,
    pub capacity: u64,
}

impl ZmqHubConfig {
    pub fn new(query_addr: impl Into<String>, pull_addr: impl Into<String>) -> Self {
        Self {
            query_addr: query_addr.into(),
            pull_addr: pull_addr.into(),
            capacity: 100_000,
        }
    }

    pub fn with_ports(host: &str, query_port: u16, pull_port: u16) -> Self {
        Self::new(
            format!("tcp://{}:{}", host, query_port),
            format!("tcp://{}:{}", host, pull_port),
        )
    }

    pub fn bind_all(query_port: u16, pull_port: u16) -> Self {
        Self::with_ports("*", query_port, pull_port)
    }
}

impl Default for ZmqHubConfig {
    fn default() -> Self {
        Self::bind_all(5555, 5556)
    }
}

/// ZMQ-based registry hub.
pub struct ZmqHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V> + Send + Sync + 'static,
    C: RegistryCodec<K, V, M> + Send + Sync + 'static,
{
    config: ZmqHubConfig,
    storage: Arc<S>,
    codec: Arc<C>,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, S, C> ZmqHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V> + Send + Sync + 'static,
    C: RegistryCodec<K, V, M> + Send + Sync + 'static,
{
    pub fn new(config: ZmqHubConfig, storage: S, codec: C) -> Self {
        Self {
            config,
            storage: Arc::new(storage),
            codec: Arc::new(codec),
            _phantom: PhantomData,
        }
    }

    /// Get reference to storage for seeding data.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Run the hub until cancelled.
    pub async fn serve(&self, cancel: CancellationToken) -> Result<()> {
        info!(
            query_addr = %self.config.query_addr,
            pull_addr = %self.config.pull_addr,
            "ZMQ hub starting"
        );

        // Spawn query handler
        let query_cancel = cancel.clone();
        let query_storage = self.storage.clone();
        let query_codec = self.codec.clone();
        let query_addr = self.config.query_addr.clone();

        let query_handle = tokio::spawn(async move {
            Self::run_query_handler(query_storage, query_codec, query_addr, query_cancel).await
        });

        // Spawn registration handler
        let pull_cancel = cancel.clone();
        let pull_storage = self.storage.clone();
        let pull_codec = self.codec.clone();
        let pull_addr = self.config.pull_addr.clone();

        let pull_handle = tokio::spawn(async move {
            Self::run_pull_handler(pull_storage, pull_codec, pull_addr, pull_cancel).await
        });

        // Wait for either to finish or cancellation
        tokio::select! {
            result = query_handle => {
                if let Err(e) = result {
                    error!(error = %e, "Query handler panicked");
                }
            }
            result = pull_handle => {
                if let Err(e) = result {
                    error!(error = %e, "Pull handler panicked");
                }
            }
            _ = cancel.cancelled() => {
                info!("Hub received shutdown signal");
            }
        }

        info!(entries = self.storage.len(), "ZMQ hub stopped");
        Ok(())
    }

    /// Run the query handler (ROUTER socket).
    async fn run_query_handler(
        storage: Arc<S>,
        codec: Arc<C>,
        addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let context = Context::new();
        let router = router::router(&context)
            .bind(&addr)
            .map_err(|e| anyhow!("Failed to bind ROUTER to {}: {}", addr, e))?;

        let (mut send_half, mut recv_half) = router.split();

        info!(addr = %addr, "Query handler started (ROUTER)");

        // Response channel for pipelining
        let (tx, mut rx) = mpsc::channel::<(Vec<u8>, Vec<u8>)>(1024);

        // Sender task
        let send_cancel = cancel.clone();
        let sender_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = send_cancel.cancelled() => break,
                    Some((identity, response)) = rx.recv() => {
                        let mut frames = VecDeque::new();
                        frames.push_back(Message::from(identity));
                        frames.push_back(Message::from(response));

                        if let Err(e) = send_half.send(Multipart(frames)).await {
                            error!(error = %e, "Failed to send response");
                        }
                    }
                    else => break,
                }
            }
        });

        // Receive loop
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("Query handler shutting down");
                    break;
                }
                result = recv_half.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            let frames: Vec<_> = msg.iter().collect();
                            if frames.len() < 2 {
                                warn!(frames = frames.len(), "Invalid ROUTER message");
                                continue;
                            }

                            let identity = frames[0].to_vec();
                            let data = frames[frames.len() - 1].to_vec();

                            let response = Self::handle_query(&storage, &codec, &data);

                            if tx.send((identity, response)).await.is_err() {
                                error!("Response channel closed");
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "ROUTER receive error");
                        }
                        None => {
                            warn!("ROUTER socket closed");
                            break;
                        }
                    }
                }
            }
        }

        drop(tx);
        let _ = sender_handle.await;
        Ok(())
    }

    /// Run the registration handler (PULL socket).
    async fn run_pull_handler(
        storage: Arc<S>,
        codec: Arc<C>,
        addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let context = Context::new();
        let mut puller = pull::pull(&context)
            .bind(&addr)
            .map_err(|e| anyhow!("Failed to bind PULL to {}: {}", addr, e))?;

        info!(addr = %addr, "Pull handler started (PULL)");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("Pull handler shutting down");
                    break;
                }
                result = puller.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            for frame in msg.iter() {
                                Self::handle_registration(&storage, &codec, frame.as_ref());
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "PULL receive error");
                        }
                        None => {
                            warn!("PULL socket closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a query and return response bytes.
    fn handle_query(storage: &S, codec: &C, data: &[u8]) -> Vec<u8> {
        let mut response = Vec::new();

        let Some(query) = codec.decode_query(data) else {
            warn!("Failed to decode query");
            return response;
        };

        match query {
            QueryType::CanOffload(keys) => {
                let statuses: Vec<_> = keys
                    .iter()
                    .map(|k| {
                        if storage.contains(k) {
                            OffloadStatus::AlreadyStored
                        } else {
                            OffloadStatus::Granted
                        }
                    })
                    .collect();

                debug!(
                    keys = keys.len(),
                    granted = statuses
                        .iter()
                        .filter(|s| **s == OffloadStatus::Granted)
                        .count(),
                    "can_offload query"
                );

                if let Err(e) = codec.encode_response(&ResponseType::CanOffload(statuses), &mut response) {
                    warn!("Failed to encode response: {}", e);
                }
            }
            QueryType::Match(keys) => {
                let entries: Vec<_> = keys
                    .iter()
                    .filter_map(|k| storage.get(k).map(|v| (*k, v, M::default())))
                    .collect();

                debug!(
                    requested = keys.len(),
                    matched = entries.len(),
                    "match query"
                );

                if let Err(e) = codec.encode_response(&ResponseType::Match(entries), &mut response) {
                    warn!("Failed to encode response: {}", e);
                }
            }
        }

        response
    }

    /// Handle a registration message.
    fn handle_registration(storage: &S, codec: &C, data: &[u8]) {
        let Some(entries) = codec.decode_register(data) else {
            warn!("Failed to decode registration");
            return;
        };

        let count = entries.len();
        for (key, value, _metadata) in entries {
            storage.insert(key, value);
        }

        debug!(count, total = storage.len(), "registration processed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::{
        BinaryCodec, HashMapStorage, NoMetadata,
    };
    use std::time::Duration;

    #[test]
    fn test_config_builder() {
        let config = ZmqHubConfig::bind_all(6000, 6001);
        assert_eq!(config.query_addr, "tcp://*:6000");
        assert_eq!(config.pull_addr, "tcp://*:6001");
    }

    #[tokio::test]
    #[ignore] // Requires ZMQ, run with: cargo test -- --ignored
    async fn test_zmq_hub_e2e() {
        use crate::block_manager::distributed::registry::core::{RegistryTransport, ZmqTransport};

        let port_base = 16555;
        let config = ZmqHubConfig::bind_all(port_base, port_base + 1);

        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);

        let hub = ZmqHub::new(config, storage, BinaryCodec::<u64, u64, NoMetadata>::new());
        let cancel = CancellationToken::new();

        // Start hub
        let hub_cancel = cancel.clone();
        let hub_handle = tokio::spawn(async move { hub.serve(hub_cancel).await });

        // Wait for hub to bind
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Connect client
        let transport = ZmqTransport::connect_to("localhost", port_base, port_base + 1)
            .expect("Failed to connect");

        // Test query
        let codec = BinaryCodec::<u64, u64, NoMetadata>::new();
        let mut buf = Vec::new();
        codec.encode_query(&QueryType::CanOffload(vec![1, 3]), &mut buf).unwrap();

        let response = transport.request(&buf).await.expect("Request failed");
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses[0], OffloadStatus::AlreadyStored);
                assert_eq!(statuses[1], OffloadStatus::Granted);
            }
            _ => panic!("Wrong response type"),
        }

        // Shutdown
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(1), hub_handle).await;
    }
}
