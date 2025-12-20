// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::runtime::Runtime;
use anyhow::{Context, Result};

use async_nats::jetstream::kv;
use derive_builder::Builder;
use derive_getters::Dissolve;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use validator::Validate;

use etcd_client::{
    Certificate, Compare, CompareOp, DeleteOptions, GetOptions, Identity, LockClient, LockOptions,
    LockResponse, PutOptions, PutResponse, TlsOptions, Txn, TxnOp, TxnOpResponse, WatchOptions,
    WatchStream, Watcher,
};
pub use etcd_client::{ConnectOptions, KeyValue, LeaseClient};
use tokio::time::{Duration, interval};
use tokio_util::sync::CancellationToken;

mod connector;
mod lease;
mod lock;

use connector::Connector;
use lease::*;
pub use lock::*;

use super::utils::build_in_runtime;
use crate::config::environment_names::etcd as env_etcd;

/// ETCD Client
#[derive(Clone)]
pub struct Client {
    connector: Arc<Connector>,
    primary_lease: u64,
    runtime: Runtime,
    // Exclusive runtime for etcd lease keep-alive and watch tasks
    // Avoid those tasks from being starved when the main runtime is busy
    // WARNING: Do not await on main runtime from this runtime or deadlocks may occur
    rt: Arc<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etcd::Client primary_lease={}", self.primary_lease)
    }
}

impl Client {
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// Create a new discovery client
    ///
    /// This will establish a connection to the etcd server, create a primary lease,
    /// and spawn a task to keep the lease alive and tie the lifetime of the [`Runtime`]
    /// to the lease.
    ///
    /// If the lease expires, the [`Runtime`] will be shutdown.
    /// If the [`Runtime`] is shutdown, the lease will be revoked.
    pub async fn new(config: ClientOptions, runtime: Runtime) -> Result<Self> {
        let token = runtime.primary_token();

        let ((connector, lease_id), rt) = build_in_runtime(
            async move {
                let etcd_urls = config.etcd_url.clone();
                let connect_options = config.etcd_connect_options.clone();

                // Create the connector
                let connector = Connector::new(etcd_urls, connect_options)
                    .await
                    .with_context(|| {
                        format!(
                            "Unable to connect to etcd server at {}. Check etcd server status",
                            config.etcd_url.join(", ")
                        )
                    })?;

                let lease_id = if config.attach_lease {
                    create_lease(connector.clone(), 10, token)
                        .await
                        .with_context(|| {
                            format!(
                                "Unable to create lease. Check etcd server status at {}",
                                config.etcd_url.join(", ")
                            )
                        })?
                } else {
                    0
                };

                Ok((connector, lease_id))
            },
            1,
        )
        .await?;

        Ok(Client {
            connector,
            primary_lease: lease_id,
            rt,
            runtime,
        })
    }

    /// Get a clone of the underlying [`etcd_client::Client`] instance.
    /// This returns a clone since the client is behind an RwLock.
    fn etcd_client(&self) -> etcd_client::Client {
        self.connector.get_client()
    }

    /// Get the primary lease ID.
    pub fn lease_id(&self) -> u64 {
        self.primary_lease
    }

    /// Returns Ok(None) if value was created, Ok(Some(revision)) if the value already exists.
    pub async fn kv_create(
        &self,
        key: &str,
        value: Vec<u8>,
        lease_id: Option<u64>,
    ) -> Result<Option<u64>> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id as i64);

        // Build transaction that creates key only if it doesn't exist
        let txn = Txn::new()
            .when(vec![Compare::version(key, CompareOp::Equal, 0)]) // Ensure the lock does not exist
            .and_then(vec![
                TxnOp::put(key, value, Some(put_options)), // Create the object
            ])
            .or_else(vec![
                TxnOp::get(key, None), // Key exists, get its info
            ]);

        // Execute the transaction
        let result = self.connector.get_client().kv_client().txn(txn).await?;

        // Created
        if result.succeeded() {
            return Ok(None);
        }

        // Already exists
        if let Some(etcd_client::TxnOpResponse::Get(get_resp)) =
            result.op_responses().into_iter().next()
            && let Some(kv) = get_resp.kvs().first()
        {
            let version = kv.version() as u64;
            return Ok(Some(version));
        }

        // Error
        for resp in result.op_responses() {
            tracing::warn!(response = ?resp, "kv_create etcd op response");
        }
        anyhow::bail!("Unable to create key. Check etcd server status")
    }

    /// Atomically create a key if it does not exist, or validate the values are identical if the key exists.
    pub async fn kv_create_or_validate(
        &self,
        key: String,
        value: Vec<u8>,
        lease_id: Option<u64>,
    ) -> Result<()> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id as i64);

        // Build the transaction that either creates the key if it doesn't exist,
        // or validates the existing value matches what we expect
        let txn = Txn::new()
            .when(vec![Compare::version(key.as_str(), CompareOp::Equal, 0)]) // Key doesn't exist
            .and_then(vec![
                TxnOp::put(key.as_str(), value.clone(), Some(put_options)), // Create it
            ])
            .or_else(vec![
                // If key exists but values don't match, this will fail the transaction
                TxnOp::txn(Txn::new().when(vec![Compare::value(
                    key.as_str(),
                    CompareOp::Equal,
                    value.clone(),
                )])),
            ]);

        // Execute the transaction
        let result = self.connector.get_client().kv_client().txn(txn).await?;

        // We have to enumerate the response paths to determine if the transaction succeeded
        if result.succeeded() {
            Ok(())
        } else {
            match result.op_responses().first() {
                Some(response) => match response {
                    TxnOpResponse::Txn(response) => match response.succeeded() {
                        true => Ok(()),
                        false => anyhow::bail!(
                            "Unable to create or validate key. Check etcd server status"
                        ),
                    },
                    _ => {
                        anyhow::bail!("Unable to validate key operation. Check etcd server status")
                    }
                },
                None => anyhow::bail!("Unable to create or validate key. Check etcd server status"),
            }
        }
    }

    pub async fn kv_put(
        &self,
        key: impl AsRef<str>,
        value: impl AsRef<[u8]>,
        lease_id: Option<u64>,
    ) -> Result<()> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id as i64);
        let _ = self
            .connector
            .get_client()
            .kv_client()
            .put(key.as_ref(), value.as_ref(), Some(put_options))
            .await?;
        Ok(())
    }

    pub async fn kv_put_with_options(
        &self,
        key: impl AsRef<str>,
        value: impl AsRef<[u8]>,
        options: Option<PutOptions>,
    ) -> Result<PutResponse> {
        let options = options
            .unwrap_or_default()
            .with_lease(self.lease_id() as i64);
        self.connector
            .get_client()
            .kv_client()
            .put(key.as_ref(), value.as_ref(), Some(options))
            .await
            .map_err(|err| err.into())
    }

    pub async fn kv_get(
        &self,
        key: impl Into<Vec<u8>>,
        options: Option<GetOptions>,
    ) -> Result<Vec<KeyValue>> {
        let mut get_response = self
            .connector
            .get_client()
            .kv_client()
            .get(key, options)
            .await?;
        Ok(get_response.take_kvs())
    }

    pub async fn kv_delete(
        &self,
        key: impl Into<Vec<u8>>,
        options: Option<DeleteOptions>,
    ) -> Result<u64> {
        self.connector
            .get_client()
            .kv_client()
            .delete(key, options)
            .await
            .map(|del_response| del_response.deleted() as u64)
            .map_err(|err| err.into())
    }

    pub async fn kv_get_prefix(&self, prefix: impl AsRef<str>) -> Result<Vec<KeyValue>> {
        let mut get_response = self
            .connector
            .get_client()
            .kv_client()
            .get(prefix.as_ref(), Some(GetOptions::new().with_prefix()))
            .await?;

        Ok(get_response.take_kvs())
    }

    /// Acquire a distributed lock using etcd's native lock mechanism
    /// Returns a LockResponse that can be used to unlock later
    pub async fn lock(
        &self,
        key: impl Into<Vec<u8>>,
        lease_id: Option<u64>,
    ) -> Result<LockResponse> {
        let mut lock_client = self.connector.get_client().lock_client();
        let id = lease_id.unwrap_or(self.lease_id());
        let options = LockOptions::new().with_lease(id as i64);
        lock_client
            .lock(key, Some(options))
            .await
            .map_err(|err| err.into())
    }

    /// Release a distributed lock using the key from the LockResponse
    pub async fn unlock(&self, lock_key: impl Into<Vec<u8>>) -> Result<()> {
        let mut lock_client = self.connector.get_client().lock_client();
        lock_client
            .unlock(lock_key)
            .await
            .map_err(|err: etcd_client::Error| anyhow::anyhow!(err))?;
        Ok(())
    }

    /// Like kv_get_and_watch_prefix but only for new changes, does not include existing values.
    pub async fn kv_watch_prefix(
        &self,
        prefix: impl AsRef<str> + std::fmt::Display,
    ) -> Result<PrefixWatcher> {
        self.watch_internal(prefix, false).await
    }

    pub async fn kv_get_and_watch_prefix(
        &self,
        prefix: impl AsRef<str> + std::fmt::Display,
    ) -> Result<PrefixWatcher> {
        self.watch_internal(prefix, true).await
    }

    /// Core watch implementation that sets up a resilient watcher for a key prefix.
    ///
    /// Creates a background task that maintains a watch stream with automatic reconnection
    /// on recoverable errors. If `include_existing` is true, existing keys are included
    /// in the initial watch events.
    async fn watch_internal(
        &self,
        prefix: impl AsRef<str> + std::fmt::Display,
        include_existing: bool,
    ) -> Result<PrefixWatcher> {
        let (tx, rx) = mpsc::channel(32);

        // Get start revision and send existing KVs
        let mut start_revision = self
            .get_start_revision(
                prefix.as_ref(),
                if include_existing { Some(&tx) } else { None },
            )
            .await?;

        // Resilience watch stream in background
        let connector = self.connector.clone();
        let prefix_str = prefix.as_ref().to_string();
        self.rt.spawn(async move {
            let mut reconnect = true;
            while reconnect {
                // Start a new watch stream
                let watch_stream =
                    match Self::new_watch_stream(&connector, &prefix_str, start_revision).await {
                        Ok(stream) => stream,
                        Err(_) => return,
                    };

                // Watch the stream
                reconnect =
                    Self::monitor_watch_stream(watch_stream, &prefix_str, &mut start_revision, &tx)
                        .await;
            }
        });

        Ok(PrefixWatcher {
            prefix: prefix.as_ref().to_string(),
            rx,
        })
    }

    /// Fetch the initial revision for watching and optionally send existing key-values.
    ///
    /// Returns the next revision to watch from. If `existing_kvs_tx` is provided,
    /// all existing keys with the prefix are sent through the channel first.
    async fn get_start_revision(
        &self,
        prefix: impl AsRef<str> + std::fmt::Display,
        existing_kvs_tx: Option<&mpsc::Sender<WatchEvent>>,
    ) -> Result<i64> {
        let mut kv_client = self.connector.get_client().kv_client();
        let mut get_response = kv_client
            .get(prefix.as_ref(), Some(GetOptions::new().with_prefix()))
            .await?;

        // Get the start revision
        let mut start_revision = get_response
            .header()
            .ok_or(anyhow::anyhow!("missing header; unable to get revision"))?
            .revision();
        tracing::trace!("{prefix}: start_revision: {start_revision}");
        start_revision += 1;

        // Send existing KVs from response if requested
        if let Some(tx) = existing_kvs_tx {
            let kvs = get_response.take_kvs();
            tracing::trace!("initial kv count: {:?}", kvs.len());
            for kv in kvs.into_iter() {
                tx.send(WatchEvent::Put(kv)).await?;
            }
        }

        Ok(start_revision)
    }

    /// Establish a new watch stream with automatic retry and reconnection.
    ///
    /// Attempts to create a watch stream, reconnecting to ETCD if necessary.
    /// Uses a 10-second timeout for reconnection attempts before giving up.
    async fn new_watch_stream(
        connector: &Arc<Connector>,
        prefix: &String,
        start_revision: i64,
    ) -> Result<WatchStream> {
        loop {
            match connector
                .get_client()
                .watch_client()
                .watch(
                    prefix.as_str(),
                    Some(
                        WatchOptions::new()
                            .with_prefix()
                            .with_start_revision(start_revision)
                            .with_prev_key(),
                    ),
                )
                .await
            {
                Ok((_, watch_stream)) => {
                    tracing::debug!("Watch stream established for prefix '{}'", prefix);
                    return Ok(watch_stream);
                }
                Err(err) => {
                    tracing::debug!(error = %err, "Failed to establish watch stream for prefix '{}'", prefix);
                    let deadline = std::time::Instant::now() + Duration::from_secs(10);
                    if let Err(err) = connector.reconnect(deadline).await {
                        tracing::error!(
                            "Failed to reconnect to ETCD within 10 secs for watching prefix '{}': {}",
                            prefix,
                            err
                        );
                        return Err(err);
                    }
                    // continue - retry establishing the watch stream
                }
            }
        }
    }

    /// Monitor a watch stream and forward events to receivers.
    ///
    /// Returns `true` for recoverable errors (network issues, stream closure) that warrant
    /// reconnection attempts. Returns `false` for permanent failures (protocol violations,
    /// channel errors, no receivers) where watching should stop.
    async fn monitor_watch_stream(
        mut watch_stream: WatchStream,
        prefix: &String,
        start_revision: &mut i64,
        tx: &mpsc::Sender<WatchEvent>,
    ) -> bool {
        loop {
            tokio::select! {
                maybe_resp = watch_stream.next() => {
                    // Handle the watch response
                    let response = match maybe_resp {
                        Some(Ok(res)) => res,
                        Some(Err(err)) => {
                            tracing::warn!(error = %err, "Error watching stream for prefix '{}'", prefix);
                            return true; // Exit to reconnect
                        }
                        None => {
                            tracing::warn!("Watch stream unexpectedly closed for prefix '{}'", prefix);
                            return true; // Exit to reconnect
                        }
                    };

                    // Update revision for reconnect
                    *start_revision = match response.header() {
                        Some(header) => header.revision() + 1,
                        None => {
                            tracing::error!("Missing header in watch response for prefix '{}'", prefix);
                            return false;
                        }
                    };

                    // Process events
                    if Self::process_watch_events(response.events(), tx).await.is_err() {
                        return false;
                    };
                }
                _ = tx.closed() => {
                    tracing::debug!("no more receivers, stopping watcher");
                    return false;
                }
            }
        }
    }

    /// Process etcd events and forward them as Put/Delete watch events.
    ///
    /// Filters out events without key-values and transforms etcd events into
    /// appropriate WatchEvent types for channel transmission.
    async fn process_watch_events(
        events: &[etcd_client::Event],
        tx: &mpsc::Sender<WatchEvent>,
    ) -> Result<()> {
        for event in events {
            // Extract the KeyValue if it exists
            let Some(kv) = event.kv() else {
                continue; // Skip events with no KV
            };

            // Handle based on event type
            match event.event_type() {
                etcd_client::EventType::Put => {
                    if let Err(err) = tx.send(WatchEvent::Put(kv.clone())).await {
                        tracing::error!("kv watcher error forwarding WatchEvent::Put: {err}");
                        return Err(err.into());
                    }
                }
                etcd_client::EventType::Delete => {
                    if tx.send(WatchEvent::Delete(kv.clone())).await.is_err() {
                        return Err(anyhow::anyhow!("failed to send WatchEvent::Delete"));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Dissolve)]
pub struct PrefixWatcher {
    prefix: String,
    rx: mpsc::Receiver<WatchEvent>,
}

#[derive(Debug)]
pub enum WatchEvent {
    Put(KeyValue),
    Delete(KeyValue),
}

/// ETCD client configuration options
#[derive(Debug, Clone, Builder, Validate)]
pub struct ClientOptions {
    #[validate(length(min = 1))]
    pub etcd_url: Vec<String>,

    #[builder(default)]
    pub etcd_connect_options: Option<ConnectOptions>,

    /// If true, the client will attach a lease to the primary [`CancellationToken`].
    #[builder(default = "true")]
    pub attach_lease: bool,
}

impl Default for ClientOptions {
    fn default() -> Self {
        let mut connect_options = None;

        if let (Ok(username), Ok(password)) = (
            std::env::var(env_etcd::auth::ETCD_AUTH_USERNAME),
            std::env::var(env_etcd::auth::ETCD_AUTH_PASSWORD),
        ) {
            // username and password are set
            connect_options = Some(ConnectOptions::new().with_user(username, password));
        } else if let (Ok(ca), Ok(cert), Ok(key)) = (
            std::env::var(env_etcd::auth::ETCD_AUTH_CA),
            std::env::var(env_etcd::auth::ETCD_AUTH_CLIENT_CERT),
            std::env::var(env_etcd::auth::ETCD_AUTH_CLIENT_KEY),
        ) {
            // TLS is set
            connect_options = Some(
                ConnectOptions::new().with_tls(
                    TlsOptions::new()
                        .ca_certificate(Certificate::from_pem(ca))
                        .identity(Identity::from_pem(cert, key)),
                ),
            );
        }

        ClientOptions {
            etcd_url: default_servers(),
            etcd_connect_options: connect_options,
            attach_lease: true,
        }
    }
}

fn default_servers() -> Vec<String> {
    match std::env::var(env_etcd::ETCD_ENDPOINTS) {
        Ok(possible_list_of_urls) => possible_list_of_urls
            .split(',')
            .map(|s| s.to_string())
            .collect(),
        Err(_) => vec!["http://localhost:2379".to_string()],
    }
}

/// A cache for etcd key-value pairs that watches for changes
pub struct KvCache {
    client: Client,
    pub prefix: String,
    cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    watcher: Option<PrefixWatcher>,
}

impl KvCache {
    /// Create a new KV cache for the given prefix
    pub async fn new(
        client: Client,
        prefix: String,
        initial_values: HashMap<String, Vec<u8>>,
    ) -> Result<Self> {
        let mut cache = HashMap::new();

        // First get all existing keys with this prefix
        let existing_kvs = client.kv_get_prefix(&prefix).await?;
        for kv in existing_kvs {
            let key = String::from_utf8_lossy(kv.key()).to_string();
            cache.insert(key, kv.value().to_vec());
        }

        // For any keys in initial_values that don't exist in etcd, write them
        // TODO: proper lease handling, this requires the first process that write to a prefix atomically
        // create a lease and write the lease to etcd. Later processes will attach to the lease and
        // help refresh the lease.
        for (key, value) in initial_values.iter() {
            let full_key = format!("{}{}", prefix, key);
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(full_key.clone()) {
                client.kv_put(&full_key, value.clone(), None).await?;
                e.insert(value.clone());
            }
        }

        // Start watching for changes
        // we won't miss events between the initial push and the watcher starting because
        // client.kv_get_and_watch_prefix() will get all kv pairs and put them back again
        let watcher = client.kv_get_and_watch_prefix(&prefix).await?;

        let cache = Arc::new(RwLock::new(cache));
        let mut result = Self {
            client,
            prefix,
            cache,
            watcher: Some(watcher),
        };

        // Start the background watcher task
        result.start_watcher().await?;

        Ok(result)
    }

    /// Start the background watcher task
    async fn start_watcher(&mut self) -> Result<()> {
        if let Some(watcher) = self.watcher.take() {
            let cache = self.cache.clone();
            let prefix = self.prefix.clone();

            tokio::spawn(async move {
                let mut rx = watcher.rx;

                while let Some(event) = rx.recv().await {
                    match event {
                        WatchEvent::Put(kv) => {
                            let key = String::from_utf8_lossy(kv.key()).to_string();
                            let value = kv.value().to_vec();

                            tracing::trace!("KvCache update: {} = {:?}", key, value);
                            let mut cache_write = cache.write().await;
                            cache_write.insert(key, value);
                        }
                        WatchEvent::Delete(kv) => {
                            let key = String::from_utf8_lossy(kv.key()).to_string();

                            tracing::trace!("KvCache delete: {}", key);
                            let mut cache_write = cache.write().await;
                            cache_write.remove(&key);
                        }
                    }
                }

                tracing::debug!("KvCache watcher for prefix '{}' stopped", prefix);
            });
        }

        Ok(())
    }

    /// Get a value from the cache
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let full_key = format!("{}{}", self.prefix, key);
        let cache_read = self.cache.read().await;
        cache_read.get(&full_key).cloned()
    }

    /// Get all key-value pairs in the cache
    pub async fn get_all(&self) -> HashMap<String, Vec<u8>> {
        let cache_read = self.cache.read().await;
        cache_read.clone()
    }

    /// Update a value in both the cache and etcd
    pub async fn put(&self, key: &str, value: Vec<u8>, lease_id: Option<u64>) -> Result<()> {
        let full_key = format!("{}{}", self.prefix, key);

        // Update etcd first
        self.client
            .kv_put(&full_key, value.clone(), lease_id)
            .await?;

        // Then update local cache
        let mut cache_write = self.cache.write().await;
        cache_write.insert(full_key, value);

        Ok(())
    }

    /// Delete a key from both the cache and etcd
    pub async fn delete(&self, key: &str) -> Result<()> {
        let full_key = format!("{}{}", self.prefix, key);

        // Delete from etcd first
        self.client.kv_delete(full_key.clone(), None).await?;

        // Then remove from local cache
        let mut cache_write = self.cache.write().await;
        cache_write.remove(&full_key);

        Ok(())
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod tests {
    use crate::{DistributedRuntime, distributed::DistributedConfig};

    use super::*;

    #[test]
    fn test_ectd_client() {
        let rt = Runtime::from_settings().unwrap();
        let rt_clone = rt.clone();
        let config = DistributedConfig::from_settings();

        rt_clone.primary().block_on(async move {
            let drt = DistributedRuntime::new(rt, config).await.unwrap();
            test_kv_create_or_validate(drt).await.unwrap();
        });
    }

    async fn test_kv_create_or_validate(drt: DistributedRuntime) -> Result<()> {
        let key = "__integration_test_key";
        let value = b"test_value";

        let client = Client::new(ClientOptions::default(), drt.runtime().clone())
            .await
            .expect("etcd client should be available");
        let lease_id = drt.connection_id();

        // Create the key
        let result = client.kv_create(key, value.to_vec(), Some(lease_id)).await;
        assert!(result.is_ok(), "");

        // Try to create the key again - this should fail
        let result = client.kv_create(key, value.to_vec(), Some(lease_id)).await;
        assert!(result.is_err());

        // Create or validate should succeed as the values match
        let result = client
            .kv_create_or_validate(key.to_string(), value.to_vec(), Some(lease_id))
            .await;
        assert!(result.is_ok());

        // Try to create the key with a different value
        let different_value = b"different_value";
        let result = client
            .kv_create_or_validate(key.to_string(), different_value.to_vec(), Some(lease_id))
            .await;
        assert!(result.is_err(), "");

        Ok(())
    }

    #[test]
    fn test_kv_cache() {
        let rt = Runtime::from_settings().unwrap();
        let rt_clone = rt.clone();
        let config = DistributedConfig::from_settings();

        rt_clone.primary().block_on(async move {
            let drt = DistributedRuntime::new(rt, config).await.unwrap();
            test_kv_cache_operations(drt).await.unwrap();
        });
    }

    async fn test_kv_cache_operations(drt: DistributedRuntime) -> Result<()> {
        // Make the client and unwrap it
        let client = Client::new(ClientOptions::default(), drt.runtime().clone())
            .await
            .expect("etcd client should be available");

        // Create a unique test prefix to avoid conflicts with other tests
        let test_id = uuid::Uuid::new_v4().to_string();
        let prefix = format!("v1/test_kv_cache_{}/", test_id);

        // Initial values
        let mut initial_values = HashMap::new();
        initial_values.insert("key1".to_string(), b"value1".to_vec());
        initial_values.insert("key2".to_string(), b"value2".to_vec());

        // Create the KV cache
        let kv_cache = KvCache::new(client.clone(), prefix.clone(), initial_values).await?;

        // Test get
        let value1 = kv_cache.get("key1").await;
        assert_eq!(value1, Some(b"value1".to_vec()));

        let value2 = kv_cache.get("key2").await;
        assert_eq!(value2, Some(b"value2".to_vec()));

        // Test get_all
        let all_values = kv_cache.get_all().await;
        assert_eq!(all_values.len(), 2);
        assert_eq!(
            all_values.get(&format!("{}key1", prefix)),
            Some(&b"value1".to_vec())
        );
        assert_eq!(
            all_values.get(&format!("{}key2", prefix)),
            Some(&b"value2".to_vec())
        );

        // Test put - using None for lease_id
        kv_cache.put("key3", b"value3".to_vec(), None).await?;

        // Allow some time for the update to propagate
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Verify the new value
        let value3 = kv_cache.get("key3").await;
        assert_eq!(value3, Some(b"value3".to_vec()));

        // Test update
        kv_cache
            .put("key1", b"updated_value1".to_vec(), None)
            .await?;

        // Allow some time for the update to propagate
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Verify the updated value
        let updated_value1 = kv_cache.get("key1").await;
        assert_eq!(updated_value1, Some(b"updated_value1".to_vec()));

        // Test external update (simulating another client updating a value)
        client
            .kv_put(
                &format!("{}key2", prefix),
                b"external_update".to_vec(),
                None,
            )
            .await?;

        // Allow some time for the update to propagate
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Verify the cache was updated
        let external_update = kv_cache.get("key2").await;
        assert_eq!(external_update, Some(b"external_update".to_vec()));

        // Clean up - delete the test keys
        let etcd_client = client.etcd_client();
        let _ = etcd_client
            .kv_client()
            .delete(
                prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await?;

        Ok(())
    }
}
