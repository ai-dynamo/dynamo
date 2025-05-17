// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{error, CancellationToken, ErrorContext, Result, Runtime};

use async_nats::jetstream::kv;
use derive_builder::Builder;
use derive_getters::Dissolve;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use validator::Validate;

use etcd_client::{
    Certificate, Compare, CompareOp, DeleteOptions, GetOptions, Identity, PutOptions, PutResponse,
    TlsOptions, Txn, TxnOp, TxnOpResponse, WatchOptions, Watcher,
};
use tokio::time::{interval, Duration};
pub use etcd_client::{ConnectOptions, KeyValue, LeaseClient};

mod lease;
use lease::*;

//pub use etcd::ConnectOptions as EtcdConnectOptions;

/// ETCD Client
#[derive(Clone)]
pub struct Client {
    client: etcd_client::Client,
    primary_lease: i64,
    runtime: Runtime,
}

#[derive(Debug, Clone)]
pub struct Lease {
    /// ETCD lease ID
    id: i64,

    /// [`CancellationToken`] associated with the lease
    cancel_token: CancellationToken,
}

impl Lease {
    /// Get the lease ID
    pub fn id(&self) -> i64 {
        self.id
    }

    /// Get the primary [`CancellationToken`] associated with the lease.
    /// This token will revoke the lease if canceled.
    pub fn primary_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Get a child [`CancellationToken`] from the lease's [`CancellationToken`].
    /// This child token will be triggered if the lease is revoked, but will not revoke the lease if canceled.
    pub fn child_token(&self) -> CancellationToken {
        self.cancel_token.child_token()
    }

    /// Revoke the lease triggering the [`CancellationToken`].
    pub fn revoke(&self) {
        self.cancel_token.cancel();
    }

    /// Check if the lease is still valid (not revoked)
    pub async fn is_valid(&self) -> Result<bool> {
        // A lease is valid if its cancellation token has not been triggered
        // We can use try_cancelled which returns immediately with a boolean
        Ok(!self.cancel_token.is_cancelled())
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
        runtime
            .secondary()
            .spawn(Self::create(config, runtime.clone()))
            .await?
    }

    /// Create a new etcd client and tie the primary [`CancellationToken`] to the primary etcd lease.
    async fn create(config: ClientOptions, runtime: Runtime) -> Result<Self> {
        let token = runtime.primary_token();
        let client =
            etcd_client::Client::connect(config.etcd_url, config.etcd_connect_options).await?;

        let lease_id = if config.attach_lease {
            let lease_client = client.lease_client();

            let lease = create_lease(lease_client, 10, token)
                .await
                .context("creating primary lease")?;

            lease.id
        } else {
            0
        };

        Ok(Client {
            client,
            primary_lease: lease_id,
            runtime,
        })
    }

    /// Get a reference to the underlying [`etcd_client::Client`] instance.
    pub fn etcd_client(&self) -> &etcd_client::Client {
        &self.client
    }

    /// Get the primary lease ID.
    pub fn lease_id(&self) -> i64 {
        self.primary_lease
    }

    /// Primary [`Lease`]
    pub fn primary_lease(&self) -> Lease {
        Lease {
            id: self.primary_lease,
            cancel_token: self.runtime.primary_token(),
        }
    }

    /// Create a [`Lease`] with a given time-to-live (TTL).
    /// This [`Lease`] will be tied to the [`Runtime`], specifically a child [`CancellationToken`].
    pub async fn create_lease(&self, ttl: i64) -> Result<Lease> {
        let token = self.runtime.child_token();
        let lease_client = self.client.lease_client();
        self.runtime
            .secondary()
            .spawn(create_lease(lease_client, ttl, token))
            .await?
    }

    // Revoke an etcd lease given its lease id. A wrapper over etcd_client::LeaseClient::revoke
    pub async fn revoke_lease(&self, lease_id: i64) -> Result<()> {
        let lease_client = self.client.lease_client();
        self.runtime
            .secondary()
            .spawn(revoke_lease(lease_client, lease_id))
            .await?
    }

    pub async fn kv_create(
        &self,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> Result<()> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id);

        // Build the transaction
        let txn = Txn::new()
            .when(vec![Compare::version(key.as_str(), CompareOp::Equal, 0)]) // Ensure the lock does not exist
            .and_then(vec![
                TxnOp::put(key.as_str(), value, Some(put_options)), // Create the object
            ]);

        // Execute the transaction
        let result = self.client.kv_client().txn(txn).await?;

        if result.succeeded() {
            Ok(())
        } else {
            for resp in result.op_responses() {
                tracing::warn!("kv_create etcd op response: {resp:?}");
            }
            Err(error!("failed to create key"))
        }
    }

    /// Atomically create a key if it does not exist, or validate the values are identical if the key exists.
    pub async fn kv_create_or_validate(
        &self,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> Result<()> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id);

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
        let result = self.client.kv_client().txn(txn).await?;

        // We have to enumerate the response paths to determine if the transaction succeeded
        if result.succeeded() {
            Ok(())
        } else {
            match result.op_responses().first() {
                Some(response) => match response {
                    TxnOpResponse::Txn(response) => match response.succeeded() {
                        true => Ok(()),
                        false => Err(error!("failed to create or validate key")),
                    },
                    _ => Err(error!("unexpected response type")),
                },
                None => Err(error!("failed to create or validate key")),
            }
        }
    }

    pub async fn kv_put(
        &self,
        key: impl AsRef<str>,
        value: impl AsRef<[u8]>,
        lease_id: Option<i64>,
    ) -> Result<()> {
        let id = lease_id.unwrap_or(self.lease_id());
        let put_options = PutOptions::new().with_lease(id);
        let _ = self
            .client
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
            .with_lease(self.primary_lease().id());
        self.client
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
        let mut get_response = self.client.kv_client().get(key, options).await?;
        Ok(get_response.take_kvs())
    }

    pub async fn kv_delete(
        &self,
        key: impl Into<Vec<u8>>,
        options: Option<DeleteOptions>,
    ) -> Result<i64> {
        self.client
            .kv_client()
            .delete(key, options)
            .await
            .map(|del_response| del_response.deleted())
            .map_err(|err| err.into())
    }

    pub async fn kv_get_prefix(&self, prefix: impl AsRef<str>) -> Result<Vec<KeyValue>> {
        let mut get_response = self
            .client
            .kv_client()
            .get(prefix.as_ref(), Some(GetOptions::new().with_prefix()))
            .await?;

        Ok(get_response.take_kvs())
    }

    pub async fn kv_get_and_watch_prefix(
        &self,
        prefix: impl AsRef<str> + std::fmt::Display,
    ) -> Result<PrefixWatcher> {
        let mut kv_client = self.client.kv_client();
        let mut watch_client = self.client.watch_client();

        let mut get_response = kv_client
            .get(prefix.as_ref(), Some(GetOptions::new().with_prefix()))
            .await?;

        let start_revision = get_response
            .header()
            .ok_or(error!("missing header; unable to get revision"))?
            .revision();

        tracing::trace!("{prefix}: start_revision: {start_revision}");
        let start_revision = start_revision + 1;

        let (watcher, mut watch_stream) = watch_client
            .watch(
                prefix.as_ref(),
                Some(
                    WatchOptions::new()
                        .with_prefix()
                        .with_start_revision(start_revision)
                        .with_prev_key(),
                ),
            )
            .await?;

        let kvs = get_response.take_kvs();
        tracing::trace!("initial kv count: {:?}", kvs.len());

        let (tx, rx) = mpsc::channel(2048);

        self.runtime.secondary().spawn(async move {
            for kv in kvs {
                if tx.send(WatchEvent::Put(kv)).await.is_err() {
                    // receiver is already closed
                    return;
                }
            }
        
            let mut ticker = interval(Duration::from_secs(15));
        
            loop {
                tokio::select! {
                    maybe_resp = watch_stream.next() => {
                        match maybe_resp {
                            Some(Ok(response)) => {
                                for event in response.events() {
                                    match event.event_type() {
                                        etcd_client::EventType::Put => {
                                            if let Some(kv) = event.kv() {
                                                if let Err(err) = tx.send(WatchEvent::Put(kv.clone())).await {
                                                    tracing::error!(
                                                        "kv watcher error forwarding WatchEvent::Put: {err}"
                                                    );
                                                    // receiver is closed → shut everything down
                                                    return;
                                                }
                                            }
                                        }
                                        etcd_client::EventType::Delete => {
                                            if let Some(kv) = event.kv() {
                                                if tx.send(WatchEvent::Delete(kv.clone())).await.is_err() {
                                                    // receiver is closed
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // the stream ended or errored out
                            _ => {
                                tracing::info!("kv watch stream closed");
                                return;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if tx.is_closed() {
                            tracing::debug!("no more receivers, stopping watcher");
                            return;
                        }
                    }
                }
            }
        });
        Ok(PrefixWatcher {
            prefix: prefix.as_ref().to_string(),
            watcher,
            rx,
        })
    }
}

#[derive(Dissolve)]
pub struct PrefixWatcher {
    prefix: String,
    watcher: Watcher,
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
            std::env::var("ETCD_AUTH_USERNAME"),
            std::env::var("ETCD_AUTH_PASSWORD"),
        ) {
            // username and password are set
            connect_options = Some(ConnectOptions::new().with_user(username, password));
        } else if let (Ok(ca), Ok(cert), Ok(key)) = (
            std::env::var("ETCD_AUTH_CA"),
            std::env::var("ETCD_AUTH_CLIENT_CERT"),
            std::env::var("ETCD_AUTH_CLIENT_KEY"),
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
    match std::env::var("ETCD_ENDPOINTS") {
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
        // we won't miss events bewteen the initial push and the watcher starting because
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

                            tracing::debug!("KvCache update: {} = {:?}", key, value);
                            let mut cache_write = cache.write().await;
                            cache_write.insert(key, value);
                        }
                        WatchEvent::Delete(kv) => {
                            let key = String::from_utf8_lossy(kv.key()).to_string();

                            tracing::debug!("KvCache delete: {}", key);
                            let mut cache_write = cache.write().await;
                            cache_write.remove(&key);
                        }
                    }
                }

                tracing::info!("KvCache watcher for prefix '{}' stopped", prefix);
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
    pub async fn put(&self, key: &str, value: Vec<u8>, lease_id: Option<i64>) -> Result<()> {
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
    use crate::{distributed::DistributedConfig, DistributedRuntime};

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

        let client = drt.etcd_client();
        let lease_id = drt.primary_lease().id();

        // Create the key
        let result = client
            .kv_create(key.to_string(), value.to_vec(), Some(lease_id))
            .await;
        assert!(result.is_ok(), "");

        // Try to create the key again - this should fail
        let result = client
            .kv_create(key.to_string(), value.to_vec(), Some(lease_id))
            .await;
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
        let config = DistributedConfig::from_settings(false);

        rt_clone.primary().block_on(async move {
            let drt = DistributedRuntime::new(rt, config).await.unwrap();
            test_kv_cache_operations(drt).await.unwrap();
        });
    }

    async fn test_kv_cache_operations(drt: DistributedRuntime) -> Result<()> {
        // Get the client and unwrap it
        let client = drt.etcd_client().expect("etcd client should be available");

        // Create a unique test prefix to avoid conflicts with other tests
        let test_id = uuid::Uuid::new_v4().to_string();
        let prefix = format!("test_kv_cache_{}/", test_id);

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
