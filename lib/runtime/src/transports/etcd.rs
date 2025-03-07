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
use tokio::sync::mpsc;
use validator::Validate;

use etcd_client::{
    Compare, CompareOp, GetOptions, PutOptions, Txn, TxnOp, TxnOpResponse, WatchOptions, Watcher,
};

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

    pub async fn kv_create(
        &self,
        key: String,
        value: Vec<u8>,
        lease_id: Option<i64>,
    ) -> Result<()> {
        let put_options = lease_id.map(|id| PutOptions::new().with_lease(id));

        // Build the transaction
        let txn = Txn::new()
            .when(vec![Compare::version(key.as_str(), CompareOp::Equal, 0)]) // Ensure the lock does not exist
            .and_then(vec![
                TxnOp::put(key.as_str(), value, put_options), // Create the object
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
        let put_options = lease_id.map(|id| PutOptions::new().with_lease(id));

        // Build the transaction that either creates the key if it doesn't exist,
        // or validates the existing value matches what we expect
        let txn = Txn::new()
            .when(vec![Compare::version(key.as_str(), CompareOp::Equal, 0)]) // Key doesn't exist
            .and_then(vec![
                TxnOp::put(key.as_str(), value.clone(), put_options), // Create it
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
        let _ = self
            .client
            .kv_client()
            .put(
                key.as_ref(),
                value.as_ref(),
                lease_id.map(|id| PutOptions::new().with_lease(id)),
            )
            .await?;

        Ok(())
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

        let (tx, rx) = mpsc::channel(32);

        self.runtime.secondary().spawn(async move {
            for kv in kvs {
                if tx.send(WatchEvent::Put(kv)).await.is_err() {
                    // receiver is closed
                    break;
                }
            }

            while let Some(Ok(response)) = watch_stream.next().await {
                for event in response.events() {
                    match event.event_type() {
                        etcd_client::EventType::Put => {
                            if let Some(kv) = event.kv() {
                                if let Err(err) = tx.send(WatchEvent::Put(kv.clone())).await {
                                    tracing::error!(
                                        "kv watcher error forwarding WatchEvent::Put: {err}"
                                    );
                                    // receiver is closed
                                    break;
                                }
                            }
                        }
                        etcd_client::EventType::Delete => {
                            if let Some(kv) = event.kv() {
                                if tx.send(WatchEvent::Delete(kv.clone())).await.is_err() {
                                    // receiver is closed
                                    break;
                                }
                            }
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
        ClientOptions {
            etcd_url: default_servers(),
            etcd_connect_options: None,
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
}
