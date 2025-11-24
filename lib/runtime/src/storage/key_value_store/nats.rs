// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, pin::Pin, time::Duration};

use crate::{
    protocols::EndpointId,
    slug::Slug,
    storage::key_value_store::{Key, KeyValue, WatchEvent},
    transports::nats::Client,
};
use async_nats::jetstream::kv::Operation;
use async_trait::async_trait;
use futures::StreamExt;

use super::{KeyValueBucket, KeyValueStore, StoreError, StoreOutcome};

#[derive(Clone)]
pub struct NATSStore {
    client: Client,
    endpoint: EndpointId,
}

pub struct NATSBucket {
    nats_store: async_nats::jetstream::kv::Store,
}

#[async_trait]
impl KeyValueStore for NATSStore {
    type Bucket = NATSBucket;

    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        ttl: Option<Duration>,
    ) -> Result<Self::Bucket, StoreError> {
        let name = Slug::slugify(bucket_name);
        let nats_store = self
            .get_or_create_key_value(&self.endpoint.namespace, &name, ttl)
            .await?;
        Ok(NATSBucket { nats_store })
    }

    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError> {
        let name = Slug::slugify(bucket_name);
        match self.get_key_value(&self.endpoint.namespace, &name).await? {
            Some(nats_store) => Ok(Some(NATSBucket { nats_store })),
            None => Ok(None),
        }
    }

    fn connection_id(&self) -> u64 {
        self.client.client().server_info().client_id
    }

    fn shutdown(&self) {
        // TODO: Track and delete any owned keys
        // The TTL should ensure NATS does it, but best we do it immediately
    }
}

impl NATSStore {
    pub fn new(client: Client, endpoint: EndpointId) -> Self {
        NATSStore { client, endpoint }
    }

    /// Get or create a key-value store (aka bucket) in NATS.
    ///
    /// ttl is only used if we are creating the bucket, so if that has
    /// changed first delete the bucket.
    async fn get_or_create_key_value(
        &self,
        namespace: &str,
        bucket_name: &Slug,
        // Delete entries older than this
        ttl: Option<Duration>,
    ) -> Result<async_nats::jetstream::kv::Store, StoreError> {
        if let Ok(Some(kv)) = self.get_key_value(namespace, bucket_name).await {
            return Ok(kv);
        }

        // It doesn't exist, create it

        let bucket_name = single_name(namespace, bucket_name);
        let js = self.client.jetstream();
        let create_result = js
            .create_key_value(
                // TODO: configure the bucket, probably need to pass some of these values in
                async_nats::jetstream::kv::Config {
                    bucket: bucket_name.clone(),
                    max_age: ttl.unwrap_or_default(),
                    ..Default::default()
                },
            )
            .await;
        let nats_store = create_result
            .map_err(|err| StoreError::KeyValueError(err.to_string(), bucket_name.clone()))?;
        tracing::debug!("Created bucket {bucket_name}");
        Ok(nats_store)
    }

    async fn get_key_value(
        &self,
        namespace: &str,
        bucket_name: &Slug,
    ) -> Result<Option<async_nats::jetstream::kv::Store>, StoreError> {
        let bucket_name = single_name(namespace, bucket_name);
        let js = self.client.jetstream();

        use async_nats::jetstream::context::KeyValueErrorKind;
        match js.get_key_value(&bucket_name).await {
            Ok(store) => Ok(Some(store)),
            Err(err) if err.kind() == KeyValueErrorKind::GetBucket => {
                // bucket doesn't exist
                Ok(None)
            }
            Err(err) => Err(StoreError::KeyValueError(err.to_string(), bucket_name)),
        }
    }
}

#[async_trait]
impl KeyValueBucket for NATSBucket {
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        if revision == 0 {
            self.create(key, value).await
        } else {
            self.update(key, value, revision).await
        }
    }

    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError> {
        self.nats_store
            .get(key)
            .await
            .map_err(|e| StoreError::NATSError(e.to_string()))
    }

    async fn delete(&self, key: &Key) -> Result<(), StoreError> {
        self.nats_store
            .delete(key)
            .await
            .map_err(|e| StoreError::NATSError(e.to_string()))
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        let watch_stream = self
            .nats_store
            .watch_all()
            .await
            .map_err(|e| StoreError::NATSError(e.to_string()))?;
        // Map the `Entry` to `Entry.value` which is Bytes of the stored value.
        Ok(Box::pin(
            watch_stream.filter_map(
                |maybe_entry: Result<
                    async_nats::jetstream::kv::Entry,
                    async_nats::error::Error<_>,
                >| async move {
                    match maybe_entry {
                        Ok(entry) => {
                            let key = Key::new(entry.key);
                            Some(match entry.operation {
                                Operation::Put => {
                                    let item = KeyValue::new(key, entry.value);
                                    WatchEvent::Put(item)
                                }
                                Operation::Delete => WatchEvent::Delete(key),
                                // TODO: What is Purge? Not urgent, NATS impl not used
                                Operation::Purge => WatchEvent::Delete(key),
                            })
                        }
                        Err(e) => {
                            tracing::error!(error=%e, "watch fatal err");
                            None
                        }
                    }
                },
            ),
        ))
    }

    async fn entries(&self) -> Result<HashMap<Key, bytes::Bytes>, StoreError> {
        let mut key_stream = self
            .nats_store
            .keys()
            .await
            .map_err(|e| StoreError::NATSError(e.to_string()))?;
        let mut out = HashMap::new();
        while let Some(Ok(key)) = key_stream.next().await {
            if let Ok(Some(entry)) = self.nats_store.entry(&key).await {
                out.insert(Key::new(key), entry.value);
            }
        }
        Ok(out)
    }
}

impl NATSBucket {
    async fn create(&self, key: &Key, value: bytes::Bytes) -> Result<StoreOutcome, StoreError> {
        match self.nats_store.create(&key, value).await {
            Ok(revision) => Ok(StoreOutcome::Created(revision)),
            Err(err) if err.kind() == async_nats::jetstream::kv::CreateErrorKind::AlreadyExists => {
                // key exists, get the revsion
                match self.nats_store.entry(key).await {
                    Ok(Some(entry)) => Ok(StoreOutcome::Exists(entry.revision)),
                    Ok(None) => {
                        tracing::error!(
                            %key,
                            "Race condition, key deleted between create and fetch. Retry."
                        );
                        Err(StoreError::Retry)
                    }
                    Err(err) => Err(StoreError::NATSError(err.to_string())),
                }
            }
            Err(err) => Err(StoreError::NATSError(err.to_string())),
        }
    }

    async fn update(
        &self,
        key: &Key,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        match self.nats_store.update(key, value.clone(), revision).await {
            Ok(revision) => Ok(StoreOutcome::Created(revision)),
            Err(err)
                if err.kind() == async_nats::jetstream::kv::UpdateErrorKind::WrongLastRevision =>
            {
                tracing::warn!(revision, %key, "Update WrongLastRevision, resync");
                self.resync_update(key, value).await
            }
            Err(err) => Err(StoreError::NATSError(err.to_string())),
        }
    }

    /// We have the wrong revision for a key. Fetch it's entry to get the correct revision,
    /// and try the update again.
    async fn resync_update(
        &self,
        key: &Key,
        value: bytes::Bytes,
    ) -> Result<StoreOutcome, StoreError> {
        match self.nats_store.entry(key).await {
            Ok(Some(entry)) => {
                // Re-try the update with new version number
                let next_rev = entry.revision + 1;
                match self.nats_store.update(key, value, next_rev).await {
                    Ok(correct_revision) => Ok(StoreOutcome::Created(correct_revision)),
                    Err(err) => Err(StoreError::NATSError(format!(
                        "Error during update of key {key} after resync: {err}"
                    ))),
                }
            }
            Ok(None) => {
                tracing::warn!(%key, "Entry does not exist during resync, creating.");
                self.create(key, value).await
            }
            Err(err) => {
                tracing::error!(%key, %err, "Failed fetching entry during resync");
                Err(StoreError::NATSError(err.to_string()))
            }
        }
    }
}

/// async-nats won't let us use a multi-part subject to create KV buckets (and probably many other
/// things).
fn single_name(namespace: &str, name: &Slug) -> String {
    format!("{namespace}_{name}")
}
