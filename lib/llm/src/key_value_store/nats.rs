// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::{collections::HashMap, pin::Pin, time::Duration};

use async_trait::async_trait;
use dynamo_runtime::{protocols::Endpoint, slug::Slug, transports::nats::Client};
use futures::StreamExt;

use super::{KeyValueBucket, KeyValueStore, StorageError, StorageOutcome};

#[derive(Clone)]
pub struct NATSStorage {
    client: Client,
    endpoint: Endpoint,
}

pub struct NATSBucket {
    nats_store: async_nats::jetstream::kv::Store,
}

#[async_trait]
impl KeyValueStore for NATSStorage {
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        ttl: Option<Duration>,
    ) -> Result<Box<dyn KeyValueBucket>, StorageError> {
        let name = Slug::slugify(bucket_name);
        let nats_store = self
            .get_or_create_key_value(&self.endpoint.namespace, &name, ttl)
            .await?;
        Ok(Box::new(NATSBucket { nats_store }))
    }

    async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn KeyValueBucket>>, StorageError> {
        let name = Slug::slugify(bucket_name);
        match self.get_key_value(&self.endpoint.namespace, &name).await? {
            Some(nats_store) => Ok(Some(Box::new(NATSBucket { nats_store }))),
            None => Ok(None),
        }
    }
}

impl NATSStorage {
    pub fn new(client: Client, endpoint: Endpoint) -> Self {
        NATSStorage { client, endpoint }
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
    ) -> Result<async_nats::jetstream::kv::Store, StorageError> {
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
        tracing::debug!("Created bucket {bucket_name}");
        create_result.map_err(|err| StorageError::KeyValueError(err.to_string(), bucket_name))
    }

    async fn get_key_value(
        &self,
        namespace: &str,
        bucket_name: &Slug,
    ) -> Result<Option<async_nats::jetstream::kv::Store>, StorageError> {
        let bucket_name = single_name(namespace, bucket_name);
        let js = self.client.jetstream();

        use async_nats::jetstream::context::KeyValueErrorKind;
        match js.get_key_value(&bucket_name).await {
            Ok(store) => Ok(Some(store)),
            Err(err) if err.kind() == KeyValueErrorKind::GetBucket => {
                // bucket doesn't exist
                Ok(None)
            }
            Err(err) => Err(StorageError::KeyValueError(err.to_string(), bucket_name)),
        }
    }
}

#[async_trait]
impl KeyValueBucket for NATSBucket {
    async fn insert(
        &self,
        key: String,
        value: String,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        if revision == 0 {
            self.create(key, value).await
        } else {
            self.update(key, value, revision).await
        }
    }

    async fn get(&self, key: &str) -> Result<Option<bytes::Bytes>, StorageError> {
        self.nats_store
            .get(key)
            .await
            .map_err(|e| StorageError::NATSError(e.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        self.nats_store
            .delete(key)
            .await
            .map_err(|e| StorageError::NATSError(e.to_string()))
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = bytes::Bytes> + Send + 'life0>>, StorageError>
    {
        let watch_stream = self
            .nats_store
            .watch_all()
            .await
            .map_err(|e| StorageError::NATSError(e.to_string()))?;
        // Map the `Entry` to `Entry.value` which is Bytes of the stored value.
        Ok(Box::pin(
            watch_stream.filter_map(
                |maybe_entry: Result<
                    async_nats::jetstream::kv::Entry,
                    async_nats::error::Error<_>,
                >| async move {
                    match maybe_entry {
                        Ok(entry) => Some(entry.value),
                        Err(e) => {
                            tracing::error!(error=%e, "watch fatal err");
                            None
                        }
                    }
                },
            ),
        ))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StorageError> {
        let mut key_stream = self
            .nats_store
            .keys()
            .await
            .map_err(|e| StorageError::NATSError(e.to_string()))?;
        let mut out = HashMap::new();
        while let Some(Ok(key)) = key_stream.next().await {
            if let Ok(Some(entry)) = self.nats_store.entry(&key).await {
                out.insert(key, entry.value);
            }
        }
        Ok(out)
    }
}

impl NATSBucket {
    async fn create(&self, key: String, value: String) -> Result<StorageOutcome, StorageError> {
        match self.nats_store.create(&key, value.into()).await {
            Ok(revision) => Ok(StorageOutcome::Created(revision)),
            Err(err) if err.kind() == async_nats::jetstream::kv::CreateErrorKind::AlreadyExists => {
                // key exists, get the revsion
                match self.nats_store.entry(&key).await {
                    Ok(Some(entry)) => Ok(StorageOutcome::Exists(entry.revision)),
                    Ok(None) => {
                        tracing::error!(
                            key,
                            "Race condition, key deleted between create and fetch. Retry."
                        );
                        Err(StorageError::Retry)
                    }
                    Err(err) => Err(StorageError::NATSError(err.to_string())),
                }
            }
            Err(err) => Err(StorageError::NATSError(err.to_string())),
        }
    }

    async fn update(
        &self,
        key: String,
        value: String,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        match self
            .nats_store
            .update(key.clone(), value.clone().into(), revision)
            .await
        {
            Ok(revision) => Ok(StorageOutcome::Created(revision)),
            Err(err)
                if err.kind() == async_nats::jetstream::kv::UpdateErrorKind::WrongLastRevision =>
            {
                tracing::warn!(revision, key, "Update WrongLastRevision, resync");
                self.resync_update(key, value).await
            }
            Err(err) => Err(StorageError::NATSError(err.to_string())),
        }
    }

    /// We have the wrong revision for a key. Fetch it's entry to get the correct revision,
    /// and try the update again.
    async fn resync_update(
        &self,
        key: String,
        value: String,
    ) -> Result<StorageOutcome, StorageError> {
        match self.nats_store.entry(&key).await {
            Ok(Some(entry)) => {
                // Re-try the update with new version number
                let next_rev = entry.revision + 1;
                match self
                    .nats_store
                    .update(key.clone(), value.into(), next_rev)
                    .await
                {
                    Ok(correct_revision) => Ok(StorageOutcome::Created(correct_revision)),
                    Err(err) => Err(StorageError::NATSError(format!(
                        "Error during update of key {key} after resync: {err}"
                    ))),
                }
            }
            Ok(None) => {
                tracing::warn!(key, "Entry does not exist during resync, creating.");
                self.create(key, value).await
            }
            Err(err) => {
                tracing::error!(key, %err, "Failed fetching entry during resync");
                Err(StorageError::NATSError(err.to_string()))
            }
        }
    }
}

/// async-nats won't let us use a multi-part subject to create KV buckets (and probably many other
/// things).
fn single_name(namespace: &str, name: &Slug) -> String {
    format!("{namespace}_{name}")
}
