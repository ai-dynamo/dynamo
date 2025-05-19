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

use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use dynamo_runtime::{slug::Slug, transports::etcd::Client};
use etcd_client::{EventType, PutOptions, WatchOptions};

use super::{KeyValueBucket, KeyValueStore, StorageError, StorageOutcome};

#[derive(Clone)]
pub struct EtcdStorage {
    client: Client,
}

impl EtcdStorage {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl KeyValueStore for EtcdStorage {
    /// A "bucket" in etcd is a path prefix
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        _ttl: Option<Duration>, // TODO ttl not used yet
    ) -> Result<Box<dyn KeyValueBucket>, StorageError> {
        Ok(self.get_bucket(bucket_name).await?.unwrap())
    }

    /// A "bucket" in etcd is a path prefix. This creates an EtcdBucket object without doing
    /// any network calls.
    async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn KeyValueBucket>>, StorageError> {
        Ok(Some(Box::new(EtcdBucket {
            client: self.client.clone(),
            bucket_name: bucket_name.to_string(),
        })))
    }
}

pub struct EtcdBucket {
    client: Client,
    bucket_name: String,
}

#[async_trait]
impl KeyValueBucket for EtcdBucket {
    async fn insert(
        &self,
        key: String,
        value: String,
        // "version" in etcd speak. revision is a global cluster-wide value
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        let version = revision;
        if version == 0 {
            self.create(&key, &value).await
        } else {
            self.update(&key, &value, version).await
        }
    }

    async fn get(&self, key: &str) -> Result<Option<bytes::Bytes>, StorageError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd get: {k}");

        let mut kvs = self
            .client
            .kv_get(k, None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Ok(None);
        }
        let (_, val) = kvs.swap_remove(0).into_key_value();
        Ok(Some(val.into()))
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let _ = self
            .client
            .kv_delete(key, None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        Ok(())
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = bytes::Bytes> + Send + 'life0>>, StorageError>
    {
        let k = make_key(&self.bucket_name, "");
        tracing::trace!("etcd watch: {k}");
        let (_watcher, mut watch_stream) = self
            .client
            .etcd_client()
            .clone()
            .watch(k.as_bytes(), Some(WatchOptions::new().with_prefix()))
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        let output = stream! {
            while let Ok(Some(resp)) = watch_stream.message().await {
                for e in resp.events() {
                    if matches!(e.event_type(), EventType::Put) && e.kv().is_some() {
                        let b: bytes::Bytes = e.kv().unwrap().value().to_vec().into();
                        yield b;
                    }
                }
            }
        };
        Ok(Box::pin(output))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StorageError> {
        let k = make_key(&self.bucket_name, "");
        tracing::trace!("etcd entries: {k}");

        let resp = self
            .client
            .kv_get_prefix(k)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        let out: HashMap<String, bytes::Bytes> = resp
            .into_iter()
            .map(|kv| {
                let (k, v) = kv.into_key_value();
                (String::from_utf8_lossy(&k).to_string(), v.into())
            })
            .collect();

        Ok(out)
    }
}

impl EtcdBucket {
    async fn create(&self, key: &str, value: &str) -> Result<StorageOutcome, StorageError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd create: {k}");

        // Does it already exists? For 'create' it shouldn't.
        let kvs = self
            .client
            .kv_get(k.clone(), None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        if !kvs.is_empty() {
            let version = kvs.first().unwrap().version();
            return Ok(StorageOutcome::Exists(version as u64));
        }

        // Write it
        let mut put_resp = self
            .client
            .kv_put_with_options(k, value, Some(PutOptions::new().with_prev_key()))
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        // Check if we overwrite something
        if put_resp.take_prev_key().is_some() {
            // Key created between our get and put
            return Err(StorageError::Retry);
        }

        // version of a new key is always 1
        Ok(StorageOutcome::Created(1))
    }

    async fn update(
        &self,
        key: &str,
        value: &str,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        let version = revision;
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd update: {k}");

        let kvs = self
            .client
            .kv_get(k.clone(), None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Err(StorageError::MissingKey(key.to_string()));
        }
        let current_version = kvs.first().unwrap().version() as u64;
        if current_version != version + 1 {
            tracing::warn!(
                current_version,
                attempted_next_version = version,
                key,
                "update: Wrong revision"
            );
            // NATS does a resync_update, overwriting the key anyway and getting the new revision.
            // So we do too in etcd.
        }

        let mut put_resp = self
            .client
            .kv_put_with_options(k, value, Some(PutOptions::new().with_prev_key()))
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        Ok(match put_resp.take_prev_key() {
            // Should this be an error?
            // The key was deleted between our get and put. We re-created it.
            // Version of new key is always 1.
            // <https://etcd.io/docs/v3.5/learning/data_model/>
            None => StorageOutcome::Created(1),
            // Expected case, success
            Some(kv) if kv.version() as u64 == version + 1 => StorageOutcome::Created(version),
            // Should this be an error? Something updated the version between our get and put
            Some(kv) => StorageOutcome::Created(kv.version() as u64 + 1),
        })
    }
}

fn make_key(bucket_name: &str, key: &str) -> String {
    [
        Slug::slugify(bucket_name).to_string(),
        Slug::slugify(key).to_string(),
    ]
    .join("/")
}
