// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

use crate::{
    storage::key_value_store::{Key, KeyValue, WatchEvent},
    transports::etcd,
};
use async_stream::stream;
use async_trait::async_trait;
use etcd_client::{Compare, CompareOp, EventType, PutOptions, Txn, TxnOp, WatchOptions};

use super::{KeyValueBucket, KeyValueStore, StoreError, StoreOutcome};

#[derive(Clone)]
pub struct EtcdStore {
    client: etcd::Client,
}

impl EtcdStore {
    pub fn new(client: etcd::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl KeyValueStore for EtcdStore {
    type Bucket = EtcdBucket;

    /// A "bucket" in etcd is a path prefix
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        _ttl: Option<Duration>, // TODO ttl not used yet
    ) -> Result<Self::Bucket, StoreError> {
        Ok(EtcdBucket {
            client: self.client.clone(),
            bucket_name: bucket_name.to_string(),
        })
    }

    /// A "bucket" in etcd is a path prefix. This creates an EtcdBucket object without doing
    /// any network calls.
    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError> {
        Ok(Some(EtcdBucket {
            client: self.client.clone(),
            bucket_name: bucket_name.to_string(),
        }))
    }

    fn connection_id(&self) -> u64 {
        self.client.lease_id()
    }

    fn shutdown(&self) {
        // Revoke the lease? etcd will do it for us on disconnect.
    }
}

pub struct EtcdBucket {
    client: etcd::Client,
    bucket_name: String,
}

#[async_trait]
impl KeyValueBucket for EtcdBucket {
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        // "version" in etcd speak. revision is a global cluster-wide value
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        let version = revision;
        if version == 0 {
            self.create(key, value).await
        } else {
            self.update(key, value, version).await
        }
    }

    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd get: {k}");

        let mut kvs = self
            .client
            .kv_get(k, None)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Ok(None);
        }
        let (_, val) = kvs.swap_remove(0).into_key_value();
        Ok(Some(val.into()))
    }

    async fn delete(&self, key: &Key) -> Result<(), StoreError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd delete: {k}");
        let _ = self
            .client
            .kv_delete(k, None)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        Ok(())
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        let prefix = make_key(&self.bucket_name, &"".into());
        tracing::trace!("etcd watch: {prefix}");
        let watcher = self
            .client
            .kv_watch_prefix(&prefix)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        let (_, mut watch_stream) = watcher.dissolve();
        let output = stream! {
            while let Some(event) = watch_stream.recv().await {
                match event {
                    etcd::WatchEvent::Put(kv) => {
                        let (k, v) = kv.into_key_value();
                        let key = match String::from_utf8(k) {
                            Ok(k) => Key::new(k),
                            Err(err) => {
                                tracing::error!(%err, prefix, "Invalid UTF8 in etcd key");
                                continue;
                            }
                        };
                        let item = KeyValue::new(key, v.into());
                        yield WatchEvent::Put(item);
                    }
                    etcd::WatchEvent::Delete(kv) => {
                        let (k, _) = kv.into_key_value();
                        let key = match String::from_utf8(k) {
                            Ok(k) => Key::new(k),
                            Err(err) => {
                                tracing::error!(%err, prefix, "Invalid UTF8 in etcd key");
                                continue;
                            }
                        };
                        yield WatchEvent::Delete(key);
                    }
                }
            }
        };
        Ok(Box::pin(output))
    }

    async fn entries(&self) -> Result<HashMap<Key, bytes::Bytes>, StoreError> {
        let k = make_key(&self.bucket_name, &"".into());
        tracing::trace!("etcd entries: {k}");

        let resp = self
            .client
            .kv_get_prefix(k)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        let out: HashMap<Key, bytes::Bytes> = resp
            .into_iter()
            .map(|kv| {
                let (k, v) = kv.into_key_value();
                (Key::new(String::from_utf8_lossy(&k).to_string()), v.into())
            })
            .collect();

        Ok(out)
    }
}

impl EtcdBucket {
    async fn create(
        &self,
        key: &Key,
        value: impl Into<Vec<u8>>,
    ) -> Result<StoreOutcome, StoreError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd create: {k}");

        match self
            .client
            .kv_create(k.as_str(), value.into(), None)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?
        {
            None => {
                // Key was created successfully
                Ok(StoreOutcome::Created(1)) // version of new key is always 1
            }
            Some(revision) => Ok(StoreOutcome::Exists(revision)),
        }
    }

    async fn update(
        &self,
        key: &Key,
        value: impl AsRef<[u8]>,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        let version = revision;
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd update: {k}");

        let kvs = self
            .client
            .kv_get(k.clone(), None)
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Err(StoreError::MissingKey(key.to_string()));
        }
        let current_version = kvs.first().unwrap().version() as u64;
        if current_version != version + 1 {
            tracing::warn!(
                current_version,
                attempted_next_version = version,
                %key,
                "update: Wrong revision"
            );
            // NATS does a resync_update, overwriting the key anyway and getting the new revision.
            // So we do too in etcd.
        }

        let put_options = PutOptions::new()
            .with_lease(self.client.lease_id() as i64)
            .with_prev_key();
        let mut put_resp = self
            .client
            .kv_put_with_options(k, value, Some(put_options))
            .await
            .map_err(|e| StoreError::EtcdError(e.to_string()))?;
        Ok(match put_resp.take_prev_key() {
            // Should this be an error?
            // The key was deleted between our get and put. We re-created it.
            // Version of new key is always 1.
            // <https://etcd.io/docs/v3.5/learning/data_model/>
            None => StoreOutcome::Created(1),
            // Expected case, success
            Some(kv) if kv.version() as u64 == version + 1 => StoreOutcome::Created(version),
            // Should this be an error? Something updated the version between our get and put
            Some(kv) => StoreOutcome::Created(kv.version() as u64 + 1),
        })
    }
}

fn make_key(bucket_name: &str, key: &Key) -> String {
    [bucket_name.to_string(), key.to_string()].join("/")
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod concurrent_create_tests {
    use super::*;
    use crate::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::sync::Arc;
    use tokio::sync::Barrier;

    #[test]
    fn test_concurrent_etcd_create_race_condition() {
        let rt = Runtime::from_settings().unwrap();
        let rt_clone = rt.clone();
        let config = DistributedConfig::from_settings();

        rt_clone.primary().block_on(async move {
            let drt = DistributedRuntime::new(rt, config).await.unwrap();
            test_concurrent_create(drt).await.unwrap();
        });
    }

    async fn test_concurrent_create(drt: DistributedRuntime) -> Result<(), StoreError> {
        let storage = drt.store();

        // Create a bucket for testing
        let bucket = Arc::new(tokio::sync::Mutex::new(
            storage
                .get_or_create_bucket("test_concurrent_bucket", None)
                .await?,
        ));

        // Number of concurrent workers
        let num_workers = 10;
        let barrier = Arc::new(Barrier::new(num_workers));

        // Shared test data
        let test_key: Key = Key::new(format!("concurrent_test_key_{}", uuid::Uuid::new_v4()));
        let test_value = "test_value";

        // Spawn multiple tasks that will all try to create the same key simultaneously
        let mut handles = Vec::new();
        let success_count = Arc::new(tokio::sync::Mutex::new(0));
        let exists_count = Arc::new(tokio::sync::Mutex::new(0));

        for worker_id in 0..num_workers {
            let bucket_clone = bucket.clone();
            let barrier_clone = barrier.clone();
            let key_clone = test_key.clone();
            let value_clone = format!("{}_from_worker_{}", test_value, worker_id);
            let success_count_clone = success_count.clone();
            let exists_count_clone = exists_count.clone();

            let handle = tokio::spawn(async move {
                // Wait for all workers to be ready
                barrier_clone.wait().await;

                // All workers try to create the same key at the same time
                let result = bucket_clone
                    .lock()
                    .await
                    .insert(&key_clone, value_clone.into(), 0)
                    .await;

                match result {
                    Ok(StoreOutcome::Created(version)) => {
                        println!(
                            "Worker {} successfully created key with version {}",
                            worker_id, version
                        );
                        let mut count = success_count_clone.lock().await;
                        *count += 1;
                        Ok(version)
                    }
                    Ok(StoreOutcome::Exists(version)) => {
                        println!(
                            "Worker {} found key already exists with version {}",
                            worker_id, version
                        );
                        let mut count = exists_count_clone.lock().await;
                        *count += 1;
                        Ok(version)
                    }
                    Err(e) => {
                        println!("Worker {} got error: {:?}", worker_id, e);
                        Err(e)
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            if let Ok(version) = result {
                results.push(version);
            }
        }

        // Verify results
        let final_success_count = *success_count.lock().await;
        let final_exists_count = *exists_count.lock().await;

        println!(
            "Final counts - Created: {}, Exists: {}",
            final_success_count, final_exists_count
        );

        // CRITICAL ASSERTIONS:
        // 1. Exactly ONE worker should have successfully created the key
        assert_eq!(
            final_success_count, 1,
            "Exactly one worker should create the key"
        );

        // 2. All other workers should have gotten "Exists" response
        assert_eq!(
            final_exists_count,
            num_workers - 1,
            "All other workers should see key exists"
        );

        // 3. Total successful operations should equal number of workers
        assert_eq!(
            results.len(),
            num_workers,
            "All workers should complete successfully"
        );

        // 4. Verify the key actually exists in etcd
        let stored_value = bucket.lock().await.get(&test_key).await?;
        assert!(stored_value.is_some(), "Key should exist in etcd");

        // 5. The stored value should be from one of the workers
        let stored_str = String::from_utf8(stored_value.unwrap().to_vec()).unwrap();
        assert!(
            stored_str.starts_with(test_value),
            "Stored value should match expected prefix"
        );

        // Clean up
        bucket.lock().await.delete(&test_key).await?;

        Ok(())
    }
}
