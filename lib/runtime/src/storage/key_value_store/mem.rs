// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rand::Rng as _;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::storage::key_value_store::{Key, KeyValue, WatchEvent};

use super::{KeyValueBucket, KeyValueStore, StoreError, StoreOutcome};

#[derive(Clone, Debug)]
enum MemoryEvent {
    Put { key: String, value: bytes::Bytes },
    Delete { key: String },
}

#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<MemoryStoreInner>,
    connection_id: u64,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

struct MemoryStoreInner {
    data: parking_lot::Mutex<HashMap<String, MemoryBucket>>,
    change_sender: UnboundedSender<MemoryEvent>,
    change_receiver: tokio::sync::Mutex<UnboundedReceiver<MemoryEvent>>,
}

pub struct MemoryBucketRef {
    name: String,
    inner: Arc<MemoryStoreInner>,
}

struct MemoryBucket {
    data: HashMap<String, (u64, bytes::Bytes)>,
}

impl MemoryBucket {
    fn new() -> Self {
        MemoryBucket {
            data: HashMap::new(),
        }
    }
}

impl MemoryStore {
    pub(super) fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        MemoryStore {
            inner: Arc::new(MemoryStoreInner {
                data: parking_lot::Mutex::new(HashMap::new()),
                change_sender: tx,
                change_receiver: tokio::sync::Mutex::new(rx),
            }),
            connection_id: rand::rng().random(),
        }
    }
}

#[async_trait]
impl KeyValueStore for MemoryStore {
    type Bucket = MemoryBucketRef;

    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // MemoryStore doesn't respect TTL yet
        _ttl: Option<Duration>,
    ) -> Result<Self::Bucket, StoreError> {
        let mut locked_data = self.inner.data.lock();
        // Ensure the bucket exists
        locked_data
            .entry(bucket_name.to_string())
            .or_insert_with(MemoryBucket::new);
        // Return an object able to access it
        Ok(MemoryBucketRef {
            name: bucket_name.to_string(),
            inner: self.inner.clone(),
        })
    }

    /// This operation cannot fail on MemoryStore. Always returns Ok.
    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError> {
        let locked_data = self.inner.data.lock();
        match locked_data.get(bucket_name) {
            Some(_) => Ok(Some(MemoryBucketRef {
                name: bucket_name.to_string(),
                inner: self.inner.clone(),
            })),
            None => Ok(None),
        }
    }

    fn connection_id(&self) -> u64 {
        self.connection_id
    }

    fn shutdown(&self) {}
}

#[async_trait]
impl KeyValueBucket for MemoryBucketRef {
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        let mut locked_data = self.inner.data.lock();
        let mut b = locked_data.get_mut(&self.name);
        let Some(bucket) = b.as_mut() else {
            return Err(StoreError::MissingBucket(self.name.to_string()));
        };
        let outcome = match bucket.data.entry(key.to_string()) {
            Entry::Vacant(e) => {
                e.insert((revision, value.clone()));
                let _ = self.inner.change_sender.send(MemoryEvent::Put {
                    key: key.to_string(),
                    value,
                });
                StoreOutcome::Created(revision)
            }
            Entry::Occupied(mut entry) => {
                let (rev, _v) = entry.get();
                if *rev == revision {
                    StoreOutcome::Exists(revision)
                } else {
                    entry.insert((revision, value));
                    StoreOutcome::Created(revision)
                }
            }
        };
        Ok(outcome)
    }

    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError> {
        let locked_data = self.inner.data.lock();
        let Some(bucket) = locked_data.get(&self.name) else {
            return Ok(None);
        };
        Ok(bucket.data.get(&key.0).map(|(_, v)| v.clone()))
    }

    async fn delete(&self, key: &Key) -> Result<(), StoreError> {
        let mut locked_data = self.inner.data.lock();
        let Some(bucket) = locked_data.get_mut(&self.name) else {
            return Err(StoreError::MissingBucket(self.name.to_string()));
        };
        if bucket.data.remove(&key.0).is_some() {
            let _ = self.inner.change_sender.send(MemoryEvent::Delete {
                key: key.to_string(),
            });
        }
        Ok(())
    }

    /// All current values in the bucket first, then block waiting for new
    /// values to be published.
    /// Caller takes the lock so only a single caller may use this at once.
    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        // All the existing ones first
        let mut existing_items = vec![];
        let mut seen_keys = HashSet::new();
        let data_lock = self.inner.data.lock();
        let Some(bucket) = data_lock.get(&self.name) else {
            return Err(StoreError::MissingBucket(self.name.to_string()));
        };
        for (key, (_rev, v)) in &bucket.data {
            seen_keys.insert(key.clone());
            let item = KeyValue::new(key.clone(), v.clone());
            existing_items.push(WatchEvent::Put(item));
        }
        drop(data_lock);

        Ok(Box::pin(async_stream::stream! {
            for event in existing_items {
                yield event;
            }
            // Now any new ones
            let mut rcv_lock = self.inner.change_receiver.lock().await;
            loop {
                match rcv_lock.recv().await {
                    None => {
                        // Channel is closed, no more values coming
                        break;
                    },
                    Some(MemoryEvent::Put { key, value }) => {
                        if seen_keys.contains(&key) {
                            continue;
                        }
                        let item = KeyValue::new(key, value);
                        yield WatchEvent::Put(item);
                    },
                    Some(MemoryEvent::Delete { key }) => {
                        yield WatchEvent::Delete(Key::from_raw(key));
                    }
                }
            }
        }))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StoreError> {
        let locked_data = self.inner.data.lock();
        match locked_data.get(&self.name) {
            Some(bucket) => Ok(bucket
                .data
                .iter()
                .map(|(k, (_rev, v))| (k.to_string(), v.clone()))
                .collect()),
            None => Err(StoreError::MissingBucket(self.name.clone())),
        }
    }
}
