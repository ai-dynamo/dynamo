// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rand::Rng as _;
use tokio::sync::broadcast;

use super::{Bucket, Key, KeyValue, Store, StoreError, StoreOutcome, WatchEvent};

const MEMORY_EVENT_BUFFER_CAPACITY: usize = 16_384;

#[derive(Clone, Debug)]
enum MemoryEvent {
    Put {
        bucket: String,
        key: String,
        value: bytes::Bytes,
    },
    Delete {
        bucket: String,
        key: String,
    },
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
    change_sender: broadcast::Sender<MemoryEvent>,
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
        let (tx, _) = broadcast::channel(MEMORY_EVENT_BUFFER_CAPACITY);
        MemoryStore {
            inner: Arc::new(MemoryStoreInner {
                data: parking_lot::Mutex::new(HashMap::new()),
                change_sender: tx,
            }),
            connection_id: rand::rng().random(),
        }
    }
}

#[async_trait]
impl Store for MemoryStore {
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
impl Bucket for MemoryBucketRef {
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
                    bucket: self.name.clone(),
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
                    entry.insert((revision, value.clone()));
                    let _ = self.inner.change_sender.send(MemoryEvent::Put {
                        bucket: self.name.clone(),
                        key: key.to_string(),
                        value,
                    });
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
                bucket: self.name.clone(),
                key: key.to_string(),
            });
        }
        Ok(())
    }

    /// All current values in the bucket first, then block waiting for new
    /// values to be published.
    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        // Subscribe while holding the data lock so the snapshot and incremental stream have no
        // race: mutations take the same lock before broadcasting their update.
        let data_lock = self.inner.data.lock();
        let Some(bucket) = data_lock.get(&self.name) else {
            return Err(StoreError::MissingBucket(self.name.to_string()));
        };
        let mut changes = self.inner.change_sender.subscribe();
        let existing_items: Vec<_> = bucket
            .data
            .iter()
            .map(|(key, (_revision, value))| {
                WatchEvent::Put(KeyValue::new(Key::new(key.clone()), value.clone()))
            })
            .collect();
        drop(data_lock);
        let bucket_name = self.name.clone();
        let inner = self.inner.clone();

        Ok(Box::pin(async_stream::stream! {
            for event in existing_items {
                yield event;
            }
            loop {
                match changes.recv().await {
                    Ok(MemoryEvent::Put { bucket, key, value }) => {
                        if bucket != bucket_name {
                            continue;
                        }
                        let item = KeyValue::new(Key::new(key), value);
                        yield WatchEvent::Put(item);
                    },
                    Ok(MemoryEvent::Delete { bucket, key }) => {
                        if bucket != bucket_name {
                            continue;
                        }
                        yield WatchEvent::Delete(Key::new(key));
                    },
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let snapshot = {
                            let data = inner.data.lock();
                            // Discard retained events that predate this authoritative snapshot.
                            // Mutations take the same lock before broadcasting, so the replacement
                            // receiver observes every update that follows the snapshot.
                            changes = inner.change_sender.subscribe();
                            data.get(&bucket_name)
                                .map(|bucket| {
                                    bucket.data
                                        .iter()
                                        .map(|(key, (_revision, value))| {
                                            (Key::new(key.clone()), value.clone())
                                        })
                                        .collect()
                                })
                                .unwrap_or_default()
                        };
                        yield WatchEvent::Resync(snapshot);
                    },
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }))
    }

    async fn entries(&self) -> Result<HashMap<Key, bytes::Bytes>, StoreError> {
        let locked_data = self.inner.data.lock();
        match locked_data.get(&self.name) {
            Some(bucket) => {
                let mut out = HashMap::new();
                for (k, (_rev, v)) in bucket.data.iter() {
                    let key = Key::new([self.name.clone(), k.to_string()].join("/"));
                    let value = v.clone();
                    out.insert(key, value);
                }
                Ok(out)
            }
            None => Err(StoreError::MissingBucket(self.name.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MEMORY_EVENT_BUFFER_CAPACITY;
    use crate::storage::kv::{Bucket as _, Key, MemoryStore, Store as _, WatchEvent};
    use futures::StreamExt;
    use std::collections::HashSet;
    use std::time::Duration;

    #[tokio::test]
    async fn multiple_watchers_receive_updates_without_cross_bucket_stealing() {
        let store = MemoryStore::new();
        let bucket_a = store.get_or_create_bucket("bucket-a", None).await.unwrap();
        let bucket_b = store.get_or_create_bucket("bucket-b", None).await.unwrap();
        let mut a_first = bucket_a.watch().await.unwrap();
        let mut a_second = bucket_a.watch().await.unwrap();
        let mut b = bucket_b.watch().await.unwrap();

        bucket_a
            .insert(&Key::new("shared-key".to_string()), "a".into(), 1)
            .await
            .unwrap();
        bucket_b
            .insert(&Key::new("shared-key".to_string()), "b".into(), 1)
            .await
            .unwrap();

        for watcher in [&mut a_first, &mut a_second] {
            let WatchEvent::Put(item) = watcher.next().await.unwrap() else {
                panic!("expected bucket-a put");
            };
            assert_eq!(item.value(), b"a");
        }
        let WatchEvent::Put(item) = b.next().await.unwrap() else {
            panic!("expected bucket-b put");
        };
        assert_eq!(item.value(), b"b");
    }

    #[tokio::test]
    async fn watcher_observes_updates_to_an_existing_key() {
        let store = MemoryStore::new();
        let bucket = store.get_or_create_bucket("bucket", None).await.unwrap();
        bucket
            .insert(&Key::new("key".to_string()), "old".into(), 1)
            .await
            .unwrap();
        let mut watcher = bucket.watch().await.unwrap();
        assert!(matches!(watcher.next().await, Some(WatchEvent::Put(_))));

        bucket
            .insert(&Key::new("key".to_string()), "new".into(), 2)
            .await
            .unwrap();
        let WatchEvent::Put(item) = watcher.next().await.unwrap() else {
            panic!("expected updated put");
        };
        assert_eq!(item.value(), b"new");
    }

    #[tokio::test]
    async fn lag_resync_discards_retained_pre_snapshot_events() {
        let store = MemoryStore::new();
        let bucket = store.get_or_create_bucket("bucket", None).await.unwrap();
        let mut watcher = bucket.watch().await.unwrap();
        let key = Key::new("key".to_string());

        for revision in 1..=MEMORY_EVENT_BUFFER_CAPACITY + 1 {
            bucket
                .insert(&key, format!("value-{revision}").into(), revision as u64)
                .await
                .unwrap();
        }

        let WatchEvent::Resync(snapshot) = watcher.next().await.unwrap() else {
            panic!("expected authoritative resync after lag");
        };
        assert_eq!(snapshot.len(), 1);
        let latest_value = format!("value-{}", MEMORY_EVENT_BUFFER_CAPACITY + 1);
        assert_eq!(snapshot.get(&key).unwrap(), latest_value.as_bytes());

        bucket
            .insert(
                &key,
                "post-resync".into(),
                (MEMORY_EVENT_BUFFER_CAPACITY + 2) as u64,
            )
            .await
            .unwrap();
        let next = tokio::time::timeout(Duration::from_secs(1), watcher.next())
            .await
            .expect("post-resync update timed out")
            .unwrap();
        let WatchEvent::Put(item) = next else {
            panic!("expected post-resync put");
        };
        assert_eq!(item.value(), b"post-resync");
    }

    #[tokio::test]
    async fn test_entries_full_path() {
        let m = MemoryStore::new();
        let bucket = m.get_or_create_bucket("bucket1", None).await.unwrap();
        let _ = bucket
            .insert(&Key::new("key1".to_string()), "value1".into(), 0)
            .await
            .unwrap();
        let _ = bucket
            .insert(&Key::new("key2".to_string()), "value2".into(), 0)
            .await
            .unwrap();
        let entries = bucket.entries().await.unwrap();
        let keys: HashSet<Key> = entries.into_keys().collect();
        assert!(keys.contains(&Key::new("bucket1/key1".to_string())));
        assert!(keys.contains(&Key::new("bucket1/key2".to_string())));
    }
}
