// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::{
    component::Component, traits::DistributedRuntimeProvider,
    pipeline::MultimodalCacheIndex,
    transports::event_plane::EventSubscriber,
};
use tokio_stream::StreamExt;

use crate::kv_router::{
    MULTIMODAL_EMBEDDING_CACHE_SUBJECT,
    publisher::{MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCacheUpdate},
};

#[derive(Clone, Default)]
pub struct EmbeddingCacheIndexer {
    key_workers: Arc<DashMap<String, HashSet<WorkerId>>>,
    worker_cache_keys: Arc<DashMap<WorkerId, HashSet<String>>>,
    started: Arc<AtomicBool>,
}

impl EmbeddingCacheIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn for_component(component: &Component) -> anyhow::Result<Self> {
        let indexer = Self::new();
        indexer.start_subscriber(component).await?;
        Ok(indexer)
    }

    pub fn cache_keys_for_worker(&self, worker_id: WorkerId) -> Vec<String> {
        self.worker_cache_keys
            .get(&worker_id)
            .map(|keys| {
                let mut keys = keys.iter().cloned().collect::<Vec<_>>();
                keys.sort();
                keys
            })
            .unwrap_or_default()
    }

    pub fn workers_with_cached_keys<'a, I>(&self, cache_keys: I) -> Vec<WorkerId>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let requested = cache_keys.into_iter().collect::<HashSet<_>>();
        if requested.is_empty() {
            return Vec::new();
        }

        let mut per_key_sets = requested
            .iter()
            .map(|key| {
                self.key_workers
                    .get(*key)
                    .map(|workers| workers.iter().copied().collect::<HashSet<_>>())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();

        if per_key_sets.iter().any(HashSet::is_empty) {
            return Vec::new();
        }

        per_key_sets.sort_by_key(HashSet::len);
        let mut intersection = per_key_sets.remove(0);
        for workers in per_key_sets {
            intersection.retain(|worker_id| workers.contains(worker_id));
            if intersection.is_empty() {
                return Vec::new();
            }
        }

        let mut worker_ids = intersection.into_iter().collect::<Vec<_>>();
        worker_ids.sort_unstable();
        worker_ids
    }

    pub fn apply_event(&self, event: &MultimodalEmbeddingCacheEvent) {
        match &event.update {
            MultimodalEmbeddingCacheUpdate::Snapshot { cache_keys } => {
                let new_keys = cache_keys.iter().cloned().collect::<HashSet<_>>();
                self.apply_snapshot(event.worker_id, new_keys);
            }
            MultimodalEmbeddingCacheUpdate::Delta {
                added_keys,
                removed_keys,
            } => {
                self.apply_delta(
                    event.worker_id,
                    added_keys.iter().cloned().collect(),
                    removed_keys.iter().cloned().collect(),
                );
            }
        }
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        let Some((_, keys)) = self.worker_cache_keys.remove(&worker_id) else {
            return;
        };

        for key in keys {
            self.remove_worker_from_key(&key, worker_id);
        }
    }

    pub async fn start_subscriber(&self, component: &Component) -> anyhow::Result<()> {
        if self.started.swap(true, Ordering::SeqCst) {
            tracing::debug!("Embedding cache indexer subscriber already started, skipping");
            return Ok(());
        }

        let cancellation_token = component.drt().child_token();
        let subscriber = match EventSubscriber::for_namespace(
            component.namespace(),
            MULTIMODAL_EMBEDDING_CACHE_SUBJECT,
        )
        .await
        {
            Ok(subscriber) => subscriber.typed::<MultimodalEmbeddingCacheEvent>(),
            Err(error) => {
                self.started.store(false, Ordering::SeqCst);
                return Err(error.into());
            }
        };

        let indexer = self.clone();
        tokio::spawn(async move {
            let mut subscriber = subscriber;

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Embedding cache indexer subscriber cancelled");
                        break;
                    }
                    maybe_event = subscriber.next() => {
                        let Some(result) = maybe_event else {
                            tracing::debug!("Embedding cache indexer stream closed");
                            break;
                        };

                        match result {
                            Ok((_envelope, event)) => indexer.apply_event(&event),
                            Err(error) => {
                                tracing::error!(
                                    "Error receiving multimodal embedding cache event: {error:?}"
                                );
                            }
                        }
                    }
                }
            }

            indexer.started.store(false, Ordering::SeqCst);
        });

        Ok(())
    }

    fn apply_snapshot(&self, worker_id: WorkerId, new_keys: HashSet<String>) {
        let existing_keys = self
            .worker_cache_keys
            .get(&worker_id)
            .map(|keys| keys.clone())
            .unwrap_or_default();

        let removed_keys = existing_keys
            .difference(&new_keys)
            .cloned()
            .collect::<Vec<_>>();
        let added_keys = new_keys
            .difference(&existing_keys)
            .cloned()
            .collect::<Vec<_>>();

        for key in removed_keys {
            self.remove_worker_from_key(&key, worker_id);
        }
        for key in &added_keys {
            self.add_worker_to_key(key.clone(), worker_id);
        }

        if new_keys.is_empty() {
            self.worker_cache_keys.remove(&worker_id);
        } else {
            self.worker_cache_keys.insert(worker_id, new_keys);
        }
    }

    fn apply_delta(
        &self,
        worker_id: WorkerId,
        added_keys: HashSet<String>,
        removed_keys: HashSet<String>,
    ) {
        let mut worker_keys = self
            .worker_cache_keys
            .get(&worker_id)
            .map(|keys| keys.clone())
            .unwrap_or_default();

        for key in removed_keys {
            if worker_keys.remove(&key) {
                self.remove_worker_from_key(&key, worker_id);
            }
        }
        for key in added_keys {
            if worker_keys.insert(key.clone()) {
                self.add_worker_to_key(key, worker_id);
            }
        }

        if worker_keys.is_empty() {
            self.worker_cache_keys.remove(&worker_id);
        } else {
            self.worker_cache_keys.insert(worker_id, worker_keys);
        }
    }

    fn add_worker_to_key(&self, key: String, worker_id: WorkerId) {
        self.key_workers.entry(key).or_default().insert(worker_id);
    }

    fn remove_worker_from_key(&self, key: &str, worker_id: WorkerId) {
        let should_remove = if let Some(mut workers) = self.key_workers.get_mut(key) {
            workers.remove(&worker_id);
            workers.is_empty()
        } else {
            false
        };

        if should_remove {
            self.key_workers.remove(key);
        }
    }
}

impl MultimodalCacheIndex for EmbeddingCacheIndexer {
    fn workers_with_all_cache_keys(&self, cache_keys: &[String]) -> Vec<WorkerId> {
        self.workers_with_cached_keys(cache_keys.iter().map(|key| key.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingCacheIndexer;
    use crate::kv_router::publisher::{
        MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCacheUpdate,
    };

    #[test]
    fn embedding_cache_state_replaces_worker_snapshot() {
        let indexer = EmbeddingCacheIndexer::new();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 7,
            update: MultimodalEmbeddingCacheUpdate::Snapshot {
                cache_keys: vec!["b".to_string(), "a".to_string()],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 7,
            update: MultimodalEmbeddingCacheUpdate::Snapshot {
                cache_keys: vec!["c".to_string()],
            },
        });

        assert_eq!(indexer.cache_keys_for_worker(7), vec!["c".to_string()]);
    }

    #[test]
    fn workers_with_cached_keys_requires_all_requested_keys() {
        let indexer = EmbeddingCacheIndexer::new();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 1,
            update: MultimodalEmbeddingCacheUpdate::Snapshot {
                cache_keys: vec!["a".to_string(), "b".to_string()],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 2,
            update: MultimodalEmbeddingCacheUpdate::Snapshot {
                cache_keys: vec!["a".to_string()],
            },
        });

        assert_eq!(indexer.workers_with_cached_keys(["a", "b"]), vec![1]);
    }

    #[test]
    fn delta_updates_reverse_index() {
        let indexer = EmbeddingCacheIndexer::new();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 3,
            update: MultimodalEmbeddingCacheUpdate::Delta {
                added_keys: vec!["a".to_string(), "b".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 4,
            update: MultimodalEmbeddingCacheUpdate::Delta {
                added_keys: vec!["a".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 3,
            update: MultimodalEmbeddingCacheUpdate::Delta {
                added_keys: vec![],
                removed_keys: vec!["b".to_string()],
            },
        });

        assert_eq!(indexer.workers_with_cached_keys(["a"]), vec![3, 4]);
        assert!(indexer.workers_with_cached_keys(["a", "b"]).is_empty());
    }
}