// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::{
    component::Endpoint, pipeline::MultimodalCacheIndex, traits::DistributedRuntimeProvider,
    transports::event_plane::EventSubscriber,
};
use tokio::sync::Mutex;

use crate::kv_router::{
    MULTIMODAL_EMBEDDING_CACHE_SUBJECT, publisher::MultimodalEmbeddingCacheEvent,
};

#[derive(Clone, Default)]
pub struct EmbeddingCacheIndexer {
    key_workers: Arc<DashMap<String, HashSet<WorkerId>>>,
    worker_cache_keys: Arc<DashMap<WorkerId, HashSet<String>>>,
    started: Arc<AtomicBool>,
}

static SHARED_INDEXERS: OnceLock<Mutex<HashMap<String, Arc<dyn MultimodalCacheIndex>>>> =
    OnceLock::new();

pub async fn try_build_cache_indexer(endpoint: &Endpoint) -> Option<Arc<dyn MultimodalCacheIndex>> {
    let endpoint_id = endpoint.id().to_string();
    let mut indexers = SHARED_INDEXERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .await;

    if let Some(indexer) = indexers.get(&endpoint_id) {
        return Some(Arc::clone(indexer));
    }

    match EmbeddingCacheIndexer::for_endpoint(endpoint).await {
        Ok(indexer) => {
            let indexer = Arc::new(indexer) as Arc<dyn MultimodalCacheIndex>;
            indexers.insert(endpoint_id, Arc::clone(&indexer));
            Some(indexer)
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "embedding cache indexer subscriber not available; skipping cache-state sync"
            );
            None
        }
    }
}

impl EmbeddingCacheIndexer {
    pub async fn for_endpoint(endpoint: &Endpoint) -> anyhow::Result<Self> {
        let indexer = Self::default();
        indexer.start_subscriber(endpoint).await?;
        Ok(indexer)
    }

    pub fn workers_with_cached_keys<'a, I>(&self, cache_keys: I) -> Vec<WorkerId>
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.worker_cache_key_hits(cache_keys)
            .into_iter()
            .map(|(worker_id, _hits)| worker_id)
            .collect()
    }

    pub fn worker_cache_key_hits<'a, I>(&self, cache_keys: I) -> Vec<(WorkerId, usize)>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let requested = cache_keys.into_iter().collect::<HashSet<_>>();
        if requested.is_empty() {
            return Vec::new();
        }

        // Partial cache hits are useful: for a request with multiple images,
        // a worker that already has some embeddings can still avoid work.
        // Count per-worker hits across the de-duplicated requested keys, then
        // prefer workers with more hits.
        let mut worker_hits = HashMap::<WorkerId, usize>::with_capacity(requested.len());
        for key in requested {
            if let Some(workers) = self.key_workers.get(key) {
                for worker_id in workers.iter() {
                    *worker_hits.entry(*worker_id).or_default() += 1;
                }
            }
        }

        let mut worker_hits = worker_hits.into_iter().collect::<Vec<_>>();
        worker_hits.sort_by(|(left_id, left_hits), (right_id, right_hits)| {
            right_hits
                .cmp(left_hits)
                .then_with(|| left_id.cmp(right_id))
        });
        worker_hits
    }

    pub fn apply_event(&self, event: &MultimodalEmbeddingCacheEvent) {
        self.apply_delta(
            event.worker_id,
            event.update.added_keys.iter().cloned().collect(),
            event.update.removed_keys.iter().cloned().collect(),
        );
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        let Some((_, keys)) = self.worker_cache_keys.remove(&worker_id) else {
            return;
        };

        for key in keys {
            self.remove_worker_from_key(&key, worker_id);
        }
    }

    pub async fn start_subscriber(&self, endpoint: &Endpoint) -> anyhow::Result<()> {
        if self.started.swap(true, Ordering::AcqRel) {
            tracing::debug!("Embedding cache indexer subscriber already started, skipping");
            return Ok(());
        }

        let cancellation_token = endpoint.drt().child_token();
        let endpoint = endpoint.clone();
        let subscriber = match EventSubscriber::for_endpoint(
            &endpoint,
            MULTIMODAL_EMBEDDING_CACHE_SUBJECT,
        )
        .await
        {
            Ok(subscriber) => subscriber.typed::<MultimodalEmbeddingCacheEvent>(),
            Err(error) => {
                self.started.store(false, Ordering::Release);
                return Err(error);
            }
        };

        let indexer = self.clone();
        tokio::spawn(async move {
            let mut subscriber = subscriber;
            const RECONNECT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);

            'reconnect: loop {
                loop {
                    tokio::select! {
                        _ = cancellation_token.cancelled() => {
                            tracing::debug!("Embedding cache indexer subscriber cancelled");
                            break 'reconnect;
                        }
                        maybe_event = subscriber.next() => {
                            let Some(result) = maybe_event else {
                                tracing::warn!(
                                    "Embedding cache indexer stream ended; reconnecting"
                                );
                                break;
                            };

                            match result {
                                Ok((_envelope, event)) => indexer.apply_event(&event),
                                Err(error) => {
                                    tracing::warn!(
                                        "Error receiving multimodal embedding cache event: {error:?}; reconnecting"
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }

                subscriber = loop {
                    tokio::select! {
                        _ = tokio::time::sleep(RECONNECT_BACKOFF) => {}
                        _ = cancellation_token.cancelled() => {
                            tracing::debug!("Embedding cache indexer subscriber cancelled");
                            break 'reconnect;
                        }
                    }

                    match EventSubscriber::for_endpoint(
                        &endpoint,
                        MULTIMODAL_EMBEDDING_CACHE_SUBJECT,
                    )
                    .await
                    {
                        Ok(subscriber) => {
                            break subscriber.typed::<MultimodalEmbeddingCacheEvent>();
                        }
                        Err(error) => {
                            tracing::warn!(
                                "Failed to reconnect embedding cache indexer subscriber (will retry): {error}"
                            );
                        }
                    }
                };
            }

            indexer.started.store(false, Ordering::Release);
        });

        Ok(())
    }

    fn apply_delta(
        &self,
        worker_id: WorkerId,
        added_keys: HashSet<String>,
        removed_keys: HashSet<String>,
    ) {
        let should_remove_worker_entry;

        {
            let mut worker_keys = self.worker_cache_keys.entry(worker_id).or_default();

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

            should_remove_worker_entry = worker_keys.is_empty();
        }

        if should_remove_worker_entry {
            self.worker_cache_keys
                .remove_if(&worker_id, |_, keys| keys.is_empty());
        }
    }

    fn add_worker_to_key(&self, key: String, worker_id: WorkerId) {
        self.key_workers.entry(key).or_default().insert(worker_id);
    }

    fn remove_worker_from_key(&self, key: &str, worker_id: WorkerId) {
        if let Some(mut workers) = self.key_workers.get_mut(key) {
            workers.remove(&worker_id);
        }
        self.key_workers
            .remove_if(key, |_, workers| workers.is_empty());
    }
}

impl MultimodalCacheIndex for EmbeddingCacheIndexer {
    fn workers_with_cache_key_hits(&self, cache_keys: &[String]) -> Vec<(WorkerId, usize)> {
        self.worker_cache_key_hits(cache_keys.iter().map(|key| key.as_str()))
    }

    fn remove_worker(&self, worker_id: WorkerId) {
        EmbeddingCacheIndexer::remove_worker(self, worker_id);
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingCacheIndexer;
    use crate::kv_router::publisher::{
        MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCacheUpdate,
    };

    #[test]
    fn delta_removes_stale_worker_keys() {
        let indexer = EmbeddingCacheIndexer::default();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 7,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["b".to_string(), "a".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 7,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["c".to_string()],
                removed_keys: vec!["a".to_string(), "b".to_string()],
            },
        });

        let worker_keys = indexer.worker_cache_keys.get(&7).unwrap();
        assert_eq!(worker_keys.len(), 1);
        assert!(worker_keys.contains("c"));
    }

    #[test]
    fn workers_with_cached_keys_returns_partial_matches_first() {
        let indexer = EmbeddingCacheIndexer::default();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 1,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["a".to_string(), "b".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 2,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["a".to_string()],
                removed_keys: vec![],
            },
        });

        assert_eq!(indexer.workers_with_cached_keys(["a", "b"]), vec![1, 2]);
        assert_eq!(
            indexer.worker_cache_key_hits(["a", "b"]),
            vec![(1, 2), (2, 1)]
        );
    }

    #[test]
    fn delta_updates_reverse_index() {
        let indexer = EmbeddingCacheIndexer::default();

        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 3,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["a".to_string(), "b".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 4,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec!["a".to_string()],
                removed_keys: vec![],
            },
        });
        indexer.apply_event(&MultimodalEmbeddingCacheEvent {
            worker_id: 3,
            update: MultimodalEmbeddingCacheUpdate {
                added_keys: vec![],
                removed_keys: vec!["b".to_string()],
            },
        });

        assert_eq!(indexer.workers_with_cached_keys(["a"]), vec![3, 4]);
        assert_eq!(indexer.workers_with_cached_keys(["a", "b"]), vec![3, 4]);
    }
}
