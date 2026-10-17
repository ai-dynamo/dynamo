// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::{
    component::Endpoint, pipeline::MultimodalCacheIndex, traits::DistributedRuntimeProvider,
    transports::event_plane::EventSubscriber,
};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::kv_router::{
    MULTIMODAL_EMBEDDING_CACHE_SUBJECT, publisher::MultimodalEmbeddingCacheEvent,
};
use crate::protocols::common::{llm_backend::PreprocessedRequest, preprocessor::MultimodalData};

fn multimodal_cache_key_from_url(url: &str) -> String {
    blake3::hash(url.as_bytes()).to_hex().to_string()
}

pub fn preprocessed_multimodal_cache_keys(request: &PreprocessedRequest) -> Vec<String> {
    let Some(items) = request
        .multi_modal_data
        .as_ref()
        .and_then(|media| media.get("image_url"))
    else {
        return Vec::new();
    };

    let mut keys = Vec::with_capacity(items.len());
    for item in items {
        match item {
            MultimodalData::Url(url) => keys.push(multimodal_cache_key_from_url(url.as_str())),
            MultimodalData::RawUrl(url) => keys.push(multimodal_cache_key_from_url(url)),
            MultimodalData::Decoded(descriptor) => {
                if let Some(key) = descriptor.content_hash_key() {
                    keys.push(key.to_string());
                }
            }
            MultimodalData::UuidOnly(_) => {}
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Cache-key alternatives for each distinct image, including session-scoped workers.
///
/// Workers choose their cache policy locally. Retaining the unscoped key also
/// supports global-cache workers and older workers in a mixed deployment.
/// A scoped worker only publishes the scoped key, so another session cannot match it.
pub fn preprocessed_multimodal_cache_key_alternatives(
    request: &PreprocessedRequest,
) -> Vec<Vec<String>> {
    let keys = preprocessed_multimodal_cache_keys(request);
    let scope = request
        .image_cache_scope
        .as_deref()
        .map(|scope| {
            // Python str.strip also treats the ASCII information separators as whitespace.
            scope.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
        })
        .filter(|scope| !scope.is_empty());
    let Some(scope) = scope else {
        return keys.into_iter().map(|key| vec![key]).collect();
    };
    let scope_digest = format!("{:x}", Sha256::digest(scope.as_bytes()));
    keys.into_iter()
        .map(|key| vec![format!("{scope_digest}:{key}"), key])
        .collect()
}

#[derive(Clone, Default)]
pub struct EmbeddingCacheIndexer {
    key_workers: Arc<DashMap<String, HashSet<WorkerId>>>,
    worker_cache_keys: Arc<DashMap<WorkerId, HashSet<String>>>,
    started: Arc<AtomicBool>,
}

type SharedIndexerKey = (u64, String);
type SharedIndexerMap = HashMap<SharedIndexerKey, Weak<dyn MultimodalCacheIndex>>;

static SHARED_INDEXERS: OnceLock<Mutex<SharedIndexerMap>> = OnceLock::new();

fn shared_indexer(
    indexers: &mut SharedIndexerMap,
    indexer_key: &SharedIndexerKey,
) -> Option<Arc<dyn MultimodalCacheIndex>> {
    indexers.retain(|_, indexer| indexer.strong_count() > 0);
    indexers.get(indexer_key).and_then(Weak::upgrade)
}

pub async fn try_build_cache_indexer(endpoint: &Endpoint) -> Option<Arc<dyn MultimodalCacheIndex>> {
    let indexer_key = (endpoint.drt().connection_id(), endpoint.id().to_string());
    let mut indexers = SHARED_INDEXERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .await;

    if let Some(indexer) = shared_indexer(&mut indexers, &indexer_key) {
        return Some(indexer);
    }

    match EmbeddingCacheIndexer::for_endpoint(endpoint).await {
        Ok(indexer) => {
            let indexer: Arc<dyn MultimodalCacheIndex> = indexer;
            indexers.insert(indexer_key, Arc::downgrade(&indexer));
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
    pub async fn for_endpoint(endpoint: &Endpoint) -> anyhow::Result<Arc<Self>> {
        let indexer = Arc::new(Self::default());
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

    pub async fn start_subscriber(self: &Arc<Self>, endpoint: &Endpoint) -> anyhow::Result<()> {
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

        let indexer = Arc::clone(self);
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
    use super::*;
    use crate::kv_router::publisher::{
        MultimodalEmbeddingCacheEvent, MultimodalEmbeddingCacheUpdate,
    };

    #[tokio::test]
    async fn scoped_embedding_cache_routing_matches_worker_cache_policies() {
        use crate::protocols::common::llm_backend::LLMEngineOutput;
        use dynamo_runtime::{
            DistributedRuntime, Runtime,
            distributed::DistributedConfig,
            pipeline::{PushRouter, RouterMode},
        };
        use sha2::{Digest, Sha256};

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("scoped_embedding_cache_probe")
            .unwrap()
            .component("encoder")
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let mut failures = Vec::new();

        for (name, request_scope, worker_scope, images, cached_images, expected_full_hit) in [
            ("legacy global", None, None, vec!["one"], vec!["one"], true),
            (
                "global worker with scoped request",
                Some("session-a"),
                None,
                vec!["one"],
                vec!["one"],
                true,
            ),
            (
                "same scope",
                Some("session-a"),
                Some("session-a"),
                vec!["one"],
                vec!["one"],
                true,
            ),
            (
                "other scope",
                Some("session-a"),
                Some("session-b"),
                vec!["one"],
                vec!["one"],
                false,
            ),
            (
                "duplicate image",
                Some("session-a"),
                Some("session-a"),
                vec!["one", "one"],
                vec!["one"],
                true,
            ),
            (
                "partial hit",
                Some("session-a"),
                Some("session-a"),
                vec!["one", "two"],
                vec!["one"],
                false,
            ),
            (
                "full two-image hit",
                Some("session-a"),
                Some("session-a"),
                vec!["one", "two"],
                vec!["one", "two"],
                true,
            ),
        ] {
            let indexer = Arc::new(EmbeddingCacheIndexer::default());
            let added_keys = cached_images
                .iter()
                .map(|image| {
                    let key =
                        multimodal_cache_key_from_url(&format!("https://example.com/{image}.png"));
                    match worker_scope {
                        Some(scope) => format!("{:x}:{key}", Sha256::digest(scope.as_bytes())),
                        None => key,
                    }
                })
                .collect();
            indexer.apply_event(&MultimodalEmbeddingCacheEvent {
                worker_id,
                update: MultimodalEmbeddingCacheUpdate {
                    added_keys,
                    removed_keys: vec![],
                },
            });
            if name == "partial hit" {
                indexer.apply_event(&MultimodalEmbeddingCacheEvent {
                    worker_id,
                    update: MultimodalEmbeddingCacheUpdate {
                        added_keys: vec![multimodal_cache_key_from_url(
                            "https://example.com/one.png",
                        )],
                        removed_keys: vec![],
                    },
                });
            }
            let request = PreprocessedRequest::builder()
                .model("model".to_string())
                .token_ids(vec![1])
                .multi_modal_data(Some(HashMap::from([(
                    "image_url".to_string(),
                    images
                        .iter()
                        .map(|image| {
                            MultimodalData::RawUrl(format!("https://example.com/{image}.png"))
                        })
                        .collect(),
                )])))
                .image_cache_scope(request_scope.map(str::to_owned))
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap();
            let router =
                PushRouter::<PreprocessedRequest, LLMEngineOutput>::from_client_with_state(
                    client.clone(),
                    RouterMode::DeviceAwareWeighted,
                    None,
                    Some(indexer),
                    Some(Arc::new(preprocessed_multimodal_cache_keys)),
                )
                .await
                .unwrap()
                .with_multimodal_cache_key_alternatives(
                    preprocessed_multimodal_cache_key_alternatives,
                );
            let selection = router
                .select_device_aware_and_reserve(&request, None)
                .unwrap();
            let hit = selection.embedding_cache_hit();
            let key_count = selection.request_cache_keys();
            let full_hit = selection.into_reservation().is_none();
            eprintln!(
                "{name}: cache_hit={hit}, distinct_images={key_count}, full_hit={full_hit}, expected_full_hit={expected_full_hit}"
            );
            if full_hit != expected_full_hit {
                failures.push(name);
            }
        }
        runtime.shutdown();
        assert!(
            failures.is_empty(),
            "incorrect cache selection/admission: {failures:?}"
        );
    }

    #[test]
    fn scoped_cache_key_alternatives_preserve_media_identity_and_global_fallback() {
        use crate::preprocessor::media::RdmaMediaDataDescriptor;
        use dynamo_memory::nixl::{MemType, NixlDescriptor};

        let url = "https://example.com/one.png";
        let descriptor: RdmaMediaDataDescriptor = serde_json::from_value(serde_json::json!({
            "nixl_metadata": "",
            "nixl_descriptor": NixlDescriptor { addr: 0, size: 3, mem_type: MemType::Dram, device_id: 0 },
            "shape": [1, 1, 3],
            "dtype": "UINT8",
            "content_hash": "0123456789abcdef",
        })).unwrap();
        let mut missing_hash_descriptor = descriptor.clone();
        missing_hash_descriptor.content_hash = None;
        let mut request = PreprocessedRequest::builder()
            .model("model".into())
            .token_ids(vec![1])
            .multi_modal_data(Some(HashMap::from([(
                "image_url".into(),
                vec![
                    MultimodalData::Url(url.parse().unwrap()),
                    MultimodalData::RawUrl(url.into()),
                    MultimodalData::Decoded(descriptor.clone()),
                    MultimodalData::Decoded(missing_hash_descriptor),
                    MultimodalData::UuidOnly("backend-owned".into()),
                ],
            )])))
            .image_cache_scope(Some(" session-a ".into()))
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let raw_keys = preprocessed_multimodal_cache_keys(&request);
        assert_eq!(raw_keys.len(), 2);
        // Matches scope_image_cache_key in the Python worker for session-a.
        let scope_digest = "fa57a52dbf08190218529730a3e99db6946c6c29220fb6e0551e21598b0b05db";
        assert_eq!(
            preprocessed_multimodal_cache_key_alternatives(&request),
            raw_keys
                .iter()
                .map(|key| vec![format!("{scope_digest}:{key}"), key.clone()])
                .collect::<Vec<_>>()
        );
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            request.image_cache_scope = Some(format!("{separator}session-a{separator}"));
            assert_eq!(
                preprocessed_multimodal_cache_key_alternatives(&request),
                raw_keys
                    .iter()
                    .map(|key| vec![format!("{scope_digest}:{key}"), key.clone()])
                    .collect::<Vec<_>>()
            );
        }
        for scope in [None, Some("  ".into()), Some("\u{1c}\u{1f}".into())] {
            request.image_cache_scope = scope;
            assert_eq!(
                preprocessed_multimodal_cache_key_alternatives(&request),
                raw_keys
                    .iter()
                    .map(|key| vec![key.clone()])
                    .collect::<Vec<_>>()
            );
        }
        request.multi_modal_data = None;
        assert!(preprocessed_multimodal_cache_key_alternatives(&request).is_empty());
    }

    #[test]
    fn shared_indexer_cache_prunes_dropped_entries() {
        let live_key = (1, "live".to_string());
        let stale_key = (2, "stale".to_string());
        let live: Arc<dyn MultimodalCacheIndex> = Arc::new(EmbeddingCacheIndexer::default());
        let stale: Arc<dyn MultimodalCacheIndex> = Arc::new(EmbeddingCacheIndexer::default());
        let mut indexers = HashMap::from([
            (live_key.clone(), Arc::downgrade(&live)),
            (stale_key.clone(), Arc::downgrade(&stale)),
        ]);
        drop(stale);

        assert!(shared_indexer(&mut indexers, &live_key).is_some());
        assert!(!indexers.contains_key(&stale_key));

        drop(live);
        assert!(shared_indexer(&mut indexers, &live_key).is_none());
        assert!(indexers.is_empty());
    }

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
