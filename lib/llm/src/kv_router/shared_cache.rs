// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HiCache shared KV cache client for SGLang + Mooncake.
//!
//! This client:
//! 1. Reads Mooncake HiCache metadata published by SGLang workers in runtime config.
//! 2. Recomputes the logical HiCache page hashes from request tokens using the
//!    same token -> page-hash logic as SGLang.
//! 3. Expands those logical page hashes into the concrete Mooncake object keys
//!    SGLang uses for the configured TP/PP/MLA layout.
//! 4. Tracks those object keys from the Mooncake master's KV event stream.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::{
    SharedKvCache,
    indexer::KvRouterError,
    protocols::{SharedCacheHits, WorkerId},
};
use dynamo_runtime::{component::Component, traits::DistributedRuntimeProvider};

use crate::{
    discovery::RuntimeConfigWatch,
    local_model::runtime_config::ModelRuntimeConfig,
    utils::zmq::{connect_sub_socket, multipart_message},
};

const SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY: &str = "sglang_hicache_mooncake";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct SglangHicacheMooncakeConfig {
    backend: String,
    page_size: u32,
    tp_size: u32,
    pp_size: u32,
    is_mla_model: bool,
    is_eagle: bool,
    tp_lcm_size: Option<u32>,
    should_split_heads: bool,
    extra_backend_tag: Option<String>,
    master_server_address: Option<String>,
    master_metrics_port: u16,
    kv_events_endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MooncakeObjectEvent {
    event_type: String,
    #[serde(default)]
    object_key: Option<String>,
    #[serde(default)]
    tenant_id: String,
    #[serde(default)]
    group_id: Option<String>,
}

type MooncakeEventBatch = (i64, Vec<MooncakeObjectEvent>, u32);

#[derive(Debug, Clone, Copy)]
enum QueryToken {
    Single(u32),
    Bigram(u32, u32),
}

/// Event-driven shared KV cache index for SGLang HiCache (L3) state.
#[derive(Clone)]
pub struct HicacheSharedKvCache {
    runtime_configs: RuntimeConfigWatch,
    present_keys: Arc<DashSet<String>>,
    group_states: Arc<DashMap<String, (u64, bool)>>,
    last_sequence: Arc<AtomicU64>,
}

impl HicacheSharedKvCache {
    pub fn new(runtime_configs: RuntimeConfigWatch) -> Self {
        Self {
            runtime_configs,
            present_keys: Arc::new(DashSet::new()),
            group_states: Arc::new(DashMap::new()),
            last_sequence: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn start_subscriber(&self, component: &Component) {
        let cache = self.clone();
        let cancellation_token = component.drt().child_token();
        tokio::spawn(async move { cache.run_subscriber(cancellation_token).await });
    }

    fn resolve_mooncake_config(&self) -> Option<SglangHicacheMooncakeConfig> {
        let workers = self.runtime_configs.borrow();
        let mut configs = Vec::new();

        for (worker_id, runtime_config) in workers.iter() {
            if let Some(config) = mooncake_config_from_runtime(*worker_id, runtime_config) {
                configs.push((*worker_id, config));
            }
        }

        let (_, first) = configs.first()?;

        if configs.iter().any(|(_, config)| config != first) {
            tracing::warn!(
                workers = ?configs.iter().map(|(worker_id, _)| *worker_id).collect::<Vec<_>>(),
                "SGLang Mooncake HiCache runtime configs differ across workers; skipping shared-cache lookup"
            );
            return None;
        }

        Some(first.clone())
    }

    fn apply_batch(&self, sequence: u64, events: Vec<MooncakeObjectEvent>) {
        let previous = self.last_sequence.swap(sequence, Ordering::AcqRel);
        if (previous == 0 && sequence != 1) || (previous != 0 && sequence != previous + 1) {
            self.present_keys.clear();
            self.group_states.clear();
            tracing::warn!(
                previous,
                sequence,
                "Mooncake KV event sequence gap; cleared shared-cache state"
            );
        }

        for event in events {
            // The shared-cache query contract currently has no tenant input and
            // historically queried Mooncake's default tenant only.
            if !event.tenant_id.is_empty() && event.tenant_id != "default" {
                continue;
            }
            let Some(object_key) = event.object_key else {
                continue;
            };
            let group_id = event.group_id.filter(|id| !id.is_empty());
            match event.event_type.as_str() {
                "stored" => {
                    self.present_keys.insert(object_key);
                }
                "removed" => {
                    self.present_keys.remove(&object_key);
                }
                _ => {}
            }
            if let Some(group_id) = group_id {
                self.group_states.insert(group_id, (sequence, false));
            }
        }
    }

    fn clear(&self) {
        self.present_keys.clear();
        self.group_states.clear();
        self.last_sequence.store(0, Ordering::Release);
    }

    async fn run_subscriber(mut self, cancellation_token: CancellationToken) {
        let endpoint = loop {
            if let Some(endpoint) = self
                .resolve_mooncake_config()
                .and_then(|config| config.kv_events_endpoint)
            {
                break endpoint;
            }

            tokio::select! {
                _ = cancellation_token.cancelled() => return,
                result = self.runtime_configs.changed() => {
                    if result.is_err() {
                        return;
                    }
                }
            }
        };

        let mut socket = match connect_sub_socket(&endpoint, None).await {
            Ok(socket) => socket,
            Err(error) => {
                tracing::warn!(%endpoint, %error, "Failed to connect to Mooncake KV events");
                return;
            }
        };
        tracing::info!(%endpoint, "Connected to Mooncake KV events");

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => return,
                message = socket.next() => {
                    let frames = match message {
                        Some(Ok(frames)) => multipart_message(frames),
                        Some(Err(error)) => {
                            tracing::warn!(%endpoint, %error, "Mooncake KV event stream failed");
                            self.clear();
                            return;
                        }
                        None => {
                            tracing::warn!(%endpoint, "Mooncake KV event stream ended");
                            self.clear();
                            return;
                        }
                    };
                    if frames.len() != 3 || frames[1].len() != 8 {
                        tracing::warn!(frame_count = frames.len(), "Invalid Mooncake KV event frame");
                        self.clear();
                        continue;
                    }
                    let sequence = u64::from_be_bytes(frames[1].as_slice().try_into().unwrap());
                    match rmp_serde::from_slice::<MooncakeEventBatch>(&frames[2]) {
                        Ok((_, events, _)) => self.apply_batch(sequence, events),
                        Err(error) => {
                            tracing::warn!(%error, "Failed to decode Mooncake KV event batch");
                            self.clear();
                        }
                    }
                }
            }
        }
    }
}

#[async_trait]
impl SharedKvCache for HicacheSharedKvCache {
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
    ) -> Result<SharedCacheHits, KvRouterError> {
        let Some(config) = self.resolve_mooncake_config() else {
            tracing::debug!("No SGLang Mooncake HiCache runtime config available");
            return Ok(SharedCacheHits::default());
        };

        if config.backend != "mooncake" {
            tracing::debug!(backend = %config.backend, "Skipping non-Mooncake HiCache config");
            return Ok(SharedCacheHits::default());
        }

        if config.page_size == 0 || block_size == 0 {
            tracing::warn!(
                worker_page_size = config.page_size,
                router_page_size = block_size,
                "Invalid HiCache page size; skipping shared-cache lookup"
            );
            return Ok(SharedCacheHits::default());
        }

        if config.page_size != block_size {
            tracing::warn!(
                worker_page_size = config.page_size,
                router_page_size = block_size,
                "HiCache page size mismatch; skipping shared-cache lookup"
            );
            return Ok(SharedCacheHits::default());
        }

        if config.kv_events_endpoint.is_none() {
            tracing::debug!("Mooncake KV event endpoint is unavailable");
            return Ok(SharedCacheHits::default());
        }

        let page_hashes = logical_page_hashes(tokens, config.page_size, config.is_eagle);
        if page_hashes.is_empty() {
            return Ok(SharedCacheHits::default());
        }

        let page_hits = page_hashes
            .iter()
            .map(|page_hash| {
                let group_id = sglang_group_id(page_hash, &config);
                let generation = self.group_states.get(&group_id).map(|state| *state);
                if generation.is_some_and(|(_, verified)| verified) {
                    return true;
                }

                let hit = expand_actual_query_keys(page_hash, &config)
                    .iter()
                    .all(|key| self.present_keys.contains(key));
                if hit
                    && let Some((generation, _)) = generation
                    && let Some(mut state) = self.group_states.get_mut(&group_id)
                {
                    if state.0 != generation {
                        return false;
                    }
                    state.1 = true;
                }
                hit
            })
            .collect::<Vec<_>>();

        Ok(SharedCacheHits::from_hits(&page_hits))
    }
}

fn mooncake_config_from_runtime(
    worker_id: WorkerId,
    runtime_config: &ModelRuntimeConfig,
) -> Option<SglangHicacheMooncakeConfig> {
    match runtime_config
        .get_engine_specific::<SglangHicacheMooncakeConfig>(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY)
    {
        Ok(Some(config)) => Some(config),
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(
                worker_id,
                runtime_key = SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY,
                %error,
                "Failed to parse SGLang Mooncake HiCache runtime config"
            );
            None
        }
    }
}

fn logical_page_hashes(tokens: &[u32], page_size: u32, is_eagle: bool) -> Vec<String> {
    let page_size = page_size as usize;
    if page_size == 0 {
        return Vec::new();
    }

    let query_tokens = if is_eagle {
        tokens
            .windows(2)
            .map(|pair| QueryToken::Bigram(pair[0], pair[1]))
            .collect::<Vec<_>>()
    } else {
        tokens
            .iter()
            .copied()
            .map(QueryToken::Single)
            .collect::<Vec<_>>()
    };

    let aligned_len = (query_tokens.len() / page_size) * page_size;
    let aligned_tokens = &query_tokens[..aligned_len];

    let mut page_hashes = Vec::with_capacity(aligned_tokens.len() / page_size);
    let mut prior_hash = None;

    for page_tokens in aligned_tokens.chunks(page_size) {
        let digest = hash_query_tokens(page_tokens, prior_hash.as_ref());
        page_hashes.push(hex_encode(&digest));
        prior_hash = Some(digest);
    }

    page_hashes
}

fn hash_query_tokens(page_tokens: &[QueryToken], prior_hash: Option<&[u8; 32]>) -> [u8; 32] {
    let mut hasher = Sha256::new();

    if let Some(prior_hash) = prior_hash {
        hasher.update(prior_hash);
    }

    for token in page_tokens {
        match token {
            QueryToken::Single(token) => hasher.update(token.to_le_bytes()),
            QueryToken::Bigram(lhs, rhs) => {
                hasher.update(lhs.to_le_bytes());
                hasher.update(rhs.to_le_bytes());
            }
        }
    }

    hasher.finalize().into()
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn sglang_group_id(logical_page_hash: &str, config: &SglangHicacheMooncakeConfig) -> String {
    match config
        .extra_backend_tag
        .as_deref()
        .filter(|tag| !tag.is_empty())
    {
        Some(tag) => format!("sglang-hicache:{tag}_{logical_page_hash}"),
        None => format!("sglang-hicache:{logical_page_hash}"),
    }
}

fn expand_actual_query_keys(
    logical_page_hash: &str,
    config: &SglangHicacheMooncakeConfig,
) -> Vec<String> {
    let logical_key = maybe_prefix_key(logical_page_hash, config.extra_backend_tag.as_deref());
    let pp_size = config.pp_size.max(1);

    if config.is_mla_model {
        return if pp_size > 1 {
            (0..pp_size)
                .map(|pp_rank| format!("{logical_key}_{pp_rank}_k"))
                .collect()
        } else {
            vec![format!("{logical_key}__k")]
        };
    }

    let rank_count = if config.should_split_heads {
        config
            .tp_lcm_size
            .unwrap_or(config.tp_size)
            .max(config.tp_size)
            .max(1)
    } else {
        config.tp_size.max(1)
    };

    let mut query_keys = Vec::with_capacity((pp_size * rank_count * 2) as usize);
    for pp_rank in 0..pp_size {
        for rank in 0..rank_count {
            let suffix = if pp_size > 1 {
                format!("{rank}_{pp_rank}")
            } else {
                rank.to_string()
            };

            query_keys.push(format!("{logical_key}_{suffix}_k"));
            query_keys.push(format!("{logical_key}_{suffix}_v"));
        }
    }

    query_keys
}

fn maybe_prefix_key(logical_key: &str, extra_backend_tag: Option<&str>) -> String {
    match extra_backend_tag.filter(|tag| !tag.is_empty()) {
        Some(prefix) => format!("{prefix}_{logical_key}"),
        None => logical_key.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range};

    use super::*;
    use tokio::sync::watch;

    fn mooncake_config() -> SglangHicacheMooncakeConfig {
        SglangHicacheMooncakeConfig {
            backend: "mooncake".to_string(),
            page_size: 4,
            tp_size: 1,
            pp_size: 1,
            is_mla_model: false,
            is_eagle: false,
            tp_lcm_size: None,
            should_split_heads: false,
            extra_backend_tag: None,
            master_server_address: Some("127.0.0.1:50051".to_string()),
            master_metrics_port: 9003,
            kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
        }
    }

    fn runtime_watch_with_config(config: SglangHicacheMooncakeConfig) -> RuntimeConfigWatch {
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, config)
            .unwrap();

        let mut workers = HashMap::new();
        workers.insert(1, runtime_config);

        let (_tx, rx) = watch::channel(workers);
        rx
    }

    #[test]
    fn test_logical_page_hashes_match_sglang_for_normal_tokens() {
        let hashes = logical_page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, false);
        assert_eq!(
            hashes,
            vec![
                "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72".to_string(),
                "4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6".to_string(),
            ]
        );
    }

    #[test]
    fn test_logical_page_hashes_match_sglang_for_eagle_tokens() {
        let hashes = logical_page_hashes(&[10, 11, 12, 13, 14], 2, true);
        assert_eq!(
            hashes,
            vec![
                "4bde82677ba8b6de843da1713b58a439678ec01b642bbdcffec4acfa81b0ec8e".to_string(),
                "75ab93a767bad1e254945d1a0ccfa1588d6ebb803303e412d984baedcbbf04b9".to_string(),
            ]
        );
    }

    #[test]
    fn test_expand_actual_query_keys_for_mha_tp_pp_layout() {
        let config = SglangHicacheMooncakeConfig {
            tp_size: 2,
            pp_size: 2,
            ..mooncake_config()
        };

        let query_keys = expand_actual_query_keys("hash", &config);
        assert_eq!(
            query_keys,
            vec![
                "hash_0_0_k",
                "hash_0_0_v",
                "hash_1_0_k",
                "hash_1_0_v",
                "hash_0_1_k",
                "hash_0_1_v",
                "hash_1_1_k",
                "hash_1_1_v",
            ]
        );
    }

    #[test]
    fn test_expand_actual_query_keys_for_mla_without_pp_uses_double_underscore() {
        let config = SglangHicacheMooncakeConfig {
            is_mla_model: true,
            ..mooncake_config()
        };

        let query_keys = expand_actual_query_keys("hash", &config);
        assert_eq!(query_keys, vec!["hash__k"]);
    }

    #[test]
    fn test_expand_actual_query_keys_for_split_heads() {
        let config = SglangHicacheMooncakeConfig {
            tp_size: 2,
            tp_lcm_size: Some(4),
            should_split_heads: true,
            extra_backend_tag: Some("tag".to_string()),
            ..mooncake_config()
        };

        let query_keys = expand_actual_query_keys("hash", &config);
        assert_eq!(
            query_keys,
            vec![
                "tag_hash_0_k",
                "tag_hash_0_v",
                "tag_hash_1_k",
                "tag_hash_1_v",
                "tag_hash_2_k",
                "tag_hash_2_v",
                "tag_hash_3_k",
                "tag_hash_3_v",
            ]
        );
    }

    #[tokio::test]
    async fn test_check_blocks_uses_mooncake_events() {
        let hash0 = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72".to_string();
        let hash1 = "4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6".to_string();
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            1,
            vec![
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash0}_0_k")),
                    tenant_id: "default".to_string(),
                    group_id: None,
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash0}_0_v")),
                    tenant_id: "default".to_string(),
                    group_id: None,
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash1}_0_k")),
                    tenant_id: "default".to_string(),
                    group_id: None,
                },
            ],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4)
            .await
            .unwrap();

        assert_eq!(hits.ranges, vec![Range { start: 0, end: 1 }]);
        assert_eq!(hits.total_hits, 1);

        cache.apply_batch(
            2,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some(format!("{hash0}_0_v")),
                tenant_id: "default".to_string(),
                group_id: None,
            }],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4)
            .await
            .unwrap();
        assert_eq!(hits.total_hits, 0);
    }

    #[tokio::test]
    async fn test_check_blocks_uses_group_id() {
        let hash = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72";
        let group_id = format!("sglang-hicache:{hash}");
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            1,
            vec![
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash}_0_k")),
                    tenant_id: "default".to_string(),
                    group_id: Some(group_id.clone()),
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash}_0_v")),
                    tenant_id: "default".to_string(),
                    group_id: Some(group_id.clone()),
                },
            ],
        );

        let hits = cache.check_blocks(&[1, 2, 3, 4], 4).await.unwrap();
        assert_eq!(hits.total_hits, 1);
        assert!(cache.group_states.get(&group_id).is_some_and(|v| v.1));

        cache.apply_batch(
            2,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some(format!("{hash}_0_v")),
                tenant_id: "default".to_string(),
                group_id: Some(group_id),
            }],
        );
        let hits = cache.check_blocks(&[1, 2, 3, 4], 4).await.unwrap();
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_sequence_gap_clears_stale_keys() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            1,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("old_0_k".to_string()),
                tenant_id: "default".to_string(),
                group_id: Some("old-group".to_string()),
            }],
        );
        assert!(cache.group_states.contains_key("old-group"));
        cache.apply_batch(
            3,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("new_0_k".to_string()),
                tenant_id: "default".to_string(),
                group_id: None,
            }],
        );

        assert!(!cache.present_keys.contains("old_0_k"));
        assert!(cache.group_states.is_empty());
        assert!(cache.present_keys.contains("new_0_k"));
    }
}
