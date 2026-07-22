// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HiCache shared KV cache client for SGLang + Mooncake.
//!
//! Instead of querying a worker endpoint over the request plane, this client:
//! 1. Reads Mooncake HiCache metadata published by SGLang workers in runtime config.
//! 2. Recomputes the logical HiCache page hashes from request tokens using the
//!    same token -> page-hash logic as SGLang.
//! 3. Expands those logical page hashes into the concrete Mooncake object keys
//!    SGLang uses for the configured TP/PP/MLA layout.
//! 4. Queries the Mooncake master HTTP service directly via `/batch_query_keys`.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MOONCAKE_HTTP_TIMEOUT: Duration = Duration::from_secs(2);

use dynamo_kv_router::{
    SharedKvCache,
    indexer::KvRouterError,
    protocols::{SharedCacheHits, WorkerId},
};

use crate::{discovery::RuntimeConfigWatch, local_model::runtime_config::ModelRuntimeConfig};

const SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY: &str = "sglang_hicache_mooncake";
const MOONCAKE_BATCH_QUERY_KEYS_CHUNK_SIZE: usize = 128;

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
}

#[derive(Debug, Deserialize)]
struct MooncakeBatchQueryKeysResponse {
    success: bool,
    #[serde(default)]
    data: HashMap<String, MooncakeBatchQueryKeyResult>,
}

#[derive(Debug, Deserialize, Default)]
struct MooncakeBatchQueryKeyResult {
    #[serde(default)]
    ok: bool,
}

#[derive(Debug, Clone, Copy)]
enum QueryToken {
    Single(u32),
    Bigram(u32, u32),
}

/// Shared KV cache client that queries the Mooncake master HTTP service for
/// SGLang HiCache (L3) state.
pub struct HicacheSharedKvCache {
    runtime_configs: RuntimeConfigWatch,
    http_client: reqwest::Client,
}

impl HicacheSharedKvCache {
    pub fn new(runtime_configs: RuntimeConfigWatch) -> Self {
        Self {
            runtime_configs,
            http_client: reqwest::Client::builder()
                .timeout(MOONCAKE_HTTP_TIMEOUT)
                .build()
                .expect("failed to build reqwest client"),
        }
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

    async fn fetch_key_presence(
        &self,
        endpoint: &Url,
        actual_keys: &[String],
    ) -> Result<HashMap<String, bool>, KvRouterError> {
        let mut key_presence = HashMap::with_capacity(actual_keys.len());

        for chunk in actual_keys.chunks(MOONCAKE_BATCH_QUERY_KEYS_CHUNK_SIZE) {
            let joined_keys = chunk.join(",");

            let mut url = endpoint.clone();
            // Mooncake expects a raw comma-separated `keys=` list. If commas are
            // percent-encoded (`%2C`), Mooncake treats the entire value as one key.
            url.set_query(Some(&format!("keys={joined_keys}")));

            let response = self.http_client.get(url.clone()).send().await.map_err(|e| {
                tracing::warn!(error = %e, url = %url, "Mooncake batch_query_keys request failed");
                KvRouterError::IndexerOffline
            })?;

            let status = response.status();
            if !status.is_success() {
                tracing::warn!(
                    status = %status,
                    url = %url,
                    "Mooncake batch_query_keys returned non-success status"
                );
                return Err(KvRouterError::IndexerOffline);
            }

            let body: MooncakeBatchQueryKeysResponse = response.json().await.map_err(|e| {
                tracing::warn!(
                    error = %e,
                    url = %url,
                    "Failed to decode Mooncake batch_query_keys response"
                );
                KvRouterError::IndexerOffline
            })?;

            if !body.success {
                tracing::warn!(url = %url, "Mooncake batch_query_keys reported failure");
                return Err(KvRouterError::IndexerOffline);
            }

            for key in chunk {
                let exists = body.data.get(key).map(|entry| entry.ok).unwrap_or(false);
                key_presence.insert(key.clone(), exists);
            }
        }

        Ok(key_presence)
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

        let Some(endpoint) = mooncake_batch_query_endpoint(&config) else {
            tracing::debug!("Mooncake master HTTP endpoint is unavailable");
            return Ok(SharedCacheHits::default());
        };

        let page_hashes = logical_page_hashes(tokens, config.page_size, config.is_eagle);
        if page_hashes.is_empty() {
            return Ok(SharedCacheHits::default());
        }

        let page_query_keys = build_page_query_keys(&page_hashes, &config);
        let all_actual_keys = page_query_keys
            .iter()
            .flat_map(|keys| keys.iter().cloned())
            .collect::<Vec<_>>();

        let key_presence = self.fetch_key_presence(&endpoint, &all_actual_keys).await?;
        let page_hits = page_query_keys
            .iter()
            .map(|keys| {
                keys.iter()
                    .all(|key| key_presence.get(key).copied().unwrap_or(false))
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

fn mooncake_batch_query_endpoint(config: &SglangHicacheMooncakeConfig) -> Option<Url> {
    let master_server_address = config.master_server_address.as_deref()?;

    let mut url = Url::parse(&format!("http://{master_server_address}"))
        .inspect_err(|error| {
            tracing::warn!(
                master_server_address,
                %error,
                "Failed to parse Mooncake master address"
            );
        })
        .ok()?;

    if url.set_port(Some(config.master_metrics_port)).is_err() {
        tracing::warn!(
            master_server_address,
            master_metrics_port = config.master_metrics_port,
            "Failed to set Mooncake master HTTP port"
        );
        return None;
    }

    url.set_path("/batch_query_keys");
    url.set_query(None);
    Some(url)
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

fn build_page_query_keys(
    page_hashes: &[String],
    config: &SglangHicacheMooncakeConfig,
) -> Vec<Vec<String>> {
    page_hashes
        .iter()
        .map(|page_hash| expand_actual_query_keys(page_hash, config))
        .collect()
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
    use std::ops::Range;

    use super::*;
    use mockito::{Matcher, Server};
    use serde_json::json;
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
    async fn test_check_blocks_queries_mooncake_master() {
        let mut server = Server::new_async().await;
        let server_url = Url::parse(&server.url()).unwrap();

        let hash0 = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72".to_string();
        let hash1 = "4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6".to_string();

        let response = json!({
            "success": true,
            "data": {
                format!("{hash0}_0_k"): {"ok": true, "values": []},
                format!("{hash0}_0_v"): {"ok": true, "values": []},
                format!("{hash1}_0_k"): {"ok": true, "values": []},
                format!("{hash1}_0_v"): {"ok": false, "error": "not found"},
            }
        });

        let mock = server
            .mock("GET", "/batch_query_keys")
            .match_query(Matcher::Exact(format!(
                "keys={hash0}_0_k,{hash0}_0_v,{hash1}_0_k,{hash1}_0_v"
            )))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response.to_string())
            .create_async()
            .await;

        let config = SglangHicacheMooncakeConfig {
            master_server_address: Some(format!("{}:50051", server_url.host_str().unwrap())),
            master_metrics_port: server_url.port().unwrap(),
            ..mooncake_config()
        };

        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4)
            .await
            .unwrap();

        assert_eq!(hits.ranges, vec![Range { start: 0, end: 1 }]);
        assert_eq!(hits.total_hits, 1);

        mock.assert_async().await;
    }
}
