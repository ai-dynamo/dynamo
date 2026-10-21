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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::metrics::RoutingOverheadMetrics,
    local_model::runtime_config::ModelRuntimeConfig,
    utils::zmq::{connect_sub_socket, multipart_message},
};
use dynamo_kv_router::{
    SessionPrefixIndexer, SessionResidencyOwner, SharedCacheBackend, SharedKvCache,
    indexer::KvRouterError,
    protocols::{ExternalSequenceBlockHash, SharedCacheHits, StorageTier, WorkerId},
};

const SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY: &str = "sglang_hicache_mooncake";
const MOONCAKE_EVENT_RECONNECT_DELAY: Duration = Duration::from_secs(1);
const MAX_MOONCAKE_INDEX_ENTRIES: usize = 1_000_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SglangHicacheMooncakeConfig {
    backend: String,
    page_size: u32,
    tp_size: u32,
    pp_size: u32,
    is_mla_model: bool,
    is_eagle: bool,
    #[serde(default)]
    tp_lcm_size: Option<u32>,
    should_split_heads: bool,
    #[serde(default)]
    extra_backend_tag: Option<String>,
    /// Full prefix SGLang's MooncakeStore prepends to stored keys (backend tag and model
    /// name). Absent from workers that predate it.
    #[serde(default)]
    key_prefix: Option<String>,
    #[serde(default)]
    kv_events_endpoint: Option<String>,
    /// `hicache` (default) or `linker` for SGLang's direct external linker.
    #[serde(default)]
    key_layout: Option<String>,
}

impl SglangHicacheMooncakeConfig {
    fn has_same_layout(&self, other: &Self) -> bool {
        self.backend == other.backend
            && self.page_size == other.page_size
            && self.tp_size == other.tp_size
            && self.pp_size == other.pp_size
            && self.is_mla_model == other.is_mla_model
            && self.is_eagle == other.is_eagle
            && self.tp_lcm_size == other.tp_lcm_size
            && self.should_split_heads == other.should_split_heads
            && self.extra_backend_tag == other.extra_backend_tag
            && self.key_prefix == other.key_prefix
            && self.key_layout == other.key_layout
    }

    fn is_linker_layout(&self) -> bool {
        self.key_layout.as_deref() == Some("linker")
    }

    /// Prefix used for both physical keys and group IDs. Older workers publish only the
    /// backend tag, so fall back to it when `key_prefix` is absent.
    fn effective_key_prefix(&self) -> Option<&str> {
        self.key_prefix.as_deref().or_else(|| {
            self.extra_backend_tag
                .as_deref()
                .filter(|tag| !tag.is_empty())
        })
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct MooncakeObjectEvent {
    event_type: String,
    #[serde(default)]
    object_key: Option<String>,
    #[serde(default)]
    tenant_id: String,
    #[serde(default)]
    group_id: Option<String>,
    /// `cpu` or `disk`.
    #[serde(default)]
    medium: Option<String>,
    /// Session that wrote the object (Mooncake `ReplicateConfig.session_ids`), on `stored` only.
    #[serde(default)]
    session_id: Option<String>,
}

/// Session prefix index owner for Mooncake residency: the store is shared, not a worker rank.
const MOONCAKE_SESSION_OWNER: SessionResidencyOwner =
    SessionResidencyOwner::SharedCache(SharedCacheBackend::Mooncake);

/// Mooncake's `medium` is the store's own medium. The owner already says the copy is
/// shared, so `cpu` is host memory on the store, not a worker's host tier.
fn mooncake_session_tier(medium: Option<&str>) -> StorageTier {
    match medium {
        Some("disk") => StorageTier::Disk,
        _ => StorageTier::HostPinned,
    }
}

/// `{prefix}_{page_hash}` of a Mooncake object key: everything through the page's SHA-256
/// hex, dropping the rank and component suffix. SGLang's direct linker writes one object
/// per page component (for example DeepSeek-V4 `_cp0_pp0_deepseek_v4_c4`), so presence is
/// tracked per page.
fn mooncake_logical_key(object_key: &str) -> Option<&str> {
    let mut offset = 0;
    for part in object_key.split('_') {
        offset += part.len();
        if part.len() == 64 && part.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Some(&object_key[..offset]);
        }
        offset += 1;
    }
    None
}

/// SGLang's event block hash for the page named by a Mooncake object key: the first 64 bits
/// of the page's SHA-256 hex (`hash_str_to_int64`), wherever it sits between the key prefix
/// and the rank suffix.
fn mooncake_key_block_hash(object_key: &str) -> Option<ExternalSequenceBlockHash> {
    let page_hash = object_key
        .split('_')
        .find(|part| part.len() == 64 && part.bytes().all(|byte| byte.is_ascii_hexdigit()))?;
    u64::from_str_radix(&page_hash[..16], 16)
        .ok()
        .map(ExternalSequenceBlockHash)
}

/// Mooncake KV publisher payload: `(timestamp_ms, events, dp_rank)` after the empty-topic and
/// big-endian sequence ZMQ frames.
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
    /// Engine hash -> original object keys -> media still observed for that key.
    object_keys: Arc<DashMap<ExternalSequenceBlockHash, BTreeMap<String, BTreeSet<String>>>>,
    /// Linker layout: logical page keys with at least one stored component object.
    present_pages: Arc<DashSet<String>>,
    group_states: Arc<DashMap<String, (u64, bool)>>,
    last_sequence: Arc<AtomicU64>,
    has_sequence: Arc<AtomicBool>,
    last_layout: Arc<ArcSwapOption<SglangHicacheMooncakeConfig>>,
    cancellation_token: CancellationToken,
    frontend_kv_events_endpoint: Option<String>,
    session_index: Arc<ArcSwapOption<SessionPrefixIndexer>>,
}

impl HicacheSharedKvCache {
    pub fn new(runtime_configs: RuntimeConfigWatch) -> Self {
        Self::new_with_cancellation_and_endpoint(runtime_configs, CancellationToken::new(), None)
    }

    pub fn new_with_cancellation(
        runtime_configs: RuntimeConfigWatch,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self::new_with_cancellation_and_endpoint(runtime_configs, cancellation_token, None)
    }

    pub fn new_with_cancellation_and_endpoint(
        runtime_configs: RuntimeConfigWatch,
        cancellation_token: CancellationToken,
        frontend_kv_events_endpoint: Option<String>,
    ) -> Self {
        Self {
            runtime_configs,
            present_keys: Arc::new(DashSet::new()),
            object_keys: Arc::new(DashMap::new()),
            present_pages: Arc::new(DashSet::new()),
            group_states: Arc::new(DashMap::new()),
            last_sequence: Arc::new(AtomicU64::new(0)),
            has_sequence: Arc::new(AtomicBool::new(false)),
            last_layout: Arc::new(ArcSwapOption::empty()),
            cancellation_token,
            frontend_kv_events_endpoint,
            session_index: Arc::new(ArcSwapOption::empty()),
        }
    }

    pub fn start_subscriber(&self) {
        let cache = self.clone();
        let cancellation_token = self.cancellation_token.clone();
        tokio::spawn(async move { cache.run_subscriber(cancellation_token).await });
    }

    pub fn shutdown(&self) {
        self.cancellation_token.cancel();
        self.clear();
    }

    fn clear_on_layout_change(&self, layout: &SglangHicacheMooncakeConfig) {
        let last_layout = self.last_layout.load();
        if last_layout
            .as_ref()
            .is_some_and(|previous| previous.has_same_layout(layout))
        {
            return;
        }
        if last_layout.is_some() {
            self.clear();
            tracing::warn!("SGLang Mooncake HiCache layout changed; cleared shared-cache state");
        }
        self.last_layout.store(Some(Arc::new(layout.clone())));
    }

    fn resolve_mooncake_config_and_endpoint(
        &self,
    ) -> Option<(SglangHicacheMooncakeConfig, String)> {
        let workers = self.runtime_configs.borrow();
        let mut configs = Vec::new();

        for (worker_id, runtime_config) in workers.iter() {
            if let Some(config) = mooncake_config_from_runtime(*worker_id, runtime_config) {
                configs.push((*worker_id, config));
            }
        }

        let (_, first) = configs.first()?;

        if configs
            .iter()
            .any(|(_, config)| !config.has_same_layout(first))
        {
            tracing::warn!(
                workers = ?configs.iter().map(|(worker_id, _)| *worker_id).collect::<Vec<_>>(),
                "SGLang Mooncake HiCache runtime configs differ across workers; skipping shared-cache lookup"
            );
            return None;
        }

        self.clear_on_layout_change(first);

        if let Some(endpoint) = &self.frontend_kv_events_endpoint {
            return Some((first.clone(), endpoint.clone()));
        }

        let mut endpoints = configs
            .iter()
            .filter_map(|(_, config)| config.kv_events_endpoint.as_deref())
            .filter(|endpoint| !endpoint.is_empty());
        let endpoint = endpoints.next()?;
        if endpoints.any(|candidate| candidate != endpoint) {
            tracing::warn!(
                "SGLang Mooncake KV event endpoints differ across workers; skipping shared-cache lookup"
            );
            return None;
        }
        Some((first.clone(), endpoint.to_string()))
    }

    fn kv_events_endpoint(&self) -> Option<String> {
        self.resolve_mooncake_config_and_endpoint()
            .map(|(_, endpoint)| endpoint)
    }

    fn apply_batch(&self, sequence: u64, events: Vec<MooncakeObjectEvent>) {
        // SGLang's ZmqEventPublisher increments this sequence once per published batch, so a
        // non-consecutive value means one or more whole batches were missed.
        let has_previous = self.has_sequence.swap(true, Ordering::AcqRel);
        let previous = self.last_sequence.swap(sequence, Ordering::AcqRel);
        if has_previous && sequence == previous {
            return;
        }
        let session_index = self.session_index.load_full();
        if has_previous && sequence != previous.wrapping_add(1) {
            self.present_keys.clear();
            self.object_keys.clear();
            self.present_pages.clear();
            self.group_states.clear();
            if let Some(index) = &session_index {
                index.clear_owner_frontiers(MOONCAKE_SESSION_OWNER);
            }
            tracing::warn!(
                previous,
                sequence,
                "Mooncake KV event sequence gap; cleared shared-cache state"
            );
        }

        let (mut session_stored, mut session_advanced, mut session_receded) =
            (0usize, 0usize, 0usize);
        for event in events {
            // The shared-cache query contract currently has no tenant input and
            // historically queried Mooncake's default tenant only.
            if !event.tenant_id.is_empty() && event.tenant_id != "default" {
                continue;
            }
            if event.event_type == "cleared" {
                self.clear_observations();
                continue;
            }
            let Some(object_key) = event.object_key else {
                continue;
            };
            if let Some(hash) = mooncake_key_block_hash(&object_key) {
                match event.event_type.as_str() {
                    "stored" => {
                        self.object_keys
                            .entry(hash)
                            .or_default()
                            .entry(object_key.clone())
                            .or_default()
                            .insert(event.medium.clone().unwrap_or_else(|| "cpu".into()));
                    }
                    "removed" => {
                        if let Some(mut keys) = self.object_keys.get_mut(&hash)
                            && let Some(media) = keys.get_mut(&object_key)
                        {
                            if let Some(medium) = &event.medium {
                                media.remove(medium);
                            } else {
                                media.clear();
                            }
                            if media.is_empty() {
                                keys.remove(&object_key);
                            }
                        }
                        self.object_keys.remove_if(&hash, |_, keys| keys.is_empty());
                    }
                    _ => {}
                }
            }
            if let Some(index) = &session_index
                && let Some(block_hash) = mooncake_key_block_hash(&object_key)
            {
                let tier = mooncake_session_tier(event.medium.as_deref());
                match (event.event_type.as_str(), event.session_id.as_deref()) {
                    ("stored", Some(session_id)) if !session_id.is_empty() => {
                        session_stored += 1;
                        // Mooncake events carry no lineage. Reuse the parent the device events
                        // recorded so the store does not conflict with it; a block that is not
                        // known yet becomes a root that a later device event grafts.
                        let parent_hash = index
                            .get_node_from_hash(block_hash)
                            .and_then(|node| index.get_node(node))
                            .and_then(|node| node.parent())
                            .and_then(|parent| index.get_node(parent))
                            .map(|parent| parent.block_hash());
                        match index.update_session_from_stored_blocks_in_tier(
                            session_id,
                            MOONCAKE_SESSION_OWNER,
                            tier,
                            parent_hash,
                            &[block_hash],
                        ) {
                            Ok(true) => session_advanced += 1,
                            Ok(false) => {}
                            Err(error) => {
                                tracing::warn!(%error, session_id, "failed to record Mooncake session residency")
                            }
                        }
                    }
                    ("removed", _) => {
                        session_receded += index.update_session_from_removed_blocks_in_tier(
                            MOONCAKE_SESSION_OWNER,
                            tier,
                            &[block_hash],
                        );
                    }
                    _ => {}
                }
            }
            let group_id = event.group_id.filter(|id| !id.is_empty());
            if let Some(page) = mooncake_logical_key(&object_key) {
                match event.event_type.as_str() {
                    "stored" => {
                        self.present_pages.insert(page.to_string());
                    }
                    "removed" => {
                        // Any missing component makes the page unrestorable.
                        self.present_pages.remove(page);
                    }
                    _ => {}
                }
            }
            match event.event_type.as_str() {
                "stored" => {
                    self.present_keys.insert(object_key);
                    if let Some(group_id) = group_id {
                        self.group_states.insert(group_id, (sequence, false));
                    }
                }
                "removed" => {
                    // Another medium may still hold the same object. Keep the
                    // key inventory (and its size bound) aligned with the resolver.
                    let still_present = mooncake_key_block_hash(&object_key)
                        .and_then(|hash| self.object_keys.get(&hash))
                        .is_some_and(|keys| keys.contains_key(&object_key));
                    if !still_present {
                        self.present_keys.remove(&object_key);
                    }
                    if let Some(group_id) = group_id {
                        self.group_states.remove(&group_id);
                    } else {
                        // An older Mooncake publisher may omit `group_id` on removal. Clearing all
                        // verified groups is conservative and prevents a stale group fast-path hit.
                        self.group_states.clear();
                    }
                }
                _ => {}
            }
        }

        if session_stored > 0 || session_receded > 0 {
            tracing::debug!(
                sequence,
                session_stored,
                session_advanced,
                session_receded,
                "Mooncake session residency updated"
            );
        }

        self.clear_if_index_too_large(MAX_MOONCAKE_INDEX_ENTRIES);
    }

    fn clear_observations(&self) {
        self.present_keys.clear();
        self.present_pages.clear();
        self.group_states.clear();
        self.object_keys.clear();
        if let Some(index) = self.session_index.load_full() {
            index.clear_owner_frontiers(MOONCAKE_SESSION_OWNER);
        }
    }

    fn clear(&self) {
        self.clear_observations();
        self.last_sequence.store(0, Ordering::Release);
        self.has_sequence.store(false, Ordering::Release);
    }

    fn clear_if_index_too_large(&self, max_entries: usize) {
        let present_keys = self.present_keys.len();
        let group_states = self.group_states.len();
        if present_keys.saturating_add(group_states) > max_entries {
            self.clear();
            tracing::warn!(
                present_keys,
                group_states,
                max_entries,
                "Mooncake KV event index exceeded its size limit; cleared shared-cache state"
            );
        }
    }

    fn record_subscriber_error(&self) {
        if let Some(metrics) = RoutingOverheadMetrics::get() {
            metrics.inc_shared_cache_errors();
        }
    }

    async fn run_subscriber(mut self, cancellation_token: CancellationToken) {
        loop {
            let endpoint = loop {
                if let Some(endpoint) = self.kv_events_endpoint() {
                    break endpoint;
                }

                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    result = self.runtime_configs.changed() => {
                        if result.is_err() {
                            self.clear();
                            return;
                        }
                    }
                }
            };

            self.clear();
            let mut socket = match connect_sub_socket(&endpoint, None).await {
                Ok(socket) => socket,
                Err(error) => {
                    self.record_subscriber_error();
                    tracing::warn!(%endpoint, %error, "Failed to connect to Mooncake KV events; retrying");
                    tokio::select! {
                        _ = cancellation_token.cancelled() => return,
                        _ = tokio::time::sleep(MOONCAKE_EVENT_RECONNECT_DELAY) => continue,
                    }
                }
            };
            tracing::info!(%endpoint, "Connected to Mooncake KV events");

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    result = self.runtime_configs.changed() => {
                        if result.is_err() {
                            self.clear();
                            return;
                        }
                        let next_endpoint = self.kv_events_endpoint();
                        if next_endpoint.as_deref() != Some(endpoint.as_str()) {
                            tracing::info!(%endpoint, next_endpoint = ?next_endpoint, "Mooncake KV event endpoint changed; reconnecting");
                            break;
                        }
                    }
                    message = socket.next() => {
                        let frames = match message {
                            Some(Ok(frames)) => multipart_message(frames),
                            Some(Err(error)) => {
                                self.record_subscriber_error();
                                tracing::warn!(%endpoint, %error, "Mooncake KV event stream failed; reconnecting");
                                break;
                            }
                            None => {
                                self.record_subscriber_error();
                                tracing::warn!(%endpoint, "Mooncake KV event stream ended; reconnecting");
                                break;
                            }
                        };
                        match parse_mooncake_event_frames(&frames) {
                            Ok((sequence, events)) => self.apply_batch(sequence, events),
                            Err(error) => {
                                self.record_subscriber_error();
                                tracing::warn!(%error, "Dropping invalid Mooncake KV event frame");
                            }
                        }
                    }
                }
            }

            tokio::select! {
                _ = cancellation_token.cancelled() => return,
                _ = tokio::time::sleep(MOONCAKE_EVENT_RECONNECT_DELAY) => {}
            }
        }
    }
}

fn parse_mooncake_event_frames(
    frames: &[Vec<u8>],
) -> anyhow::Result<(u64, Vec<MooncakeObjectEvent>)> {
    let [_, sequence, payload] = frames else {
        anyhow::bail!("expected three frames, got {}", frames.len());
    };
    let sequence = u64::from_be_bytes(
        sequence
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected an 8-byte sequence frame"))?,
    );
    let (_, events, _) = rmp_serde::from_slice::<MooncakeEventBatch>(payload)?;
    Ok((sequence, events))
}

#[async_trait]
impl SharedKvCache for HicacheSharedKvCache {
    fn resolve_object_keys(
        &self,
        blocks: &[ExternalSequenceBlockHash],
    ) -> Result<Vec<Vec<String>>, KvRouterError> {
        let (config, _) = self.resolve_mooncake_config_and_endpoint().ok_or_else(|| {
            KvRouterError::Unsupported("no unambiguous Mooncake runtime configuration".into())
        })?;
        blocks
            .iter()
            .map(|hash| {
                let Some(observed) = self.object_keys.get(hash) else {
                    return Ok(Vec::new());
                };
                let mut page_identity = None;
                let mut keys = Vec::new();
                for key in observed.keys() {
                    let Some(page) = mooncake_logical_key(key) else {
                        continue;
                    };
                    let prefix = if page.len() == 64 {
                        None
                    } else {
                        Some(&page[..page.len() - 65])
                    };
                    if prefix
                        != config
                            .effective_key_prefix()
                            .filter(|prefix| !prefix.is_empty())
                    {
                        continue;
                    }
                    if page_identity.is_some_and(|previous| previous != page) {
                        return Err(KvRouterError::Unsupported(format!(
                            "ambiguous Mooncake block hash: {}",
                            hash.0
                        )));
                    }
                    page_identity = Some(page);
                    keys.push(key.clone());
                }
                Ok(keys)
            })
            .collect()
    }

    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache_namespace: Option<&str>,
    ) -> Result<SharedCacheHits, KvRouterError> {
        if cache_namespace
            .filter(|namespace| !namespace.is_empty())
            .is_some()
        {
            tracing::debug!("Skipping SGLang Mooncake HiCache lookup for cache-namespaced request");
            return Ok(SharedCacheHits::default());
        }

        let Some((config, _endpoint)) = self.resolve_mooncake_config_and_endpoint() else {
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

                let hit = if config.is_linker_layout() {
                    self.present_pages
                        .contains(&maybe_prefix_key(page_hash, config.effective_key_prefix()))
                } else {
                    expand_actual_query_keys(page_hash, &config)
                        .iter()
                        .all(|key| self.present_keys.contains(key))
                };
                if hit
                    && let Some((generation, _)) = generation
                    && let Some(mut state) = self.group_states.get_mut(&group_id)
                {
                    // A concurrent stored event invalidates verification, not the physical key
                    // check that already proved this request is a hit.
                    if state.0 == generation {
                        state.1 = true;
                    }
                }
                hit
            })
            .collect::<Vec<_>>();

        Ok(SharedCacheHits::from_hits(&page_hits))
    }

    fn attach_session_prefix_index(&self, index: Arc<SessionPrefixIndexer>) {
        self.session_index.store(Some(index));
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
    let logical_key = maybe_prefix_key(logical_page_hash, config.effective_key_prefix());
    format!("sglang-hicache:{logical_key}")
}

fn expand_actual_query_keys(
    logical_page_hash: &str,
    config: &SglangHicacheMooncakeConfig,
) -> Vec<String> {
    let logical_key = maybe_prefix_key(logical_page_hash, config.effective_key_prefix());
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

// SGLang prepends `{prefix}_` whenever the prefix is not None, even when it is empty.
fn maybe_prefix_key(logical_key: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(prefix) => format!("{prefix}_{logical_key}"),
        None => logical_key.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range};

    use super::*;
    use dynamo_kv_router::protocols::WorkerWithDpRank;
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
            key_prefix: None,
            kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
            key_layout: None,
        }
    }

    fn runtime_watch_with_config_and_sender(
        config: SglangHicacheMooncakeConfig,
    ) -> (
        RuntimeConfigWatch,
        watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>,
    ) {
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, config)
            .unwrap();

        let mut workers = HashMap::new();
        workers.insert(1, runtime_config);

        let (tx, rx) = watch::channel(workers);
        (rx, tx)
    }

    fn runtime_watch_with_config(config: SglangHicacheMooncakeConfig) -> RuntimeConfigWatch {
        runtime_watch_with_config_and_sender(config).0
    }

    #[tokio::test]
    async fn pin_uses_worker_lineage_without_store_session_tags() {
        use crate::kv_router::mooncake_client::{
            MooncakeClient, command_resolved_blocks, session_block_chains,
        };

        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let index = Arc::new(SessionPrefixIndexer::new());
        cache.attach_session_prefix_index(Arc::clone(&index));
        let worker = WorkerWithDpRank::new(7, 0);
        for (session, chain) in [
            ("worker-only", vec![11, 22]),
            ("worker-only", vec![11, 33]),
            ("other", vec![44]),
        ] {
            index
                .update_session_from_stored_blocks(
                    session,
                    worker,
                    None,
                    &chain
                        .into_iter()
                        .map(ExternalSequenceBlockHash)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let keys = |hashes: &[u64]| {
            hashes
                .iter()
                .flat_map(|hash| {
                    ["k", "v"].map(|kind| format!("{hash:016x}{}_0_{kind}", "0".repeat(48)))
                })
                .collect::<Vec<_>>()
        };
        cache.apply_batch(
            1,
            keys(&[11, 22, 33, 44])
                .into_iter()
                .map(|key| MooncakeObjectEvent {
                    event_type: "stored".into(),
                    object_key: Some(key),
                    tenant_id: "default".into(),
                    session_id: None,
                    ..Default::default()
                })
                .collect(),
        );
        assert!(
            index
                .get_session_block_lineage_in_tier(
                    "worker-only",
                    MOONCAKE_SESSION_OWNER,
                    StorageTier::HostPinned,
                    None
                )
                .unwrap()
                .is_empty()
        );
        let all = session_block_chains(&index, "worker-only", None).unwrap();
        assert_eq!(
            all,
            vec![
                vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(22)],
                vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(33)]
            ]
        );
        let selected =
            session_block_chains(&index, "worker-only", Some(ExternalSequenceBlockHash(22)))
                .unwrap();
        assert_eq!(selected, vec![all[0].clone()]);
        // Ancestors, another session's frontier, and unknown hashes must not select a branch.
        for hash in [11, 44, 99] {
            assert!(
                session_block_chains(&index, "worker-only", Some(ExternalSequenceBlockHash(hash)))
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(
            session_block_chains(&index, "missing", None)
                .unwrap()
                .is_empty()
        );

        let mut server = mockito::Server::new_async().await;
        let client = MooncakeClient::new(&server.url(), "default").unwrap();
        for (hashes, expected) in [
            (vec![ExternalSequenceBlockHash(22); 2], keys(&[22])),
            (all.into_iter().flatten().collect(), keys(&[11, 22, 33])),
            (selected.into_iter().flatten().collect(), keys(&[11, 22])),
        ] {
            let response: Vec<_> = expected.iter().map(|key| serde_json::json!({"key":key,"error_code":if key.ends_with("_v") {-704} else {0}})).collect();
            let command = server
                .mock("POST", "/api/v1/soft_pin")
                .match_body(mockito::Matcher::Json(
                    serde_json::json!({"tenant_id":"default","keys":expected,"ttl_ms":60000}),
                ))
                .with_body(serde_json::json!({"results":response}).to_string())
                .expect(1)
                .create_async()
                .await;
            let results = command_resolved_blocks(&cache, &client, &hashes, Some(60000))
                .await
                .unwrap();
            assert_eq!(
                results.iter().map(|r| r.key.clone()).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(results[1].error_code, -704); // Stale observations remain per-key failures.
            command.assert_async().await;
            let release = server
                .mock("POST", "/api/v1/soft_unpin")
                .match_body(mockito::Matcher::Json(
                    serde_json::json!({"tenant_id":"default","keys":expected}),
                ))
                .with_body(serde_json::json!({"results":response}).to_string())
                .expect(1)
                .create_async()
                .await;
            let mut partly_missing = hashes;
            partly_missing.push(ExternalSequenceBlockHash(99));
            assert_eq!(
                command_resolved_blocks(&cache, &client, &partly_missing, None)
                    .await
                    .unwrap(),
                results
            );
            release.assert_async().await;
        }
        let no_command = server
            .mock("POST", "/api/v1/soft_pin")
            .expect(0)
            .create_async()
            .await;
        for hashes in [
            vec![],
            vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(99)],
        ] {
            assert!(
                command_resolved_blocks(&cache, &client, &hashes, Some(60000))
                    .await
                    .is_err()
            );
        }
        let tenant_client = MooncakeClient::new(&server.url(), "other").unwrap();
        assert!(
            command_resolved_blocks(
                &cache,
                &tenant_client,
                &[ExternalSequenceBlockHash(11)],
                Some(60000)
            )
            .await
            .is_err()
        );
        no_command.assert_async().await;
        let no_release = server
            .mock("POST", "/api/v1/soft_unpin")
            .expect(0)
            .create_async()
            .await;
        for hashes in [vec![], vec![ExternalSequenceBlockHash(99)]] {
            assert!(
                command_resolved_blocks(&cache, &client, &hashes, None)
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
        // A lost observation must not send stale keys, even if the Store may retain them.
        cache.clear();
        assert!(
            command_resolved_blocks(&cache, &client, &[ExternalSequenceBlockHash(22)], None)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            command_resolved_blocks(&cache, &tenant_client, &[], None)
                .await
                .is_err()
        );
        no_release.assert_async().await;
    }

    #[test]
    fn object_key_resolution_preserves_components_and_remaining_media() {
        let config = SglangHicacheMooncakeConfig {
            key_prefix: Some("model".into()),
            key_layout: Some("linker".into()),
            ..mooncake_config()
        };
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
        let hash = ExternalSequenceBlockHash(42);
        let page = format!("{:016x}{}", hash.0, "0".repeat(48));
        let a = format!("model_{page}_cp0_pp0_c4");
        let b = format!("model_{page}_cp0_pp0_c4_indexer");
        let other = format!("other_{page}_cp0_pp0_c4");
        let event = |kind: &str, key: &str, medium: &str| MooncakeObjectEvent {
            event_type: kind.into(),
            object_key: Some(key.into()),
            medium: Some(medium.into()),
            ..Default::default()
        };
        cache.apply_batch(
            1,
            vec![
                event("stored", &a, "cpu"),
                event("stored", &a, "disk"),
                event("stored", &b, "cpu"),
                event("stored", &other, "cpu"),
            ],
        );
        assert_eq!(
            cache
                .resolve_object_keys(&[hash, ExternalSequenceBlockHash(99)])
                .unwrap(),
            vec![vec![a.clone(), b.clone()], vec![]]
        );
        cache.apply_batch(2, vec![event("removed", &a, "cpu")]);
        assert_eq!(
            cache.resolve_object_keys(&[hash]).unwrap()[0],
            vec![a.clone(), b.clone()]
        );
        cache.apply_batch(3, vec![event("removed", &a, "disk")]);
        assert_eq!(cache.resolve_object_keys(&[hash]).unwrap()[0], vec![b]);
        // A different full page hash sharing the truncated engine hash is unsafe.
        let collision = format!("model_{:016x}{}_cp0_pp0_c4", hash.0, "1".repeat(48));
        cache.apply_batch(4, vec![event("stored", &collision, "cpu")]);
        assert!(cache.resolve_object_keys(&[hash]).is_err());
        cache.apply_batch(6, vec![]); // lost events invalidate the mapping
        assert!(cache.resolve_object_keys(&[hash]).unwrap()[0].is_empty());
    }

    #[test]
    fn clear_invalidates_object_keys_and_session_frontiers() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let index = Arc::new(SessionPrefixIndexer::new());
        cache.attach_session_prefix_index(Arc::clone(&index));
        let key = format!("{}{}_0_k", "000000000000002a", "0".repeat(48));
        let stored = || MooncakeObjectEvent {
            event_type: "stored".into(),
            object_key: Some(key.clone()),
            session_id: Some("s".into()),
            ..Default::default()
        };
        cache.apply_batch(1, vec![stored()]);
        assert!(!index.get_session_residency("s").is_empty());
        cache.apply_batch(
            2,
            vec![MooncakeObjectEvent {
                event_type: "cleared".into(),
                ..Default::default()
            }],
        );
        assert!(cache.object_keys.is_empty());
        assert!(cache.present_keys.is_empty());
        assert!(index.get_session_residency("s").is_empty());
        cache.apply_batch(3, vec![stored()]);
        cache.clear();
        assert!(cache.object_keys.is_empty());
        assert!(index.get_session_residency("s").is_empty());
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

    #[tokio::test]
    async fn test_check_blocks_counts_linker_pages_with_any_component() {
        let hash0 = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72".to_string();
        let config = SglangHicacheMooncakeConfig {
            key_layout: Some("linker".to_string()),
            key_prefix: Some("Org-Model".to_string()),
            ..mooncake_config()
        };
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
        let component = |name: &str| format!("Org-Model_{hash0}_cp0_pp0_{name}");
        let event = |event_type: &str, name: &str| MooncakeObjectEvent {
            event_type: event_type.to_string(),
            object_key: Some(component(name)),
            tenant_id: "default".to_string(),
            ..Default::default()
        };
        cache.apply_batch(
            1,
            vec![event("stored", "deepseek_v4_c4"), event("stored", "swa")],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
            .await
            .unwrap();
        assert_eq!(hits.ranges, vec![Range { start: 0, end: 1 }]);

        // Losing any component makes the page unrestorable.
        cache.apply_batch(2, vec![event("removed", "swa")]);
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
            .await
            .unwrap();
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_mooncake_logical_key_drops_rank_and_component_suffix() {
        let hash = "ab".repeat(32);
        let key = format!("deepseek-ai-DeepSeek-V4-Flash_{hash}_cp0_pp0_deepseek_v4_c4_indexer");
        assert_eq!(
            mooncake_logical_key(&key),
            Some(format!("deepseek-ai-DeepSeek-V4-Flash_{hash}").as_str())
        );
        assert_eq!(
            mooncake_logical_key("sglang_mooncake_store_warmup_key0"),
            None
        );
    }

    #[test]
    fn test_mooncake_session_event_after_device_chain_extends_shared_frontier() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let index = Arc::new(SessionPrefixIndexer::new());
        cache.attach_session_prefix_index(Arc::clone(&index));
        let hashes = [0x1111_u64, 0x2222, 0x3333];
        let device = WorkerWithDpRank::new(7, 0);
        index
            .update_session_from_stored_blocks(
                "sess-b",
                device,
                None,
                &hashes.map(ExternalSequenceBlockHash),
            )
            .unwrap();

        let key = |hash: u64| format!("M_{hash:016x}{}_cp0_pp0_deepseek_v4_c4", "0".repeat(48));
        let stored = |hash: u64| MooncakeObjectEvent {
            event_type: "stored".to_string(),
            object_key: Some(key(hash)),
            tenant_id: "default".to_string(),
            group_id: None,
            medium: Some("cpu".to_string()),
            session_id: Some("sess-b".to_string()),
        };
        cache.apply_batch(1, vec![stored(0x1111), stored(0x3333)]);

        let shared: Vec<_> = index
            .get_session_residency("sess-b")
            .into_iter()
            .filter(|(owner, _, _)| *owner == MOONCAKE_SESSION_OWNER)
            .collect();
        assert_eq!(shared.len(), 1);
        let node = index.get_node(shared[0].2).unwrap();
        assert_eq!(node.block_hash(), ExternalSequenceBlockHash(0x3333));
    }

    #[test]
    fn test_mooncake_session_events_update_session_prefix_index() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let index = Arc::new(SessionPrefixIndexer::new());
        cache.attach_session_prefix_index(Arc::clone(&index));

        let page_hash = format!("{:016x}{}", 0xfeed_beef_u64, "0".repeat(48));
        let key = format!("Org-Model_{page_hash}_tp0_cp0_pp0_k");
        let block_hash = ExternalSequenceBlockHash(0xfeed_beef);
        assert_eq!(mooncake_key_block_hash(&key), Some(block_hash));

        let event = |event_type: &str, session_id: Option<&str>| MooncakeObjectEvent {
            event_type: event_type.to_string(),
            object_key: Some(key.clone()),
            tenant_id: "default".to_string(),
            group_id: None,
            medium: Some("cpu".to_string()),
            session_id: session_id.map(str::to_string),
        };

        cache.apply_batch(1, vec![event("stored", Some("sess-a"))]);
        let residency = index.get_session_residency("sess-a");
        assert_eq!(residency.len(), 1);
        let (owner, tier, node) = residency[0];
        assert_eq!(
            (owner, tier),
            (MOONCAKE_SESSION_OWNER, StorageTier::HostPinned)
        );
        assert_eq!(index.get_node(node).unwrap().block_hash(), block_hash);

        // A disk replica is the same owner in its own tier, and a removal recedes only
        // the medium it names.
        let tiers = |index: &SessionPrefixIndexer| {
            let mut tiers: Vec<_> = index
                .get_session_residency("sess-a")
                .into_iter()
                .map(|(owner, tier, _)| (owner, tier))
                .collect();
            tiers.sort_by_key(|(_, tier)| *tier as u8);
            tiers
        };
        let on_disk = |event_type: &str, session_id: Option<&str>| MooncakeObjectEvent {
            medium: Some("disk".to_string()),
            ..event(event_type, session_id)
        };
        cache.apply_batch(2, vec![on_disk("stored", Some("sess-a"))]);
        assert_eq!(
            tiers(&index),
            vec![
                (MOONCAKE_SESSION_OWNER, StorageTier::HostPinned),
                (MOONCAKE_SESSION_OWNER, StorageTier::Disk),
            ]
        );

        cache.apply_batch(3, vec![event("removed", None)]);
        assert_eq!(
            tiers(&index),
            vec![(MOONCAKE_SESSION_OWNER, StorageTier::Disk)]
        );

        cache.apply_batch(4, vec![on_disk("removed", None)]);
        assert!(index.get_session_residency("sess-a").is_empty());
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

    #[test]
    fn test_sglang_group_id_uses_extra_backend_tag() {
        let config = SglangHicacheMooncakeConfig {
            extra_backend_tag: Some("tag".to_string()),
            ..mooncake_config()
        };

        assert_eq!(sglang_group_id("hash", &config), "sglang-hicache:tag_hash");
    }

    #[test]
    fn test_key_prefix_takes_precedence_over_extra_backend_tag() {
        let config = SglangHicacheMooncakeConfig {
            extra_backend_tag: Some("tag".to_string()),
            key_prefix: Some("tag_Qwen-Qwen3-0.6B".to_string()),
            ..mooncake_config()
        };

        assert_eq!(
            sglang_group_id("hash", &config),
            "sglang-hicache:tag_Qwen-Qwen3-0.6B_hash"
        );
        assert_eq!(
            expand_actual_query_keys("hash", &config),
            vec![
                "tag_Qwen-Qwen3-0.6B_hash_0_k",
                "tag_Qwen-Qwen3-0.6B_hash_0_v"
            ]
        );
    }

    #[test]
    fn test_parse_mooncake_event_frames() {
        let payload = rmp_serde::to_vec(&(
            0_i64,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: Some("group-0".to_string()),
                ..Default::default()
            }],
            0_u32,
        ))
        .unwrap();

        let (sequence, events) =
            parse_mooncake_event_frames(&[Vec::new(), 7_u64.to_be_bytes().to_vec(), payload])
                .unwrap();

        assert_eq!(sequence, 7);
        assert_eq!(events[0].object_key.as_deref(), Some("key-0"));
        assert_eq!(events[0].group_id.as_deref(), Some("group-0"));
    }

    #[test]
    fn test_kv_events_endpoint_tracks_runtime_config_updates() {
        let (runtime_configs, tx) = runtime_watch_with_config_and_sender(mooncake_config());
        let cache = HicacheSharedKvCache::new(runtime_configs);
        assert_eq!(
            cache.kv_events_endpoint().as_deref(),
            Some("tcp://127.0.0.1:5557")
        );

        let mut updated = mooncake_config();
        updated.kv_events_endpoint = Some("tcp://127.0.0.1:5558".to_string());
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, updated)
            .unwrap();
        tx.send(HashMap::from([(1, runtime_config)])).unwrap();

        assert_eq!(
            cache.kv_events_endpoint().as_deref(),
            Some("tcp://127.0.0.1:5558")
        );
    }

    #[test]
    fn test_kv_events_endpoint_tolerates_worker_metadata_omission() {
        let mut advertised = mooncake_config();
        advertised.kv_events_endpoint = Some("tcp://127.0.0.1:5557".to_string());
        let mut missing = advertised.clone();
        missing.kv_events_endpoint = None;
        let mut advertised_runtime = ModelRuntimeConfig::new();
        advertised_runtime
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, advertised)
            .unwrap();
        let mut missing_runtime = ModelRuntimeConfig::new();
        missing_runtime
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, missing)
            .unwrap();
        let (_tx, runtime_configs) = watch::channel(HashMap::from([
            (1, advertised_runtime),
            (2, missing_runtime),
        ]));
        let cache = HicacheSharedKvCache::new(runtime_configs);

        assert_eq!(
            cache.kv_events_endpoint().as_deref(),
            Some("tcp://127.0.0.1:5557")
        );
    }

    #[test]
    fn test_frontend_kv_events_endpoint_overrides_worker_metadata() {
        let mut worker_config = mooncake_config();
        worker_config.kv_events_endpoint = None;
        let cache = HicacheSharedKvCache::new_with_cancellation_and_endpoint(
            runtime_watch_with_config(worker_config),
            CancellationToken::new(),
            Some("tcp://frontend-config:5557".to_string()),
        );

        assert_eq!(
            cache.kv_events_endpoint().as_deref(),
            Some("tcp://frontend-config:5557")
        );
    }

    #[tokio::test]
    async fn test_layout_change_clears_cached_hits() {
        let config = mooncake_config();
        let (runtime_configs, tx) = runtime_watch_with_config_and_sender(config.clone());
        let cache = HicacheSharedKvCache::new(runtime_configs);
        let hash = logical_page_hashes(&[1, 2, 3, 4], config.page_size, config.is_eagle)
            .pop()
            .unwrap();
        let group_id = sglang_group_id(&hash, &config);
        cache.apply_batch(
            1,
            expand_actual_query_keys(&hash, &config)
                .into_iter()
                .map(|object_key| MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(object_key),
                    tenant_id: "default".to_string(),
                    group_id: Some(group_id.clone()),
                    ..Default::default()
                })
                .collect(),
        );
        assert_eq!(
            cache
                .check_blocks(&[1, 2, 3, 4], config.page_size, None)
                .await
                .unwrap()
                .total_hits,
            1
        );

        let mut new_layout = config;
        new_layout.tp_size = 2;
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, new_layout)
            .unwrap();
        tx.send(HashMap::from([(1, runtime_config)])).unwrap();

        assert_eq!(
            cache
                .check_blocks(&[1, 2, 3, 4], 4, None)
                .await
                .unwrap()
                .total_hits,
            0
        );
        assert!(cache.present_keys.is_empty());
        assert!(cache.group_states.is_empty());
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
                    ..Default::default()
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash0}_0_v")),
                    tenant_id: "default".to_string(),
                    group_id: None,
                    ..Default::default()
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash1}_0_k")),
                    tenant_id: "default".to_string(),
                    group_id: None,
                    ..Default::default()
                },
            ],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
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
                ..Default::default()
            }],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
            .await
            .unwrap();
        assert_eq!(hits.total_hits, 0);
    }

    fn stored_events(object_keys: &[String], group_id: &str) -> Vec<MooncakeObjectEvent> {
        object_keys
            .iter()
            .map(|object_key| MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some(object_key.clone()),
                tenant_id: "default".to_string(),
                group_id: Some(group_id.to_string()),
                ..Default::default()
            })
            .collect()
    }

    // Key names as stored by an SGLang worker serving `Qwen/Qwen3-0.6B` with no
    // `extra_backend_tag`.
    #[tokio::test]
    async fn test_check_blocks_matches_sglang_model_prefixed_keys() {
        let hash = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72";
        let config = SglangHicacheMooncakeConfig {
            key_prefix: Some("Qwen-Qwen3-0.6B".to_string()),
            ..mooncake_config()
        };
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
        cache.apply_batch(
            1,
            stored_events(
                &[
                    format!("Qwen-Qwen3-0.6B_{hash}_0_k"),
                    format!("Qwen-Qwen3-0.6B_{hash}_0_v"),
                ],
                &format!("sglang-hicache:Qwen-Qwen3-0.6B_{hash}"),
            ),
        );

        let hits = cache.check_blocks(&[1, 2, 3, 4], 4, None).await.unwrap();
        assert_eq!(hits.total_hits, 1);
        assert!(
            cache
                .group_states
                .get(&format!("sglang-hicache:Qwen-Qwen3-0.6B_{hash}"))
                .is_some_and(|v| v.1)
        );
    }

    #[tokio::test]
    async fn test_check_blocks_invalidates_group_on_unlabeled_removal() {
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
                    ..Default::default()
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash}_0_v")),
                    tenant_id: "default".to_string(),
                    group_id: Some(group_id.clone()),
                    ..Default::default()
                },
            ],
        );

        let hits = cache.check_blocks(&[1, 2, 3, 4], 4, None).await.unwrap();
        assert_eq!(hits.total_hits, 1);
        assert!(cache.group_states.get(&group_id).is_some_and(|v| v.1));

        cache.apply_batch(
            2,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some(format!("{hash}_0_v")),
                tenant_id: "default".to_string(),
                group_id: None,
                ..Default::default()
            }],
        );
        assert!(cache.group_states.is_empty());
        let hits = cache.check_blocks(&[1, 2, 3, 4], 4, None).await.unwrap();
        assert_eq!(hits.total_hits, 0);
    }

    #[tokio::test]
    async fn test_labeled_removal_preserves_other_verified_groups() {
        let hash0 = "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72";
        let hash1 = "4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6";
        let group0 = format!("sglang-hicache:{hash0}");
        let group1 = format!("sglang-hicache:{hash1}");
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            1,
            [(&hash0, &group0), (&hash1, &group1)]
                .into_iter()
                .flat_map(|(hash, group_id)| {
                    ["k", "v"].into_iter().map(move |kind| MooncakeObjectEvent {
                        event_type: "stored".to_string(),
                        object_key: Some(format!("{hash}_0_{kind}")),
                        tenant_id: "default".to_string(),
                        group_id: Some(group_id.clone()),
                        ..Default::default()
                    })
                })
                .collect(),
        );
        assert_eq!(
            cache
                .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
                .await
                .unwrap()
                .total_hits,
            2
        );

        cache.apply_batch(
            2,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some(format!("{hash0}_0_k")),
                tenant_id: "default".to_string(),
                group_id: Some(group0.clone()),
                ..Default::default()
            }],
        );

        assert!(!cache.group_states.contains_key(&group0));
        assert!(cache.group_states.get(&group1).is_some_and(|state| state.1));
    }

    #[test]
    fn test_duplicate_sequence_preserves_shared_cache_state() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            0,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: None,
                ..Default::default()
            }],
        );
        cache.apply_batch(
            0,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: None,
                ..Default::default()
            }],
        );

        assert!(cache.present_keys.contains("key-0"));
    }

    #[test]
    fn test_index_size_limit_clears_shared_cache_state() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        cache.apply_batch(
            1,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: Some("group-0".to_string()),
                ..Default::default()
            }],
        );

        cache.clear_if_index_too_large(1);

        assert!(cache.present_keys.is_empty());
        assert!(cache.group_states.is_empty());
    }

    #[tokio::test]
    async fn test_subscriber_retries_failed_connection_until_cancelled() {
        let mut config = mooncake_config();
        config.kv_events_endpoint = Some("invalid://mooncake-events".to_string());
        let (runtime_configs, _tx) = runtime_watch_with_config_and_sender(config);
        let cache = HicacheSharedKvCache::new(runtime_configs);
        let cancellation_token = CancellationToken::new();
        let task = tokio::spawn(cache.run_subscriber(cancellation_token.clone()));

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished());
        cancellation_token.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn test_subscriber_clears_state_when_runtime_config_watch_closes() {
        let (tx, runtime_configs) = watch::channel(HashMap::<WorkerId, ModelRuntimeConfig>::new());
        let cache = HicacheSharedKvCache::new(runtime_configs);
        cache.apply_batch(
            1,
            vec![MooncakeObjectEvent {
                event_type: "stored".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: None,
                ..Default::default()
            }],
        );
        let task = tokio::spawn(cache.clone().run_subscriber(CancellationToken::new()));

        drop(tx);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();

        assert!(cache.present_keys.is_empty());
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
                ..Default::default()
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
                ..Default::default()
            }],
        );

        assert!(!cache.present_keys.contains("old_0_k"));
        assert!(cache.group_states.is_empty());
        assert!(cache.present_keys.contains("new_0_k"));
    }

    #[tokio::test]
    async fn test_check_blocks_skips_mooncake_for_cache_namespace() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, Some("tenant-a"))
            .await
            .unwrap();

        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);
    }
}
