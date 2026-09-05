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

#[cfg(test)]
mod ha_tests;
mod mooncake_ha;

use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

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
    SharedKvCache,
    indexer::KvRouterError,
    protocols::{SharedCacheHits, WorkerId},
};

use self::mooncake_ha::{MooncakeHaConfig, MooncakeHaResolver, is_mooncake_leader_unavailable};

const SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY: &str = "sglang_hicache_mooncake";
const MOONCAKE_EVENT_RECONNECT_DELAY: Duration = Duration::from_secs(1);
const MOONCAKE_LEADER_REFRESH_INTERVAL: Duration = Duration::from_secs(1);
const MOONCAKE_LEADER_RESOLVE_TIMEOUT: Duration = Duration::from_secs(2);
const MOONCAKE_LEADER_WARNING_INTERVAL: Duration = Duration::from_secs(30);
const MAX_MOONCAKE_INDEX_ENTRIES: usize = 1_000_000;

type MooncakeEndpointResolution =
    Pin<Box<dyn Future<Output = anyhow::Result<Option<String>>> + Send>>;

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
    #[serde(default)]
    kv_events_endpoint: Option<String>,
    #[serde(flatten)]
    ha: MooncakeHaConfig,
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
            && self.ha.is_compatible_with(&other.ha)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct MooncakeObjectEvent {
    event_type: String,
    #[serde(default)]
    object_key: Option<String>,
    #[serde(default)]
    tenant_id: String,
    #[serde(default)]
    group_id: Option<String>,
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
    group_states: Arc<DashMap<String, (u64, bool)>>,
    last_sequence: Arc<AtomicU64>,
    has_sequence: Arc<AtomicBool>,
    last_layout: Arc<ArcSwapOption<SglangHicacheMooncakeConfig>>,
    cancellation_token: CancellationToken,
    frontend_kv_events_endpoint: Option<String>,
    ha_resolver: MooncakeHaResolver,
    last_leader_resolution_warning: Arc<Mutex<Option<Instant>>>,
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
            group_states: Arc::new(DashMap::new()),
            last_sequence: Arc::new(AtomicU64::new(0)),
            has_sequence: Arc::new(AtomicBool::new(false)),
            last_layout: Arc::new(ArcSwapOption::empty()),
            cancellation_token,
            frontend_kv_events_endpoint,
            ha_resolver: MooncakeHaResolver::default(),
            last_leader_resolution_warning: Arc::new(Mutex::new(None)),
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
        if let Some(previous) = last_layout.as_ref()
            && previous.has_same_layout(layout)
        {
            // During a rolling upgrade, old workers omit the HA fields. Enrich the saved
            // snapshot once a new worker advertises them so a later locator change is no
            // longer compared against a permanent wildcard.
            let enriched_ha = layout.ha.enriched_with(&previous.ha);
            if enriched_ha != previous.ha {
                let mut enriched = layout.clone();
                enriched.ha = enriched_ha;
                self.last_layout.store(Some(Arc::new(enriched)));
            }
            return;
        }
        if last_layout.is_some() {
            self.clear();
            self.ha_resolver.config_changed();
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

        let mut resolved = first.clone();
        resolved.ha = match MooncakeHaConfig::merge(configs.iter().map(|(_, config)| &config.ha)) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(
                    workers = ?configs.iter().map(|(worker_id, _)| *worker_id).collect::<Vec<_>>(),
                    %error,
                    "SGLang Mooncake HA runtime configs differ across workers; skipping shared-cache lookup"
                );
                return None;
            }
        };

        self.clear_on_layout_change(&resolved);

        if let Some(endpoint) = &self.frontend_kv_events_endpoint {
            return Some((resolved, endpoint.clone()));
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
        Some((resolved, endpoint.to_string()))
    }

    #[cfg(test)]
    fn kv_events_endpoint(&self) -> Option<String> {
        self.resolve_mooncake_config_and_endpoint()
            .map(|(_, endpoint)| endpoint)
    }

    async fn resolved_kv_events_endpoint(
        &self,
        force_refresh: bool,
    ) -> anyhow::Result<Option<String>> {
        let Some((config, configured_endpoint)) = self.resolve_mooncake_config_and_endpoint()
        else {
            return Ok(None);
        };
        // This environment variable is an explicit frontend-side override and may point to a
        // stable Service or relay which already follows the leader. Only worker-advertised
        // endpoints are treated as host templates for direct leader discovery.
        if self.frontend_kv_events_endpoint.is_some() {
            return Ok(Some(configured_endpoint));
        }
        self.ha_resolver
            .resolve_event_endpoint(&config.ha, &configured_endpoint, force_refresh)
            .await
            .map(Some)
    }

    fn endpoint_resolution(&self, force_refresh: bool) -> MooncakeEndpointResolution {
        let cache = self.clone();
        Box::pin(async move {
            tokio::time::timeout(
                MOONCAKE_LEADER_RESOLVE_TIMEOUT,
                cache.resolved_kv_events_endpoint(force_refresh),
            )
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Mooncake leader resolution exceeded {:?}",
                    MOONCAKE_LEADER_RESOLVE_TIMEOUT
                )
            })?
        })
    }

    fn apply_batch(&self, sequence: u64, events: Vec<MooncakeObjectEvent>) {
        // SGLang's ZmqEventPublisher increments this sequence once per published batch, so a
        // non-consecutive value means one or more whole batches were missed.
        let has_previous = self.has_sequence.swap(true, Ordering::AcqRel);
        let previous = self.last_sequence.swap(sequence, Ordering::AcqRel);
        if has_previous && sequence == previous {
            return;
        }
        if has_previous && sequence != previous.wrapping_add(1) {
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
                    if let Some(group_id) = group_id {
                        self.group_states.insert(group_id, (sequence, false));
                    }
                }
                "removed" => {
                    self.present_keys.remove(&object_key);
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

        self.clear_if_index_too_large(MAX_MOONCAKE_INDEX_ENTRIES);
    }

    fn clear(&self) {
        self.present_keys.clear();
        self.group_states.clear();
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

    fn should_warn_leader_resolution_failure(&self) -> bool {
        self.should_warn_leader_resolution_failure_at(Instant::now())
    }

    fn should_warn_leader_resolution_failure_at(&self, now: Instant) -> bool {
        let mut last_warned_at = self
            .last_leader_resolution_warning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if last_warned_at.is_some_and(|last| {
            now.saturating_duration_since(last) < MOONCAKE_LEADER_WARNING_INTERVAL
        }) {
            return false;
        }
        *last_warned_at = Some(now);
        true
    }

    fn reset_leader_resolution_warning(&self) {
        *self
            .last_leader_resolution_warning
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    fn should_reconnect_after_endpoint_resolution(
        &self,
        endpoint: &str,
        result: anyhow::Result<Option<String>>,
    ) -> bool {
        match result {
            Ok(next_endpoint) => {
                self.reset_leader_resolution_warning();
                if next_endpoint.as_deref() != Some(endpoint) {
                    tracing::info!(%endpoint, next_endpoint = ?next_endpoint, "Mooncake HA leader changed; reconnecting KV events");
                    true
                } else {
                    false
                }
            }
            Err(error) if is_mooncake_leader_unavailable(&error) => {
                self.record_subscriber_error();
                self.clear();
                if self.should_warn_leader_resolution_failure() {
                    tracing::warn!(%endpoint, %error, "Mooncake HA cluster has no active leader; cleared shared-cache state and reconnecting");
                }
                true
            }
            Err(error) => {
                self.record_subscriber_error();
                if self.should_warn_leader_resolution_failure() {
                    tracing::warn!(%endpoint, %error, "Failed to refresh Mooncake HA leader; keeping current KV event subscription");
                }
                false
            }
        }
    }

    async fn run_subscriber(mut self, cancellation_token: CancellationToken) {
        loop {
            let endpoint = loop {
                let mut resolution = self.endpoint_resolution(false);
                let result = tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    result = self.runtime_configs.changed() => {
                        if result.is_err() {
                            self.clear();
                            return;
                        }
                        continue;
                    }
                    result = &mut resolution => result,
                };

                match result {
                    Ok(Some(endpoint)) => {
                        self.reset_leader_resolution_warning();
                        break endpoint;
                    }
                    Ok(None) => self.reset_leader_resolution_warning(),
                    Err(error) => {
                        self.record_subscriber_error();
                        if self.should_warn_leader_resolution_failure() {
                            tracing::warn!(%error, "Failed to resolve Mooncake KV event leader; retrying");
                        }
                    }
                }

                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    result = self.runtime_configs.changed() => {
                        if result.is_err() {
                            self.clear();
                            return;
                        }
                    }
                    _ = tokio::time::sleep(MOONCAKE_EVENT_RECONNECT_DELAY) => {}
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
            let mut leader_refresh = tokio::time::interval(MOONCAKE_LEADER_REFRESH_INTERVAL);
            leader_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            leader_refresh.tick().await;
            let mut endpoint_resolution: Option<MooncakeEndpointResolution> = None;

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => return,
                    result = self.runtime_configs.changed() => {
                        if result.is_err() {
                            self.clear();
                            return;
                        }
                        if endpoint_resolution.is_none() {
                            endpoint_resolution = Some(self.endpoint_resolution(false));
                        }
                    }
                    _ = leader_refresh.tick(), if endpoint_resolution.is_none() => {
                        endpoint_resolution = Some(self.endpoint_resolution(true));
                    }
                    result = async {
                        endpoint_resolution
                            .as_mut()
                            .expect("endpoint resolution is present when branch is enabled")
                            .await
                    }, if endpoint_resolution.is_some() => {
                        endpoint_resolution = None;
                        if self.should_reconnect_after_endpoint_resolution(&endpoint, result) {
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

            self.clear();
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

                let hit = expand_actual_query_keys(page_hash, &config)
                    .iter()
                    .all(|key| self.present_keys.contains(key));
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
            kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
            ha: MooncakeHaConfig::default(),
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

    #[test]
    fn test_sglang_group_id_uses_extra_backend_tag() {
        let config = SglangHicacheMooncakeConfig {
            extra_backend_tag: Some("tag".to_string()),
            ..mooncake_config()
        };

        assert_eq!(sglang_group_id("hash", &config), "sglang-hicache:tag_hash");
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
            }],
        );
        let hits = cache
            .check_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)
            .await
            .unwrap();
        assert_eq!(hits.total_hits, 0);
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
                },
                MooncakeObjectEvent {
                    event_type: "stored".to_string(),
                    object_key: Some(format!("{hash}_0_v")),
                    tenant_id: "default".to_string(),
                    group_id: Some(group_id.clone()),
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
            }],
        );
        cache.apply_batch(
            0,
            vec![MooncakeObjectEvent {
                event_type: "removed".to_string(),
                object_key: Some("key-0".to_string()),
                tenant_id: "default".to_string(),
                group_id: None,
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

    #[test]
    fn test_leader_resolution_warnings_are_throttled_and_reset() {
        let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
        let now = Instant::now();

        assert!(cache.should_warn_leader_resolution_failure_at(now));
        assert!(!cache.should_warn_leader_resolution_failure_at(now + Duration::from_secs(1)));
        assert!(
            cache.should_warn_leader_resolution_failure_at(now + MOONCAKE_LEADER_WARNING_INTERVAL)
        );

        cache.reset_leader_resolution_warning();
        assert!(cache.should_warn_leader_resolution_failure_at(
            now + MOONCAKE_LEADER_WARNING_INTERVAL + Duration::from_secs(1)
        ));
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
