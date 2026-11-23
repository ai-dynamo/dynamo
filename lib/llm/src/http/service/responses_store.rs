// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-side storage for OpenAI Responses API objects.
//!
//! A stored response is an immutable node in a conversation chain: it keeps
//! the input items and output items of its own turn plus the id of the
//! response it continued from. History is rebuilt by walking the chain, so a
//! chain can fork freely and nothing is ever rewritten. Items are kept as
//! Responses API JSON, independent of the engine that produced them, and are
//! replayed by deserializing them back into request input items.
//!
//! This module owns the [`ResponseStore`] contract and a bounded in-memory
//! backend for development and tests. Durable backends plug in through the
//! same trait.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm as env_llm;
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Default lifetime of a stored response.
pub const DEFAULT_TTL_SECS: u64 = 3600;
/// Default record cap of the memory backend.
pub const DEFAULT_MAX_RECORDS: usize = 10_000;
/// Default serialized-bytes cap of the memory backend (256 MiB).
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Default continuation depth limit.
pub const DEFAULT_MAX_CHAIN_DEPTH: usize = 256;

/// One stored response: the delta of a single turn plus its chain links.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredResponse {
    /// Public response id (`resp_...`).
    pub id: String,
    /// The response this turn continued from, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// First response id of the chain. Equals `id` for a chain of length one.
    pub root_id: String,
    /// Served model name the response was generated with.
    pub model: String,
    /// Unix seconds when the response was created.
    pub created_at: u64,
    /// Input items of this turn only, as Responses API item JSON.
    #[serde(default)]
    pub input_items: Vec<serde_json::Value>,
    /// Output items of this turn only, as Responses API item JSON.
    #[serde(default)]
    pub output_items: Vec<serde_json::Value>,
    /// The public response object exactly as it was returned to the client.
    pub response: serde_json::Value,
}

impl StoredResponse {
    /// Serialized size used for the memory backend's byte accounting.
    pub fn serialized_bytes(&self) -> usize {
        serde_json::to_vec(self).map(|v| v.len()).unwrap_or(0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResponseStoreError {
    /// The record exceeds the backend's per-record or total capacity.
    #[error("stored response {id} is too large for the store ({bytes} bytes, limit {limit})")]
    TooLarge {
        id: String,
        bytes: usize,
        limit: usize,
    },
    /// The backend could not be reached or failed internally.
    #[error("response store unavailable: {0}")]
    Unavailable(String),
}

/// Storage contract for Responses API objects.
#[async_trait]
pub trait ResponseStore: Send + Sync {
    /// Store a response node. Ids are unique per response, so a second `put`
    /// with the same id replaces the first.
    async fn put(&self, record: StoredResponse) -> Result<(), ResponseStoreError>;

    /// Fetch a response node, or `None` when it is unknown, deleted, or expired.
    async fn get(&self, id: &str) -> Result<Option<Arc<StoredResponse>>, ResponseStoreError>;

    /// Delete a response node. Returns whether a live node was removed.
    async fn delete(&self, id: &str) -> Result<bool, ResponseStoreError>;

    /// Backend name for logs and diagnostics.
    fn backend_name(&self) -> &'static str;
}

/// Why a continuation chain could not be rebuilt.
#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    /// A response in the chain is unknown, deleted, or expired.
    #[error("previous response '{0}' not found")]
    NotFound(String),
    /// The chain is longer than the configured limit.
    #[error("response chain exceeds the maximum depth of {limit}")]
    DepthExceeded { limit: usize },
    #[error(transparent)]
    Store(#[from] ResponseStoreError),
}

/// The items a continuation replays, oldest first, plus the chain's root.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedHistory {
    pub root_id: String,
    /// Number of stored responses in the chain.
    pub depth: usize,
    /// Input items then output items of every node, oldest node first.
    pub items: Vec<serde_json::Value>,
}

/// Walk the chain that ends at `previous_response_id` and flatten it into the
/// items a continuation has to replay before its own input.
pub async fn resolve_history(
    store: &dyn ResponseStore,
    previous_response_id: &str,
    max_depth: usize,
) -> Result<ResolvedHistory, HistoryError> {
    let mut nodes: Vec<Arc<StoredResponse>> = Vec::new();
    let mut cursor = Some(previous_response_id.to_string());
    while let Some(id) = cursor {
        if nodes.len() >= max_depth {
            return Err(HistoryError::DepthExceeded { limit: max_depth });
        }
        let Some(node) = store.get(&id).await? else {
            return Err(HistoryError::NotFound(id));
        };
        cursor = node.previous_response_id.clone();
        nodes.push(node);
    }
    // `nodes` is newest first; the first pushed node is the one the client named.
    let root_id = nodes
        .last()
        .map(|node| node.root_id.clone())
        .unwrap_or_else(|| previous_response_id.to_string());
    let depth = nodes.len();
    let mut items = Vec::new();
    for node in nodes.iter().rev() {
        items.extend(node.input_items.iter().cloned());
        items.extend(node.output_items.iter().cloned());
    }
    Ok(ResolvedHistory {
        root_id,
        depth,
        items,
    })
}

/// Unix seconds now.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Limits of the in-memory backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStoreConfig {
    /// Lifetime of a node from `put`. `None` keeps nodes until evicted.
    pub ttl: Option<Duration>,
    /// Maximum number of live nodes.
    pub max_records: usize,
    /// Maximum total serialized bytes of live nodes.
    pub max_bytes: usize,
}

impl Default for MemoryStoreConfig {
    fn default() -> Self {
        Self {
            ttl: Some(Duration::from_secs(DEFAULT_TTL_SECS)),
            max_records: DEFAULT_MAX_RECORDS,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

struct MemoryEntry {
    record: Arc<StoredResponse>,
    bytes: usize,
    expires_at: Option<Instant>,
}

struct MemoryInner {
    entries: LruCache<String, MemoryEntry>,
    bytes: usize,
}

/// Bounded in-memory [`ResponseStore`]: TTL, record cap, byte cap, LRU eviction.
///
/// Intended for a single frontend process in development and tests. Nodes
/// live only in this process, so a second frontend replica cannot continue
/// them; production deployments use an external backend.
pub struct MemoryResponseStore {
    config: MemoryStoreConfig,
    inner: Mutex<MemoryInner>,
}

impl MemoryResponseStore {
    pub fn new(config: MemoryStoreConfig) -> Self {
        let capacity = NonZeroUsize::new(config.max_records.max(1)).expect("non-zero");
        Self {
            config,
            inner: Mutex::new(MemoryInner {
                entries: LruCache::new(capacity),
                bytes: 0,
            }),
        }
    }

    pub fn config(&self) -> &MemoryStoreConfig {
        &self.config
    }

    /// Live record count after dropping expired nodes.
    pub fn len(&self) -> usize {
        let mut inner = self.inner.lock();
        Self::drop_expired(&mut inner, Instant::now());
        inner.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialized bytes of live records after dropping expired nodes.
    pub fn bytes(&self) -> usize {
        let mut inner = self.inner.lock();
        Self::drop_expired(&mut inner, Instant::now());
        inner.bytes
    }

    fn drop_expired(inner: &mut MemoryInner, now: Instant) {
        let expired: Vec<String> = inner
            .entries
            .iter()
            .filter(|(_, entry)| entry.expires_at.is_some_and(|at| at <= now))
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            if let Some(entry) = inner.entries.pop(&id) {
                inner.bytes -= entry.bytes;
            }
        }
    }

    fn evict_until_fits(inner: &mut MemoryInner, incoming: usize, max_bytes: usize) {
        while inner.bytes + incoming > max_bytes {
            match inner.entries.pop_lru() {
                Some((_, entry)) => inner.bytes -= entry.bytes,
                None => break,
            }
        }
    }
}

#[async_trait]
impl ResponseStore for MemoryResponseStore {
    async fn put(&self, record: StoredResponse) -> Result<(), ResponseStoreError> {
        let bytes = record.serialized_bytes();
        if bytes > self.config.max_bytes {
            return Err(ResponseStoreError::TooLarge {
                id: record.id,
                bytes,
                limit: self.config.max_bytes,
            });
        }
        let now = Instant::now();
        let mut inner = self.inner.lock();
        Self::drop_expired(&mut inner, now);
        if let Some(previous) = inner.entries.pop(&record.id) {
            inner.bytes -= previous.bytes;
        }
        Self::evict_until_fits(&mut inner, bytes, self.config.max_bytes);
        // The LRU capacity bounds record count; a push at capacity evicts the
        // least recently used entry, whose bytes must leave the account too.
        let id = record.id.clone();
        let entry = MemoryEntry {
            record: Arc::new(record),
            bytes,
            expires_at: self.config.ttl.map(|ttl| now + ttl),
        };
        if let Some((_, evicted)) = inner.entries.push(id, entry) {
            inner.bytes -= evicted.bytes;
        }
        inner.bytes += bytes;
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Arc<StoredResponse>>, ResponseStoreError> {
        let now = Instant::now();
        let mut inner = self.inner.lock();
        let expired = inner
            .entries
            .peek(id)
            .is_some_and(|entry| entry.expires_at.is_some_and(|at| at <= now));
        if expired {
            if let Some(entry) = inner.entries.pop(id) {
                inner.bytes -= entry.bytes;
            }
            return Ok(None);
        }
        Ok(inner.entries.get(id).map(|entry| entry.record.clone()))
    }

    async fn delete(&self, id: &str) -> Result<bool, ResponseStoreError> {
        let now = Instant::now();
        let mut inner = self.inner.lock();
        match inner.entries.pop(id) {
            Some(entry) => {
                inner.bytes -= entry.bytes;
                Ok(entry.expires_at.is_none_or(|at| at > now))
            }
            None => Ok(false),
        }
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }
}

/// Store selection from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseStoreSelector {
    None,
    Memory(MemoryStoreConfig),
}

fn env_usize(name: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("{name}={value:?} is not a non-negative integer: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(anyhow::anyhow!("{name} is not valid unicode: {e}")),
    }
}

impl ResponseStoreSelector {
    /// Read `DYN_HTTP_RESPONSES_STORE` and its sizing variables.
    pub fn from_env() -> anyhow::Result<Self> {
        let backend = match std::env::var(env_llm::DYN_HTTP_RESPONSES_STORE) {
            Ok(value) => value.trim().to_ascii_lowercase(),
            Err(std::env::VarError::NotPresent) => return Ok(Self::None),
            Err(e) => anyhow::bail!(
                "{} is not valid unicode: {e}",
                env_llm::DYN_HTTP_RESPONSES_STORE
            ),
        };
        match backend.as_str() {
            "" | "none" | "off" | "disabled" => Ok(Self::None),
            "memory" => {
                let ttl_secs = env_usize(
                    env_llm::DYN_HTTP_RESPONSES_STORE_TTL_SECS,
                    DEFAULT_TTL_SECS as usize,
                )?;
                let max_records = env_usize(
                    env_llm::DYN_HTTP_RESPONSES_STORE_MAX_RECORDS,
                    DEFAULT_MAX_RECORDS,
                )?;
                let max_bytes = env_usize(
                    env_llm::DYN_HTTP_RESPONSES_STORE_MAX_BYTES,
                    DEFAULT_MAX_BYTES,
                )?;
                if max_records == 0 || max_bytes == 0 {
                    anyhow::bail!(
                        "{} and {} must be greater than zero",
                        env_llm::DYN_HTTP_RESPONSES_STORE_MAX_RECORDS,
                        env_llm::DYN_HTTP_RESPONSES_STORE_MAX_BYTES
                    );
                }
                Ok(Self::Memory(MemoryStoreConfig {
                    ttl: (ttl_secs > 0).then(|| Duration::from_secs(ttl_secs as u64)),
                    max_records,
                    max_bytes,
                }))
            }
            other => anyhow::bail!(
                "{}={other:?} is not supported; expected one of none, memory",
                env_llm::DYN_HTTP_RESPONSES_STORE
            ),
        }
    }

    /// Build the selected backend, `None` when storage is disabled.
    pub fn build(self) -> Option<Arc<dyn ResponseStore>> {
        match self {
            Self::None => None,
            Self::Memory(config) => Some(Arc::new(MemoryResponseStore::new(config))),
        }
    }
}

/// Continuation depth limit from `DYN_HTTP_RESPONSES_MAX_CHAIN_DEPTH`.
pub fn max_chain_depth_from_env() -> anyhow::Result<usize> {
    let depth = env_usize(
        env_llm::DYN_HTTP_RESPONSES_MAX_CHAIN_DEPTH,
        DEFAULT_MAX_CHAIN_DEPTH,
    )?;
    if depth == 0 {
        anyhow::bail!(
            "{} must be greater than zero",
            env_llm::DYN_HTTP_RESPONSES_MAX_CHAIN_DEPTH
        );
    }
    Ok(depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str, previous: Option<&str>, root: &str, turn: &str) -> StoredResponse {
        StoredResponse {
            id: id.to_string(),
            previous_response_id: previous.map(str::to_string),
            root_id: root.to_string(),
            model: "m".to_string(),
            created_at: 1,
            input_items: vec![
                json!({"type": "message", "role": "user", "content": format!("in-{turn}")}),
            ],
            output_items: vec![
                json!({"type": "message", "role": "assistant", "content": format!("out-{turn}")}),
            ],
            response: json!({"id": id, "object": "response"}),
        }
    }

    fn store(config: MemoryStoreConfig) -> MemoryResponseStore {
        MemoryResponseStore::new(config)
    }

    fn unbounded() -> MemoryStoreConfig {
        MemoryStoreConfig {
            ttl: None,
            max_records: 1000,
            max_bytes: usize::MAX,
        }
    }

    #[tokio::test]
    async fn put_get_delete_round_trip() {
        let store = store(unbounded());
        store.put(node("a", None, "a", "1")).await.unwrap();
        let got = store.get("a").await.unwrap().expect("stored");
        assert_eq!(got.id, "a");
        assert_eq!(store.len(), 1);
        assert!(store.bytes() > 0);
        assert!(store.delete("a").await.unwrap());
        assert!(!store.delete("a").await.unwrap());
        assert!(store.get("a").await.unwrap().is_none());
        assert_eq!(store.bytes(), 0);
    }

    #[tokio::test]
    async fn record_cap_evicts_least_recently_used() {
        let store = store(MemoryStoreConfig {
            max_records: 2,
            ..unbounded()
        });
        store.put(node("a", None, "a", "1")).await.unwrap();
        store.put(node("b", None, "b", "1")).await.unwrap();
        // Touch `a` so `b` becomes the eviction candidate.
        store.get("a").await.unwrap();
        store.put(node("c", None, "c", "1")).await.unwrap();
        assert!(store.get("a").await.unwrap().is_some());
        assert!(store.get("b").await.unwrap().is_none());
        assert!(store.get("c").await.unwrap().is_some());
        assert_eq!(store.len(), 2);
        let expected: usize = ["a", "c"]
            .iter()
            .map(|id| node(id, None, id, "1").serialized_bytes())
            .sum();
        assert_eq!(store.bytes(), expected);
    }

    #[tokio::test]
    async fn byte_cap_evicts_until_the_record_fits() {
        let one = node("a", None, "a", "1").serialized_bytes();
        let store = store(MemoryStoreConfig {
            max_bytes: one * 2 + one / 2,
            ..unbounded()
        });
        store.put(node("a", None, "a", "1")).await.unwrap();
        store.put(node("b", None, "b", "1")).await.unwrap();
        store.put(node("c", None, "c", "1")).await.unwrap();
        assert!(store.get("a").await.unwrap().is_none());
        assert_eq!(store.len(), 2);
        assert!(store.bytes() <= one * 2 + one / 2);
    }

    #[tokio::test]
    async fn oversized_record_is_rejected() {
        let store = store(MemoryStoreConfig {
            max_bytes: 8,
            ..unbounded()
        });
        let err = store.put(node("a", None, "a", "1")).await.unwrap_err();
        assert!(matches!(err, ResponseStoreError::TooLarge { .. }), "{err}");
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn replacing_an_id_keeps_the_byte_account_exact() {
        let store = store(unbounded());
        store.put(node("a", None, "a", "1")).await.unwrap();
        let mut bigger = node("a", None, "a", "1");
        bigger
            .output_items
            .push(json!({"type": "message", "content": "x".repeat(64)}));
        let expected = bigger.serialized_bytes();
        store.put(bigger).await.unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes(), expected);
    }

    #[tokio::test]
    async fn expired_records_vanish() {
        let store = store(MemoryStoreConfig {
            ttl: Some(Duration::from_millis(5)),
            ..unbounded()
        });
        store.put(node("a", None, "a", "1")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(store.get("a").await.unwrap().is_none());
        assert_eq!(store.len(), 0);
        assert_eq!(store.bytes(), 0);
        // Deleting an expired entry reports nothing live was removed.
        store.put(node("b", None, "b", "1")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!store.delete("b").await.unwrap());
    }

    #[tokio::test]
    async fn history_flattens_the_chain_oldest_first() {
        let store = store(unbounded());
        store.put(node("r1", None, "r1", "1")).await.unwrap();
        store.put(node("r2", Some("r1"), "r1", "2")).await.unwrap();
        store.put(node("r3", Some("r2"), "r1", "3")).await.unwrap();
        let history = resolve_history(&store, "r3", 16).await.unwrap();
        assert_eq!(history.root_id, "r1");
        assert_eq!(history.depth, 3);
        let texts: Vec<&str> = history
            .items
            .iter()
            .map(|item| item["content"].as_str().unwrap())
            .collect();
        assert_eq!(texts, ["in-1", "out-1", "in-2", "out-2", "in-3", "out-3"]);
    }

    #[tokio::test]
    async fn history_supports_forks() {
        let store = store(unbounded());
        store.put(node("r1", None, "r1", "1")).await.unwrap();
        store.put(node("a", Some("r1"), "r1", "a")).await.unwrap();
        store.put(node("b", Some("r1"), "r1", "b")).await.unwrap();
        let a = resolve_history(&store, "a", 16).await.unwrap();
        let b = resolve_history(&store, "b", 16).await.unwrap();
        assert_eq!(a.root_id, "r1");
        assert_eq!(b.root_id, "r1");
        assert_eq!(a.items.len(), 4);
        assert_eq!(a.items[2]["content"], "in-a");
        assert_eq!(b.items[2]["content"], "in-b");
    }

    #[tokio::test]
    async fn history_reports_missing_and_too_deep_chains() {
        let store = store(unbounded());
        store.put(node("r2", Some("r1"), "r1", "2")).await.unwrap();
        let err = resolve_history(&store, "r2", 16).await.unwrap_err();
        assert!(
            matches!(err, HistoryError::NotFound(ref id) if id == "r1"),
            "{err}"
        );
        let err = resolve_history(&store, "missing", 16).await.unwrap_err();
        assert!(
            matches!(err, HistoryError::NotFound(ref id) if id == "missing"),
            "{err}"
        );

        store.put(node("r1", None, "r1", "1")).await.unwrap();
        let err = resolve_history(&store, "r2", 1).await.unwrap_err();
        assert!(
            matches!(err, HistoryError::DepthExceeded { limit: 1 }),
            "{err}"
        );
        assert!(resolve_history(&store, "r2", 2).await.is_ok());
    }

    #[test]
    fn selector_defaults_to_none_and_parses_memory() {
        temp_env::with_vars(
            [
                (env_llm::DYN_HTTP_RESPONSES_STORE, None::<&str>),
                (env_llm::DYN_HTTP_RESPONSES_STORE_TTL_SECS, None),
            ],
            || {
                assert_eq!(
                    ResponseStoreSelector::from_env().unwrap(),
                    ResponseStoreSelector::None
                );
            },
        );
        temp_env::with_vars(
            [
                (env_llm::DYN_HTTP_RESPONSES_STORE, Some("memory")),
                (env_llm::DYN_HTTP_RESPONSES_STORE_TTL_SECS, Some("0")),
                (env_llm::DYN_HTTP_RESPONSES_STORE_MAX_RECORDS, Some("7")),
                (env_llm::DYN_HTTP_RESPONSES_STORE_MAX_BYTES, Some("4096")),
            ],
            || {
                assert_eq!(
                    ResponseStoreSelector::from_env().unwrap(),
                    ResponseStoreSelector::Memory(MemoryStoreConfig {
                        ttl: None,
                        max_records: 7,
                        max_bytes: 4096,
                    })
                );
            },
        );
        temp_env::with_vars(
            [(env_llm::DYN_HTTP_RESPONSES_STORE, Some("postgres"))],
            || {
                assert!(ResponseStoreSelector::from_env().is_err());
            },
        );
        temp_env::with_vars(
            [(env_llm::DYN_HTTP_RESPONSES_MAX_CHAIN_DEPTH, Some("0"))],
            || {
                assert!(max_chain_depth_from_env().is_err());
            },
        );
    }
}
