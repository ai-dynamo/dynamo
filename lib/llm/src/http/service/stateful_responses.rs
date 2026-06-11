// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful `/v1/responses` support for the in-process Dynamo endpoint.

use std::collections::HashMap;
use std::env;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "stateful-responses-redb")]
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use dynamo_protocols::types::responses::InputParam;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::protocols::openai::responses::NvCreateResponse;

const DEFAULT_TTL_SECS: u64 = 30 * 24 * 60 * 60;
const DEFAULT_TTL_SWEEP_SECS: u64 = 60 * 60;
#[cfg(feature = "stateful-responses-tikv")]
const DEFAULT_TIKV_PREFIX: &str = "dynamo/stateful-responses/";

const DEFAULT_TTL_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_DEFAULT_TTL_SECS";
const TTL_SWEEP_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_TTL_SWEEP_SECS";
const STORE_URL_ENV: &str = "DYN_STATEFUL_RESPONSES_STORE_URL";

fn is_falsey(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn duration_secs_from_env(
    name: &str,
    default: Option<Duration>,
) -> anyhow::Result<Option<Duration>> {
    let Ok(value) = env::var(name) else {
        return Ok(default);
    };

    parse_optional_duration_secs(name, &value)
}

fn store_settings_from_env() -> anyhow::Result<(Option<Duration>, Option<Duration>)> {
    Ok((
        duration_secs_from_env(
            DEFAULT_TTL_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SECS)),
        )?,
        duration_secs_from_env(
            TTL_SWEEP_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SWEEP_SECS)),
        )?,
    ))
}

fn parse_optional_duration_secs(name: &str, value: &str) -> anyhow::Result<Option<Duration>> {
    let value = value.trim();
    if value.is_empty()
        || is_falsey(value)
        || matches!(value.to_ascii_lowercase().as_str(), "none" | "never")
    {
        return Ok(None);
    }

    let secs = value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))?;
    Ok(Some(Duration::from_secs(secs)))
}

fn expires_at_from_ttl(ttl: Option<Duration>) -> Option<i64> {
    Some(unix_timestamp().saturating_add(i64::try_from(ttl?.as_secs()).unwrap_or(i64::MAX)))
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn is_expired(expires_at: Option<i64>, now: i64) -> bool {
    expires_at.is_some_and(|expires_at| expires_at <= now)
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StoredResponseContext {
    pub input_items: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredResponseRecord {
    context: StoredResponseContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[cfg(any(
    feature = "stateful-responses-redb",
    feature = "stateful-responses-tikv"
))]
fn decode_record(bytes: &[u8]) -> Result<StoredResponseRecord, StoreError> {
    Ok(serde_json::from_slice(bytes)?)
}

#[async_trait]
pub trait ResponseContextStore: Send + Sync + 'static {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError>;
    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError>;
    async fn delete(&self, key: &str) -> Result<bool, StoreError>;
    async fn purge_expired(&self, _now: i64) -> Result<usize, StoreError> {
        Ok(0)
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("failed to serialize Responses context: {0}")]
    Serde(#[from] serde_json::Error),
    #[cfg(feature = "stateful-responses-redb")]
    #[error("redb state store error: {0}")]
    Redb(#[from] redb::Error),
    #[cfg(feature = "stateful-responses-redb")]
    #[error("redb state store task join error: {0}")]
    RedbTask(#[from] tokio::task::JoinError),
    #[cfg(feature = "stateful-responses-tikv")]
    #[error("TiKV state store error: {0}")]
    Tikv(#[from] tikv_client::Error),
}

#[derive(Default)]
pub struct MemoryResponseContextStore {
    contexts: RwLock<HashMap<String, StoredResponseRecord>>,
}

impl MemoryResponseContextStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ResponseContextStore for MemoryResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        let record = self.contexts.read().get(key).cloned();
        let Some(record) = record else {
            return Ok(None);
        };
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        self.contexts.write().insert(
            key.to_string(),
            StoredResponseRecord {
                context: context.clone(),
                expires_at,
            },
        );
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        Ok(self.contexts.write().remove(key).is_some())
    }

    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError> {
        let mut contexts = self.contexts.write();
        let before = contexts.len();
        contexts.retain(|_, record| !is_expired(record.expires_at, now));
        Ok(before - contexts.len())
    }
}

#[cfg(feature = "stateful-responses-redb")]
const REDB_CONTEXTS: redb::TableDefinition<&str, &[u8]> =
    redb::TableDefinition::new("response_contexts");

#[cfg(feature = "stateful-responses-redb")]
pub struct RedbResponseContextStore {
    db: Arc<redb::Database>,
}

#[cfg(feature = "stateful-responses-redb")]
impl RedbResponseContextStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let db = if path.as_ref().exists() {
            redb::Database::open(path).map_err(redb::Error::from)?
        } else {
            redb::Database::create(path).map_err(redb::Error::from)?
        };

        let write_txn = db.begin_write().map_err(redb::Error::from)?;
        {
            write_txn
                .open_table(REDB_CONTEXTS)
                .map_err(redb::Error::from)?;
        }
        write_txn.commit().map_err(redb::Error::from)?;

        Ok(Self { db: Arc::new(db) })
    }

    pub async fn open_async(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || Self::open(path)).await?
    }

    async fn with_db<T>(
        &self,
        f: impl FnOnce(Arc<redb::Database>) -> Result<T, StoreError> + Send + 'static,
    ) -> Result<T, StoreError>
    where
        T: Send + 'static,
    {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || f(db)).await?
    }
}

#[cfg(feature = "stateful-responses-redb")]
#[async_trait]
impl ResponseContextStore for RedbResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        let key = key.to_string();
        let lookup_key = key.clone();
        let record = self
            .with_db(move |db| {
                use redb::ReadableDatabase;

                let read_txn = db.begin_read().map_err(redb::Error::from)?;
                let table = read_txn
                    .open_table(REDB_CONTEXTS)
                    .map_err(redb::Error::from)?;
                table
                    .get(lookup_key.as_str())
                    .map_err(redb::Error::from)?
                    .map(|bytes| decode_record(bytes.value()))
                    .transpose()
            })
            .await?;

        let Some(record) = record else {
            return Ok(None);
        };
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(&key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        let key = key.to_string();
        let value = serde_json::to_vec(&StoredResponseRecord {
            context: context.clone(),
            expires_at,
        })?;
        self.with_db(move |db| {
            let write_txn = db.begin_write().map_err(redb::Error::from)?;
            {
                let mut table = write_txn
                    .open_table(REDB_CONTEXTS)
                    .map_err(redb::Error::from)?;
                table
                    .insert(key.as_str(), value.as_slice())
                    .map_err(redb::Error::from)?;
            }
            write_txn.commit().map_err(redb::Error::from)?;
            Ok(())
        })
        .await
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        let key = key.to_string();
        self.with_db(move |db| {
            let write_txn = db.begin_write().map_err(redb::Error::from)?;
            let removed = {
                let mut table = write_txn
                    .open_table(REDB_CONTEXTS)
                    .map_err(redb::Error::from)?;
                table
                    .remove(key.as_str())
                    .map_err(redb::Error::from)?
                    .is_some()
            };
            write_txn.commit().map_err(redb::Error::from)?;
            Ok(removed)
        })
        .await
    }

    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError> {
        self.with_db(move |db| {
            use redb::ReadableTable;

            let write_txn = db.begin_write().map_err(redb::Error::from)?;
            let expired = {
                let mut table = write_txn
                    .open_table(REDB_CONTEXTS)
                    .map_err(redb::Error::from)?;
                let mut expired = Vec::new();
                for entry in table.iter().map_err(redb::Error::from)? {
                    let (key, value) = entry.map_err(redb::Error::from)?;
                    if is_expired(decode_record(value.value())?.expires_at, now) {
                        expired.push(key.value().to_string());
                    }
                }
                for key in &expired {
                    table.remove(key.as_str()).map_err(redb::Error::from)?;
                }
                expired.len()
            };
            write_txn.commit().map_err(redb::Error::from)?;
            Ok(expired)
        })
        .await
    }
}

#[cfg(feature = "stateful-responses-tikv")]
pub struct TikvResponseContextStore {
    client: tikv_client::RawClient,
    prefix: Vec<u8>,
}

#[cfg(feature = "stateful-responses-tikv")]
impl TikvResponseContextStore {
    pub async fn connect(endpoints: Vec<String>, prefix: String) -> Result<Self, StoreError> {
        let client = tikv_client::RawClient::new(endpoints).await?;
        Ok(Self {
            client,
            prefix: normalize_tikv_prefix(&prefix),
        })
    }

    fn key(&self, key: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.prefix.len() + key.len());
        out.extend_from_slice(&self.prefix);
        out.extend_from_slice(key.as_bytes());
        out
    }
}

#[cfg(feature = "stateful-responses-tikv")]
#[async_trait]
impl ResponseContextStore for TikvResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        let Some(value) = self.client.get(self.key(key)).await? else {
            return Ok(None);
        };
        let record = decode_record(&value)?;
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        let value = serde_json::to_vec(&StoredResponseRecord {
            context: context.clone(),
            expires_at,
        })?;
        let key = self.key(key);
        match ttl_secs_for_put(expires_at) {
            Some(0) => self.client.delete(key).await?,
            Some(ttl) => self.client.put_with_ttl(key, value, ttl).await?,
            None => self.client.put(key, value).await?,
        }
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        let key = self.key(key);
        let existed = self.client.get(key.clone()).await?.is_some();
        self.client.delete(key).await?;
        Ok(existed)
    }
}

#[cfg(feature = "stateful-responses-tikv")]
fn normalize_tikv_prefix(prefix: &str) -> Vec<u8> {
    let mut prefix = prefix.trim_matches('/').as_bytes().to_vec();
    if prefix.is_empty() {
        prefix.extend_from_slice(DEFAULT_TIKV_PREFIX.as_bytes());
    } else {
        prefix.push(b'/');
    }
    prefix
}

#[cfg(feature = "stateful-responses-tikv")]
fn ttl_secs_for_put(expires_at: Option<i64>) -> Option<u64> {
    let expires_at = expires_at?;
    let ttl = expires_at.saturating_sub(unix_timestamp());
    Some(u64::try_from(ttl).unwrap_or(0))
}

#[derive(Clone, Debug, PartialEq)]
pub enum StoreConfig {
    Memory,
    #[cfg(feature = "stateful-responses-redb")]
    Redb(PathBuf),
    #[cfg(feature = "stateful-responses-tikv")]
    Tikv {
        endpoints: Vec<String>,
        prefix: String,
    },
}

impl StoreConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&env::var(STORE_URL_ENV).unwrap_or_else(|_| "memory".to_string()))
    }

    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw = raw.trim();
        if raw.eq_ignore_ascii_case("memory") {
            return Ok(Self::Memory);
        }
        #[cfg(feature = "stateful-responses-redb")]
        if let Some(path) = raw
            .strip_prefix("redb://")
            .or_else(|| raw.strip_prefix("redb:"))
        {
            return Ok(Self::Redb(PathBuf::from(path)));
        }
        #[cfg(feature = "stateful-responses-tikv")]
        if let Some(config) = raw
            .strip_prefix("tikv://")
            .or_else(|| raw.strip_prefix("tikv:"))
        {
            return parse_tikv_store_config(config);
        }
        anyhow::bail!(
            "unsupported store URL `{raw}`; enabled stores: {}",
            enabled_store_url_schemes()
        );
    }

    pub async fn open(&self) -> anyhow::Result<Arc<dyn ResponseContextStore>> {
        match self {
            Self::Memory => Ok(Arc::new(MemoryResponseContextStore::new())),
            #[cfg(feature = "stateful-responses-redb")]
            Self::Redb(path) => Ok(Arc::new(RedbResponseContextStore::open_async(path).await?)),
            #[cfg(feature = "stateful-responses-tikv")]
            Self::Tikv { endpoints, prefix } => Ok(Arc::new(
                TikvResponseContextStore::connect(endpoints.to_vec(), prefix.to_string()).await?,
            )),
        }
    }
}

fn enabled_store_url_schemes() -> String {
    let schemes = [
        "memory",
        #[cfg(feature = "stateful-responses-redb")]
        "redb:/path",
        #[cfg(feature = "stateful-responses-tikv")]
        "tikv://pd1:2379,pd2:2379/prefix",
    ];
    schemes.join(", ")
}

#[cfg(feature = "stateful-responses-tikv")]
fn parse_tikv_store_config(config: &str) -> anyhow::Result<StoreConfig> {
    let (endpoints, prefix) = config
        .split_once('/')
        .unwrap_or((config, DEFAULT_TIKV_PREFIX));
    let endpoints = endpoints
        .split(',')
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if endpoints.is_empty() {
        anyhow::bail!("tikv store URL must include at least one PD endpoint");
    }
    Ok(StoreConfig::Tikv {
        endpoints,
        prefix: prefix.to_string(),
    })
}

#[derive(Clone, Debug)]
pub struct ResponseContextStoreConfig {
    pub store_config: StoreConfig,
    pub default_ttl: Option<Duration>,
    pub ttl_sweep_interval: Option<Duration>,
}

impl ResponseContextStoreConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let store_config = StoreConfig::from_env()?;
        let (default_ttl, ttl_sweep_interval) = store_settings_from_env()?;

        Ok(Self {
            store_config,
            default_ttl,
            ttl_sweep_interval,
        })
    }
}

pub struct ResponseContextStoreManager {
    config: ResponseContextStoreConfig,
    store: OnceCell<Arc<dyn ResponseContextStore>>,
    sweeper_started: AtomicBool,
    shutdown: CancellationToken,
}

impl ResponseContextStoreManager {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_env_with_shutdown(CancellationToken::new())
    }

    pub fn from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<Self> {
        Ok(Self::with_shutdown(
            ResponseContextStoreConfig::from_env()?,
            shutdown,
        ))
    }

    pub fn new(config: ResponseContextStoreConfig) -> Self {
        Self::with_shutdown(config, CancellationToken::new())
    }

    pub fn with_shutdown(config: ResponseContextStoreConfig, shutdown: CancellationToken) -> Self {
        Self {
            config,
            store: OnceCell::new(),
            sweeper_started: AtomicBool::new(false),
            shutdown,
        }
    }

    pub async fn store(&self) -> anyhow::Result<Arc<dyn ResponseContextStore>> {
        let store = self
            .store
            .get_or_try_init(|| async { self.config.store_config.open().await })
            .await?
            .clone();

        if self.config.default_ttl.is_some()
            && let Some(interval) = self.config.ttl_sweep_interval
            && self
                .sweeper_started
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            spawn_store_ttl_sweeper(store.clone(), interval, self.shutdown.child_token());
        }

        Ok(store)
    }

    pub async fn warmup(&self) -> anyhow::Result<()> {
        self.store().await.map(|_| ())
    }

    pub async fn prepare_request(
        &self,
        request: &mut NvCreateResponse,
    ) -> anyhow::Result<ExpandedResponseContext> {
        let previous_context =
            match request.inner.previous_response_id.as_deref() {
                Some(previous_response_id) => {
                    let store = self.store().await?;
                    Some(store.get(previous_response_id).await?.ok_or_else(|| {
                        PreviousResponseNotFound(previous_response_id.to_string())
                    })?)
                }
                None => None,
            };

        expand_typed_request(request, previous_context.as_ref())
    }

    pub async fn persist_response<T: Serialize>(
        &self,
        expanded: &ExpandedResponseContext,
        response_id: &str,
        output_items: &[T],
    ) -> anyhow::Result<()> {
        if !expanded.should_store {
            return Ok(());
        }

        let mut input_items = expanded.input_items.clone();
        for item in output_items {
            input_items.push(serde_json::to_value(item)?);
        }
        self.store()
            .await?
            .put(
                response_id,
                &StoredResponseContext { input_items },
                expires_at_from_ttl(self.config.default_ttl),
            )
            .await?;
        Ok(())
    }

    pub async fn delete(&self, response_id: &str) -> anyhow::Result<bool> {
        Ok(self.store().await?.delete(response_id).await?)
    }
}

impl Drop for ResponseContextStoreManager {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedResponseContext {
    pub input_items: Vec<Value>,
    pub should_store: bool,
}

#[derive(Debug, Error)]
#[error("`previous_response_id` `{0}` was not found")]
pub struct PreviousResponseNotFound(pub String);

fn spawn_store_ttl_sweeper(
    store: Arc<dyn ResponseContextStore>,
    interval: Duration,
    shutdown: CancellationToken,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = tick.tick() => {
                    if let Err(err) = store.purge_expired(unix_timestamp()).await {
                        tracing::warn!(%err, "failed to purge expired stateful Responses contexts");
                    }
                }
            }
        }
    });
}

fn expand_typed_request(
    request: &mut NvCreateResponse,
    previous_context: Option<&StoredResponseContext>,
) -> anyhow::Result<ExpandedResponseContext> {
    let current_items = typed_input_as_items(&request.inner.input)?;
    let mut input_items = Vec::new();
    if let Some(context) = previous_context {
        input_items.extend(context.input_items.iter().cloned());
    }
    input_items.extend(current_items);

    request.inner.input = serde_json::from_value(Value::Array(input_items.clone()))?;

    let should_store = request.inner.store.unwrap_or(true);
    request.inner.store = Some(should_store);

    Ok(ExpandedResponseContext {
        input_items,
        should_store,
    })
}

fn typed_input_as_items(input: &InputParam) -> Result<Vec<Value>, serde_json::Error> {
    match input {
        InputParam::Text(text) => Ok(vec![message_item("user", "input_text", text)]),
        InputParam::Items(items) => items
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>(),
    }
}

fn message_item(role: &str, content_type: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [{ "type": content_type, "text": text }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::extensions::NvExt;
    use dynamo_protocols::types::responses::{
        AssistantRole, FunctionToolCall, InputItem, Item, MessageItem, OutputItem, OutputMessage,
        OutputMessageContent, OutputStatus, OutputTextContent,
    };

    fn manager(store_config: StoreConfig) -> ResponseContextStoreManager {
        ResponseContextStoreManager::new(ResponseContextStoreConfig {
            store_config,
            default_ttl: None,
            ttl_sweep_interval: None,
        })
    }

    fn memory_manager() -> ResponseContextStoreManager {
        manager(StoreConfig::Memory)
    }

    fn context(input_items: Vec<Value>) -> StoredResponseContext {
        StoredResponseContext { input_items }
    }

    fn text_request(text: &str) -> NvCreateResponse {
        NvCreateResponse {
            extensions: Default::default(),
            inner: dynamo_protocols::types::responses::CreateResponse {
                input: InputParam::Text(text.into()),
                model: Some("m".into()),
                ..Default::default()
            },
            nvext: None,
        }
    }

    #[tokio::test]
    async fn memory_store_round_trips_expires_and_deletes_context() {
        assert_eq!(StoreConfig::parse("memory").unwrap(), StoreConfig::Memory);
        assert_eq!(
            parse_optional_duration_secs("ttl", "60").unwrap(),
            Some(Duration::from_secs(60))
        );
        assert_eq!(parse_optional_duration_secs("ttl", "0").unwrap(), None);
        assert_eq!(parse_optional_duration_secs("ttl", "never").unwrap(), None);

        let store = MemoryResponseContextStore::new();
        let context = context(vec![json!({"type": "message"})]);

        store.put("resp_1", &context, None).await.unwrap();
        assert_eq!(store.get("resp_1").await.unwrap(), Some(context.clone()));

        store
            .put("resp_1", &context, Some(unix_timestamp() - 1))
            .await
            .unwrap();

        assert_eq!(store.get("resp_1").await.unwrap(), None);

        store.put("resp_2", &context, None).await.unwrap();
        assert!(store.delete("resp_2").await.unwrap());
        assert_eq!(store.get("resp_2").await.unwrap(), None);
    }

    #[tokio::test]
    async fn manager_expands_typed_responses_request() {
        let manager = memory_manager();
        let mut request = text_request("hello");
        request.nvext = Some(
            NvExt::builder()
                .token_data(vec![1, 2, 3])
                .extra_fields(vec!["worker_id".to_string()])
                .build()
                .unwrap(),
        );

        let expanded = manager.prepare_request(&mut request).await.unwrap();

        assert!(expanded.should_store);
        assert_eq!(expanded.input_items.len(), 1);
        assert_eq!(expanded.input_items[0]["role"], "user");
        assert_eq!(request.inner.store, Some(true));
        let nvext = request.nvext.as_ref().unwrap();
        assert_eq!(nvext.token_data.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(
            nvext.extra_fields.as_ref().unwrap(),
            &vec!["worker_id".to_string()]
        );
        assert!(manager.store.get().is_none());
    }

    #[test]
    fn stored_output_items_round_trip_as_followup_input_items() {
        let output_items = vec![
            OutputItem::Message(OutputMessage {
                id: "msg_1".into(),
                role: AssistantRole::Assistant,
                status: OutputStatus::Completed,
                phase: None,
                content: vec![OutputMessageContent::OutputText(OutputTextContent {
                    text: "hello".into(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                })],
            }),
            OutputItem::FunctionCall(FunctionToolCall {
                arguments: "{}".into(),
                call_id: "call_1".into(),
                namespace: None,
                name: "tool".into(),
                id: Some("fc_1".into()),
                status: Some(OutputStatus::Completed),
            }),
        ];
        let stored = output_items
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let input: InputParam = serde_json::from_value(Value::Array(stored)).unwrap();
        let InputParam::Items(items) = input else {
            panic!("expected item input");
        };
        assert!(matches!(
            &items[0],
            InputItem::Item(Item::Message(MessageItem::Output(_)))
        ));
        assert!(matches!(&items[1], InputItem::Item(Item::FunctionCall(_))));
    }

    #[tokio::test]
    async fn manager_store_false_skips_persist_and_expired_previous_is_not_found() {
        let manager = memory_manager();
        let store = manager.store().await.unwrap();
        store
            .put(
                "resp_1",
                &context(vec![message_item("assistant", "output_text", "old")]),
                None,
            )
            .await
            .unwrap();

        let mut request = text_request("new");
        request.inner.previous_response_id = Some("resp_1".into());
        request.inner.store = Some(false);

        let expanded = manager.prepare_request(&mut request).await.unwrap();
        manager
            .persist_response(&expanded, "resp_2", &[] as &[Value])
            .await
            .unwrap();

        assert!(!expanded.should_store);
        assert_eq!(expanded.input_items.len(), 2);
        assert_eq!(request.inner.store, Some(false));
        assert_eq!(store.get("resp_2").await.unwrap(), None);
        store
            .put(
                "resp_expired",
                &context(vec![json!({"type": "message"})]),
                Some(unix_timestamp() - 1),
            )
            .await
            .unwrap();

        let mut request = text_request("again");
        request.inner.previous_response_id = Some("resp_expired".into());
        let err = manager.prepare_request(&mut request).await.unwrap_err();

        assert!(err.downcast_ref::<PreviousResponseNotFound>().is_some());
        assert_eq!(store.get("resp_expired").await.unwrap(), None);
    }

    struct FailingStore;

    #[async_trait]
    impl ResponseContextStore for FailingStore {
        async fn get(&self, _key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
            Ok(None)
        }

        async fn put(
            &self,
            _key: &str,
            _context: &StoredResponseContext,
            _expires_at: Option<i64>,
        ) -> Result<(), StoreError> {
            Err(serde_json::from_str::<Value>("").unwrap_err().into())
        }

        async fn delete(&self, _key: &str) -> Result<bool, StoreError> {
            Ok(false)
        }
    }

    #[tokio::test]
    async fn manager_handles_store_failures_based_on_store_flag() {
        let manager = memory_manager();
        assert!(manager.store.set(Arc::new(FailingStore)).is_ok());

        assert!(
            manager
                .persist_response(
                    &ExpandedResponseContext {
                        input_items: Vec::new(),
                        should_store: true,
                    },
                    "resp_fail",
                    &[] as &[Value],
                )
                .await
                .is_err()
        );
        manager
            .persist_response(
                &ExpandedResponseContext {
                    input_items: Vec::new(),
                    should_store: false,
                },
                "resp_skip",
                &[] as &[Value],
            )
            .await
            .unwrap();
    }

    #[cfg(feature = "stateful-responses-redb")]
    #[tokio::test]
    async fn redb_store_round_trips_and_purges_expired_context() {
        assert_eq!(
            StoreConfig::parse("redb:/tmp/state.redb").unwrap(),
            StoreConfig::Redb(PathBuf::from("/tmp/state.redb"))
        );
        let dir = tempfile::tempdir().unwrap();
        let store = RedbResponseContextStore::open(dir.path().join("state.redb")).unwrap();
        let context = context(vec![json!({"type": "message"})]);

        store.put("resp_1", &context, None).await.unwrap();

        assert_eq!(store.get("resp_1").await.unwrap(), Some(context.clone()));

        store
            .put("expired", &context, Some(unix_timestamp() - 1))
            .await
            .unwrap();
        store.put("fresh", &context, None).await.unwrap();

        assert_eq!(store.purge_expired(unix_timestamp()).await.unwrap(), 1);
        assert_eq!(store.get("expired").await.unwrap(), None);
        assert_eq!(store.get("fresh").await.unwrap(), Some(context));
    }

    #[cfg(feature = "stateful-responses-redb")]
    #[tokio::test]
    async fn manager_expands_previous_response_from_redb() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(StoreConfig::Redb(dir.path().join("state.redb")));
        manager
            .store()
            .await
            .unwrap()
            .put(
                "resp_1",
                &context(vec![
                    message_item("user", "input_text", "hello"),
                    message_item("assistant", "output_text", "hi"),
                ]),
                None,
            )
            .await
            .unwrap();

        let mut second = text_request("again");
        second.inner.previous_response_id = Some("resp_1".into());

        let second_expanded = manager.prepare_request(&mut second).await.unwrap();

        assert_eq!(second_expanded.input_items.len(), 3);
        assert_eq!(
            second_expanded.input_items[0]["content"][0]["text"],
            "hello"
        );
        assert_eq!(second_expanded.input_items[1]["role"], "assistant");
        assert_eq!(
            second_expanded.input_items[2]["content"][0]["text"],
            "again"
        );
        assert!(matches!(second.inner.input, InputParam::Items(_)));
        assert_eq!(second.inner.previous_response_id.as_deref(), Some("resp_1"));
    }

    #[cfg(feature = "stateful-responses-tikv")]
    #[test]
    fn parses_tikv_store_url_and_normalizes_prefix() {
        assert_eq!(
            StoreConfig::parse("tikv://127.0.0.1:2379,127.0.0.2:2379/prod/responses").unwrap(),
            StoreConfig::Tikv {
                endpoints: vec!["127.0.0.1:2379".to_string(), "127.0.0.2:2379".to_string()],
                prefix: "prod/responses".to_string(),
            }
        );
        assert_eq!(
            normalize_tikv_prefix("prod/responses"),
            b"prod/responses/".to_vec()
        );
    }
}
