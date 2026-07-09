// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    mem::{size_of, size_of_val},
    sync::{
        Arc, RwLock as StdRwLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tokio::{
    sync::Mutex,
    time::{Instant as TokioInstant, timeout, timeout_at},
};

use super::{
    PrefixHash, TokenizerCacheEntry, TokenizerCacheHit, TokenizerCacheL2Backend,
    validate_key_segment,
};
use crate::valkey_transport::{
    RespConnection, RespPolicy, RespValue, ValkeySentinelConfig, parse_endpoint,
};

const TOKEN_VALUE_VERSION: u8 = 1;
const MAX_L2_CANDIDATES: usize = 1024;
const MAX_CACHED_TOKENS: usize = 262_144;
const MAX_RESP_LINE_BYTES: usize = 4096;
const MAX_RESP_ARRAY_ITEMS: usize = 2;
const TOKEN_CHECKSUM_BYTES: usize = 32;
const MAX_RESP_BULK_BYTES: usize =
    1 + size_of::<u32>() + MAX_CACHED_TOKENS * size_of::<u32>() + TOKEN_CHECKSUM_BYTES;
const MAX_RESP_BYTES: usize = MAX_RESP_BULK_BYTES + MAX_RESP_LINE_BYTES + 128;
const SENTINEL_FAILURE_BACKOFF: Duration = Duration::from_millis(500);
const SENTINEL_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(2);
const LONGEST_PREFIX_SCRIPT: &[u8] = br#"for i = #KEYS, 1, -1 do
  local value = redis.call('GET', KEYS[i])
  if value then return {i, value} end
end
return nil"#;

pub(super) struct ValkeyTokenizerCache {
    endpoint: StdRwLock<Option<Arc<str>>>,
    sentinel_failure: StdRwLock<Option<CachedSentinelFailure>>,
    sentinel: Option<ValkeySentinelConfig>,
    sentinel_refresh: Mutex<()>,
    key_prefix: Arc<str>,
    ttl_seconds: Arc<[u8]>,
    command_timeout: Duration,
    resp_policy: RespPolicy,
    connections: Vec<Mutex<Option<RespConnection>>>,
    connection_cursor: AtomicUsize,
}

#[derive(Clone)]
struct CachedSentinelFailure {
    observed_at: Instant,
    message: Arc<str>,
}

impl ValkeyTokenizerCache {
    pub(super) fn new(
        url: &str,
        key_prefix: &str,
        ttl_seconds: u64,
        command_timeout: Duration,
        pool_size: usize,
    ) -> Result<Self> {
        let endpoint: Arc<str> = parse_endpoint(url)?.into();
        Self::build(
            Some(endpoint),
            None,
            key_prefix,
            ttl_seconds,
            command_timeout,
            pool_size,
        )
    }

    pub(super) fn new_with_sentinel(
        sentinel: ValkeySentinelConfig,
        key_prefix: &str,
        ttl_seconds: u64,
        command_timeout: Duration,
        pool_size: usize,
    ) -> Result<Self> {
        Self::build(
            None,
            Some(sentinel),
            key_prefix,
            ttl_seconds,
            command_timeout,
            pool_size,
        )
    }

    fn build(
        endpoint: Option<Arc<str>>,
        sentinel: Option<ValkeySentinelConfig>,
        key_prefix: &str,
        ttl_seconds: u64,
        command_timeout: Duration,
        pool_size: usize,
    ) -> Result<Self> {
        validate_key_segment(key_prefix, "DYN_TOKENIZER_CACHE_L2_KEY_PREFIX")?;
        if ttl_seconds == 0 || ttl_seconds > 7 * 24 * 60 * 60 {
            bail!("tokenizer L2 TTL must be in 1..=604800 seconds");
        }
        if command_timeout.is_zero() || command_timeout > Duration::from_secs(10) {
            bail!("tokenizer L2 timeout must be in 1..=10000 milliseconds");
        }
        if !(1..=64).contains(&pool_size) {
            bail!("tokenizer L2 connection pool size must be in 1..=64");
        }
        Ok(Self {
            endpoint: StdRwLock::new(endpoint),
            sentinel_failure: StdRwLock::new(None),
            sentinel,
            sentinel_refresh: Mutex::new(()),
            key_prefix: key_prefix.into(),
            ttl_seconds: ttl_seconds.to_string().into_bytes().into(),
            command_timeout,
            resp_policy: RespPolicy::bounded(
                command_timeout,
                MAX_RESP_LINE_BYTES,
                MAX_RESP_BULK_BYTES,
                MAX_RESP_ARRAY_ITEMS,
                MAX_RESP_BYTES,
            ),
            connections: (0..pool_size).map(|_| Mutex::new(None)).collect(),
            connection_cursor: AtomicUsize::new(0),
        })
    }

    fn current_endpoint(&self) -> Option<Arc<str>> {
        self.endpoint
            .read()
            .expect("tokenizer Valkey endpoint lock poisoned")
            .clone()
    }

    fn invalidate_endpoint(&self, failed_endpoint: &str) {
        if self.sentinel.is_none() {
            return;
        }
        let mut current = self
            .endpoint
            .write()
            .expect("tokenizer Valkey endpoint lock poisoned");
        if current
            .as_deref()
            .is_some_and(|endpoint| endpoint == failed_endpoint)
        {
            *current = None;
            *self
                .sentinel_failure
                .write()
                .expect("tokenizer Valkey Sentinel failure lock poisoned") = None;
        }
    }

    fn cached_sentinel_failure(&self) -> Option<Arc<str>> {
        self.sentinel_failure
            .read()
            .expect("tokenizer Valkey Sentinel failure lock poisoned")
            .as_ref()
            .filter(|failure| failure.observed_at.elapsed() <= SENTINEL_FAILURE_BACKOFF)
            .map(|failure| Arc::clone(&failure.message))
    }

    fn record_sentinel_failure(&self, message: Arc<str>) {
        *self
            .sentinel_failure
            .write()
            .expect("tokenizer Valkey Sentinel failure lock poisoned") =
            Some(CachedSentinelFailure {
                observed_at: Instant::now(),
                message,
            });
    }

    fn clear_sentinel_failure(&self) {
        *self
            .sentinel_failure
            .write()
            .expect("tokenizer Valkey Sentinel failure lock poisoned") = None;
    }

    async fn resolve_endpoint(&self) -> Result<Arc<str>> {
        if let Some(current) = self.current_endpoint() {
            return Ok(current);
        }
        if let Some(message) = self.cached_sentinel_failure() {
            bail!("{message}");
        }

        let sentinel = self
            .sentinel
            .as_ref()
            .context("tokenizer Valkey has neither a fixed endpoint nor Sentinel discovery")?;
        let deadline = TokioInstant::now() + SENTINEL_RESOLUTION_TIMEOUT;
        let _refresh = timeout_at(deadline, self.sentinel_refresh.lock())
            .await
            .context("wait for tokenizer Valkey Sentinel discovery")?;
        if let Some(current) = self.current_endpoint() {
            return Ok(current);
        }
        if let Some(message) = self.cached_sentinel_failure() {
            bail!("{message}");
        }
        let endpoint: Arc<str> =
            match timeout_at(deadline, sentinel.resolve_validated_primary_endpoint()).await {
                Ok(Ok(endpoint)) => endpoint.into(),
                Ok(Err(error)) => {
                    let message: Arc<str> =
                        format!("resolve tokenizer Valkey primary through Sentinel: {error}")
                            .into();
                    self.record_sentinel_failure(Arc::clone(&message));
                    bail!("{message}");
                }
                Err(_) => {
                    let message: Arc<str> = format!(
                        "tokenizer Valkey Sentinel discovery timed out after {} ms",
                        SENTINEL_RESOLUTION_TIMEOUT.as_millis()
                    )
                    .into();
                    self.record_sentinel_failure(Arc::clone(&message));
                    bail!("{message}");
                }
            };
        *self
            .endpoint
            .write()
            .expect("tokenizer Valkey endpoint lock poisoned") = Some(Arc::clone(&endpoint));
        self.clear_sentinel_failure();
        Ok(endpoint)
    }

    async fn command_once(
        &self,
        index: usize,
        endpoint: Arc<str>,
        arguments: &[&[u8]],
    ) -> Result<RespValue> {
        let mut slot = self.connections[index].lock().await;
        let mut connection = match slot.take() {
            Some(connection) if connection.endpoint == endpoint => connection,
            _ => {
                RespConnection::connect_with_policy(endpoint.as_ref(), 0, self.resp_policy).await?
            }
        };
        let result = connection.command(arguments).await;
        if result.is_ok() {
            *slot = Some(connection);
        }
        Ok(result?)
    }

    async fn command_once_with_timeout(
        &self,
        index: usize,
        endpoint: Arc<str>,
        arguments: &[&[u8]],
    ) -> Result<RespValue> {
        match timeout(
            self.command_timeout,
            self.command_once(index, endpoint, arguments),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => bail!(
                "tokenizer Valkey data command timed out after {} ms",
                self.command_timeout.as_millis()
            ),
        }
    }

    async fn command_with_retry(&self, index: usize, arguments: &[&[u8]]) -> Result<RespValue> {
        let endpoint = self.resolve_endpoint().await?;
        match self
            .command_once_with_timeout(index, Arc::clone(&endpoint), arguments)
            .await
        {
            Ok(response) => Ok(response),
            Err(error) if self.sentinel.is_some() => {
                tracing::debug!(%error, failed_endpoint = %endpoint, "refreshing tokenizer Valkey primary through Sentinel");
                self.invalidate_endpoint(&endpoint);
                let refreshed = self.resolve_endpoint().await?;
                let result = self
                    .command_once_with_timeout(index, Arc::clone(&refreshed), arguments)
                    .await;
                if result.is_err() {
                    self.invalidate_endpoint(&refreshed);
                }
                result
            }
            Err(error) => Err(error),
        }
    }

    pub(super) async fn command(&self, arguments: &[&[u8]]) -> Result<RespValue> {
        let index = self.connection_cursor.fetch_add(1, Ordering::Relaxed) % self.connections.len();
        self.command_with_retry(index, arguments).await
    }

    fn key(&self, namespace: &str, hash: &PrefixHash) -> Vec<u8> {
        let mut key = Vec::with_capacity(self.key_prefix.len() + namespace.len() + 67);
        key.extend_from_slice(self.key_prefix.as_bytes());
        key.push(b':');
        key.extend_from_slice(namespace.as_bytes());
        key.push(b':');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in hash {
            key.push(HEX[usize::from(byte >> 4)]);
            key.push(HEX[usize::from(byte & 0x0f)]);
        }
        key
    }
}

#[async_trait]
impl TokenizerCacheL2Backend for ValkeyTokenizerCache {
    async fn get_longest(
        &self,
        namespace: &str,
        candidates: &[PrefixHash],
    ) -> Result<Option<TokenizerCacheHit>> {
        let first = candidates.len().saturating_sub(MAX_L2_CANDIDATES);
        let keys: Vec<Vec<u8>> = candidates[first..]
            .iter()
            .map(|hash| self.key(namespace, hash))
            .collect();
        if keys.is_empty() {
            return Ok(None);
        }
        let key_count = keys.len().to_string();
        let mut arguments: Vec<&[u8]> = Vec::with_capacity(keys.len() + 3);
        arguments.extend([
            b"EVAL".as_slice(),
            LONGEST_PREFIX_SCRIPT,
            key_count.as_bytes(),
        ]);
        arguments.extend(keys.iter().map(Vec::as_slice));
        parse_longest_prefix_response(self.command(&arguments).await?, first, keys.len())
    }

    async fn put(&self, namespace: &str, entries: Vec<TokenizerCacheEntry>) -> Result<()> {
        for entry in entries {
            let value = match encode_tokens(&entry.tokens) {
                Ok(value) => value,
                Err(error) => {
                    tracing::debug!(%error, "skipping oversized tokenizer L2 entry");
                    continue;
                }
            };
            let key = self.key(namespace, &entry.hash);
            let response = self
                .command(&[b"SET", &key, &value, b"EX", &self.ttl_seconds])
                .await?;
            match response {
                RespValue::Simple(value) if value == "OK" => {}
                other => bail!("tokenizer Valkey SET returned {other:?}"),
            }
        }
        Ok(())
    }
}

pub(super) fn parse_longest_prefix_response(
    response: RespValue,
    first_candidate: usize,
    candidate_count: usize,
) -> Result<Option<TokenizerCacheHit>> {
    let RespValue::Array(mut values) = response else {
        if matches!(response, RespValue::Null) {
            return Ok(None);
        }
        bail!("tokenizer Valkey longest-prefix script returned an invalid response");
    };
    if values.len() != 2 {
        bail!("tokenizer Valkey longest-prefix script returned an invalid response");
    }
    let payload = values.pop().expect("two response values");
    let relative_index = values.pop().expect("two response values");
    let RespValue::Integer(relative_index) = relative_index else {
        bail!("tokenizer Valkey longest-prefix script returned an invalid candidate index");
    };
    let relative_index = usize::try_from(relative_index)
        .context("tokenizer Valkey longest-prefix candidate index is negative")?;
    if relative_index == 0 || relative_index > candidate_count {
        bail!("tokenizer Valkey longest-prefix candidate index is out of range");
    }
    let RespValue::Bulk(payload) = payload else {
        bail!("tokenizer Valkey longest-prefix script returned an invalid token value");
    };
    let candidate_index = first_candidate + relative_index - 1;
    match decode_tokens(&payload) {
        Ok(tokens) => Ok(Some(TokenizerCacheHit {
            candidate_index,
            tokens,
        })),
        Err(error) => {
            tracing::warn!(%error, candidate_index, "ignoring malformed tokenizer L2 entry");
            Ok(None)
        }
    }
}

pub(super) fn encode_tokens(tokens: &[u32]) -> Result<Vec<u8>> {
    if tokens.len() > MAX_CACHED_TOKENS {
        bail!(
            "tokenizer cache entry has {} tokens; maximum is {MAX_CACHED_TOKENS}",
            tokens.len()
        );
    }
    let count = u32::try_from(tokens.len()).context("tokenizer token count does not fit u32")?;
    let mut encoded =
        Vec::with_capacity(1 + size_of::<u32>() + size_of_val(tokens) + TOKEN_CHECKSUM_BYTES);
    encoded.push(TOKEN_VALUE_VERSION);
    encoded.extend_from_slice(&count.to_le_bytes());
    for token in tokens {
        encoded.extend_from_slice(&token.to_le_bytes());
    }
    let checksum = blake3::hash(&encoded);
    encoded.extend_from_slice(checksum.as_bytes());
    Ok(encoded)
}

pub(super) fn decode_tokens(encoded: &[u8]) -> Result<Arc<[u32]>> {
    if encoded.len() < 1 + size_of::<u32>() + TOKEN_CHECKSUM_BYTES {
        bail!("tokenizer cache value is too short");
    }
    if encoded[0] != TOKEN_VALUE_VERSION {
        bail!("unsupported tokenizer cache value version {}", encoded[0]);
    }
    let count = u32::from_le_bytes(encoded[1..5].try_into().expect("fixed-width slice")) as usize;
    if count > MAX_CACHED_TOKENS {
        bail!("tokenizer cache value contains too many tokens: {count}");
    }
    let expected = 1 + size_of::<u32>() + count * size_of::<u32>() + TOKEN_CHECKSUM_BYTES;
    if encoded.len() != expected {
        bail!(
            "tokenizer cache value has {} bytes; expected {expected}",
            encoded.len()
        );
    }
    let payload_end = encoded.len() - TOKEN_CHECKSUM_BYTES;
    let expected_checksum = blake3::hash(&encoded[..payload_end]);
    if encoded[payload_end..] != *expected_checksum.as_bytes() {
        bail!("tokenizer cache value checksum mismatch");
    }
    let tokens: Vec<u32> = encoded[5..payload_end]
        .chunks_exact(size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("fixed-width chunk")))
        .collect();
    Ok(tokens.into())
}
