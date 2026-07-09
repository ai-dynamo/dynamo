// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-level tokenizer prefix cache used by the async frontend preprocessor.

use std::{
    mem::size_of_val,
    sync::Arc,
    time::{Duration, Instant},
};

use aho_corasick::AhoCorasick;
use anyhow::{Result, bail};
use async_trait::async_trait;
use dynamo_tokenizers::{Encoding, traits::Tokenizer};
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::valkey_transport::ValkeySentinelConfig;

mod valkey;

#[cfg(test)]
use crate::valkey_transport::{RespValue, parse_endpoint};
use valkey::ValkeyTokenizerCache;
#[cfg(test)]
use valkey::{decode_tokens, encode_tokens, parse_longest_prefix_response};

type PrefixHash = [u8; 32];

const DEFAULT_L1_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_L2_TTL_SECONDS: u64 = 60 * 60;
const DEFAULT_L2_TIMEOUT_MS: u64 = 20;
const DEFAULT_L2_POOL_SIZE: usize = 8;
const DEFAULT_MAX_PENDING_WRITES: usize = 128;
const MAX_PREFIX_BOUNDARIES: usize = 16_384;
const DEFAULT_KEY_PREFIX: &str = "dynamo:tokenizer:v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenizerCacheConfig {
    pub enabled: bool,
    pub l1_bytes: u64,
    pub extend: bool,
    pub l2: Option<TokenizerCacheL2Config>,
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            l1_bytes: DEFAULT_L1_BYTES,
            extend: true,
            l2: None,
        }
    }
}

impl TokenizerCacheConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=u32::MAX as u64).contains(&self.l1_bytes) {
            bail!("tokenizer_cache.l1_bytes must be in 1..={}", u32::MAX);
        }
        if let Some(l2_config) = &self.l2 {
            validate_l2_config(l2_config)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerCacheL2Config {
    pub allow_insecure_plaintext: bool,
    pub url: Option<String>,
    pub sentinel_urls: Option<String>,
    pub sentinel_master_name: Option<String>,
    pub sentinel_quorum: Option<usize>,
    pub scope: String,
    pub key_prefix: String,
    pub ttl_seconds: u64,
    pub timeout_ms: u64,
    pub connection_pool_size: usize,
    pub max_pending_writes: usize,
}

impl Default for TokenizerCacheL2Config {
    fn default() -> Self {
        Self {
            allow_insecure_plaintext: false,
            url: None,
            sentinel_urls: None,
            sentinel_master_name: None,
            sentinel_quorum: None,
            scope: "default".to_string(),
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            ttl_seconds: DEFAULT_L2_TTL_SECONDS,
            timeout_ms: DEFAULT_L2_TIMEOUT_MS,
            connection_pool_size: DEFAULT_L2_POOL_SIZE,
            max_pending_writes: DEFAULT_MAX_PENDING_WRITES,
        }
    }
}

#[derive(Clone)]
struct TokenizerCacheEntry {
    hash: PrefixHash,
    tokens: Arc<[u32]>,
}

struct TokenizerCacheHit {
    candidate_index: usize,
    tokens: Arc<[u32]>,
}

#[derive(Clone, Copy)]
struct PrefixCandidate {
    end: usize,
    hash: PrefixHash,
}

#[async_trait]
trait TokenizerCacheL2Backend: Send + Sync {
    async fn get_longest(
        &self,
        namespace: &str,
        candidates: &[PrefixHash],
    ) -> Result<Option<TokenizerCacheHit>>;

    async fn put(&self, namespace: &str, entries: Vec<TokenizerCacheEntry>) -> Result<()>;
}

fn validate_key_segment(value: &str, name: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'-' | b'_' | b'.'))
    {
        bail!("{name} must contain 1..=128 ASCII letters, digits, ':', '-', '_' or '.'");
    }
    Ok(())
}

fn validate_l2_config(l2_config: &TokenizerCacheL2Config) -> Result<()> {
    if !l2_config.allow_insecure_plaintext {
        bail!(
            "tokenizer cache plaintext Valkey transport requires allow_insecure_plaintext=true and a separate tenant-isolated trusted network"
        );
    }
    validate_key_segment(&l2_config.scope, "tokenizer_cache.scope")?;
    validate_key_segment(&l2_config.key_prefix, "tokenizer_cache.key_prefix")?;
    if !(1..=7 * 24 * 60 * 60).contains(&l2_config.ttl_seconds) {
        bail!("tokenizer_cache.ttl_seconds must be in 1..=604800");
    }
    if !(1..=10_000).contains(&l2_config.timeout_ms) {
        bail!("tokenizer_cache.timeout_ms must be in 1..=10000");
    }
    if !(1..=64).contains(&l2_config.connection_pool_size) {
        bail!("tokenizer_cache.connection_pool_size must be in 1..=64");
    }
    if !(1..=4096).contains(&l2_config.max_pending_writes) {
        bail!("tokenizer_cache.max_pending_writes must be in 1..=4096");
    }
    match (
        l2_config.url.as_deref(),
        l2_config.sentinel_urls.as_deref(),
        l2_config.sentinel_master_name.as_deref(),
    ) {
        (Some(_), None, None) | (None, Some(_), Some(_)) => Ok(()),
        (Some(_), _, _) => {
            bail!("tokenizer_cache.url is mutually exclusive with Sentinel discovery")
        }
        (None, Some(_), None) => {
            bail!("tokenizer_cache.sentinel_urls requires sentinel_master_name")
        }
        (None, None, Some(_)) => {
            bail!("tokenizer_cache.sentinel_master_name requires sentinel_urls")
        }
        (None, None, None) => bail!("tokenizer_cache L2 requires url or Sentinel discovery"),
    }
}

pub(crate) struct SharedTokenizerCache {
    inner: Arc<dyn Tokenizer>,
    matcher: Option<AhoCorasick>,
    l1: Cache<PrefixHash, Arc<[u32]>>,
    namespace: Arc<str>,
    l2: Option<Arc<dyn TokenizerCacheL2Backend>>,
    write_permits: Arc<Semaphore>,
    extend_on_hit: bool,
}

impl SharedTokenizerCache {
    pub(crate) fn from_config(
        inner: Arc<dyn Tokenizer>,
        special_tokens: Vec<String>,
        namespace: impl Into<Arc<str>>,
        config: &TokenizerCacheConfig,
    ) -> Result<Option<Self>> {
        let Some(l2_config) = config.l2.as_ref().filter(|_| config.enabled) else {
            return Ok(None);
        };
        if special_tokens.is_empty() {
            tracing::warn!(
                "tokenizer L2 requested but this tokenizer exposes no safe special-token boundaries; shared prefix caching is disabled"
            );
            return Ok(None);
        }

        let tokenizer_namespace = namespace.into();
        let namespace: Arc<str> = format!("{}:{tokenizer_namespace}", l2_config.scope).into();
        config.validate()?;
        let valkey = match (
            l2_config.url.as_deref(),
            l2_config.sentinel_urls.as_deref(),
            l2_config.sentinel_master_name.as_deref(),
        ) {
            (Some(url), None, None) => ValkeyTokenizerCache::new(
                url,
                &l2_config.key_prefix,
                l2_config.ttl_seconds,
                Duration::from_millis(l2_config.timeout_ms),
                l2_config.connection_pool_size,
            )?,
            (None, Some(urls), Some(master_name)) => {
                let sentinel =
                    ValkeySentinelConfig::new(urls, master_name, l2_config.sentinel_quorum)?;
                ValkeyTokenizerCache::new_with_sentinel(
                    sentinel,
                    &l2_config.key_prefix,
                    l2_config.ttl_seconds,
                    Duration::from_millis(l2_config.timeout_ms),
                    l2_config.connection_pool_size,
                )?
            }
            _ => unreachable!("validated tokenizer L2 endpoint"),
        };
        let l2: Arc<dyn TokenizerCacheL2Backend> = Arc::new(valkey);
        Ok(Some(Self::new_with_limits(
            inner,
            special_tokens,
            namespace,
            Some(l2),
            config.l1_bytes,
            l2_config.max_pending_writes,
            config.extend,
        )?))
    }

    #[cfg(test)]
    fn new(
        inner: Arc<dyn Tokenizer>,
        special_tokens: Vec<String>,
        namespace: impl Into<Arc<str>>,
        l2: Option<Arc<dyn TokenizerCacheL2Backend>>,
    ) -> Result<Self> {
        Self::new_with_limits(
            inner,
            special_tokens,
            namespace,
            l2,
            DEFAULT_L1_BYTES,
            DEFAULT_MAX_PENDING_WRITES,
            true,
        )
    }

    fn new_with_limits(
        inner: Arc<dyn Tokenizer>,
        special_tokens: Vec<String>,
        namespace: impl Into<Arc<str>>,
        l2: Option<Arc<dyn TokenizerCacheL2Backend>>,
        l1_bytes: u64,
        max_pending_writes: usize,
        extend_on_hit: bool,
    ) -> Result<Self> {
        let namespace = namespace.into();
        validate_key_segment(&namespace, "tokenizer cache namespace")?;
        let l1 = Cache::builder()
            .max_capacity(l1_bytes)
            .weigher(|_hash: &PrefixHash, tokens: &Arc<[u32]>| -> u32 {
                size_of_val(tokens.as_ref()).min(u32::MAX as usize) as u32
            })
            .build();
        let matcher = if special_tokens.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(special_tokens)?)
        };
        Ok(Self {
            inner,
            matcher,
            l1,
            namespace,
            l2,
            write_permits: Arc::new(Semaphore::new(max_pending_writes)),
            extend_on_hit,
        })
    }

    pub(crate) async fn encode(&self, input: &str) -> Result<Encoding> {
        let candidates = self.candidates(input);
        if candidates.is_empty() {
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_MISSES_TOTAL.inc();
            return self.encode_plain(input).await;
        }

        let mut l1_hit = None;
        for (candidate_index, candidate) in candidates.iter().enumerate().rev() {
            if let Some(tokens) = self.l1.get(&candidate.hash).await {
                l1_hit = Some((candidate_index, tokens));
                break;
            }
        }
        if l1_hit.is_some() {
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_HITS_TOTAL.inc();
        } else {
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_MISSES_TOTAL.inc();
        }

        let l2_start = l1_hit
            .as_ref()
            .map_or(0, |(candidate_index, _)| candidate_index + 1);
        if l2_start == candidates.len() {
            return self.encode_from_prefix(input, &candidates, l1_hit).await;
        }

        if let Some(l2) = &self.l2 {
            let hashes: Vec<PrefixHash> = candidates[l2_start..]
                .iter()
                .map(|candidate| candidate.hash)
                .collect();
            let lookup_started = Instant::now();
            let lookup = l2.get_longest(&self.namespace, &hashes).await;
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_LOOKUP_SECONDS
                .observe(lookup_started.elapsed().as_secs_f64());
            match lookup {
                Ok(Some(hit)) if hit.candidate_index < hashes.len() => {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_HITS_TOTAL.inc();
                    let candidate_index = l2_start + hit.candidate_index;
                    let hash = candidates[candidate_index].hash;
                    self.l1.insert(hash, Arc::clone(&hit.tokens)).await;
                    return self
                        .encode_from_prefix(input, &candidates, Some((candidate_index, hit.tokens)))
                        .await;
                }
                Ok(Some(_)) => {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_ERRORS_TOTAL.inc();
                    tracing::warn!("tokenizer L2 returned an invalid candidate index")
                }
                Ok(None) => {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_MISSES_TOTAL.inc();
                }
                Err(error) => {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_ERRORS_TOTAL.inc();
                    tracing::debug!(%error, "tokenizer L2 lookup failed; using local tokenizer")
                }
            }
        }

        self.encode_from_prefix(input, &candidates, l1_hit).await
    }

    fn candidates(&self, input: &str) -> Vec<PrefixCandidate> {
        let Some(matcher) = &self.matcher else {
            return Vec::new();
        };
        let mut boundaries: Vec<usize> = matcher
            .find_overlapping_iter(input)
            .map(|matched| matched.end())
            .filter(|end| *end < input.len())
            .collect();
        boundaries.sort_unstable();
        boundaries.dedup();
        if boundaries.len() > MAX_PREFIX_BOUNDARIES {
            tracing::debug!(
                boundaries = boundaries.len(),
                maximum = MAX_PREFIX_BOUNDARIES,
                "bypassing tokenizer prefix cache because the prompt has too many special-token boundaries"
            );
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(boundaries.len());
        let mut hasher = blake3::Hasher::new();
        let mut previous = 0;
        for end in boundaries {
            hasher.update(&input.as_bytes()[previous..end]);
            candidates.push(PrefixCandidate {
                end,
                hash: *hasher.finalize().as_bytes(),
            });
            previous = end;
        }
        candidates
    }

    async fn encode_plain(&self, input: &str) -> Result<Encoding> {
        let input = input.to_owned();
        let tokenizer = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || tokenizer.encode(&input)).await?
    }

    async fn encode_from_prefix(
        &self,
        input: &str,
        candidates: &[PrefixCandidate],
        matched: Option<(usize, Arc<[u32]>)>,
    ) -> Result<Encoding> {
        let input = input.to_owned();
        let candidates = candidates.to_vec();
        let tokenizer = Arc::clone(&self.inner);
        let extend_on_hit = self.extend_on_hit;
        let (tokens, entry) = tokio::task::spawn_blocking(move || {
            let deepest_index = candidates.len() - 1;
            let deepest = candidates[deepest_index];
            let (matched_index, previous, mut tokens) = match matched {
                Some((candidate_index, tokens)) => (
                    Some(candidate_index),
                    candidates[candidate_index].end,
                    tokens.to_vec(),
                ),
                None => (None, 0, Vec::new()),
            };

            if matched_index.is_some_and(|index| index == deepest_index || !extend_on_hit) {
                let suffix = tokenizer.encode(&input[previous..])?;
                tokens.reserve(suffix.token_ids().len());
                tokens.extend_from_slice(suffix.token_ids());
                return Ok::<_, anyhow::Error>((tokens, None));
            }

            let prefix_extension = tokenizer.encode(&input[previous..deepest.end])?;
            let tail = tokenizer.encode(&input[deepest.end..])?;
            tokens.reserve(prefix_extension.token_ids().len() + tail.token_ids().len());
            tokens.extend_from_slice(prefix_extension.token_ids());
            let entry = TokenizerCacheEntry {
                hash: deepest.hash,
                tokens: tokens.as_slice().into(),
            };
            tokens.extend_from_slice(tail.token_ids());
            Ok::<_, anyhow::Error>((tokens, Some(entry)))
        })
        .await??;

        if let Some(entry) = entry {
            self.l1.insert(entry.hash, Arc::clone(&entry.tokens)).await;
            self.schedule_l2_put(entry);
        }
        Ok(Encoding::Sp(tokens))
    }

    fn schedule_l2_put(&self, entry: TokenizerCacheEntry) {
        let Some(l2) = self.l2.as_ref().map(Arc::clone) else {
            return;
        };
        let Ok(permit) = Arc::clone(&self.write_permits).try_acquire_owned() else {
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_WRITE_DROPS_TOTAL.inc();
            tracing::debug!("dropping tokenizer L2 cache write because the queue is full");
            return;
        };
        let namespace = Arc::clone(&self.namespace);
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(error) = l2.put(&namespace, vec![entry]).await {
                dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_L2_WRITE_ERRORS_TOTAL.inc();
                tracing::debug!(%error, "tokenizer L2 cache write failed");
            }
        });
    }
}

#[cfg(test)]
mod tests;
