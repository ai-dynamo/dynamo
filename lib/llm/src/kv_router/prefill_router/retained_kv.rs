// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local metadata for one-shot reads of retained decoder KV.

use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{Map, Value};

pub(super) const PRIVATE_KEY: &str = "dynamo_retained_kv";
pub(super) const VERSION: u64 = 1;
pub(super) const MAX_TOKENS: usize = 1_048_576;
const MAX_ID_BYTES: usize = 1024;
const MAX_SCOPE_BYTES: usize = 4096;
const MAX_GROUPS: usize = 128;
const MAX_PARALLEL_SIZE: u64 = 1024;

pub(super) struct Config {
    trust_domain: String,
    max_entries: usize,
    max_bytes: usize,
    max_tokens: usize,
    ttl: Duration,
}

impl Config {
    pub(super) fn from_env() -> Option<Self> {
        let config = Self::new(std::env::var("DYN_VLLM_RETAINED_KV_REUSE_TRUST_DOMAIN").ok()?)?;
        match std::env::var("DYN_VLLM_RETAINED_KV_REUSE_MAX_AGE_SECS") {
            Ok(value) => config.with_max_age(&value).or_else(|| {
                tracing::warn!(
                    "Disabling retained KV reuse: MAX_AGE_SECS must be an integer from 1 to 600"
                );
                None
            }),
            Err(std::env::VarError::NotPresent) => Some(config),
            Err(_) => None,
        }
    }

    fn with_max_age(mut self, value: &str) -> Option<Self> {
        let seconds = value.parse::<u64>().ok()?;
        if !(1..=600).contains(&seconds) {
            return None;
        }
        self.ttl = Duration::from_secs(seconds);
        Some(self)
    }

    /// An explicit operator declaration; this value is never read from a request.
    pub(super) fn new(trust_domain: String) -> Option<Self> {
        let trust_domain = trust_domain.trim();
        valid_text(trust_domain, 256).then(|| Self {
            trust_domain: trust_domain.to_owned(),
            max_entries: 128,
            max_bytes: 64 * 1024 * 1024,
            max_tokens: MAX_TOKENS,
            ttl: Duration::from_secs(20),
        })
    }
}

/// `pipeline` is the selected deployment/configuration identity, including its generation.
/// A client salt only subdivides the operator's trust domain; it does not authenticate a tenant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Scope {
    pub(super) pipeline: String,
    pub(super) cache_namespace: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub(super) worker_id: u64,
    pub(super) generation: String,
}

pub(super) struct Donor {
    pub(super) source: Source,
    pub(super) kv_transfer_params: Value,
}

struct Entry {
    scope: Scope,
    donor: Donor,
    peer: blake3::Hash,
    prefix: Vec<u32>,
    prefix_hash: blake3::Hash,
    deadline: Instant,
    bytes: usize,
}

#[derive(Default)]
struct State {
    entries: VecDeque<Entry>,
    bytes: usize,
}

impl State {
    fn retain(&mut self, mut keep: impl FnMut(&Entry) -> bool) {
        let mut bytes = 0;
        self.entries.retain(|entry| {
            let retained = keep(entry);
            if retained {
                bytes += entry.bytes;
            }
            retained
        });
        self.bytes = bytes;
    }
}

pub(super) struct Catalog {
    config: Config,
    state: Mutex<State>,
}

impl Catalog {
    pub(super) fn new(config: Config) -> Self {
        Self {
            config,
            state: Mutex::new(State::default()),
        }
    }

    pub(super) fn trust_domain(&self) -> &str {
        &self.config.trust_domain
    }

    /// Call only for a successful terminal, single-choice internal decoder response.
    pub(super) fn insert(
        &self,
        scope: Scope,
        source: Source,
        authoritative_prompt: &[u32],
        raw_generated_ids: &[u32],
        private_payload: &Value,
        now: Instant,
    ) -> bool {
        let Some(entry) = self.entry(
            scope,
            source,
            authoritative_prompt,
            raw_generated_ids,
            private_payload,
            now,
        ) else {
            return false;
        };

        let mut state = self.state.lock();
        state.retain(|stored| stored.deadline > now);
        // A worker's changed native peer or pipeline incarnation retires all old leases.
        state.retain(|stored| {
            stored.donor.source.worker_id != entry.donor.source.worker_id
                || (stored.donor.source == entry.donor.source && stored.peer == entry.peer)
        });
        if state.entries.iter().any(|stored| {
            stored.donor.source == entry.donor.source
                && stored.donor.kv_transfer_params["remote_request_id"]
                    == entry.donor.kv_transfer_params["remote_request_id"]
        }) {
            return false;
        }
        while state.entries.len() >= self.config.max_entries
            || state.bytes > self.config.max_bytes - entry.bytes
        {
            let Some(evicted) = state.entries.pop_front() else {
                return false;
            };
            state.bytes -= evicted.bytes;
        }
        state.bytes += entry.bytes;
        state.entries.push_back(entry);
        true
    }

    /// Removes the donor before returning; the caller rechecks admitted membership before dispatch.
    pub(super) fn claim(
        &self,
        scope: &Scope,
        incoming_tokens: &[u32],
        now: Instant,
    ) -> Option<Donor> {
        let mut state = self.state.lock();
        state.retain(|entry| entry.deadline > now);
        let mut lengths: Vec<_> = state
            .entries
            .iter()
            .filter(|entry| entry.scope == *scope && entry.prefix.len() <= incoming_tokens.len())
            .map(|entry| entry.prefix.len())
            .collect();
        lengths.sort_unstable();
        lengths.dedup();
        let max_len = *lengths.last()?;
        let encoded = encode_tokens(&incoming_tokens[..max_len]);
        let mut hasher = self.hasher();
        let mut offset = 0;
        let hashes: HashMap<_, _> = lengths
            .into_iter()
            .map(|len| {
                let end = len * size_of::<u32>();
                hasher.update(&encoded[offset..end]);
                offset = end;
                (len, hasher.finalize())
            })
            .collect();

        let index = state
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.scope == *scope
                    && hashes.get(&entry.prefix.len()) == Some(&entry.prefix_hash)
                    && incoming_tokens.get(..entry.prefix.len()) == Some(entry.prefix.as_slice())
            })
            .max_by_key(|(_, entry)| entry.prefix.len())
            .map(|(index, _)| index)?;
        let entry = state.entries.remove(index)?;
        state.bytes -= entry.bytes;
        Some(entry.donor)
    }

    pub(super) fn clear(&self) {
        *self.state.lock() = State::default();
    }

    fn hasher(&self) -> blake3::Hasher {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.config.trust_domain.as_bytes());
        hasher
    }

    fn entry(
        &self,
        scope: Scope,
        source: Source,
        authoritative_prompt: &[u32],
        raw_generated_ids: &[u32],
        private_payload: &Value,
        now: Instant,
    ) -> Option<Entry> {
        if self.config.max_entries == 0
            || !valid_text(&scope.pipeline, MAX_SCOPE_BYTES)
            || !valid_text(&source.generation, MAX_SCOPE_BYTES)
            || scope
                .cache_namespace
                .as_ref()
                .is_some_and(|salt| salt.len() > MAX_SCOPE_BYTES)
            || authoritative_prompt.is_empty()
            || authoritative_prompt.len() > self.config.max_tokens
        {
            return None;
        }
        let payload_bytes = value_bytes(private_payload, self.config.max_bytes, 0)?;
        let payload = private_payload.as_object()?;
        if payload.get("version")?.as_u64()? != VERSION {
            return None;
        }
        let actual_prompt = payload.get("prompt_token_ids")?.as_array()?;
        if actual_prompt.len() != authoritative_prompt.len()
            || !actual_prompt
                .iter()
                .zip(authoritative_prompt)
                .all(|(actual, expected)| actual.as_u64() == Some(u64::from(*expected)))
        {
            return None;
        }
        let params = payload.get("kv_transfer_params")?.as_object()?;
        let (extent, peer) = validate_params(params, self.config.max_tokens)?;
        let generated_extent = extent.checked_sub(authoritative_prompt.len())?;
        if generated_extent > raw_generated_ids.len() {
            return None;
        }
        let mut ttl = self.config.ttl;
        if let Some(hint) = payload.get("ttl_seconds") {
            let hint = hint.as_f64()?;
            if !hint.is_finite() || hint <= 0.0 {
                return None;
            }
            ttl = Duration::try_from_secs_f64(hint.min(ttl.as_secs_f64())).ok()?;
        }
        if ttl.is_zero() {
            return None;
        }
        let bytes = payload_bytes
            .checked_add(extent.checked_mul(size_of::<u32>())?)?
            .checked_add(scope.pipeline.len())?
            .checked_add(scope.cache_namespace.as_ref().map_or(0, String::len))?
            .checked_add(source.generation.len())?
            .checked_add(size_of::<Entry>())?;
        if bytes > self.config.max_bytes {
            return None;
        }
        let mut prefix = Vec::with_capacity(extent);
        prefix.extend_from_slice(authoritative_prompt);
        prefix.extend_from_slice(&raw_generated_ids[..generated_extent]);
        let mut hasher = self.hasher();
        hasher.update(&encode_tokens(&prefix));
        Some(Entry {
            scope,
            donor: Donor {
                source,
                kv_transfer_params: Value::Object(params.clone()),
            },
            peer,
            prefix,
            prefix_hash: hasher.finalize(),
            deadline: now.checked_add(ttl)?,
            bytes,
        })
    }
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn encode_tokens(tokens: &[u32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect()
}

fn validate_params(
    params: &Map<String, Value>,
    max_tokens: usize,
) -> Option<(usize, blake3::Hash)> {
    if params.contains_key("_remote_blocks_processed")
        || params.contains_key("_p_side_truncated")
        || params.get("do_remote_decode")?.as_bool() != Some(true)
        || params.get("do_remote_prefill")?.as_bool() != Some(false)
        || params
            .get("transfer_mode")
            .is_some_and(|mode| mode.as_str() != Some("pull"))
    {
        return None;
    }
    let extent = usize::try_from(params.get("remote_num_tokens")?.as_u64()?).ok()?;
    if extent == 0 || extent > max_tokens {
        return None;
    }
    let groups = params.get("remote_block_ids")?.as_array()?;
    if groups.is_empty() || groups.len() > MAX_GROUPS {
        return None;
    }
    let mut blocks = 0usize;
    for group in groups {
        let group = group.as_array()?;
        blocks = blocks.checked_add(group.len())?;
        if blocks > MAX_TOKENS || group.iter().any(|id| id.as_u64().is_none()) {
            return None;
        }
    }
    if blocks == 0 {
        return None;
    }
    let mut peer = blake3::Hasher::new();
    for field in ["remote_engine_id", "remote_host", "remote_request_id"] {
        let value = params.get(field)?.as_str()?;
        if !valid_text(value, MAX_ID_BYTES) {
            return None;
        }
        if field == "remote_host"
            && (value.chars().any(char::is_whitespace) || value.contains(['/', '?', '#', '@']))
        {
            return None;
        }
        if field != "remote_request_id" {
            peer.update(&(value.len() as u64).to_le_bytes());
            peer.update(value.as_bytes());
        }
    }
    let port = params.get("remote_port")?.as_u64()?;
    if !(1..=u64::from(u16::MAX)).contains(&port) {
        return None;
    }
    peer.update(&port.to_le_bytes());
    for field in ["tp_size", "dcp_size", "pp_size"] {
        let size = params.get(field).map_or(Some(1), Value::as_u64)?;
        if !(1..=MAX_PARALLEL_SIZE).contains(&size) {
            return None;
        }
        peer.update(&size.to_le_bytes());
    }
    if let Some(generation) = params.get("remote_registration_generation") {
        peer.update(&generation.as_u64()?.to_le_bytes());
    }
    // This is a source monotonic clock value; only the native engine interprets it.
    let expiry = params.get("remote_blocks_expiry_time")?.as_f64()?;
    if !expiry.is_finite() || expiry <= 0.0 {
        return None;
    }
    Some((extent, peer.finalize()))
}

/// Conservative logical ownership cost, with bounded traversal before cloning opaque values.
fn value_bytes(value: &Value, limit: usize, depth: usize) -> Option<usize> {
    if depth > 16 {
        return None;
    }
    let mut bytes = size_of::<Value>();
    match value {
        Value::String(value) => bytes = bytes.checked_add(value.len())?,
        Value::Array(values) => {
            for value in values {
                bytes = bytes.checked_add(value_bytes(value, limit, depth + 1)?)?;
                if bytes > limit {
                    return None;
                }
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                bytes = bytes
                    .checked_add(size_of::<String>())?
                    .checked_add(key.len())?
                    .checked_add(value_bytes(value, limit, depth + 1)?)?;
                if bytes > limit {
                    return None;
                }
            }
        }
        _ => {}
    }
    (bytes <= limit).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use serde_json::json;

    use super::*;

    fn scope() -> Scope {
        Scope {
            pipeline: "namespace/model/mdc/generation-1".into(),
            cache_namespace: Some("tenant-cache-partition".into()),
        }
    }

    fn source() -> Source {
        Source {
            worker_id: 7,
            generation: "decode/prefill-generation-1".into(),
        }
    }

    fn payload() -> Value {
        json!({
            "version": VERSION,
            "prompt_token_ids": [1, 2],
            "kv_transfer_params": {
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": "engine-a",
                "remote_request_id": "request-a",
                "remote_host": "10.0.0.7",
                "remote_port": 5600,
                "remote_block_ids": [[4, 5], []],
                "remote_num_tokens": 3,
                "remote_blocks_expiry_time": 12345.0,
                "tp_size": 4,
                "dcp_size": 1,
                "pp_size": 1,
                "transfer_mode": "pull"
            }
        })
    }

    fn catalog() -> Catalog {
        Catalog::new(Config::new("operator-trust-domain".into()).unwrap())
    }

    fn insert(catalog: &Catalog, payload: &Value, now: Instant) -> bool {
        catalog.insert(scope(), source(), &[1, 2], &[3, 4], payload, now)
    }

    #[test]
    fn trust_domain_requires_an_explicit_bounded_value() {
        for value in ["", " \t ", "domain\nother"] {
            assert!(Config::new(value.into()).is_none());
        }
        assert!(Config::new("x".repeat(257)).is_none());
        assert!(Config::new("domain".into()).is_some());
    }

    #[test]
    fn configured_max_age_is_bounded_and_never_extends_the_worker_ttl() {
        for invalid in ["", "0", "601", "-1", "1.5", " 20", "18446744073709551616"] {
            assert!(
                Config::new("domain".into())
                    .unwrap()
                    .with_max_age(invalid)
                    .is_none()
            );
        }
        let now = Instant::now();
        for (configured, worker_hint, expected) in [("600", 5, 5), ("1", 600, 1)] {
            let config = Config::new("domain".into())
                .unwrap()
                .with_max_age(configured)
                .unwrap();
            let catalog = Catalog::new(config);
            let mut value = payload();
            value["ttl_seconds"] = json!(worker_hint);
            assert!(insert(&catalog, &value, now));
            assert!(
                catalog
                    .claim(&scope(), &[1, 2, 3], now + Duration::from_secs(expected))
                    .is_none()
            );
        }
    }

    #[test]
    fn claims_the_entire_computed_prefix_once() {
        let catalog = catalog();
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        assert!(catalog.claim(&scope(), &[1, 2], now).is_none());
        assert!(catalog.claim(&scope(), &[1, 2, 4, 8], now).is_none());
        let donor = catalog.claim(&scope(), &[1, 2, 3, 8], now).unwrap();
        assert_eq!(donor.source, source());
        assert_eq!(donor.kv_transfer_params["remote_num_tokens"], 3);
        assert_eq!(
            donor.kv_transfer_params["remote_block_ids"],
            json!([[4, 5], []])
        );
        assert!(catalog.claim(&scope(), &[1, 2, 3, 8], now).is_none());
        assert_eq!(catalog.state.lock().bytes, 0);
    }

    #[test]
    fn model_namespace_generation_and_cache_partitions_must_match() {
        let catalog = catalog();
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        for pipeline in [
            "namespace/alias/mdc/generation-1",
            "other/model/mdc/generation-1",
            "namespace/model/new-mdc/generation-1",
            "namespace/model/mdc/generation-2",
        ] {
            let different = Scope {
                pipeline: pipeline.into(),
                ..scope()
            };
            assert!(catalog.claim(&different, &[1, 2, 3], now).is_none());
        }
        for cache_namespace in [None, Some("other-tenant".into()), Some(String::new())] {
            let different = Scope {
                cache_namespace,
                ..scope()
            };
            assert!(catalog.claim(&different, &[1, 2, 3], now).is_none());
        }
        assert!(catalog.claim(&scope(), &[1, 2, 3], now).is_some());
    }

    #[test]
    fn digest_matches_do_not_replace_exact_token_comparison() {
        let catalog = catalog();
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        let mut hasher = catalog.hasher();
        hasher.update(&encode_tokens(&[1, 2, 9]));
        catalog.state.lock().entries[0].prefix_hash = hasher.finalize();
        assert!(catalog.claim(&scope(), &[1, 2, 9], now).is_none());
    }

    #[test]
    fn missing_or_conflicting_engine_prompt_provenance_is_rejected() {
        let catalog = catalog();
        let now = Instant::now();
        for prompt in [Value::Null, json!([1]), json!([1, 9]), json!([1, true])] {
            let mut invalid = payload();
            invalid["prompt_token_ids"] = prompt;
            assert!(!insert(&catalog, &invalid, now));
        }
        let mut missing = payload();
        missing.as_object_mut().unwrap().remove("prompt_token_ids");
        assert!(!insert(&catalog, &missing, now));
        assert!(catalog.state.lock().entries.is_empty());
    }

    #[test]
    fn rejects_incomplete_or_invalid_computed_extents() {
        let catalog = catalog();
        let now = Instant::now();
        for extent in [
            json!(0),
            json!(1),
            json!(5),
            json!(-1),
            json!(true),
            json!(3.5),
        ] {
            let mut invalid = payload();
            invalid["kv_transfer_params"]["remote_num_tokens"] = extent;
            assert!(!insert(&catalog, &invalid, now));
        }
        assert!(!catalog.insert(scope(), source(), &[1, 2], &[], &payload(), now));
        let mut valid = payload();
        valid["kv_transfer_params"]["remote_num_tokens"] = json!(2);
        assert!(insert(&catalog, &valid, now));
        assert!(catalog.claim(&scope(), &[1, 2, 99], now).is_some());
    }

    #[test]
    fn rejects_unknown_schema_and_malformed_native_coordinates() {
        let catalog = catalog();
        let now = Instant::now();
        for version in [Value::Null, json!(0), json!(2), json!(true), json!("1")] {
            let mut invalid = payload();
            invalid["version"] = version;
            assert!(!insert(&catalog, &invalid, now));
        }
        for (field, value) in [
            ("remote_engine_id", json!("")),
            ("remote_request_id", Value::Null),
            ("remote_host", json!("https://outside.invalid")),
            ("remote_host", json!("10.0.0.7\n")),
            ("remote_port", json!(0)),
            ("remote_port", json!(65536)),
            ("remote_port", json!(true)),
            ("remote_block_ids", json!([])),
            ("remote_block_ids", json!([[]])),
            ("remote_block_ids", json!([[true]])),
            ("remote_block_ids", json!([[-1]])),
            ("remote_block_ids", json!([1, 2])),
            ("tp_size", json!(0)),
            ("dcp_size", json!(-1)),
            ("pp_size", json!(u64::MAX)),
            ("remote_registration_generation", json!(false)),
            ("remote_blocks_expiry_time", Value::Null),
            ("remote_blocks_expiry_time", json!(-1)),
            ("do_remote_decode", json!(false)),
            ("do_remote_prefill", json!(true)),
            ("transfer_mode", json!("push")),
            ("_remote_blocks_processed", json!(false)),
            ("_p_side_truncated", json!(false)),
        ] {
            let mut invalid = payload();
            invalid["kv_transfer_params"][field] = value;
            assert!(!insert(&catalog, &invalid, now), "accepted {field}");
        }
    }

    #[test]
    fn expiry_is_local_bounded_and_never_refreshed_by_lookup_or_duplicate() {
        let catalog = catalog();
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        assert!(!insert(&catalog, &payload(), now + Duration::from_secs(19)));
        assert!(
            catalog
                .claim(&scope(), &[1, 2], now + Duration::from_secs(19))
                .is_none()
        );
        assert!(
            catalog
                .claim(&scope(), &[1, 2, 3], now + Duration::from_secs(20))
                .is_none()
        );
        let mut short = payload();
        short["ttl_seconds"] = json!(0.5);
        // A source clock value is opaque even when numerically far from frontend time.
        short["kv_transfer_params"]["remote_blocks_expiry_time"] = json!(0.01);
        assert!(insert(&catalog, &short, now));
        assert!(
            catalog
                .claim(&scope(), &[1, 2, 3], now + Duration::from_millis(500))
                .is_none()
        );
        for hint in [json!(0), json!(-1), json!("20"), Value::Null] {
            short["ttl_seconds"] = hint;
            assert!(!insert(&catalog, &short, now));
        }
    }

    #[test]
    fn source_generation_and_native_peer_changes_invalidate_old_donors() {
        for (field, value) in [
            ("remote_engine_id", json!("engine-b")),
            ("remote_host", json!("10.0.0.8")),
            ("remote_port", json!(5601)),
            ("tp_size", json!(8)),
            ("remote_registration_generation", json!(1)),
        ] {
            let catalog = catalog();
            let now = Instant::now();
            assert!(insert(&catalog, &payload(), now));
            let mut changed = payload();
            changed["kv_transfer_params"][field] = value;
            changed["kv_transfer_params"]["remote_request_id"] = json!("request-b");
            assert!(insert(&catalog, &changed, now));
            assert_eq!(catalog.state.lock().entries.len(), 1);
            assert_eq!(
                catalog
                    .claim(&scope(), &[1, 2, 3], now)
                    .unwrap()
                    .kv_transfer_params["remote_request_id"],
                "request-b"
            );
        }
        let catalog = catalog();
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        let mut changed_source = source();
        changed_source.generation = "successor-generation".into();
        assert!(catalog.insert(
            scope(),
            changed_source.clone(),
            &[1, 2],
            &[3],
            &payload(),
            now
        ));
        assert_eq!(catalog.state.lock().entries.len(), 1);
        assert_eq!(
            catalog.claim(&scope(), &[1, 2, 3], now).unwrap().source,
            changed_source
        );
    }

    #[test]
    fn simultaneous_claims_have_exactly_one_winner() {
        let catalog = Arc::new(catalog());
        let now = Instant::now();
        assert!(insert(&catalog, &payload(), now));
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let catalog = catalog.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    catalog.claim(&scope(), &[1, 2, 3], now).is_some()
                })
            })
            .collect();
        assert_eq!(
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
    }

    #[test]
    fn limits_evict_oldest_and_release_logical_bytes() {
        let mut config = Config::new("test".into()).unwrap();
        config.max_entries = 2;
        let catalog = Catalog::new(config);
        let now = Instant::now();
        for id in ["a", "b", "c"] {
            let mut value = payload();
            value["kv_transfer_params"]["remote_request_id"] = json!(id);
            assert!(insert(&catalog, &value, now));
        }
        let state = catalog.state.lock();
        assert_eq!(state.entries.len(), 2);
        assert_eq!(
            state.entries[0].donor.kv_transfer_params["remote_request_id"],
            "b"
        );
        assert_eq!(
            state.bytes,
            state.entries.iter().map(|entry| entry.bytes).sum::<usize>()
        );
        drop(state);
        catalog.clear();
        assert!(catalog.state.lock().entries.is_empty());
        assert_eq!(catalog.state.lock().bytes, 0);

        let entry_bytes = catalog
            .entry(scope(), source(), &[1, 2], &[3], &payload(), now)
            .unwrap()
            .bytes;
        let mut config = Config::new("test".into()).unwrap();
        config.max_bytes = entry_bytes + 1;
        let catalog = Catalog::new(config);
        assert!(insert(&catalog, &payload(), now));
        let mut newer = payload();
        newer["kv_transfer_params"]["remote_request_id"] = json!("request-b");
        assert!(insert(&catalog, &newer, now));
        assert_eq!(catalog.state.lock().entries.len(), 1);
        assert_eq!(
            catalog
                .claim(&scope(), &[1, 2, 3], now)
                .unwrap()
                .kv_transfer_params["remote_request_id"],
            "request-b"
        );

        let mut config = Config::new("test".into()).unwrap();
        config.max_bytes = 32;
        assert!(!insert(&Catalog::new(config), &payload(), now));
        let mut config = Config::new("test".into()).unwrap();
        config.max_tokens = 2;
        assert!(!insert(&Catalog::new(config), &payload(), now));
    }

    #[test]
    fn longest_complete_prefix_wins_without_clipping_other_donors() {
        let catalog = catalog();
        let now = Instant::now();
        let mut short = payload();
        short["kv_transfer_params"]["remote_num_tokens"] = json!(2);
        short["kv_transfer_params"]["remote_request_id"] = json!("short");
        assert!(insert(&catalog, &short, now));
        assert!(insert(&catalog, &payload(), now));
        assert_eq!(
            catalog
                .claim(&scope(), &[1, 2, 3, 9], now)
                .unwrap()
                .kv_transfer_params["remote_num_tokens"],
            3
        );
        assert_eq!(
            catalog
                .claim(&scope(), &[1, 2, 9], now)
                .unwrap()
                .kv_transfer_params["remote_num_tokens"],
            2
        );
    }

    #[test]
    fn opaque_metadata_is_bounded_before_clone_and_does_not_authorize_a_donor() {
        let catalog = catalog();
        let now = Instant::now();
        let mut invalid = payload();
        let mut nested = json!(null);
        for _ in 0..18 {
            nested = json!([nested]);
        }
        invalid["status"] = nested;
        assert!(!insert(&catalog, &invalid, now));
        let mut bounded = payload();
        bounded["status"] = json!("not-an-authorization-input");
        assert!(insert(&catalog, &bounded, now));
    }
}
