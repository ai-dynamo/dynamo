// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use serde_json::Value;

const ENABLE_ENV: &str = "DYN_ENABLE_VLLM_NIXL_BIDIRECTIONAL_KV";
const TTL_ENV: &str = "DYN_VLLM_NIXL_BIDIRECTIONAL_KV_TTL_SECS";
const DEFAULT_TTL_SECS: u64 = 450;

#[derive(Debug)]
struct CacheEntry {
    kv_transfer_params: Value,
    expires_at: Instant,
}

#[derive(Debug)]
pub(super) struct SessionKvTransferCache {
    ttl: Duration,
    entries: Mutex<HashMap<String, CacheEntry>>,
}

impl SessionKvTransferCache {
    pub(super) fn from_env() -> Option<Self> {
        if !env_is_truthy(ENABLE_ENV) {
            return None;
        }
        let ttl_secs = std::env::var(TTL_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|ttl_secs| *ttl_secs > 0)
            .unwrap_or(DEFAULT_TTL_SECS);
        Some(Self::new(ttl_secs))
    }

    pub(super) fn new(ttl_secs: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_secs),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn take(&self, session_id: &str) -> Option<Value> {
        let now = Instant::now();
        let mut entries = self
            .entries
            .lock()
            .expect("session KV cache mutex poisoned");
        let entry = entries.remove(session_id)?;
        (entry.expires_at > now).then_some(entry.kv_transfer_params)
    }

    pub(super) fn put(&self, session_id: String, kv_transfer_params: Value) {
        let mut entries = self
            .entries
            .lock()
            .expect("session KV cache mutex poisoned");
        entries.retain(|_, entry| entry.expires_at > Instant::now());
        entries.insert(
            session_id,
            CacheEntry {
                kv_transfer_params,
                expires_at: Instant::now() + self.ttl,
            },
        );
    }
}

fn env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use serde_json::json;

    use super::*;

    #[test]
    fn cache_is_single_use() {
        let cache = SessionKvTransferCache::new(60);
        cache.put("s1".to_string(), json!({"remote_block_ids": [1, 2]}));

        assert_eq!(cache.take("s1"), Some(json!({"remote_block_ids": [1, 2]})));
        assert_eq!(cache.take("s1"), None);
    }

    #[test]
    fn expired_entry_is_not_returned() {
        let cache = SessionKvTransferCache::new(0);
        cache.put("s1".to_string(), json!({"remote_block_ids": [1, 2]}));
        thread::sleep(Duration::from_millis(1));

        assert_eq!(cache.take("s1"), None);
    }
}
