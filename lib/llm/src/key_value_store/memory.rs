// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::RwLock;

use super::KeyValueStore;

#[derive(Clone)]
struct Entry {
    value: Vec<u8>,
    expires_at: Option<Instant>,
}

#[derive(Default)]
pub struct MemoryKeyValueStore {
    entries: RwLock<HashMap<String, Entry>>,
}

impl MemoryKeyValueStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl KeyValueStore for MemoryKeyValueStore {
    async fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let mut entries = self.entries.write();
        if entries.get(key).is_some_and(|entry| {
            entry
                .expires_at
                .is_some_and(|expiry| expiry <= Instant::now())
        }) {
            entries.remove(key);
            return Ok(None);
        }
        Ok(entries.get(key).map(|entry| entry.value.clone()))
    }

    async fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> anyhow::Result<()> {
        if ttl == Some(Duration::ZERO) {
            self.entries.write().remove(key);
            return Ok(());
        }
        let expires_at = ttl
            .map(|ttl| {
                Instant::now()
                    .checked_add(ttl)
                    .ok_or_else(|| anyhow::anyhow!("memory store TTL is too large"))
            })
            .transpose()?;
        self.entries.write().insert(
            key.to_string(),
            Entry {
                value: value.to_vec(),
                expires_at,
            },
        );
        Ok(())
    }

    async fn delete(&self, key: &str) -> anyhow::Result<bool> {
        let entry = self.entries.write().remove(key);
        Ok(entry.is_some_and(|entry| {
            entry
                .expires_at
                .is_none_or(|expiry| expiry > Instant::now())
        }))
    }

    async fn purge_expired(&self) -> anyhow::Result<usize> {
        let mut entries = self.entries.write();
        let before = entries.len();
        let now = Instant::now();
        entries.retain(|_, entry| entry.expires_at.is_none_or(|expiry| expiry > now));
        Ok(before - entries.len())
    }
}
