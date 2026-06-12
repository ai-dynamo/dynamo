// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async key-value storage for small frontend-owned state.

mod memory;
#[cfg(feature = "key-value-store-postgres")]
mod postgres;
#[cfg(feature = "key-value-store-redis")]
mod redis;
use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;

pub use memory::MemoryKeyValueStore;
#[cfg(feature = "key-value-store-postgres")]
pub use postgres::PostgresKeyValueStore;
#[cfg(feature = "key-value-store-redis")]
pub use redis::RedisKeyValueStore;

/// A namespaced byte store with optional expiration.
///
/// Implementations must not return expired values. `put` atomically replaces
/// the value and TTL for a key. A concurrent write can make `delete`'s return
/// value stale immediately after the operation completes.
#[async_trait]
pub trait KeyValueStore: Send + Sync + 'static {
    async fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>>;
    async fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> anyhow::Result<()>;
    /// Remove a key and report whether a live value was removed.
    async fn delete(&self, key: &str) -> anyhow::Result<bool>;

    /// Eagerly remove expired keys when the backend does not provide native TTL.
    async fn purge_expired(&self) -> anyhow::Result<usize> {
        Ok(0)
    }
}

#[derive(Clone, PartialEq)]
pub enum KeyValueStoreConfig {
    Memory,
    #[cfg(feature = "key-value-store-redis")]
    Redis(String),
    #[cfg(feature = "key-value-store-postgres")]
    Postgres(String),
}

impl fmt::Debug for KeyValueStoreConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => f.write_str("Memory"),
            #[cfg(feature = "key-value-store-redis")]
            Self::Redis(_) => f.write_str("Redis(\"<redacted>\")"),
            #[cfg(feature = "key-value-store-postgres")]
            Self::Postgres(_) => f.write_str("Postgres(\"<redacted>\")"),
        }
    }
}

impl KeyValueStoreConfig {
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw = raw.trim();
        if raw.eq_ignore_ascii_case("memory") {
            return Ok(Self::Memory);
        }
        #[cfg(feature = "key-value-store-redis")]
        if raw.starts_with("redis://") || raw.starts_with("rediss://") {
            return Ok(Self::Redis(raw.to_string()));
        }
        #[cfg(feature = "key-value-store-postgres")]
        if raw.starts_with("postgres://") || raw.starts_with("postgresql://") {
            return Ok(Self::Postgres(raw.to_string()));
        }
        anyhow::bail!(
            "unsupported key-value store URL; enabled stores: {}",
            enabled_store_url_schemes()
        );
    }

    pub async fn open(&self, namespace: &str) -> anyhow::Result<Arc<dyn KeyValueStore>> {
        let _ = namespace;
        match self {
            Self::Memory => Ok(Arc::new(MemoryKeyValueStore::new())),
            #[cfg(feature = "key-value-store-redis")]
            Self::Redis(url) => Ok(Arc::new(RedisKeyValueStore::connect(url, namespace).await?)),
            #[cfg(feature = "key-value-store-postgres")]
            Self::Postgres(url) => Ok(Arc::new(
                PostgresKeyValueStore::connect(url, namespace).await?,
            )),
        }
    }
}

pub fn enabled_store_url_schemes() -> String {
    let schemes = [
        "memory",
        #[cfg(feature = "key-value-store-redis")]
        "redis:// or rediss://",
        #[cfg(feature = "key-value-store-postgres")]
        "postgres:// or postgresql://",
    ];
    schemes.join(", ")
}

#[cfg(test)]
pub(crate) async fn test_store_contract(store: &dyn KeyValueStore) {
    let key = format!("contract:{}", uuid::Uuid::new_v4());
    store.put(&key, b"first", None).await.unwrap();
    assert_eq!(
        store.get(&key).await.unwrap().as_deref(),
        Some(b"first".as_slice())
    );

    store.put(&key, b"second", None).await.unwrap();
    assert_eq!(
        store.get(&key).await.unwrap().as_deref(),
        Some(b"second".as_slice())
    );

    store
        .put(&key, b"expired", Some(Duration::ZERO))
        .await
        .unwrap();
    assert_eq!(store.get(&key).await.unwrap(), None);
    assert!(!store.delete(&key).await.unwrap());
    store.purge_expired().await.unwrap();

    store.put(&key, b"delete", None).await.unwrap();
    assert!(store.delete(&key).await.unwrap());
    assert!(!store.delete(&key).await.unwrap());
}

#[cfg(all(
    test,
    any(
        feature = "key-value-store-redis",
        feature = "key-value-store-postgres"
    )
))]
pub(crate) async fn test_namespace_isolation(
    first: &dyn KeyValueStore,
    second: &dyn KeyValueStore,
) {
    first.put("same", b"one", None).await.unwrap();
    second.put("same", b"two", None).await.unwrap();
    assert_eq!(
        first.get("same").await.unwrap().as_deref(),
        Some(b"one".as_slice())
    );
    assert_eq!(
        second.get("same").await.unwrap().as_deref(),
        Some(b"two".as_slice())
    );
    first.delete("same").await.unwrap();
    second.delete("same").await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_memory_store() {
        assert_eq!(
            KeyValueStoreConfig::parse("memory").unwrap(),
            KeyValueStoreConfig::Memory
        );
    }

    #[test]
    fn unsupported_store_errors_do_not_include_credentials() {
        let err = KeyValueStoreConfig::parse("unknown://user:secret").unwrap_err();
        assert!(!err.to_string().contains("secret"));
    }

    #[cfg(feature = "key-value-store-redis")]
    #[test]
    fn parses_redis_urls_without_rewriting_credentials_or_tls() {
        for url in [
            "redis://user:password@redis.example:6379/2",
            "rediss://user:password@redis.example:6380/2",
        ] {
            let config = KeyValueStoreConfig::parse(url).unwrap();
            assert_eq!(config, KeyValueStoreConfig::Redis(url.to_string()));
            assert!(!format!("{config:?}").contains("password"));
        }
    }

    #[cfg(feature = "key-value-store-postgres")]
    #[test]
    fn parses_postgres_urls_without_rewriting_tls_options() {
        let url = "postgresql://user:password@db.example/responses?sslmode=require";
        let config = KeyValueStoreConfig::parse(url).unwrap();
        assert_eq!(config, KeyValueStoreConfig::Postgres(url.to_string()));
        assert!(!format!("{config:?}").contains("password"));
    }
}
