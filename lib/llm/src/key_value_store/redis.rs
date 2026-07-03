// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use async_trait::async_trait;
use redis::{AsyncCommands, aio::ConnectionManager};
use std::time::Duration;

use super::KeyValueStore;

pub struct RedisKeyValueStore {
    connection: ConnectionManager,
    prefix: String,
}

impl RedisKeyValueStore {
    pub async fn connect(url: &str, namespace: &str) -> anyhow::Result<Self> {
        let client =
            redis::Client::open(url).map_err(|_| anyhow::anyhow!("invalid Redis store URL"))?;
        let connection = client
            .get_connection_manager()
            .await
            .context("failed to connect to Redis store")?;
        Ok(Self {
            connection,
            prefix: format!("{}:{namespace}:", namespace.len()),
        })
    }

    fn key(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }
}

#[async_trait]
impl KeyValueStore for RedisKeyValueStore {
    async fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let mut connection = self.connection.clone();
        connection
            .get(self.key(key))
            .await
            .context("failed to read from Redis store")
    }

    async fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> anyhow::Result<()> {
        let mut connection = self.connection.clone();
        let key = self.key(key);
        if ttl == Some(Duration::ZERO) {
            let _: u64 = connection
                .del(key)
                .await
                .context("failed to delete expired Redis value")?;
            return Ok(());
        }

        let mut command = redis::cmd("SET");
        command.arg(key).arg(value);
        if let Some(ttl) = ttl {
            command.arg("EX").arg(ttl.as_secs().max(1));
        }
        command
            .query_async::<()>(&mut connection)
            .await
            .context("failed to write to Redis store")
    }

    async fn delete(&self, key: &str) -> anyhow::Result<bool> {
        let mut connection = self.connection.clone();
        let removed: u64 = connection
            .del(self.key(key))
            .await
            .context("failed to delete from Redis store")?;
        Ok(removed != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires DYN_TEST_REDIS_URL"]
    async fn redis_store_contract_and_namespace_isolation() {
        let url = std::env::var("DYN_TEST_REDIS_URL").expect("DYN_TEST_REDIS_URL is required");
        let namespace = format!("dynamo-test:{}", uuid::Uuid::new_v4());
        let other_namespace = format!("{namespace}:other");
        let store = RedisKeyValueStore::connect(&url, &namespace).await.unwrap();
        super::super::test_store_contract(&store).await;

        let other = RedisKeyValueStore::connect(&url, &other_namespace)
            .await
            .unwrap();
        super::super::test_namespace_isolation(&store, &other).await;
    }
}
