// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use async_trait::async_trait;
use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use std::time::Duration;

use super::KeyValueStore;

pub struct PostgresKeyValueStore {
    pool: PgPool,
    namespace: String,
}

impl PostgresKeyValueStore {
    pub async fn connect(url: &str, namespace: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(url)
            .await
            .context("failed to connect to PostgreSQL store")?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dynamo_key_value_store (
                namespace TEXT NOT NULL,
                store_key TEXT NOT NULL,
                value BYTEA NOT NULL,
                expires_at BIGINT,
                PRIMARY KEY (namespace, store_key)
            )",
        )
        .execute(&pool)
        .await
        .context("failed to initialize PostgreSQL store table")?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS dynamo_key_value_store_expiry
             ON dynamo_key_value_store (namespace, expires_at)
             WHERE expires_at IS NOT NULL",
        )
        .execute(&pool)
        .await
        .context("failed to initialize PostgreSQL store expiry index")?;
        Ok(Self {
            pool,
            namespace: namespace.to_string(),
        })
    }
}

#[async_trait]
impl KeyValueStore for PostgresKeyValueStore {
    async fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let row = sqlx::query(
            "SELECT value FROM dynamo_key_value_store
             WHERE namespace = $1 AND store_key = $2
               AND (expires_at IS NULL OR
                    expires_at > EXTRACT(EPOCH FROM clock_timestamp())::BIGINT)",
        )
        .bind(&self.namespace)
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to read from PostgreSQL store")?;
        row.map(|row| row.try_get("value"))
            .transpose()
            .context("invalid value in PostgreSQL store")
    }

    async fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> anyhow::Result<()> {
        let ttl = ttl
            .map(|ttl| {
                i64::try_from(if ttl.is_zero() {
                    0
                } else {
                    ttl.as_secs().max(1)
                })
            })
            .transpose()
            .context("PostgreSQL store TTL is too large")?;
        sqlx::query(
            "INSERT INTO dynamo_key_value_store
                 (namespace, store_key, value, expires_at)
             VALUES (
                 $1,
                 $2,
                 $3,
                 CASE WHEN $4::BIGINT IS NULL THEN NULL
                      ELSE EXTRACT(EPOCH FROM clock_timestamp())::BIGINT + $4
                 END
             )
             ON CONFLICT (namespace, store_key) DO UPDATE
             SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at",
        )
        .bind(&self.namespace)
        .bind(key)
        .bind(value)
        .bind(ttl)
        .execute(&self.pool)
        .await
        .context("failed to write to PostgreSQL store")?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM dynamo_key_value_store
             WHERE namespace = $1 AND store_key = $2
               AND (expires_at IS NULL OR
                    expires_at > EXTRACT(EPOCH FROM clock_timestamp())::BIGINT)",
        )
        .bind(&self.namespace)
        .bind(key)
        .execute(&self.pool)
        .await
        .context("failed to delete from PostgreSQL store")?;
        Ok(result.rows_affected() != 0)
    }

    async fn purge_expired(&self) -> anyhow::Result<usize> {
        let result = sqlx::query(
            "DELETE FROM dynamo_key_value_store
             WHERE namespace = $1
               AND expires_at IS NOT NULL
               AND expires_at <= EXTRACT(EPOCH FROM clock_timestamp())::BIGINT
               AND store_key IN (
                 SELECT store_key FROM dynamo_key_value_store
                 WHERE namespace = $1 AND expires_at IS NOT NULL
                   AND expires_at <= EXTRACT(EPOCH FROM clock_timestamp())::BIGINT
                 LIMIT 1000
             )",
        )
        .bind(&self.namespace)
        .execute(&self.pool)
        .await
        .context("failed to purge expired PostgreSQL store values")?;
        usize::try_from(result.rows_affected()).context("PostgreSQL purge count overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires DYN_TEST_POSTGRES_URL"]
    async fn postgres_store_contract_and_namespace_isolation() {
        let url =
            std::env::var("DYN_TEST_POSTGRES_URL").expect("DYN_TEST_POSTGRES_URL is required");
        let namespace = format!("dynamo-test:{}", uuid::Uuid::new_v4());
        let other_namespace = format!("{namespace}:other");
        let store = PostgresKeyValueStore::connect(&url, &namespace)
            .await
            .unwrap();
        super::super::test_store_contract(&store).await;

        let other = PostgresKeyValueStore::connect(&url, &other_namespace)
            .await
            .unwrap();
        super::super::test_namespace_isolation(&store, &other).await;
    }

    #[tokio::test]
    #[ignore = "requires DYN_TEST_POSTGRES_URL"]
    async fn purge_preserves_a_concurrently_refreshed_value() {
        let url =
            std::env::var("DYN_TEST_POSTGRES_URL").expect("DYN_TEST_POSTGRES_URL is required");
        let store =
            PostgresKeyValueStore::connect(&url, &format!("dynamo-test:{}", uuid::Uuid::new_v4()))
                .await
                .unwrap();
        let key = "refresh-race";
        store
            .put(key, b"expired", Some(Duration::ZERO))
            .await
            .unwrap();

        let mut tx = store.pool.begin().await.unwrap();
        sqlx::query(
            "SELECT value FROM dynamo_key_value_store
             WHERE namespace = $1 AND store_key = $2 FOR UPDATE",
        )
        .bind(&store.namespace)
        .bind(key)
        .fetch_one(&mut *tx)
        .await
        .unwrap();

        let sweeper = PostgresKeyValueStore {
            pool: store.pool.clone(),
            namespace: store.namespace.clone(),
        };
        let purge = tokio::spawn(async move { sweeper.purge_expired().await.unwrap() });
        tokio::time::sleep(Duration::from_millis(50)).await;
        sqlx::query(
            "UPDATE dynamo_key_value_store SET value = $3, expires_at = NULL
             WHERE namespace = $1 AND store_key = $2",
        )
        .bind(&store.namespace)
        .bind(key)
        .bind(b"fresh".as_slice())
        .execute(&mut *tx)
        .await
        .unwrap();
        tx.commit().await.unwrap();

        assert_eq!(purge.await.unwrap(), 0);
        assert_eq!(
            store.get(key).await.unwrap().as_deref(),
            Some(b"fresh".as_slice())
        );
        store.delete(key).await.unwrap();
    }
}
