// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;

use super::KeyValueStore;

pub struct TikvKeyValueStore {
    client: tikv_client::RawClient,
    prefix: Vec<u8>,
}

impl TikvKeyValueStore {
    pub async fn connect(endpoints: Vec<String>, namespace: &str) -> anyhow::Result<Self> {
        let client = tikv_client::RawClient::new(endpoints)
            .await
            .context("failed to connect to TiKV store")?;
        let mut prefix = namespace.trim_matches('/').as_bytes().to_vec();
        prefix.push(b'/');
        Ok(Self { client, prefix })
    }

    fn key(&self, key: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.prefix.len() + key.len());
        out.extend_from_slice(&self.prefix);
        out.extend_from_slice(key.as_bytes());
        out
    }
}

#[async_trait]
impl KeyValueStore for TikvKeyValueStore {
    async fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        self.client
            .get(self.key(key))
            .await
            .context("failed to read from TiKV store")
    }

    async fn put(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> anyhow::Result<()> {
        let key = self.key(key);
        match ttl {
            Some(ttl) if ttl.is_zero() => self.client.delete(key).await?,
            Some(ttl) => {
                self.client
                    .put_with_ttl(key, value.to_vec(), ttl.as_secs().max(1))
                    .await?
            }
            None => self.client.put(key, value.to_vec()).await?,
        }
        Ok(())
    }

    async fn delete(&self, key: &str) -> anyhow::Result<bool> {
        let key = self.key(key);
        let existed = self.client.get(key.clone()).await?.is_some();
        self.client.delete(key).await?;
        Ok(existed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires DYN_TEST_TIKV_ENDPOINTS"]
    async fn tikv_store_contract_and_namespace_isolation() {
        let endpoints = std::env::var("DYN_TEST_TIKV_ENDPOINTS")
            .expect("DYN_TEST_TIKV_ENDPOINTS is required")
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let namespace = format!("dynamo-test:{}", uuid::Uuid::new_v4());
        let other_namespace = format!("{namespace}:other");
        let store = TikvKeyValueStore::connect(endpoints.clone(), &namespace)
            .await
            .unwrap();
        super::super::test_store_contract(&store).await;

        let other = TikvKeyValueStore::connect(endpoints, &other_namespace)
            .await
            .unwrap();
        super::super::test_namespace_isolation(&store, &other).await;
    }
}
