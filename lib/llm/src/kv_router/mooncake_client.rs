// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit object-level soft-pin commands for custom KV policies.
//!
//! The caller supplies the active Mooncake master's **admin HTTP** endpoint,
//! not its Store RPC or KV-event endpoint. Commands do not manage session
//! references, renew automatically, or retry: a retry can reset a TTL or undo
//! a newer pin. TTL expiry releases protection; it does not delete the object.

use std::collections::{BTreeSet, HashSet};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use dynamo_kv_router::{SessionPrefixIndexer, SharedKvCache, protocols::ExternalSequenceBlockHash};

use super::KvRouter;

/// One result per input object key. Zero means success; other values are
/// Mooncake ErrorCode values (for example -704: missing, -703: not ready).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MooncakeKeyResult {
    pub key: String,
    pub error_code: i32,
}

#[derive(Clone, Debug)]
pub struct MooncakeClient {
    client: reqwest::Client,
    endpoint: reqwest::Url,
    tenant_id: String,
}

#[derive(Serialize)]
struct PinRequest<'a> {
    tenant_id: &'a str,
    keys: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_ms: Option<u64>,
}

#[derive(Deserialize)]
struct PinResponse {
    results: Vec<MooncakeKeyResult>,
}

impl MooncakeClient {
    pub fn new(endpoint: &str, tenant_id: &str) -> Result<Self> {
        let endpoint = reqwest::Url::parse(endpoint).context("invalid Mooncake admin URL")?;
        ensure!(
            matches!(endpoint.scheme(), "http" | "https"),
            "expected HTTP(S) admin URL"
        );
        ensure!(
            endpoint.host_str().is_some()
                && endpoint.query().is_none()
                && endpoint.fragment().is_none(),
            "expected Mooncake admin URL without query or fragment"
        );
        ensure!(!tenant_id.is_empty(), "tenant_id must not be empty");
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            endpoint,
            tenant_id: tenant_id.into(),
        })
    }

    /// Reset each object's deadline to the master's current time + ttl_ms.
    /// A zero TTL clears protection, matching Mooncake's write-time semantics.
    async fn pin(&self, keys: &[String], ttl_ms: u64) -> Result<Vec<MooncakeKeyResult>> {
        self.command("/api/v1/soft_pin", keys, Some(ttl_ms)).await
    }

    /// Clear soft pins unconditionally, without consulting session references.
    /// Hard pins, read leases and object data are unchanged.
    async fn unpin(&self, keys: &[String]) -> Result<Vec<MooncakeKeyResult>> {
        self.command("/api/v1/soft_unpin", keys, None).await
    }

    async fn command(
        &self,
        path: &str,
        keys: &[String],
        ttl_ms: Option<u64>,
    ) -> Result<Vec<MooncakeKeyResult>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(
            keys.len() <= 4096,
            "Mooncake accepts at most 4096 keys per command"
        );
        ensure!(
            keys.iter().all(|key| !key.is_empty()),
            "object keys must not be empty"
        );
        let response = self
            .client
            .post(self.endpoint.join(path)?)
            .json(&PinRequest {
                tenant_id: &self.tenant_id,
                keys,
                ttl_ms,
            })
            .send()
            .await
            .context("Mooncake command failed; outcome may be unknown")?
            .error_for_status()
            .context("Mooncake admin rejected command")?
            .json::<PinResponse>()
            .await
            .context("invalid Mooncake command response; outcome may be unknown")?;
        ensure!(
            response.results.len() == keys.len()
                && response
                    .results
                    .iter()
                    .zip(keys)
                    .all(|(result, key)| &result.key == key),
            "Mooncake returned mismatched key results; outcome may be unknown"
        );
        Ok(response.results)
    }
}

/// Distinct observed root-first chains across workers and shared storage.
/// Each owner is read separately; this is an advisory snapshot, not a lease.
pub(super) fn session_block_chains(
    index: &SessionPrefixIndexer,
    session_id: &str,
    frontier_hash: Option<ExternalSequenceBlockHash>,
) -> Result<Vec<Vec<ExternalSequenceBlockHash>>> {
    let owners: HashSet<_> = index
        .get_session_residency(session_id)
        .into_iter()
        .map(|(owner, tier, _)| (owner, tier))
        .collect();
    let mut chains = BTreeSet::new();
    for (owner, tier) in owners {
        for chain in index.get_session_block_lineage_in_tier(session_id, owner, tier, None)? {
            if frontier_hash.is_none_or(|frontier| chain.last() == Some(&frontier)) {
                chains.insert(chain);
            }
        }
    }
    Ok(chains.into_iter().collect())
}

pub(super) async fn command_resolved_blocks(
    cache: &dyn SharedKvCache,
    client: &MooncakeClient,
    block_hashes: &[ExternalSequenceBlockHash],
    ttl_ms: Option<u64>,
) -> Result<Vec<MooncakeKeyResult>> {
    ensure!(
        client.tenant_id == "default",
        "router key resolution supports only the default tenant"
    );
    ensure!(
        ttl_ms.is_none() || !block_hashes.is_empty(),
        "no observed blocks to pin"
    );
    let hashes: Vec<_> = block_hashes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let groups = cache.resolve_object_keys(&hashes)?;
    let mut keys = BTreeSet::new();
    for (hash, group) in hashes.iter().zip(groups) {
        ensure!(
            ttl_ms.is_none() || !group.is_empty(),
            "no observed Mooncake keys for block hash {}",
            hash.0
        );
        keys.extend(group);
    }
    let keys: Vec<_> = keys.into_iter().collect();
    match ttl_ms {
        Some(ttl_ms) => client.pin(&keys, ttl_ms).await,
        None => client.unpin(&keys).await,
    }
}

impl KvRouter {
    /// Read all observed session chains, or one currently observed frontier's chain.
    /// Frontier hashes identify chain ends, not anchors that truncate a chain.
    pub fn get_session_blocks(
        &self,
        session_id: &str,
        frontier_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>> {
        let index = self
            .session_prefix_index()
            .context("session prefix index is not enabled")?;
        let chains = session_block_chains(index, session_id, frontier_hash)?;
        ensure!(
            frontier_hash.is_none() || !chains.is_empty(),
            "block is not an observed frontier of session {session_id}"
        );
        Ok(chains)
    }

    /// Pin observed default-tenant Store keys for explicit SPI/engine block hashes.
    /// Reject empty/unresolved input before HTTP. Results preserve per-key Store errors.
    /// Observations do not prove all page components arrived; inspect per-key errors.
    pub async fn pin_blocks(
        &self,
        client: &MooncakeClient,
        block_hashes: &[ExternalSequenceBlockHash],
        ttl_ms: u64,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let cache = self.shared_cache().context("shared cache is not enabled")?;
        command_resolved_blocks(cache.as_ref(), client, block_hashes, Some(ttl_ms)).await
    }

    /// Pin the union of the session's observed chains across workers and shared storage.
    pub async fn pin_session(
        &self,
        client: &MooncakeClient,
        session_id: &str,
        ttl_ms: u64,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let hashes: Vec<_> = self
            .get_session_blocks(session_id, None)?
            .into_iter()
            .flatten()
            .collect();
        self.pin_blocks(client, &hashes, ttl_ms).await
    }

    /// Pin only the root-to-frontier chain, including the specified frontier.
    /// An unknown, foreign or no-longer-current frontier fails before HTTP.
    pub async fn pin_session_frontier(
        &self,
        client: &MooncakeClient,
        session_id: &str,
        frontier_hash: ExternalSequenceBlockHash,
        ttl_ms: u64,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let hashes: Vec<_> = self
            .get_session_blocks(session_id, Some(frontier_hash))?
            .into_iter()
            .flatten()
            .collect();
        self.pin_blocks(client, &hashes, ttl_ms).await
    }

    /// Release currently observed keys for these hashes, skipping unresolved hashes.
    /// Lost observations do not prove Store deletion; remaining pins expire by TTL.
    pub async fn unpin_blocks(
        &self,
        client: &MooncakeClient,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<Vec<MooncakeKeyResult>> {
        let cache = self.shared_cache().context("shared cache is not enabled")?;
        command_resolved_blocks(cache.as_ref(), client, block_hashes, None).await
    }

    /// Release the session's currently observed chains; an unknown session is a no-op.
    pub async fn unpin_session(
        &self,
        client: &MooncakeClient,
        session_id: &str,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let hashes: Vec<_> = self
            .get_session_blocks(session_id, None)?
            .into_iter()
            .flatten()
            .collect();
        self.unpin_blocks(client, &hashes).await
    }

    /// Release only the currently observed root-to-frontier chain, inclusive.
    /// An absent/stale/foreign frontier is a no-op, not a release of other chains.
    pub async fn unpin_session_frontier(
        &self,
        client: &MooncakeClient,
        session_id: &str,
        frontier_hash: ExternalSequenceBlockHash,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let index = self
            .session_prefix_index()
            .context("session prefix index is not enabled")?;
        let hashes: Vec<_> = session_block_chains(index, session_id, Some(frontier_hash))?
            .into_iter()
            .flatten()
            .collect();
        self.unpin_blocks(client, &hashes).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sends_commands_and_preserves_per_key_failures() {
        let mut server = mockito::Server::new_async().await;
        let pin = server
            .mock("POST", "/api/v1/soft_pin")
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"tenant_id":"default", "keys":["a", "missing"], "ttl_ms":1234}),
            ))
            .with_body(
                r#"{"results":[{"key":"a","error_code":0},{"key":"missing","error_code":-704}]}"#,
            )
            .create_async()
            .await;
        let unpin = server
            .mock("POST", "/api/v1/soft_unpin")
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"tenant_id":"default", "keys":["a"]}),
            ))
            .with_body(r#"{"results":[{"key":"a","error_code":0}]}"#)
            .create_async()
            .await;
        let control = MooncakeClient::new(&server.url(), "default").unwrap();
        let results = control
            .pin(&["a".into(), "missing".into()], 1234)
            .await
            .unwrap();
        assert_eq!(results[1].error_code, -704);
        assert_eq!(control.unpin(&["a".into()]).await.unwrap()[0].error_code, 0);
        pin.assert_async().await;
        unpin.assert_async().await;
    }

    #[tokio::test]
    async fn rejects_http_failure_and_mismatched_results() {
        let mut server = mockito::Server::new_async().await;
        let unavailable = server
            .mock("POST", "/api/v1/soft_pin")
            .with_status(503)
            .expect(1)
            .create_async()
            .await;
        let mismatched = server
            .mock("POST", "/api/v1/soft_unpin")
            .with_body(r#"{"results":[{"key":"wrong","error_code":0}]}"#)
            .create_async()
            .await;
        let control = MooncakeClient::new(&server.url(), "default").unwrap();
        assert!(control.pin(&["a".into()], 1).await.is_err());
        assert!(control.unpin(&["a".into()]).await.is_err());
        assert!(control.pin(&[], 1).await.unwrap().is_empty());
        assert!(control.pin(&[String::new()], 1).await.is_err());
        assert!(control.pin(&vec!["a".into(); 4097], 1).await.is_err());
        unavailable.assert_async().await;
        mismatched.assert_async().await;
    }
}
