// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit Mooncake admin commands. Policies coordinate sharing and TTL renewal.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_hashing::ExternalSequenceBlockHash;
use dynamo_kv_indexer::host::{SessionPrefixIndexer, SessionResidencyOwner, SharedCacheBackend};
use dynamo_scheduler::host::SharedKvCache;

use super::KvRouter;

const MAX_COMMAND_KEYS: usize = 4096;

/// Mooncake's per-object error code; zero means success.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MooncakeKeyResult {
    pub key: String,
    pub error_code: i32,
}

/// Missing observations are separate from Store-side per-key failures.
#[derive(Debug, Default, Serialize)]
pub struct MooncakePinResult {
    pub key_results: Vec<MooncakeKeyResult>,
    pub unresolved_hashes: Vec<u64>,
}

/// A failed batch may have applied. Earlier batches completed; later ones were not sent.
#[derive(Debug, thiserror::Error)]
#[error("Mooncake command failed after {} completed keys; failed-batch outcome may be unknown: {source:#}", completed_results.len())]
pub struct MooncakeCommandError {
    pub completed_results: Vec<MooncakeKeyResult>,
    pub uncertain_keys: Vec<String>,
    pub unattempted_keys: Vec<String>,
    pub unresolved_hashes: Vec<u64>,
    #[source]
    pub source: anyhow::Error,
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
                && endpoint.fragment().is_none()
                && endpoint.path() == "/",
            "expected Mooncake admin URL with a root path and without query or fragment"
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

    async fn command(
        &self,
        keys: &[String],
        ttl_ms: Option<u64>,
    ) -> Result<Vec<MooncakeKeyResult>> {
        ensure!(
            keys.iter().all(|key| !key.is_empty()),
            "object keys must not be empty"
        );
        let mut results = Vec::with_capacity(keys.len());
        for (batch_index, batch) in keys.chunks(MAX_COMMAND_KEYS).enumerate() {
            match self.command_batch(batch, ttl_ms).await {
                Ok(batch_results) => results.extend(batch_results),
                Err(source) => {
                    return Err(MooncakeCommandError {
                        completed_results: results,
                        uncertain_keys: batch.to_vec(),
                        unattempted_keys: keys[(batch_index * MAX_COMMAND_KEYS + batch.len())..]
                            .to_vec(),
                        unresolved_hashes: Vec::new(),
                        source,
                    }
                    .into());
                }
            }
        }
        Ok(results)
    }

    async fn command_batch(
        &self,
        keys: &[String],
        ttl_ms: Option<u64>,
    ) -> Result<Vec<MooncakeKeyResult>> {
        let path = if ttl_ms.is_some() {
            "/api/v1/soft_pin"
        } else {
            "/api/v1/soft_unpin"
        };
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
            .context("Mooncake command transport failed")?
            .error_for_status()
            .context("Mooncake admin rejected command")?
            .json::<PinResponse>()
            .await
            .context("invalid Mooncake command response")?;
        ensure!(
            response.results.len() == keys.len()
                && response
                    .results
                    .iter()
                    .zip(keys)
                    .all(|(result, key)| &result.key == key),
            "Mooncake returned mismatched key results"
        );
        Ok(response.results)
    }
}

/// Store-specific commands scoped to a router's observed cache and session index.
#[derive(Clone)]
pub struct MooncakeControl {
    cache: Arc<dyn SharedKvCache>,
    index: Option<Arc<SessionPrefixIndexer>>,
    client: MooncakeClient,
}

impl MooncakeControl {
    pub fn new(
        cache: Arc<dyn SharedKvCache>,
        index: Option<Arc<SessionPrefixIndexer>>,
        client: MooncakeClient,
    ) -> Result<Self> {
        cache.enable_object_key_tracking()?;
        Ok(Self {
            cache,
            index,
            client,
        })
    }

    /// All owners contribute lineage; a selected frontier must still be a chain end.
    pub fn get_session_blocks(
        &self,
        session_id: &str,
        frontier_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>> {
        let index = self
            .index
            .as_deref()
            .context("session prefix index is not enabled")?;
        let chains = session_block_chains(index, session_id, frontier_hash)?;
        ensure!(
            frontier_hash.is_none() || !chains.is_empty(),
            "block is not an observed frontier of session {session_id}"
        );
        Ok(chains)
    }

    pub fn get_session_mooncake_blocks(
        &self,
        session_id: &str,
        tier: StorageTier,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>> {
        let index = self
            .index
            .as_deref()
            .context("session prefix index is not enabled")?;
        Ok(index.get_session_block_lineage_in_tier(
            session_id,
            SessionResidencyOwner::SharedCache(SharedCacheBackend::Mooncake),
            tier,
            None,
        )?)
    }

    fn resolve(
        &self,
        hashes: &[ExternalSequenceBlockHash],
        strict: bool,
    ) -> Result<(Vec<String>, Vec<u64>)> {
        // SPI hashes carry no tenant: resolve only within the active writer tenant.
        let hashes: Vec<_> = hashes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let groups = self
            .cache
            .resolve_object_keys(&hashes, Some(&self.client.tenant_id))?;
        ensure!(
            groups.len() == hashes.len(),
            "cache returned mismatched block resolutions"
        );
        let mut keys = BTreeSet::new();
        let mut unresolved = Vec::new();
        for (hash, group) in hashes.into_iter().zip(groups) {
            if group.is_empty() {
                ensure!(
                    !strict,
                    "no observed Mooncake keys for block hash {}",
                    hash.0
                );
                unresolved.push(hash.0);
            }
            keys.extend(group);
        }
        Ok((keys.into_iter().collect(), unresolved))
    }

    /// Strict selection: empty input or any unresolved hash fails before HTTP.
    /// Observed keys do not prove a page's entire component inventory is present.
    pub async fn pin_blocks(
        &self,
        hashes: &[ExternalSequenceBlockHash],
        ttl_ms: u64,
    ) -> Result<Vec<MooncakeKeyResult>> {
        ensure!(!hashes.is_empty(), "no observed blocks to pin");
        let (keys, _) = self.resolve(hashes, true)?;
        self.client.command(&keys, Some(ttl_ms)).await
    }

    /// Clear current observed keys unconditionally; this does not manage session references.
    pub async fn unpin_blocks(
        &self,
        hashes: &[ExternalSequenceBlockHash],
    ) -> Result<Vec<MooncakeKeyResult>> {
        let (keys, _) = self.resolve(hashes, false)?;
        self.client.command(&keys, None).await
    }

    /// Pin what resolves and report hashes with no observed eligible keys.
    /// Pin resets TTL, including shortening it; policies coordinate shared prefixes.
    pub async fn pin_session(&self, session_id: &str, ttl_ms: u64) -> Result<MooncakePinResult> {
        self.pin_session_selection(session_id, None, ttl_ms).await
    }

    pub async fn pin_session_frontier(
        &self,
        session_id: &str,
        frontier_hash: ExternalSequenceBlockHash,
        ttl_ms: u64,
    ) -> Result<MooncakePinResult> {
        self.pin_session_selection(session_id, Some(frontier_hash), ttl_ms)
            .await
    }

    async fn pin_session_selection(
        &self,
        session_id: &str,
        frontier: Option<ExternalSequenceBlockHash>,
        ttl_ms: u64,
    ) -> Result<MooncakePinResult> {
        let hashes: Vec<_> = self
            .get_session_blocks(session_id, frontier)?
            .into_iter()
            .flatten()
            .collect();
        ensure!(
            !hashes.is_empty(),
            "no observed blocks for session {session_id}"
        );
        let (keys, unresolved_hashes) = self.resolve(&hashes, false)?;
        let key_results = self
            .client
            .command(&keys, Some(ttl_ms))
            .await
            .map_err(|mut error| {
                if let Some(batch) = error.downcast_mut::<MooncakeCommandError>() {
                    batch.unresolved_hashes.clone_from(&unresolved_hashes);
                }
                error
            })?;
        Ok(MooncakePinResult {
            key_results,
            unresolved_hashes,
        })
    }
}

impl KvRouter {
    pub fn mooncake_control(&self, client: MooncakeClient) -> Result<MooncakeControl> {
        MooncakeControl::new(
            Arc::clone(self.shared_cache().context("shared cache is not enabled")?),
            self.session_prefix_index().cloned(),
            client,
        )
    }
}

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
            .command(&["a".into(), "missing".into()], Some(1234))
            .await
            .unwrap();
        assert_eq!(results[1].error_code, -704);
        assert_eq!(
            control.command(&["a".into()], None).await.unwrap()[0].error_code,
            0
        );
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
        assert!(control.command(&["a".into()], Some(1)).await.is_err());
        assert!(control.command(&["a".into()], None).await.is_err());
        assert!(control.command(&[], Some(1)).await.unwrap().is_empty());
        assert!(control.command(&[String::new()], Some(1)).await.is_err());

        unavailable.assert_async().await;
        mismatched.assert_async().await;
    }
    #[test]
    fn rejects_admin_url_path_prefixes() {
        for endpoint in [
            "http://gw/mooncake/",
            "http://gw/?x=1",
            "http://gw/#fragment",
            "file:///store",
        ] {
            assert!(MooncakeClient::new(endpoint, "default").is_err());
        }
        assert!(MooncakeClient::new("http://gw", "default").is_ok());
    }

    #[tokio::test]
    async fn batches_preserve_progress_without_retrying_failed_requests() {
        let mut server = mockito::Server::new_async().await;
        let control = MooncakeClient::new(&server.url(), "default").unwrap();
        let keys: Vec<_> = (0..8193).map(|i| format!("key-{i:05}")).collect();
        let results: Vec<_> = keys[..4096]
            .iter()
            .map(|key| serde_json::json!({"key":key,"error_code":0}))
            .collect();
        let first = server
            .mock("POST", "/api/v1/soft_pin")
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"tenant_id":"default","keys":&keys[..4096],"ttl_ms":1}),
            ))
            .with_body(serde_json::json!({"results":results}).to_string())
            .expect(1)
            .create_async()
            .await;
        let second = server
            .mock("POST", "/api/v1/soft_pin")
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"tenant_id":"default","keys":&keys[4096..8192],"ttl_ms":1}),
            ))
            .with_status(503)
            .expect(1)
            .create_async()
            .await;
        let error = control
            .command(&keys, Some(1))
            .await
            .unwrap_err()
            .downcast::<MooncakeCommandError>()
            .unwrap();
        assert_eq!(error.completed_results.len(), 4096);
        assert_eq!(error.uncertain_keys, keys[4096..8192]);
        assert_eq!(error.unattempted_keys, keys[8192..]);
        first.assert_async().await;
        second.assert_async().await;
    }
}
