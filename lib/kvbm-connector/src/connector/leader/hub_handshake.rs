// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub handshake.
//!
//! When `leader.hub` is configured, the connector pulls `GET {url}/v1/config`
//! once at startup, resolves which hub features it will participate in, and
//! builds the must-match [`RuntimeConfigSummary`] it registers with.
//!
//! Resolution semantics (see plan decisions):
//! - `leader.hub.features` **empty** → discover the hub's enabled set, intersect
//!   with the connector's capabilities, **best-effort** (an unreachable hub or a
//!   feature whose per-instance prerequisites are missing is simply dropped).
//! - `leader.hub.features` **non-empty** → validate the requested set (and its
//!   dependency closure) against the hub's enabled set; any gap is a
//!   **hard-fail** at startup.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use kvbm_config::{BlockLayoutMode, DisaggConfig, LeaderHubConfig};
use kvbm_hub::{
    FeatureConfigRequirements, FeatureDescriptor, FeatureKey, HubConfigResponse, PrimaryConfig,
    RuntimeConfigSummary,
};

const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Hub features a connector can participate in (all standalone-selectable).
/// `conditional_disagg` additionally depends on `p2p` (co-registered).
pub const CONNECTOR_CAPS: [FeatureKey; 3] = [
    FeatureKey::Indexer,
    FeatureKey::P2P,
    FeatureKey::ConditionalDisagg,
];

/// Outcome of the hub handshake.
pub struct HubHandshake {
    /// Resolved hub base URL (from `leader.hub.url`).
    pub url: String,
    /// Effective connector-level features (a subset of [`CONNECTOR_CAPS`]).
    pub effective: HashSet<FeatureKey>,
    /// KV-index ZMQ ingest endpoint — `Some` iff Indexer is effective and the
    /// hub advertised a block-size-compatible endpoint.
    pub indexer_zmq_endpoint: Option<String>,
    /// Must-match summary to send at registration.
    pub runtime_summary: RuntimeConfigSummary,
}

impl HubHandshake {
    /// Whether `key` is in the effective set.
    pub fn has(&self, key: FeatureKey) -> bool {
        self.effective.contains(&key)
    }
}

/// Run the handshake. `page_size` is the worker layout block size; `disagg` is
/// the per-instance disagg config (carries the CD role).
pub async fn resolve(
    hub: &LeaderHubConfig,
    page_size: usize,
    block_layout: BlockLayoutMode,
    disagg: Option<&DisaggConfig>,
) -> Result<HubHandshake> {
    // Parse requested labels up front (a bad label is always a hard error).
    let mut requested: HashSet<FeatureKey> = HashSet::new();
    for label in &hub.features {
        let key = FeatureKey::from_label(label)
            .ok_or_else(|| anyhow::anyhow!("unknown feature {label:?} in leader.hub.features"))?;
        if !CONNECTOR_CAPS.contains(&key) {
            bail!(
                "feature {label:?} in leader.hub.features is not connector-selectable \
                 (choose from {:?})",
                CONNECTOR_CAPS.map(|k| k.as_str())
            );
        }
        requested.insert(key);
    }
    let explicit = !requested.is_empty();

    let runtime_summary = RuntimeConfigSummary {
        block_size: Some(page_size),
        block_layout: Some(block_layout),
    };

    // Fetch the aggregate. In auto mode a failure disables hub features
    // (best-effort); in explicit mode it is fatal.
    let aggregate = match fetch_aggregate(&hub.url).await {
        Ok(a) => a,
        Err(e) => {
            if explicit {
                return Err(e);
            }
            tracing::warn!(hub = %hub.url, error = %e, "hub /v1/config unavailable; hub features disabled");
            return Ok(HubHandshake {
                url: hub.url.clone(),
                effective: HashSet::new(),
                indexer_zmq_endpoint: None,
                runtime_summary,
            });
        }
    };

    let enabled: HashSet<FeatureKey> = aggregate.features.iter().map(|f| f.key).collect();

    // Candidate features: explicit set (each must be offered by the hub) or, in
    // auto mode, the connector's capabilities intersected with the hub's set.
    let candidates: Vec<FeatureKey> = if explicit {
        for key in &requested {
            if !enabled.contains(key) {
                bail!("leader.hub.features requires {key} but the hub does not offer it");
            }
        }
        requested.iter().copied().collect()
    } else {
        CONNECTOR_CAPS
            .iter()
            .copied()
            .filter(|k| enabled.contains(k))
            .collect()
    };

    // Validate each candidate: its dependencies are enabled on the hub, a
    // disagg role is present for CD, and its must-match fields agree with the
    // hub primary. Any of these would otherwise be rejected at registration —
    // which must NOT fail startup in auto mode. So drop the feature (auto) or
    // hard-fail with a clear reason (explicit), *before* registering.
    let primary = &aggregate.primary;
    let mut effective: HashSet<FeatureKey> = HashSet::new();
    for key in candidates {
        match unsatisfiable(&aggregate, key, &enabled, &runtime_summary, primary, disagg) {
            None => {
                effective.insert(key);
            }
            Some(reason) if explicit => {
                bail!("leader.hub feature {key} cannot be satisfied: {reason}");
            }
            Some(reason) => {
                tracing::warn!(
                    feature = %key,
                    reason,
                    "dropping hub feature (auto mode)"
                );
            }
        }
    }

    // ConditionalDisagg co-registers P2P (its dependency); reflect that in the
    // effective set so callers see p2p as active. The init dispatch prioritizes
    // the CD path, so this never double-wires.
    if effective.contains(&FeatureKey::ConditionalDisagg) {
        effective.insert(FeatureKey::P2P);
    }

    let indexer_zmq_endpoint = if effective.contains(&FeatureKey::Indexer) {
        indexer_endpoint(&aggregate, page_size)
    } else {
        None
    };

    Ok(HubHandshake {
        url: hub.url.clone(),
        effective,
        indexer_zmq_endpoint,
        runtime_summary,
    })
}

async fn fetch_aggregate(hub_url: &str) -> Result<HubConfigResponse> {
    let url = format!("{}/v1/config", hub_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("hub {url} returned {}", resp.status());
    }
    resp.json::<HubConfigResponse>()
        .await
        .context("decoding hub /v1/config")
}

/// Returns `Some(reason)` when the connector's `summary` disagrees with the hub
/// `primary` on a must-match field that `key` requires (per the aggregate's
/// per-feature `config_requirements`). `None` ⇒ compatible.
fn must_match_mismatch(
    aggregate: &HubConfigResponse,
    key: FeatureKey,
    summary: &RuntimeConfigSummary,
    primary: &PrimaryConfig,
) -> Option<String> {
    // Fold in the requirements of `key` AND its (transitive) dependencies —
    // e.g. conditional_disagg owns no must-match fields itself, but its P2P
    // dependency requires block_size + block_layout. P2P is co-registered, so
    // the connector must honor P2P's requirements when CD is effective.
    let reqs = required_with_deps(aggregate, key);
    if reqs.block_size
        && let Some(want) = primary.block_size
        && summary.block_size != Some(want)
    {
        return Some(format!(
            "block_size: hub requires {want}, connector has {:?}",
            summary.block_size
        ));
    }
    if reqs.block_layout && summary.block_layout != Some(primary.block_layout) {
        return Some(format!(
            "block_layout: hub requires {:?}, connector has {:?}",
            primary.block_layout, summary.block_layout
        ));
    }
    None
}

/// Union of `key`'s own `config_requirements` with those of its transitive
/// dependencies.
fn required_with_deps(aggregate: &HubConfigResponse, key: FeatureKey) -> FeatureConfigRequirements {
    let mut acc = FeatureConfigRequirements::default();
    let mut seen: HashSet<FeatureKey> = HashSet::new();
    let mut stack = vec![key];
    while let Some(k) = stack.pop() {
        if !seen.insert(k) {
            continue;
        }
        if let Some(fd) = aggregate.features.iter().find(|f| f.key == k) {
            acc.block_size |= fd.config_requirements.block_size;
            acc.block_layout |= fd.config_requirements.block_layout;
            stack.extend(fd.dependencies.iter().copied());
        }
    }
    acc
}

/// Reasons a candidate feature cannot be satisfied against this hub. `None` ⇒
/// the feature is usable. Used uniformly for auto (drop) and explicit (fail).
fn unsatisfiable(
    aggregate: &HubConfigResponse,
    key: FeatureKey,
    enabled: &HashSet<FeatureKey>,
    summary: &RuntimeConfigSummary,
    primary: &PrimaryConfig,
    disagg: Option<&DisaggConfig>,
) -> Option<String> {
    // Every (transitive) dependency must be enabled on the hub — they are
    // co-registered (e.g. P2P with ConditionalDisagg) and the hub rejects a
    // declared feature whose manager is absent.
    for dep in transitive_deps(aggregate, key) {
        if !enabled.contains(&dep) {
            return Some(format!("dependency {dep} is not enabled on the hub"));
        }
    }
    // ConditionalDisagg needs a per-instance role.
    if key == FeatureKey::ConditionalDisagg && disagg.is_none() {
        return Some("requires a `disagg` role but none is configured".to_string());
    }
    // Must-match fields (folding in dependency requirements) must agree.
    must_match_mismatch(aggregate, key, summary, primary)
}

/// Transitive dependency closure of `key` (excluding `key` itself).
fn transitive_deps(aggregate: &HubConfigResponse, key: FeatureKey) -> HashSet<FeatureKey> {
    let mut seen: HashSet<FeatureKey> = HashSet::new();
    let mut stack = vec![key];
    while let Some(k) = stack.pop() {
        if let Some(fd) = aggregate.features.iter().find(|f| f.key == k) {
            for dep in &fd.dependencies {
                if seen.insert(*dep) {
                    stack.push(*dep);
                }
            }
        }
    }
    seen
}

/// Pull the KV-index ZMQ endpoint from the aggregate descriptor, guarding that
/// the hub's block size matches this worker's page size.
fn indexer_endpoint(aggregate: &HubConfigResponse, page_size: usize) -> Option<String> {
    let descriptor: &FeatureDescriptor = aggregate
        .features
        .iter()
        .find(|f| f.key == FeatureKey::Indexer)?;
    if let Some(bs) = descriptor.config.get("block_size").and_then(|v| v.as_u64())
        && bs as usize != page_size
    {
        tracing::warn!(
            hub_block_size = bs,
            page_size,
            "indexer block_size mismatch; publisher disabled (registration will also reject)"
        );
        return None;
    }
    let endpoint = descriptor
        .config
        .get("zmq_endpoint")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if endpoint.is_empty() {
        tracing::warn!("indexer advertised an empty zmq_endpoint; publisher disabled");
        return None;
    }
    Some(endpoint.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use kvbm_config::DisaggregationRole;
    use kvbm_hub::{
        ConditionalDisaggManager, FeatureManager, HubServer, IndexerManager, P2pManager,
        PrimaryConfig,
    };

    const BS: usize = 16;

    async fn start_hub(features: &[&str]) -> HubServer {
        let mut b = kvbm_hub::create_server_builder()
            .bind_addr("127.0.0.1".parse().unwrap())
            .discovery_port(0)
            .control_port(0)
            .heartbeat_interval(Duration::from_secs(3600))
            .heartbeat_max_failures(u32::MAX)
            .registration_ttl(Duration::from_secs(3600))
            .primary_config(PrimaryConfig {
                block_size: Some(BS),
                ..Default::default()
            });
        if features.contains(&"p2p") {
            b = b.add_feature_manager(Arc::new(P2pManager::new()) as Arc<dyn FeatureManager>);
        }
        if features.contains(&"conditional_disagg") {
            b = b.add_feature_manager(
                Arc::new(ConditionalDisaggManager::new()) as Arc<dyn FeatureManager>
            );
        }
        if features.contains(&"indexer") {
            let kv = IndexerManager::new(
                1024,
                BS,
                Some("tcp://127.0.0.1:0".to_string()),
                Some("127.0.0.1".to_string()),
            )
            .unwrap();
            b = b.add_feature_manager(Arc::new(kv) as Arc<dyn FeatureManager>);
        }
        b.serve().await.unwrap()
    }

    fn hub_cfg(url: &str, features: &[&str]) -> LeaderHubConfig {
        LeaderHubConfig {
            url: url.to_string(),
            features: features.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn disagg() -> DisaggConfig {
        DisaggConfig {
            role: DisaggregationRole::Decode,
            max_inflight_remote_prefill_tokens: usize::MAX,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_intersects_caps_and_gates_cd_on_role() {
        let server = start_hub(&["p2p", "conditional_disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // Auto + disagg present → both indexer and conditional_disagg.
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        assert!(h.indexer_zmq_endpoint.is_some());

        // Auto + no disagg role → CD dropped (best-effort), indexer stays.
        let h = resolve(&hub_cfg(&url, &[]), BS, BlockLayoutMode::Operational, None)
            .await
            .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_subset_validation() {
        let server = start_hub(&["p2p", "conditional_disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // Explicit indexer only.
        let h = resolve(
            &hub_cfg(&url, &["indexer"]),
            BS,
            BlockLayoutMode::Operational,
            None,
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        // Explicit conditional_disagg with a role validates its p2p dep.
        assert!(
            resolve(
                &hub_cfg(&url, &["conditional_disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
            )
            .await
            .is_ok()
        );

        // Explicit conditional_disagg without a role → hard-fail.
        assert!(
            resolve(
                &hub_cfg(&url, &["conditional_disagg"]),
                BS,
                BlockLayoutMode::Operational,
                None,
            )
            .await
            .is_err()
        );

        // Unknown label → hard-fail.
        assert!(
            resolve(
                &hub_cfg(&url, &["bogus"]),
                BS,
                BlockLayoutMode::Operational,
                None
            )
            .await
            .is_err()
        );
        // p2p is now standalone-selectable (remote-controllable peer).
        let h = resolve(
            &hub_cfg(&url, &["p2p"]),
            BS,
            BlockLayoutMode::Operational,
            None,
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::P2P));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_feature_or_dep_not_offered_fails() {
        // Hub offers only indexer → requesting conditional_disagg fails.
        let server = start_hub(&["indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());
        assert!(
            resolve(
                &hub_cfg(&url, &["conditional_disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
            )
            .await
            .is_err()
        );
        server.shutdown().await.unwrap();

        // Hub offers conditional_disagg but not its p2p dependency → fails.
        let server = start_hub(&["conditional_disagg"]).await;
        let url = format!("http://{}", server.discovery_addr());
        assert!(
            resolve(
                &hub_cfg(&url, &["conditional_disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
            )
            .await
            .is_err()
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_drops_incompatible_block_size_explicit_fails() {
        let server = start_hub(&["p2p", "conditional_disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // page_size 32 != hub primary block_size 16 → must-match features
        // (indexer, conditional_disagg) are incompatible.
        // Auto: dropped, no error.
        let h = resolve(
            &hub_cfg(&url, &[]),
            32,
            BlockLayoutMode::Operational,
            Some(&disagg()),
        )
        .await
        .unwrap();
        assert!(
            h.effective.is_empty(),
            "auto mode must drop incompatible features, got {:?}",
            h.effective
        );

        // Explicit: hard-fail at startup (before registration).
        assert!(
            resolve(
                &hub_cfg(&url, &["indexer"]),
                32,
                BlockLayoutMode::Operational,
                None,
            )
            .await
            .is_err()
        );

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_drops_cd_when_p2p_dependency_missing() {
        // Hub offers conditional_disagg but not its p2p dependency. Auto mode
        // must drop CD (best-effort), never hard-fail startup.
        let server = start_hub(&["conditional_disagg"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
        )
        .await
        .unwrap();
        assert!(
            !h.has(FeatureKey::ConditionalDisagg),
            "CD must be dropped when its p2p dep is missing"
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unreachable_hub_auto_best_effort_explicit_hard_fail() {
        // Nothing listening on this port.
        let url = "http://127.0.0.1:1";

        // Auto → best-effort: no features, no error.
        let h = resolve(&hub_cfg(url, &[]), BS, BlockLayoutMode::Operational, None)
            .await
            .unwrap();
        assert!(h.effective.is_empty());

        // Explicit → hard-fail.
        assert!(
            resolve(
                &hub_cfg(url, &["indexer"]),
                BS,
                BlockLayoutMode::Operational,
                None,
            )
            .await
            .is_err()
        );
    }
}
