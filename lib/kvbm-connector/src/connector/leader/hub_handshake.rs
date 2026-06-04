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
use kvbm_config::{
    BlockLayoutMode, DisaggConfig, DisaggregationRole, LeaderHubConfig, RemoteSearch,
};
use kvbm_hub::{
    FeatureConfigRequirements, FeatureDescriptor, FeatureKey, HubConfigResponse, PrimaryConfig,
    RuntimeConfigSummary,
};

const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Hub features a connector can participate in (all standalone-selectable).
/// `disagg` additionally depends on `p2p` (co-registered). `prefill_router`
/// is participation-by-advertisement: a prefill worker that intends to be
/// reached by the hub's router pushes a `Feature::PrefillRouter(...)` payload
/// at registration; decode workers leave it untouched.
pub const CONNECTOR_CAPS: [FeatureKey; 4] = [
    FeatureKey::Indexer,
    FeatureKey::P2P,
    FeatureKey::ConditionalDisagg,
    FeatureKey::PrefillRouter,
];

/// `(parent, dep)` pairs where the connector auto-co-registers `dep` at
/// registration time whenever `parent` is in the effective set, so the
/// dep need not appear in `leader.hub.features` for the connector to
/// declare it. Used by [`resolve`] to relax the explicit-mode
/// dep-completeness check.
///
/// Today the only entry is `(ConditionalDisagg, P2P)` — see
/// `super::p2p::wire::wire_p2p`, which always prepends `Feature::P2P`
/// to the register payload. `PrefillRouter` has *no* auto-co-registered
/// parent: the connector only adds its payload inside the CD wiring
/// block, so `prefill_router` without `disagg` in
/// `leader.hub.features` is a real misconfiguration the handshake
/// rejects.
const AUTO_COREGISTERED_DEPS: &[(FeatureKey, FeatureKey)] =
    &[(FeatureKey::ConditionalDisagg, FeatureKey::P2P)];

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

/// Validate that remote search, if requested, can actually run against the
/// resolved hub. Remote search needs **both**:
///   * the `indexer` feature — to discover which instance holds a block, and
///   * the `p2p` feature — the transfer control plane it pulls over (and which
///     installs the session factory + registers this instance as a
///     hub-discoverable peer). Without P2P, discovery would succeed but
///     `open_session` / `pull_from_session` would fail at request time with
///     `NotInitialized` — so we reject the misconfiguration up front.
///
/// Both must be effective (and a hub must be configured, i.e. `handshake` is
/// `Some`). Returns an invalid-configuration error otherwise.
///
/// A `None` or `enabled = false` `remote_search` is always `Ok` (feature off).
pub fn validate_remote_search_availability(
    remote_search: Option<&RemoteSearch>,
    handshake: Option<&HubHandshake>,
) -> Result<()> {
    if !remote_search.is_some_and(|r| r.enabled) {
        return Ok(());
    }
    let has_indexer = handshake.is_some_and(|h| h.has(FeatureKey::Indexer));
    let has_p2p = handshake.is_some_and(|h| h.has(FeatureKey::P2P));
    if has_indexer && has_p2p {
        return Ok(());
    }
    bail!(
        "invalid configuration: remote_search is enabled but the hub does not offer both \
         required features (indexer effective: {has_indexer}, p2p effective: {has_p2p}). \
         Remote search discovers holders via `indexer` and pulls over `p2p`; enable both \
         on the hub (or disable remote_search)."
    )
}

/// Worker-side capability flags consulted by the handshake. Separates
/// what the connector *can* do at registration time from what the hub
/// *offers*, so [`unsatisfiable`] can drop / reject features the
/// connector won't actually be able to fulfill. Empty today — the
/// prefill-router velo backend is wired post-handshake by the embedding
/// host (e.g. `kvbm.hub.try_wrap_engine`), so there is no
/// handshake-time signal to gate on.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerCapabilities {}

/// Run the handshake. `page_size` is the worker layout block size; `disagg` is
/// the per-instance disagg config (carries the CD role); `caps` describes
/// what the worker itself can fulfill at registration time.
pub async fn resolve(
    hub: &LeaderHubConfig,
    page_size: usize,
    block_layout: BlockLayoutMode,
    disagg: Option<&DisaggConfig>,
    caps: WorkerCapabilities,
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
        // Some dep edges are auto-co-registered by the connector at
        // registration time and do not need to appear in
        // `leader.hub.features`. Today the only such edge is CD → P2P
        // (see `super::p2p::wire::wire_p2p`, which always prepends
        // `Feature::P2P` to the register payload). Expand the
        // requested set with these implicits before the generic
        // dep-completeness check so `--features disagg` (without p2p)
        // keeps working.
        for (parent, auto_dep) in AUTO_COREGISTERED_DEPS {
            if requested.contains(parent) {
                requested.insert(*auto_dep);
            }
        }
        // After auto-co-registration, every remaining transitive dep
        // must also be in the requested set. The connector's init
        // paths key off the literal effective set (e.g. the
        // `Feature::PrefillRouter` payload is only added inside the
        // CD wiring block, which itself requires
        // `handshake.has(ConditionalDisagg)`), so a missing dep
        // produces a worker that silently registers nothing. The
        // hub's server-side `validate_register` already mirrors this
        // rule (`Feature::X requires Feature::Y to also be
        // declared`); catching it client-side gives a clearer error
        // and avoids a wasted register round-trip.
        for key in &requested.clone() {
            // Direct-deps-first BFS so the error names the most
            // actionable feature: telling a user who requested
            // `prefill_router` to "add disagg" is right; "add p2p" is
            // technically true (via the dep chain) but misleading,
            // since adding disagg auto-co-registers p2p.
            if let Some(missing) = first_missing_dep(&aggregate, *key, &requested) {
                bail!(
                    "leader.hub.features requires {key} but its dependency {missing} is not \
                     also in leader.hub.features (the connector wires features off the \
                     literal list; add {missing} or drop {key})"
                );
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
        match unsatisfiable(
            &aggregate,
            key,
            &enabled,
            &runtime_summary,
            primary,
            disagg,
            &caps,
        ) {
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
    // e.g. disagg owns no must-match fields itself, but its P2P
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
    _caps: &WorkerCapabilities,
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
    // PrefillRouter is participation-by-advertisement. The connector only
    // pushes the `Feature::PrefillRouter` payload at registration when
    // the worker is a Prefill disagg role. Decode workers have nothing
    // to advertise; drop / hard-fail accordingly. Backend availability
    // (HTTP env vars vs the in-process `PrefillRouterHandler` runtime)
    // is gated at the actual push site (init.rs for HTTP, kvbm.hub for
    // velo) — the handshake only enforces the role precondition.
    if key == FeatureKey::PrefillRouter {
        let Some(d) = disagg else {
            return Some("requires a `disagg` role to participate".to_string());
        };
        if !matches!(d.role, DisaggregationRole::Prefill) {
            return Some(
                "the worker is a Decode role and cannot advertise a prefill backend".to_string(),
            );
        }
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

/// BFS for the first transitive dep of `key` that is not in `requested`.
/// Direct-deps-first so the error names the most actionable feature for
/// the operator to add (e.g. for `requested = {prefill_router}` this
/// returns `disagg`, not the transitively-reachable `p2p`).
fn first_missing_dep(
    aggregate: &HubConfigResponse,
    key: FeatureKey,
    requested: &HashSet<FeatureKey>,
) -> Option<FeatureKey> {
    let mut seen: HashSet<FeatureKey> = HashSet::new();
    let mut frontier: std::collections::VecDeque<FeatureKey> =
        std::collections::VecDeque::from([key]);
    while let Some(k) = frontier.pop_front() {
        if let Some(fd) = aggregate.features.iter().find(|f| f.key == k) {
            for dep in &fd.dependencies {
                if !requested.contains(dep) {
                    return Some(*dep);
                }
                if seen.insert(*dep) {
                    frontier.push_back(*dep);
                }
            }
        }
    }
    None
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
        PrefillRouterManager, PrimaryConfig, SelectorConfig,
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
        if features.contains(&"disagg") {
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
        if features.contains(&"prefill_router") {
            let router = PrefillRouterManager::new(SelectorConfig {
                per_worker_concurrency: 4,
                block_size: BS,
            });
            b = b.add_feature_manager(router as Arc<dyn FeatureManager>);
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
            ..Default::default()
        }
    }

    fn disagg_prefill() -> DisaggConfig {
        DisaggConfig {
            role: DisaggregationRole::Prefill,
            ..Default::default()
        }
    }

    /// Test helper for tests that historically distinguished
    /// "router-ready" from "not ready". The struct is empty today, so
    /// this is equivalent to `WorkerCapabilities::default()`; kept as
    /// a named alias so the test bodies stay readable.
    fn caps_router_ready() -> WorkerCapabilities {
        WorkerCapabilities::default()
    }

    fn handshake_with(effective: &[FeatureKey]) -> HubHandshake {
        HubHandshake {
            url: "http://hub".to_string(),
            effective: effective.iter().copied().collect(),
            indexer_zmq_endpoint: None,
            runtime_summary: RuntimeConfigSummary {
                block_size: Some(BS),
                block_layout: Some(BlockLayoutMode::Operational),
            },
        }
    }

    #[test]
    fn remote_search_disabled_is_always_ok() {
        // No remote_search → Ok regardless of handshake state.
        assert!(validate_remote_search_availability(None, None).is_ok());
        assert!(
            validate_remote_search_availability(None, Some(&handshake_with(&[]))).is_ok(),
            "absent remote_search must not error even without indexer"
        );
        // Present but disabled → Ok even without indexer.
        let disabled = RemoteSearch::default();
        assert!(
            validate_remote_search_availability(Some(&disabled), Some(&handshake_with(&[])))
                .is_ok(),
            "disabled remote_search must not error even without indexer"
        );
    }

    #[test]
    fn remote_search_requires_a_hub() {
        let rs = RemoteSearch {
            enabled: true,
            min_remote_tokens: None,
        };
        // Requested but no hub configured (handshake None) → invalid config.
        assert!(validate_remote_search_availability(Some(&rs), None).is_err());
    }

    #[test]
    fn remote_search_requires_indexer_and_p2p() {
        let rs = RemoteSearch {
            enabled: true,
            min_remote_tokens: None,
        };
        // P2P but no indexer → invalid (can't discover holders).
        let p2p_only = handshake_with(&[FeatureKey::P2P]);
        assert!(validate_remote_search_availability(Some(&rs), Some(&p2p_only)).is_err());

        // Indexer but no p2p → invalid (can't pull; open/pull would fail at
        // request time with NotInitialized).
        let indexer_only = handshake_with(&[FeatureKey::Indexer]);
        assert!(validate_remote_search_availability(Some(&rs), Some(&indexer_only)).is_err());

        // Both effective → Ok.
        let both = handshake_with(&[FeatureKey::Indexer, FeatureKey::P2P]);
        assert!(validate_remote_search_availability(Some(&rs), Some(&both)).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_prefill_router_standalone_resolves() {
        // `prefill_router` is now a self-contained velo target that
        // does not carry a hard `ConditionalDisagg` dep. The connector
        // handshake accepts an explicit `["prefill_router"]` list as
        // long as the worker is a Prefill role with a ready backend —
        // this is the path the `python -m kvbm.vllm.prefill` entrypoint
        // takes after pulling its config from the hub.
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &["prefill_router"]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg_prefill()),
            caps_router_ready(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::PrefillRouter));
        assert!(!h.has(FeatureKey::ConditionalDisagg));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_disagg_without_p2p_still_works_via_auto_coreg() {
        // The connector auto-co-registers P2P inside `wire_p2p`
        // whenever CD is being registered. `--features disagg` (no
        // p2p) must therefore keep working — otherwise this fix
        // regresses an existing supported pattern.
        let server = start_hub(&["p2p", "disagg"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &["disagg"]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::ConditionalDisagg));
        assert!(h.has(FeatureKey::P2P));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_prefill_router_with_disagg_and_ready_backend_works() {
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &["disagg", "prefill_router"]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg_prefill()),
            caps_router_ready(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::PrefillRouter));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        assert!(h.has(FeatureKey::P2P));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_prefill_router_prefill_role_resolves_regardless_of_env() {
        // The connector handshake no longer gates `prefill_router` on
        // HTTP env vars — the velo `PrefillRouterHandler` is wired
        // post-handshake by `kvbm.hub.try_wrap_engine`, so the
        // handshake only enforces the role precondition. Default caps
        // (no env vars) for a Prefill role must keep
        // `prefill_router` in the effective set.
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &["disagg", "prefill_router"]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg_prefill()),
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::PrefillRouter));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_prefill_router_with_decode_role_is_rejected() {
        // Decode workers have nothing to advertise to the router; an
        // explicit `prefill_router` selection must be rejected so the
        // operator notices the misconfiguration.
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let result = resolve(
            &hub_cfg(&url, &["disagg", "prefill_router"]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
            caps_router_ready(),
        )
        .await;
        assert!(
            result.is_err(),
            "decode-role explicit prefill_router must hard-fail"
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_keeps_prefill_router_for_prefill_role_no_env() {
        // Auto mode + Prefill role + hub offers router + no env vars
        // → keep `prefill_router` effective. The connector's init.rs
        // only pushes the Http advertisement when env vars are set;
        // without them no Feature::PrefillRouter is sent by the
        // connector, but the velo `PrefillRouterHandler` still gets
        // wired downstream and registers on its own velo instance.
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg_prefill()),
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::PrefillRouter));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_picks_up_prefill_router_from_hub() {
        // Without PrefillRouter in CONNECTOR_CAPS the auto-mode intersection
        // would drop the hub's prefill_router feature and the
        // connector's init.rs gate (handshake.has(PrefillRouter)) would
        // always return false — so workers would never advertise even
        // when the hub was started with --prefill-router.
        let server = start_hub(&["p2p", "disagg", "prefill_router"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg_prefill()),
            caps_router_ready(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::PrefillRouter));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_intersects_caps_and_gates_cd_on_role() {
        let server = start_hub(&["p2p", "disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // Auto + disagg present → both indexer and disagg.
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(h.has(FeatureKey::ConditionalDisagg));
        assert!(h.indexer_zmq_endpoint.is_some());

        // Auto + no disagg role → CD dropped (best-effort), indexer stays.
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            None,
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_subset_validation() {
        let server = start_hub(&["p2p", "disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // Explicit indexer only.
        let h = resolve(
            &hub_cfg(&url, &["indexer"]),
            BS,
            BlockLayoutMode::Operational,
            None,
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::Indexer));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        // Explicit disagg with a role validates its p2p dep.
        assert!(
            resolve(
                &hub_cfg(&url, &["disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
                WorkerCapabilities::default(),
            )
            .await
            .is_ok()
        );

        // Explicit disagg without a role → hard-fail.
        assert!(
            resolve(
                &hub_cfg(&url, &["disagg"]),
                BS,
                BlockLayoutMode::Operational,
                None,
                WorkerCapabilities::default(),
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
                None,
                WorkerCapabilities::default(),
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
            WorkerCapabilities::default(),
        )
        .await
        .unwrap();
        assert!(h.has(FeatureKey::P2P));
        assert!(!h.has(FeatureKey::ConditionalDisagg));

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_feature_or_dep_not_offered_fails() {
        // Hub offers only indexer → requesting disagg fails.
        let server = start_hub(&["indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());
        assert!(
            resolve(
                &hub_cfg(&url, &["disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
                WorkerCapabilities::default(),
            )
            .await
            .is_err()
        );
        server.shutdown().await.unwrap();

        // Hub offers disagg but not its p2p dependency → fails.
        let server = start_hub(&["disagg"]).await;
        let url = format!("http://{}", server.discovery_addr());
        assert!(
            resolve(
                &hub_cfg(&url, &["disagg"]),
                BS,
                BlockLayoutMode::Operational,
                Some(&disagg()),
                WorkerCapabilities::default(),
            )
            .await
            .is_err()
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_drops_incompatible_block_size_explicit_fails() {
        let server = start_hub(&["p2p", "disagg", "indexer"]).await;
        let url = format!("http://{}", server.discovery_addr());

        // page_size 32 != hub primary block_size 16 → must-match features
        // (indexer, disagg) are incompatible.
        // Auto: dropped, no error.
        let h = resolve(
            &hub_cfg(&url, &[]),
            32,
            BlockLayoutMode::Operational,
            Some(&disagg()),
            WorkerCapabilities::default(),
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
                WorkerCapabilities::default(),
            )
            .await
            .is_err()
        );

        server.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auto_drops_cd_when_p2p_dependency_missing() {
        // Hub offers disagg but not its p2p dependency. Auto mode
        // must drop CD (best-effort), never hard-fail startup.
        let server = start_hub(&["disagg"]).await;
        let url = format!("http://{}", server.discovery_addr());
        let h = resolve(
            &hub_cfg(&url, &[]),
            BS,
            BlockLayoutMode::Operational,
            Some(&disagg()),
            WorkerCapabilities::default(),
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
        let h = resolve(
            &hub_cfg(url, &[]),
            BS,
            BlockLayoutMode::Operational,
            None,
            WorkerCapabilities::default(),
        )
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
                WorkerCapabilities::default(),
            )
            .await
            .is_err()
        );
    }
}
