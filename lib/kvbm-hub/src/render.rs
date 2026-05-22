// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendering logic for the `kvbmctl` CLI.
//!
//! Turns a hub [`HubConfigResponse`] aggregate into a paste-ready vLLM CLI
//! fragment: the `--block-size` / `--max-model-len` flags the hub is
//! authoritative for, followed by a `--kv-transfer-config '{…}'` blob whose
//! `kv_connector_extra_config` is filled in from the hub's `primary` config and
//! enabled feature set.
//!
//! Pure (no IO): the binary fetches the aggregate over HTTP and hands it here,
//! and tests construct an aggregate in-process. The rendered
//! `kv_connector_extra_config` is round-tripped through
//! [`KvbmConfig::from_figment_with_json_for_leader`] /
//! [`KvbmConfig::from_figment_with_json_for_worker`] before it is emitted, so
//! whatever `kvbmctl` prints is guaranteed to parse identically in the
//! connector.

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};

use crate::protocol::{FeatureKey, HubConfigResponse};
use kvbm_config::overrides::{apply_overrides, deep_merge, validate_extra_config};

/// Feature keys a connector can actually participate in. `ConnectorControl` is
/// hub infrastructure (no client-side `Feature` payload) so it is never
/// selectable or rendered into `leader.hub.features`.
const SELECTABLE: &[FeatureKey] = &[
    FeatureKey::Indexer,
    FeatureKey::P2P,
    FeatureKey::ConditionalDisagg,
];

/// CLI-supplied values that are not sourced from the hub aggregate.
#[derive(Debug, Clone)]
pub struct VllmRenderOptions {
    /// Requested feature subset (labels). Empty → render the hub's full
    /// enabled (selectable) set.
    pub features: Vec<String>,
    /// Conditional-disagg role (`prefill` / `decode`). Required iff
    /// `disagg` is in the effective set; rejected otherwise.
    pub role: Option<String>,
    /// `--kvbm <dotted.path>=<value>` overrides from the connector side.
    /// Precedence among free fields, lowest to highest: `build_extra_config`
    /// (structural fields from `aggregate.primary`), then `aggregate.base_config`
    /// (hub-side `--kvbm`/`--kvbm-config`), then the `kvbm_config` blob, then
    /// these `kvbm_overrides` (highest among free fields). The
    /// `authoritative_overlay` is applied last and always wins for must-match
    /// fields. The first path segment may be a figment profile (`default` /
    /// `leader` / `worker`) or a flat config key — both are accepted by the
    /// connector.
    pub kvbm_overrides: Vec<String>,
    /// `--kvbm-config '{json}'` blob from the connector side. Deep-merged below
    /// the individual `--kvbm` overrides; see `kvbm_overrides` for full
    /// precedence order.
    pub kvbm_config: Option<String>,
    /// vLLM `kv_connector` class name.
    pub kv_connector: String,
    /// vLLM `kv_role`.
    pub kv_role: String,
    /// vLLM `kv_load_failure_policy`.
    pub kv_load_failure_policy: String,
    /// vLLM `kv_connector_module_path`.
    pub kv_connector_module_path: String,
}

/// Resolve the effective feature set against the hub aggregate.
///
/// - `requested` empty → every enabled selectable feature the hub advertises.
/// - `requested` non-empty → validated as a subset of the enabled selectable
///   set, then dependency-closed using the aggregate's per-feature
///   `dependencies` (e.g. `disagg` pulls in `p2p`). Any unknown
///   label, unmet feature, or unmet dependency is a hard error.
///
/// Returns the effective keys in [`SELECTABLE`] order (stable output).
pub fn resolve_features(
    aggregate: &HubConfigResponse,
    requested: &[String],
) -> Result<Vec<FeatureKey>> {
    let enabled: Vec<FeatureKey> = aggregate
        .features
        .iter()
        .map(|d| d.key)
        .filter(|k| SELECTABLE.contains(k))
        .collect();

    // Initial selection: the explicit subset (validated ⊆ enabled), or the
    // hub's full enabled set when none was requested.
    let mut sel: Vec<FeatureKey> = if requested.is_empty() {
        enabled.clone()
    } else {
        let mut s = Vec::new();
        for label in requested {
            let key = FeatureKey::from_label(label)
                .filter(|k| SELECTABLE.contains(k))
                .ok_or_else(|| {
                    anyhow!(
                        "unknown feature {label:?}; valid features: {}",
                        selectable_labels()
                    )
                })?;
            if !enabled.contains(&key) {
                bail!(
                    "feature {:?} is not enabled on the hub (enabled: {})",
                    key.as_str(),
                    enabled_labels(&enabled)
                );
            }
            if !s.contains(&key) {
                s.push(key);
            }
        }
        s
    };

    // Dependency closure + dep-enabled validation. Runs for BOTH paths: an
    // omitted (full-enabled) set that advertises a feature whose dependency the
    // hub did not enable is rejected here rather than emitted as a config the
    // connector would later hard-fail on.
    let mut i = 0;
    while i < sel.len() {
        let key = sel[i];
        if let Some(desc) = aggregate.features.iter().find(|d| d.key == key) {
            for dep in &desc.dependencies {
                if !enabled.contains(dep) {
                    bail!(
                        "feature {:?} depends on {:?}, which is not enabled on the hub",
                        key.as_str(),
                        dep.as_str()
                    );
                }
                if !sel.contains(dep) {
                    sel.push(*dep);
                }
            }
        }
        i += 1;
    }

    // Normalise to SELECTABLE order for deterministic output.
    let mut effective = sel;
    effective.sort_by_key(|k| SELECTABLE.iter().position(|s| s == k).unwrap_or(usize::MAX));
    effective.dedup();
    Ok(effective)
}

/// Build the `kv_connector_extra_config` object from the hub aggregate and the
/// resolved feature set, before any `--kvbm` overrides are applied.
fn build_extra_config(
    aggregate: &HubConfigResponse,
    effective: &[FeatureKey],
    hub_url: &str,
    role: Option<&str>,
) -> Result<Value> {
    let primary = &aggregate.primary;
    let cd = effective.contains(&FeatureKey::ConditionalDisagg);
    match (cd, role) {
        (true, None) => {
            bail!("disagg is enabled but no --role was given; pass --role prefill|decode")
        }
        (false, Some(r)) => {
            bail!("--role {r:?} was given but disagg is not in the effective feature set")
        }
        _ => {}
    }

    // default profile — applies to every role.
    let default = json!({ "block_layout": primary.block_layout.as_label() });

    // leader profile — the connector's hub handshake + advisory sizing live here.
    let mut leader = Map::new();
    let features: Vec<&str> = effective.iter().map(|k| k.as_str()).collect();
    leader.insert(
        "hub".to_string(),
        json!({ "url": hub_url, "features": features }),
    );
    if let Some(msl) = primary.max_seq_len {
        leader.insert("max_seq_len".to_string(), json!(msl));
    }
    // Advisory G2 sizing: explicit block count wins over GiB, mirroring the
    // connector's `cache.host` precedence.
    if let Some(blocks) = primary.g2_blocks {
        leader.insert(
            "cache".to_string(),
            json!({ "host": { "num_blocks": blocks } }),
        );
    } else if let Some(gib) = primary.g2_memory_gib {
        leader.insert(
            "cache".to_string(),
            json!({ "host": { "cache_size_gb": gib } }),
        );
    }
    if let Some(r) = role {
        leader.insert("disagg".to_string(), json!({ "role": r }));
    }

    Ok(json!({ "default": default, "leader": Value::Object(leader) }))
}

/// The fields the hub is the source of truth for, as a minimal overlay
/// deep-merged over the post-override config so they always win:
/// - `default.block_layout` — must-match; the hub rejects a registrant whose
///   `block_layout` differs from its primary.
/// - `leader.hub.{url,features}` — the URL kvbmctl queried and the
///   dependency-closed feature set it validated against that hub.
/// - `leader.disagg.role` — only when `disagg` is effective. This
///   is re-applied here, **coupled to `features`**, so an override that drops
///   the disagg block (`--kvbm leader.disagg=…` / `--kvbm leader=…`) cannot
///   leave `disagg` in `features` without its required role — a
///   config the connector would hard-fail on. `disagg` is `Option`, so the
///   round-trip validator would otherwise accept the inconsistent blob.
///
/// Other free / advisory fields (cache sizing, tokio workers, `max_seq_len`,
/// nixl, `disagg.max_inflight_remote_prefill_tokens`) are deliberately absent
/// so `--kvbm` / `--kvbm-config` can still tune them — `disagg.role` merges in
/// over a partial `disagg` object without disturbing those sibling keys.
fn authoritative_overlay(
    aggregate: &HubConfigResponse,
    effective: &[FeatureKey],
    hub_url: &str,
    role: Option<&str>,
) -> Value {
    let features: Vec<&str> = effective.iter().map(|k| k.as_str()).collect();
    let mut leader = json!({ "hub": { "url": hub_url, "features": features } });
    if effective.contains(&FeatureKey::ConditionalDisagg)
        && let Some(r) = role
    {
        leader["disagg"] = json!({ "role": r });
    }
    json!({
        "default": { "block_layout": aggregate.primary.block_layout.as_label() },
        "leader": leader,
    })
}

/// Render the full vLLM CLI fragment for `--hub <hub_url>`.
///
/// Output is a single line: the hub-authoritative vLLM flags (`--block-size`,
/// `--max-model-len`) followed by `--kv-transfer-config '<compact json>'`.
pub fn render_vllm_cli(
    aggregate: &HubConfigResponse,
    hub_url: &str,
    opts: &VllmRenderOptions,
) -> Result<String> {
    let effective = resolve_features(aggregate, &opts.features)?;
    let mut extra = build_extra_config(aggregate, &effective, hub_url, opts.role.as_deref())?;
    // Apply the hub's base_config (operator `--kvbm`/`--kvbm-config` on the hub
    // binary) as the next layer. Per-connector overrides below win over these
    // hub-side defaults; hub-authoritative fields are re-applied last and always
    // win over everything.
    if !aggregate.base_config.is_null() {
        deep_merge(&mut extra, aggregate.base_config.clone());
    }
    apply_overrides(
        &mut extra,
        opts.kvbm_config.as_deref(),
        &opts.kvbm_overrides,
    )?;
    // Re-apply the hub-authoritative fields *last* so `--kvbm` / `--kvbm-config`
    // (on either the hub or the connector side) can tune free fields (tokio
    // workers, cache sizing, nixl, …) but can never clobber what the hub owns:
    // `block_layout` is must-match (a mismatch makes the connector's registration
    // get rejected), `hub.{url,features}` are the validated identity kvbmctl
    // resolved against this hub, and `disagg.role` is re-coupled to `features`
    // so an override can't strip the role off a disagg config.
    deep_merge(
        &mut extra,
        authoritative_overlay(aggregate, &effective, hub_url, opts.role.as_deref()),
    );
    validate_extra_config(&extra)?;

    let transfer_config = json!({
        "kv_connector": opts.kv_connector,
        "kv_role": opts.kv_role,
        "kv_load_failure_policy": opts.kv_load_failure_policy,
        "kv_connector_module_path": opts.kv_connector_module_path,
        "kv_connector_extra_config": extra,
    });

    let mut parts: Vec<String> = Vec::new();
    if let Some(bs) = aggregate.primary.block_size {
        parts.push(format!("--block-size {bs}"));
    }
    if let Some(msl) = aggregate.primary.max_seq_len {
        parts.push(format!("--max-model-len {msl}"));
    }
    let json = serde_json::to_string(&transfer_config).expect("serialize transfer config");
    parts.push(format!(
        "--kv-transfer-config {}",
        shell_single_quote(&json)
    ));
    Ok(parts.join(" "))
}

/// POSIX single-quote a token so the rendered fragment is safe to paste into a
/// shell even when an operator-supplied value (`--hub`, `--kvbm-config`,
/// `--kv-connector-module-path`, …) contains a `'`. Wraps in `'…'` and rewrites
/// each inner `'` as `'\''`.
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn selectable_labels() -> String {
    SELECTABLE
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn enabled_labels(enabled: &[FeatureKey]) -> String {
    if enabled.is_empty() {
        return "<none>".to_string();
    }
    enabled
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{FeatureConfigRequirements, FeatureDescriptor, PrimaryConfig};
    use kvbm_common::BlockLayoutMode;

    fn descriptor(key: FeatureKey, deps: Vec<FeatureKey>) -> FeatureDescriptor {
        FeatureDescriptor {
            key,
            dependencies: deps,
            config_requirements: FeatureConfigRequirements::default(),
            config: Value::Null,
        }
    }

    fn aggregate(features: Vec<FeatureDescriptor>) -> HubConfigResponse {
        HubConfigResponse {
            primary: PrimaryConfig {
                block_size: Some(16),
                max_seq_len: Some(8192),
                block_layout: BlockLayoutMode::Operational,
                g2_memory_gib: Some(4.0),
                g2_blocks: None,
                advertise_host: None,
            },
            features,
            base_config: Value::Null,
        }
    }

    fn opts() -> VllmRenderOptions {
        VllmRenderOptions {
            features: vec![],
            role: None,
            kvbm_overrides: vec![],
            kvbm_config: None,
            kv_connector: "DynamoConnector".to_string(),
            kv_role: "kv_both".to_string(),
            kv_load_failure_policy: "recompute".to_string(),
            kv_connector_module_path: "kvbm.v2.vllm.connector".to_string(),
        }
    }

    // Pull `kv_connector_extra_config` back out of the rendered CLI string for
    // structural assertions.
    fn extract_config(cli: &str) -> Value {
        let marker = "--kv-transfer-config '";
        let start = cli.find(marker).expect("has kv-transfer-config") + marker.len();
        let end = cli.rfind('\'').expect("closing quote");
        let tc: Value = serde_json::from_str(&cli[start..end]).expect("valid transfer json");
        tc["kv_connector_extra_config"].clone()
    }

    #[test]
    fn indexer_only_renders_and_validates() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let cli = render_vllm_cli(&agg, "http://hub:1337", &opts()).unwrap();
        assert!(cli.contains("--block-size 16"));
        assert!(cli.contains("--max-model-len 8192"));
        let extra = extract_config(&cli);
        assert_eq!(extra["leader"]["hub"]["url"], "http://hub:1337");
        assert_eq!(extra["leader"]["hub"]["features"], json!(["indexer"]));
        assert_eq!(extra["default"]["block_layout"], "operational");
        // Advisory G2 GiB lands under leader.cache.host.
        assert_eq!(extra["leader"]["cache"]["host"]["cache_size_gb"], 4.0);
    }

    #[test]
    fn cd_pulls_in_p2p_and_requires_role() {
        let agg = aggregate(vec![
            descriptor(FeatureKey::P2P, vec![]),
            descriptor(FeatureKey::ConditionalDisagg, vec![FeatureKey::P2P]),
        ]);
        let mut o = opts();
        o.features = vec!["disagg".to_string()];
        // No role → hard error.
        assert!(render_vllm_cli(&agg, "http://hub:1337", &o).is_err());
        // With role → p2p is dependency-closed into the feature list.
        o.role = Some("prefill".to_string());
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        let feats = extra["leader"]["hub"]["features"].as_array().unwrap();
        assert!(feats.contains(&json!("p2p")));
        assert!(feats.contains(&json!("disagg")));
        assert_eq!(extra["leader"]["disagg"]["role"], "prefill");
    }

    #[test]
    fn cd_role_survives_override_that_drops_the_disagg_block() {
        // An override that overwrites the whole `leader.disagg` (or `leader`)
        // subtree must not be able to leave disagg in `features`
        // without its role: the authoritative overlay re-couples role to
        // features. Without that, `disagg: Option` lets the inconsistent blob
        // pass validation and the connector hard-fails at runtime.
        let agg = aggregate(vec![
            descriptor(FeatureKey::P2P, vec![]),
            descriptor(FeatureKey::ConditionalDisagg, vec![FeatureKey::P2P]),
        ]);
        let mut o = opts();
        o.features = vec!["disagg".to_string()];
        o.role = Some("prefill".to_string());
        // Overwrite the entire `disagg` object with one that has no role.
        o.kvbm_overrides =
            vec!["leader.disagg={\"max_inflight_remote_prefill_tokens\":7}".to_string()];
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        let feats = extra["leader"]["hub"]["features"].as_array().unwrap();
        assert!(feats.contains(&json!("disagg")));
        // Role is restored by the authoritative overlay; the free sibling key
        // the operator set is preserved.
        assert_eq!(extra["leader"]["disagg"]["role"], "prefill");
        assert_eq!(
            extra["leader"]["disagg"]["max_inflight_remote_prefill_tokens"],
            json!(7)
        );
    }

    #[test]
    fn role_without_cd_is_rejected() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.role = Some("decode".to_string());
        assert!(render_vllm_cli(&agg, "http://hub:1337", &o).is_err());
    }

    #[test]
    fn unmet_feature_is_rejected() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.features = vec!["p2p".to_string()];
        assert!(render_vllm_cli(&agg, "http://hub:1337", &o).is_err());
    }

    #[test]
    fn omitted_features_rejects_unmet_dependency() {
        // Hub advertises disagg (dep: p2p) but NOT p2p. With
        // --features omitted, the full enabled set must still be dep-validated,
        // so this is rejected rather than emitted with an unmet dependency.
        let agg = aggregate(vec![descriptor(
            FeatureKey::ConditionalDisagg,
            vec![FeatureKey::P2P],
        )]);
        let mut o = opts();
        o.role = Some("prefill".to_string());
        let err = resolve_features(&agg, &o.features).unwrap_err();
        assert!(err.to_string().contains("depends on"), "got: {err}");
    }

    #[test]
    fn hub_base_config_flows_through_to_rendered_connector_config() {
        // Hub-side --kvbm overrides (stored in aggregate.base_config) must
        // appear in the rendered kv_connector_extra_config so connectors inherit
        // them without each launcher repeating the same flags.
        let mut agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        agg.base_config = json!({
            "leader": { "tokio": { "worker_threads": 2 } },
            "worker": { "nixl": { "backends": { "UCX": {}, "POSIX": {} } } }
        });
        let cli = render_vllm_cli(&agg, "http://hub:1337", &opts()).unwrap();
        let extra = extract_config(&cli);
        assert_eq!(extra["leader"]["tokio"]["worker_threads"], json!(2));
        assert!(extra["worker"]["nixl"]["backends"]["UCX"].is_object());
    }

    #[test]
    fn per_connector_override_wins_over_hub_base_config() {
        // Per-connector --kvbm has higher precedence than the hub's base_config.
        let mut agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        agg.base_config = json!({ "leader": { "tokio": { "worker_threads": 2 } } });
        let mut o = opts();
        o.kvbm_overrides = vec!["leader.tokio.worker_threads=8".to_string()];
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        assert_eq!(extra["leader"]["tokio"]["worker_threads"], json!(8));
    }

    #[test]
    fn authoritative_fields_win_over_hub_base_config() {
        // Hub-side base_config must not be able to clobber hub-authoritative
        // fields (block_layout, hub.url, hub.features).
        let mut agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        agg.base_config = json!({
            "default": { "block_layout": "universal" },
            "leader": { "hub": { "url": "http://evil:9999", "features": ["p2p"] } }
        });
        let cli = render_vllm_cli(&agg, "http://hub:1337", &opts()).unwrap();
        let extra = extract_config(&cli);
        assert_eq!(extra["default"]["block_layout"], "operational");
        assert_eq!(extra["leader"]["hub"]["url"], "http://hub:1337");
        assert_eq!(extra["leader"]["hub"]["features"], json!(["indexer"]));
    }

    #[test]
    fn kvbm_override_wins_on_free_fields_and_is_typed() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.kvbm_overrides = vec!["leader.tokio.worker_threads=2".to_string()];
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        // Free field: override wins, parsed as an integer (not a string).
        assert_eq!(extra["leader"]["tokio"]["worker_threads"], json!(2));
    }

    #[test]
    fn authoritative_fields_cannot_be_clobbered() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.features = vec!["indexer".to_string()];
        // Try to clobber every hub-authoritative field via both override paths.
        o.kvbm_config = Some(
            r#"{"default":{"block_layout":"universal"},"leader":{"hub":{"features":["p2p"]}}}"#
                .to_string(),
        );
        o.kvbm_overrides = vec!["leader.hub.url=http://evil:9999".to_string()];
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        // Hub stays the source of truth regardless of overrides.
        assert_eq!(extra["default"]["block_layout"], "operational");
        assert_eq!(extra["leader"]["hub"]["url"], "http://hub:1337");
        assert_eq!(extra["leader"]["hub"]["features"], json!(["indexer"]));
    }

    #[test]
    fn kvbm_config_blob_merges_then_dotted_override_wins() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.kvbm_config = Some(r#"{"leader":{"tokio":{"worker_threads":8}}}"#.to_string());
        o.kvbm_overrides = vec!["leader.tokio.worker_threads=3".to_string()];
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        assert_eq!(extra["leader"]["tokio"]["worker_threads"], json!(3));
    }

    #[test]
    fn invalid_override_fails_validation() {
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        // worker_threads must be an integer; a string fails the connector parser.
        o.kvbm_overrides = vec!["leader.tokio.worker_threads=notanint".to_string()];
        assert!(render_vllm_cli(&agg, "http://hub:1337", &o).is_err());
    }

    #[test]
    fn worker_nixl_override_validates() {
        // Mirrors the skill blobs' worker profile (`nixl.backends.{UCX,POSIX}`):
        // the round-trip through `_for_worker` must accept it.
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let mut o = opts();
        o.kvbm_config =
            Some(r#"{"worker":{"nixl":{"backends":{"UCX":{},"POSIX":{}}}}}"#.to_string());
        let cli = render_vllm_cli(&agg, "http://hub:1337", &o).unwrap();
        let extra = extract_config(&cli);
        assert!(extra["worker"]["nixl"]["backends"]["UCX"].is_object());
    }

    #[test]
    fn shell_single_quote_escapes_inner_quotes() {
        // No inner quote → wrap only.
        assert_eq!(shell_single_quote("abc"), "'abc'");
        // Inner quote → POSIX `'\''` rewrite, so the whole token is one safe arg.
        assert_eq!(shell_single_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn render_shell_escapes_operator_quote() {
        // A `'` in the operator-supplied --hub value lands in leader.hub.url and
        // must not break out of the quoted `--kv-transfer-config` token.
        let agg = aggregate(vec![descriptor(FeatureKey::Indexer, vec![])]);
        let cli = render_vllm_cli(&agg, "http://h'x:1337", &opts()).unwrap();
        // The raw inner quote never appears unescaped; only the `'\''` form does.
        assert!(cli.contains("'\\''"));
    }
}
