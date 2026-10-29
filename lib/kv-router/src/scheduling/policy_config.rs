// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer};
use thiserror::Error;
use tokio::time::Instant;

use super::config::RouterQueuePolicy;
use crate::plugins::request_classifier::RawRequestClassifierConfig;
use crate::plugins::worker_selection::{WorkerCapacity, WorkerLoadInput};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::WorkerLoadProjection;
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::request_classifier::RequestClassifierConfig;
use crate::plugins::worker_selection::RawWorkerSelectionConfig;
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::worker_selection::{WorkerSelectionConfig, WorkerSelectionInstance};

const SYNTHETIC_POLICY_CLASS: &str = "default";

#[derive(Debug, Error)]
pub enum RouterPolicyConfigError {
    #[error("failed to read router policy config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse router policy config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub queue_policy: RouterQueuePolicy,
    pub quantum: usize,
    /// Time-to-first-token budget in whole milliseconds, or `None` when the class sets none.
    /// An EDF class orders a request without its own deadline by its router ingress time
    /// plus this budget. Nothing rejects or expires a request when the budget passes.
    pub slo_ttft_ms: Option<NonZeroU64>,
    /// The limits this class waits on.
    pub busy_thresholds: BusyThresholds,
    pub request_queue_limit_per_worker: Option<usize>,
    pub raw_isl_token_queue_limit_per_worker: Option<usize>,
    pub cached_token_queue_limit_per_worker: Option<usize>,
}

impl PolicyClassConfig {
    pub fn queueing_enabled(&self) -> bool {
        self.busy_thresholds.is_configured()
    }

    /// Any zero per-worker limit means `queue_rejection` rejects every queued
    /// entry (`current >= limit` holds at zero usage), so the class can never
    /// hold a queued request and must keep the direct admission path.
    pub fn never_queues(&self) -> bool {
        self.request_queue_limit_per_worker == Some(0)
            || self.raw_isl_token_queue_limit_per_worker == Some(0)
            || self.cached_token_queue_limit_per_worker == Some(0)
    }

    /// When this class's TTFT budget runs out for a request that entered the router queue at
    /// `ingress_at`, or `None` when the class sets no budget.
    pub fn slo_ttft_due_at(
        &self,
        ingress_at: Instant,
    ) -> Result<Option<Instant>, SloTtftOutOfRange> {
        let Some(slo_ttft_ms) = self.slo_ttft_ms else {
            return Ok(None);
        };
        ingress_at
            .checked_add(Duration::from_millis(slo_ttft_ms.get()))
            .map(Some)
            .ok_or_else(|| SloTtftOutOfRange {
                policy_class: self.name.clone(),
                slo_ttft_ms: slo_ttft_ms.get(),
            })
    }
}

/// A class TTFT budget the monotonic clock cannot represent from a request's ingress time.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "policy class {policy_class:?} slo_ttft_ms {slo_ttft_ms} from router ingress exceeds \
     the monotonic clock range"
)]
pub struct SloTtftOutOfRange {
    pub policy_class: String,
    pub slo_ttft_ms: u64,
}

/// Supplies the host's live reject thresholds. Selection skips a worker whose every rank is
/// past them, and rejects the request when no eligible worker remains.
pub type RejectThresholdsProvider = Arc<dyn Fn() -> BusyThresholds + Send + Sync + 'static>;

/// Per-rank load limits. A rank is full once its booked load passes any configured limit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BusyThresholds {
    /// Booked prefill tokens above which a rank is full.
    // TODO(v1.7): Remove the absolute threshold; use `prefill_busy_threshold_frac`.
    pub prefill_busy_threshold: Option<usize>,
    /// Fraction of the rank's `max_num_batched_tokens` above which it is full. When the
    /// rank does not report it, the fraction is taken of
    /// [`DEFAULT_MAX_BATCHED_TOKENS`](crate::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS).
    pub prefill_busy_threshold_frac: Option<f64>,
    /// Fraction of the rank's `total_kv_blocks` above which the KV blocks its active
    /// requests hold make it full. A rank that does not report `total_kv_blocks` skips it.
    pub decode_busy_threshold_frac: Option<f64>,
}

impl BusyThresholds {
    /// Whether any limit is set.
    pub fn is_configured(&self) -> bool {
        self.prefill_busy_threshold.is_some()
            || self.prefill_busy_threshold_frac.is_some()
            || self.decode_busy_threshold_frac.is_some()
    }

    /// Whether a rank with this `LOAD` row and capacity is within every configured limit.
    pub fn has_room(&self, load: &WorkerLoadInput, capacity: WorkerCapacity) -> bool {
        let tokens = load.active_prefill_tokens();
        let prefill_budget = capacity.prefill_budget();
        let absolute_full = self
            .prefill_busy_threshold
            .is_some_and(|threshold| tokens > threshold);
        let fractional_full = self
            .prefill_busy_threshold_frac
            .is_some_and(|threshold| (tokens as f64) > threshold * (prefill_budget as f64));
        let decode_full = self
            .decode_busy_threshold_frac
            .zip(capacity.total_kv_blocks())
            .is_some_and(|(threshold, total_kv_blocks)| {
                (load.active_decode_blocks() as f64) > threshold * (total_kv_blocks as f64)
            });
        !(absolute_full || fractional_full || decode_full)
    }

    /// Whether no rank of `worker_id` has room, judged on the `LOAD` rows in `loads`. A rank
    /// without a projection counts as idle. The policy-class queue and offline replay share
    /// this loop so they enumerate ranks and read rows the same way.
    pub fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let capacity = WorkerCapacity::of(config);
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        (start..end).all(|dp_rank| {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            let load = WorkerLoadInput::for_worker(loads, worker, WorkerLoadProjection::default);
            !self.has_room(&load, capacity)
        })
    }
}

/// Ranks past a set of busy thresholds, counted per worker while a request's load is
/// projected. Only a worker with at least one full rank has an entry, so the map stays empty
/// while no worker is near its limits and selection pays nothing to consult it.
#[derive(Debug, Default)]
pub(crate) struct FullRankCounts {
    limits: BusyThresholds,
    full_ranks: FxHashMap<WorkerId, u32>,
}

impl FullRankCounts {
    /// Start a pass judged against `limits`, forgetting the previous pass.
    pub(crate) fn reset(&mut self, limits: BusyThresholds) {
        self.limits = limits;
        self.full_ranks.clear();
    }

    /// Forget what the current pass recorded; it starts over from the first rank.
    pub(crate) fn clear(&mut self) {
        self.full_ranks.clear();
    }

    /// Judge one rank's projected load against the limits and count it when it has no room.
    #[inline]
    pub(crate) fn record(
        &mut self,
        worker: WorkerWithDpRank,
        capacity: WorkerCapacity,
        projection: &WorkerLoadProjection,
    ) {
        if !self
            .limits
            .has_room(&WorkerLoadInput::from_projection(*projection), capacity)
        {
            *self.full_ranks.entry(worker.worker_id).or_insert(0) += 1;
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.full_ranks.is_empty()
    }

    /// Whether every one of the worker's `dp_size` ranks was counted full.
    #[inline]
    pub(crate) fn worker_is_full(&self, worker_id: WorkerId, dp_size: u32) -> bool {
        self.full_ranks
            .get(&worker_id)
            .is_some_and(|&full| full >= dp_size)
    }
}

/// One flat set of policy classes, each selected directly by its name.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    class_indices: HashMap<String, usize>,
    default_class_index: usize,
}

impl PolicyProfile {
    pub fn synthetic(
        router_queue_threshold: Option<f64>,
        router_queue_policy: RouterQueuePolicy,
    ) -> Self {
        let class = PolicyClassConfig {
            name: SYNTHETIC_POLICY_CLASS.to_string(),
            queue_policy: router_queue_policy,
            quantum: 1,
            slo_ttft_ms: None,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold: None,
                prefill_busy_threshold_frac: router_queue_threshold,
                decode_busy_threshold_frac: None,
            },
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        };
        Self {
            class_indices: HashMap::from([(class.name.clone(), 0)]),
            classes: vec![class],
            default_class_index: 0,
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    pub fn default_class(&self) -> &PolicyClassConfig {
        &self.classes[self.default_class_index]
    }

    /// Resolves a requested class name to its queue. A missing or unknown name selects the
    /// default class.
    pub fn resolve_class_index(&self, requested: Option<&str>) -> usize {
        // TODO: Add bounded observability for unknown requested policy values.
        requested
            .and_then(|name| self.resolve_class_index_strict(name))
            .unwrap_or(self.default_class_index)
    }

    pub fn class(&self, index: usize) -> &PolicyClassConfig {
        &self.classes[index]
    }

    /// Resolves a configured class name, or `None` for any other name.
    pub(crate) fn resolve_class_index_strict(&self, requested: &str) -> Option<usize> {
        self.class_indices.get(requested).copied()
    }
}

/// Process-wide cache and tracking settings, separate from policy parameters.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(transparent)]
pub(super) struct RouterSettings(HashMap<String, serde_json::Value>);

impl RouterSettings {
    pub(super) fn apply(&self, config: &mut super::config::KvRouterConfig) -> Result<(), String> {
        // Deserialize into each field's existing type, including nullable fields.
        // Do not round-trip KvRouterConfig: it has process-local, non-wire fields.
        macro_rules! apply_fields {
            ($($field:ident),* $(,)?) => {
                for (name, value) in &self.0 {
                    match name.as_str() {
                        $(stringify!($field) => {
                            config.$field = serde_json::from_value(value.clone())
                                .map_err(|error| format!("router.{name}: {error}"))?;
                        })*
                        _ => return Err(format!("unknown router setting: {name}")),
                    }
                }
            };
        }
        apply_fields!(
            host_cache_hit_weight,
            disk_cache_hit_weight,
            use_kv_events,
            router_replica_sync,
            router_track_active_blocks,
            router_track_output_blocks,
            router_assume_kv_reuse,
            router_track_prefill_tokens,
            router_tracking_hash,
            router_tracking_key_file,
            router_tracking_key_id,
            router_prefill_load_model,
            router_ttl_secs,
            router_approximate_cache_policy,
            router_event_threads,
            use_remote_indexer,
            serve_indexer,
            enable_session_prefix_index,
            shared_cache_type,
            router_predicted_ttl_secs,
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicyConfig {
    router: Option<RouterSettings>,
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
    worker_selection: Option<WorkerSelectionConfig>,
    request_classifier: Option<RequestClassifierConfig>,
}

impl RouterPolicyConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RouterPolicyConfigError> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|source| RouterPolicyConfigError::Read {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_yaml(&contents).map_err(|error| match error {
            RouterPolicyConfigError::Parse { source, .. } => RouterPolicyConfigError::Parse {
                path: path.display().to_string(),
                source,
            },
            other => other,
        })
    }

    pub fn from_yaml(contents: &str) -> Result<Self, RouterPolicyConfigError> {
        let raw: RawRouterPolicyConfig =
            serde_yaml::from_str(contents).map_err(|source| RouterPolicyConfigError::Parse {
                path: "<inline>".to_string(),
                source,
            })?;
        raw.resolve()
    }

    pub fn resolve_profile(
        &self,
        model_name: Option<&str>,
        fallback_threshold: Option<f64>,
        fallback_policy: RouterQueuePolicy,
    ) -> PolicyProfile {
        // Model profiles replace the root wholesale; the synthetic profile is
        // constructed only when neither configured profile applies.
        model_name
            .and_then(|name| self.models.get(name))
            .or(self.root.as_ref())
            .cloned()
            .unwrap_or_else(|| PolicyProfile::synthetic(fallback_threshold, fallback_policy))
    }

    pub(super) fn router(&self) -> Option<&RouterSettings> {
        self.router.as_ref()
    }

    /// Returns the process-wide worker-selection policy configuration, if present.
    pub fn worker_selection(&self) -> Option<&WorkerSelectionConfig> {
        self.worker_selection.as_ref()
    }

    /// Returns the process-wide request-classifier plugin configuration, if present.
    pub fn request_classifier(&self) -> Option<&RequestClassifierConfig> {
        self.request_classifier.as_ref()
    }

    /// Whether this document configures queue policy profiles.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouterPolicyConfig {
    router: Option<RouterSettings>,
    default_policy_class: Option<String>,
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    worker_selection: Option<RawWorkerSelectionConfig>,
    request_classifier: Option<RawRequestClassifierConfig>,
    #[serde(default, deserialize_with = "key_present")]
    default_policy_family: bool,
    #[serde(default, deserialize_with = "key_present")]
    uncached_isl_buckets: bool,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        reject_removed_keys(
            "root",
            [
                ("default_policy_family", self.default_policy_family),
                ("uncached_isl_buckets", self.uncached_isl_buckets),
            ],
        )?;
        if let Some(router) = &self.router {
            router
                .apply(&mut super::config::KvRouterConfig::default())
                .map_err(RouterPolicyConfigError::Validation)?;
        }
        let root = match (self.default_policy_class, self.policy_classes) {
            (default_policy_class, Some(policy_classes)) => Some(resolve_profile(
                default_policy_class,
                policy_classes,
                "root",
            )?),
            (Some(_), None) => {
                return Err(RouterPolicyConfigError::Validation(
                    "root default_policy_class requires policy_classes".to_string(),
                ));
            }
            (None, None) => None,
        };

        let mut models = HashMap::with_capacity(self.models.len());
        for (model_name, profile) in self.models {
            if model_name.is_empty() {
                return Err(RouterPolicyConfigError::Validation(
                    "model profile name must not be empty".to_string(),
                ));
            }
            let location = format!("model {model_name:?}");
            reject_removed_keys(
                &location,
                [
                    ("default_policy_family", profile.default_policy_family),
                    ("uncached_isl_buckets", profile.uncached_isl_buckets),
                ],
            )?;
            let resolved = resolve_profile(
                profile.default_policy_class,
                profile.policy_classes,
                &location,
            )?;
            models.insert(model_name, resolved);
        }

        let worker_selection = match self.worker_selection {
            Some(config) => Some(config.resolve()?),
            None => None,
        };

        let request_classifier = self
            .request_classifier
            .map(|config| config.resolve())
            .transpose()?;
        if self.router.is_none()
            && root.is_none()
            && models.is_empty()
            && worker_selection.is_none()
            && request_classifier.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define router settings, a root profile, at least one model profile, worker_selection, or request_classifier".to_string(),
            ));
        }

        Ok(RouterPolicyConfig {
            router: self.router,
            root,
            models,
            worker_selection,
            request_classifier,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyProfile {
    default_policy_class: Option<String>,
    policy_classes: Vec<RawPolicyClassConfig>,
    #[serde(default, deserialize_with = "key_present")]
    default_policy_family: bool,
    #[serde(default, deserialize_with = "key_present")]
    uncached_isl_buckets: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    #[serde(default)]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    slo_ttft_ms: Option<u64>,
    prefill_busy_threshold: Option<usize>,
    prefill_busy_threshold_frac: Option<f64>,
    decode_busy_threshold_frac: Option<f64>,
    request_queue_limit_per_worker: Option<usize>,
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    cached_token_queue_limit_per_worker: Option<usize>,
    #[serde(default, deserialize_with = "key_present")]
    policy_family: bool,
    #[serde(default, deserialize_with = "key_present")]
    cache_bucket: bool,
}

/// Records that a key is present, whatever its value, including `null` or an empty value.
fn key_present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    serde::de::IgnoredAny::deserialize(deserializer).map(|_| true)
}

/// Rejects keys of the removed policy-family and cache-bucket hierarchy.
fn reject_removed_keys(
    location: &str,
    keys: [(&str, bool); 2],
) -> Result<(), RouterPolicyConfigError> {
    match keys.into_iter().find(|&(_, present)| present) {
        Some((key, _)) => Err(RouterPolicyConfigError::Validation(format!(
            "{location} {key} was removed; select policy classes directly by name and set \
             default_policy_class to the default class"
        ))),
        None => Ok(()),
    }
}

fn resolve_profile(
    default_policy_class: Option<String>,
    raw_classes: Vec<RawPolicyClassConfig>,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    if raw_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let mut classes = Vec::with_capacity(raw_classes.len());
    let mut class_indices = HashMap::with_capacity(raw_classes.len());
    for raw in raw_classes {
        let class = resolve_policy_class(raw, location)?;
        if class_indices
            .insert(class.name.clone(), classes.len())
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                class.name
            )));
        }
        classes.push(class);
    }

    let default_class_index = match default_policy_class {
        Some(name) => class_indices.get(&name).copied().ok_or_else(|| {
            RouterPolicyConfigError::Validation(format!(
                "{location} default_policy_class {name:?} does not name a configured policy class"
            ))
        })?,
        None if classes.len() == 1 => 0,
        None => {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} default_policy_class is required with more than one policy class"
            )));
        }
    };

    Ok(PolicyProfile {
        classes,
        class_indices,
        default_class_index,
    })
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    location: &str,
) -> Result<PolicyClassConfig, RouterPolicyConfigError> {
    reject_removed_keys(
        &format!("{location} policy class {:?}", raw.name),
        [
            ("policy_family", raw.policy_family),
            ("cache_bucket", raw.cache_bucket),
        ],
    )?;
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    if raw.queue_policy == RouterQueuePolicy::Lcfs {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} queue_policy must be edf, fcfs, or wspt",
            raw.name
        )));
    }
    for (field, value) in [
        (
            "prefill_busy_threshold_frac",
            raw.prefill_busy_threshold_frac,
        ),
        ("decode_busy_threshold_frac", raw.decode_busy_threshold_frac),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} {field} must be finite and non-negative",
                raw.name
            )));
        }
    }
    let slo_ttft_ms = match raw.slo_ttft_ms {
        None => None,
        Some(slo_ttft_ms) => Some(NonZeroU64::new(slo_ttft_ms).ok_or_else(|| {
            RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} slo_ttft_ms must be greater than zero",
                raw.name
            ))
        })?),
    };

    if raw.prefill_busy_threshold.is_some() {
        tracing::warn!(
            policy_class = %raw.name,
            "prefill_busy_threshold is deprecated and will be removed in v1.7; use prefill_busy_threshold_frac"
        );
    }
    Ok(PolicyClassConfig {
        name: raw.name,
        queue_policy: raw.queue_policy,
        quantum: raw.quantum,
        slo_ttft_ms,
        busy_thresholds: BusyThresholds {
            prefill_busy_threshold: raw.prefill_busy_threshold,
            prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
            decode_busy_threshold_frac: raw.decode_busy_threshold_frac,
        },
        request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
        raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
        cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
    })
}

pub(crate) fn validate_identifier(
    name: &str,
    kind: &str,
    location: &str,
) -> Result<(), RouterPolicyConfigError> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Ok(());
    }

    Err(RouterPolicyConfigError::Validation(format!(
        "{location} {kind} name {name:?} must match [A-Za-z0-9_.-]+"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequences::WorkerLoadProjection;

    fn prefill_load(active_prefill_tokens: usize) -> WorkerLoadInput {
        WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_prefill_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn worker_selection_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  aggregated: example
  instances:
    - name: example
      type: example-policy
      parameters:
        score_weight: 1.0
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), Some("example"));
        let instance = selection.instance("example").unwrap();
        assert_eq!(instance.policy_type(), "example-policy");
        assert!(matches!(
            instance.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
        assert_eq!(
            config
                .resolve_profile(None, Some(2.0), RouterQueuePolicy::Wspt)
                .default_class()
                .queue_policy,
            RouterQueuePolicy::Wspt
        );
    }

    #[test]
    fn request_classifier_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
request_classifier:
  type: thunderagent
  parameters:
    pause_threshold: 0.9
"#,
        )
        .unwrap();

        let classifier = config.request_classifier().unwrap();
        assert_eq!(classifier.classifier_type(), "thunderagent");
        assert!(matches!(
            classifier.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
    }

    #[test]
    fn rejects_invalid_request_classifier_config() {
        for yaml in [
            r#"
request_classifier:
  type: default
"#,
            r#"
request_classifier:
  type: thunderagent
  parameters: 1
"#,
            r#"
request_classifier:
  type: ""
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn worker_selection_accepts_every_worker_type() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  prefill: prefill-policy
  decode: default
  encode: encode-policy
  instances:
    - name: prefill-policy
      type: cache-aware
      parameters: {}
    - name: encode-policy
      type: media-aware
      parameters: {}
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), None);
        assert_eq!(selection.prefill_instance(), Some("prefill-policy"));
        assert_eq!(selection.decode_instance(), Some("default"));
        assert_eq!(selection.encode_instance(), Some("encode-policy"));
    }

    #[test]
    fn rejects_invalid_worker_selection_config() {
        for yaml in [
            r#"
worker_selection: {}
"#,
            r#"
worker_selection:
  aggregated: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  prefill: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  decode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  encode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  default: present
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: default
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: alpha
      type: alpha
      parameters: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: root-default
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_class: model-direct
    policy_classes:
      - name: model-queued
        quantum: 4
        prefill_busy_threshold_frac: 0.0
      - name: model-direct
        quantum: 2
        request_queue_limit_per_worker: 0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-direct");
        assert_eq!(
            exact
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
        assert!(!exact.default_class().queueing_enabled());
        assert!(
            exact
                .class(exact.resolve_class_index(Some("model-queued")))
                .queueing_enabled()
        );
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Edf);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        for requested in [None, Some("unknown"), Some("root-default")] {
            assert_eq!(
                exact.class(exact.resolve_class_index(requested)).name,
                "model-direct",
                "{requested:?} must use the model's default class"
            );
        }

        let unmatched = config.resolve_profile(Some("other"), Some(3.0), RouterQueuePolicy::Fcfs);
        assert_eq!(unmatched.default_class().name, "root-default");
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold,
            Some(100)
        );
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
    }

    #[test]
    fn has_room_without_limits_or_reported_budget() {
        let mut limits = BusyThresholds::default();
        assert!(!limits.is_configured());
        assert!(limits.has_room(&prefill_load(usize::MAX), WorkerCapacity::default()));

        // An unreported budget is taken as DEFAULT_MAX_BATCHED_TOKENS.
        limits.prefill_busy_threshold_frac = Some(0.5);
        let half_default =
            (crate::plugins::worker_selection::inputs::DEFAULT_MAX_BATCHED_TOKENS / 2) as usize;
        assert!(limits.has_room(&prefill_load(half_default), WorkerCapacity::default()));
        assert!(!limits.has_room(&prefill_load(half_default + 1), WorkerCapacity::default()));
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let class = profile.default_class();
        assert!(class.queueing_enabled());

        let limits = class.busy_thresholds;
        let held = |active_decode_blocks| {
            WorkerLoadInput::from_projection(WorkerLoadProjection {
                active_decode_blocks,
                // The incoming request's blocks do not count toward the limit.
                additional_active_blocks: 1_000,
                ..Default::default()
            })
        };
        let kv = |total_kv_blocks| WorkerCapacity::new(None, total_kv_blocks);
        assert!(limits.has_room(&held(50), kv(Some(100))));
        assert!(!limits.has_room(&held(51), kv(Some(100))));
        // An unreported KV capacity skips the decode limit.
        assert!(limits.has_room(&held(1_000_000), kv(None)));

        let invalid = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: -0.1
"#,
        )
        .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("decode_busy_threshold_frac must be finite and non-negative")
        );
    }

    #[test]
    fn rootless_model_config_falls_back_for_unmatched_model() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  exact-model:
    policy_classes:
      - name: absolute
        quantum: 4
        prefill_busy_threshold: 10
        prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(7.0), RouterQueuePolicy::Wspt);
        let limits = exact.default_class().busy_thresholds;
        let budget = |tokens| WorkerCapacity::new(Some(tokens), None);
        assert!(!limits.has_room(&prefill_load(11), budget(10_000_000)));
        assert!(!limits.has_room(&prefill_load(6), budget(10)));
        assert!(limits.has_room(&prefill_load(5), budget(10)));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(
            fallback.default_class().queue_policy,
            RouterQueuePolicy::Wspt
        );
        // The synthetic class serves every request; only its own name is a valid override.
        assert_eq!(fallback.resolve_class_index(Some("absolute")), 0);
        assert_eq!(
            fallback.resolve_class_index_strict(SYNTHETIC_POLICY_CLASS),
            Some(0)
        );
        assert_eq!(fallback.resolve_class_index_strict("absolute"), None);
    }

    #[test]
    fn classes_resolve_by_name_and_fall_back_to_the_default() {
        let profile = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 800
  - name: standard
    quantum: 200
  - name: batch
    quantum: 100
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);

        // Every configured class is its own queue, selected by its exact name.
        for (index, name) in ["priority", "standard", "batch"].into_iter().enumerate() {
            assert_eq!(profile.class(index).name, name);
            assert_eq!(profile.resolve_class_index(Some(name)), index);
            assert_eq!(profile.resolve_class_index_strict(name), Some(index));
        }
        // Missing and unknown names use the configured default, not the first class.
        for requested in [None, Some(""), Some("unknown"), Some("Priority")] {
            assert_eq!(profile.resolve_class_index(requested), 1, "{requested:?}");
        }
        for requested in ["", "unknown", "Priority"] {
            assert_eq!(profile.resolve_class_index_strict(requested), None);
        }

        // One class is the default without default_policy_class, and may also name itself.
        for default in ["", "default_policy_class: only\n"] {
            let single = RouterPolicyConfig::from_yaml(&format!(
                "{default}policy_classes:\n  - name: only\n    quantum: 1\n"
            ))
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
            assert_eq!(single.default_class().name, "only");
            assert_eq!(single.resolve_class_index(Some("unknown")), 0);
        }
    }

    #[test]
    fn rejects_invalid_profiles() {
        for (yaml, expected) in [
            (
                "policy_classes: []\n",
                "root policy_classes must not be empty",
            ),
            (
                "policy_classes:\n  - name: first\n    quantum: 1\n  - name: second\n    quantum: 2\n",
                "root default_policy_class is required with more than one policy class",
            ),
            (
                "default_policy_class: missing\npolicy_classes:\n  - name: only\n    quantum: 1\n",
                "root default_policy_class \"missing\" does not name a configured policy class",
            ),
            (
                "default_policy_class: only\n",
                "root default_policy_class requires policy_classes",
            ),
            (
                "default_policy_class: same\npolicy_classes:\n  - name: same\n    quantum: 1\n  - name: same\n    quantum: 2\n",
                "root contains duplicate policy class \"same\"",
            ),
            (
                "policy_classes:\n  - name: invalid/name\n    quantum: 1\n",
                "root policy class name \"invalid/name\" must match [A-Za-z0-9_.-]+",
            ),
            (
                "policy_classes:\n  - name: zero\n    quantum: 0\n",
                "root policy class \"zero\" quantum must be greater than zero",
            ),
            (
                "policy_classes:\n  - name: lcfs\n    queue_policy: lcfs\n    quantum: 1\n",
                "root policy class \"lcfs\" queue_policy must be edf, fcfs, or wspt",
            ),
            (
                "models:\n  m:\n    policy_classes:\n      - name: a\n        quantum: 1\n      - name: b\n        quantum: 1\n",
                "model \"m\" default_policy_class is required with more than one policy class",
            ),
            (
                "models:\n  m:\n    default_policy_class: absent\n    policy_classes:\n      - name: a\n        quantum: 1\n",
                "model \"m\" default_policy_class \"absent\" does not name a configured policy class",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(yaml).unwrap_err().to_string();
            assert!(error.contains(expected), "{yaml}: {error}");
        }
    }

    #[test]
    fn removed_hierarchy_keys_are_rejected_even_when_null_or_empty() {
        for value in ["", "null", "\"\"", "[]", "standard"] {
            for key in ["default_policy_family", "uncached_isl_buckets"] {
                for (yaml, location) in [
                    (
                        format!("{key}: {value}\npolicy_classes:\n  - name: c\n    quantum: 1\n"),
                        "root".to_string(),
                    ),
                    (
                        format!(
                            "models:\n  m:\n    {key}: {value}\n    policy_classes:\n      - name: c\n        quantum: 1\n"
                        ),
                        "model \"m\"".to_string(),
                    ),
                ] {
                    let error = RouterPolicyConfig::from_yaml(&yaml)
                        .unwrap_err()
                        .to_string();
                    assert!(
                        error.contains(&format!("{location} {key} was removed")),
                        "{yaml}: {error}"
                    );
                }
            }
            for key in ["policy_family", "cache_bucket"] {
                for (yaml, location) in [
                    (
                        format!(
                            "policy_classes:\n  - name: c\n    quantum: 1\n    {key}: {value}\n"
                        ),
                        "root policy class \"c\"",
                    ),
                    (
                        format!(
                            "models:\n  m:\n    policy_classes:\n      - name: c\n        quantum: 1\n        {key}: {value}\n"
                        ),
                        "model \"m\" policy class \"c\"",
                    ),
                ] {
                    let error = RouterPolicyConfig::from_yaml(&yaml)
                        .unwrap_err()
                        .to_string();
                    assert!(
                        error.contains(&format!("{location} {key} was removed")),
                        "{yaml}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn slo_ttft_ms_is_an_optional_positive_millisecond_budget() {
        let slo_ttft_ms = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
                    .default_class()
                    .slo_ttft_ms
            })
        };

        assert_eq!(slo_ttft_ms("").unwrap(), None);
        assert_eq!(slo_ttft_ms("    slo_ttft_ms: null\n").unwrap(), None);
        assert_eq!(
            slo_ttft_ms("    slo_ttft_ms: 1\n").unwrap(),
            NonZeroU64::new(1)
        );
        assert_eq!(
            slo_ttft_ms(&format!("    slo_ttft_ms: {}\n", u64::MAX)).unwrap(),
            NonZeroU64::new(u64::MAX)
        );

        let error = slo_ttft_ms("    slo_ttft_ms: 0\n").unwrap_err().to_string();
        assert!(
            error.contains("root policy class \"c\" slo_ttft_ms must be greater than zero"),
            "{error}"
        );

        let class = |slo_ttft_ms| PolicyClassConfig {
            slo_ttft_ms: NonZeroU64::new(slo_ttft_ms),
            ..PolicyProfile::synthetic(None, RouterQueuePolicy::Edf)
                .default_class()
                .clone()
        };
        let out_of_range = SloTtftOutOfRange {
            policy_class: SYNTHETIC_POLICY_CLASS.to_string(),
            slo_ttft_ms: u64::MAX,
        };
        let ingress_at = Instant::now();
        assert_eq!(class(0).slo_ttft_due_at(ingress_at), Ok(None));
        assert_eq!(
            class(1500).slo_ttft_due_at(ingress_at),
            Ok(Some(ingress_at + Duration::from_millis(1500)))
        );
        // The largest budget is exact where the platform clock represents it, and an error
        // where it does not; it never becomes "no budget".
        assert_eq!(
            class(u64::MAX).slo_ttft_due_at(ingress_at),
            ingress_at
                .checked_add(Duration::from_millis(u64::MAX))
                .map(Some)
                .ok_or(out_of_range.clone())
        );

        // Walk to within two seconds of the end of the platform clock's range.
        let mut near_clock_end = ingress_at;
        let mut step = Duration::MAX;
        while step > Duration::from_secs(1) {
            match near_clock_end.checked_add(step) {
                Some(later) => near_clock_end = later,
                None => step /= 2,
            }
        }
        let error = class(u64::MAX).slo_ttft_due_at(near_clock_end).unwrap_err();
        assert_eq!(error, out_of_range);
        assert!(
            error
                .to_string()
                .contains("exceeds the monotonic clock range"),
            "{error}"
        );
        for malformed in [
            "-1",
            "1.5",
            "\"3000\"",
            "3000ms",
            "18446744073709551616",
            "[]",
        ] {
            assert!(
                matches!(
                    slo_ttft_ms(&format!("    slo_ttft_ms: {malformed}\n")),
                    Err(RouterPolicyConfigError::Parse { .. })
                ),
                "unexpectedly accepted slo_ttft_ms {malformed}"
            );
        }
    }

    #[test]
    fn omitted_queue_policy_selects_edf_and_explicit_policies_remain() {
        let queue_policy = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Wspt)
                    .default_class()
                    .queue_policy
            })
        };

        assert_eq!(queue_policy("").unwrap(), RouterQueuePolicy::Edf);
        for policy in [
            RouterQueuePolicy::Edf,
            RouterQueuePolicy::Fcfs,
            RouterQueuePolicy::Wspt,
        ] {
            assert_eq!(
                queue_policy(&format!("    queue_policy: {policy}\n")).unwrap(),
                policy
            );
        }
        assert!(queue_policy("    queue_policy: lcfs\n").is_err());

        // The synthetic class takes the router-wide policy, still including LCFS.
        for policy in [RouterQueuePolicy::default(), RouterQueuePolicy::Lcfs] {
            assert_eq!(
                PolicyProfile::synthetic(None, policy)
                    .default_class()
                    .queue_policy,
                policy
            );
        }
        assert_eq!(RouterQueuePolicy::default(), RouterQueuePolicy::Edf);
    }

    #[test]
    fn documented_sample_exercises_root_and_model_profiles() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let classes = root
            .classes()
            .iter()
            .map(|class| (class.name.as_str(), class.quantum, class.slo_ttft_ms))
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                ("priority", 800, NonZeroU64::new(1000)),
                ("standard", 200, NonZeroU64::new(3000)),
            ]
        );
        assert!(root.classes().iter().all(|class| {
            class.queue_policy == RouterQueuePolicy::Edf && class.queueing_enabled()
        }));
        assert_eq!(
            root.class(root.resolve_class_index(Some("priority")))
                .quantum,
            800
        );
        for requested in [None, Some("unknown")] {
            assert_eq!(
                root.class(root.resolve_class_index(requested)).name,
                "standard"
            );
        }

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Wspt,
        );
        assert_eq!(model.classes().len(), 1);
        assert_eq!(model.default_class().name, "standard");
        assert_eq!(model.default_class().quantum, 1000);
        assert_eq!(
            model
                .class(model.resolve_class_index(Some("priority")))
                .quantum,
            1000,
            "model profiles must completely replace root classes"
        );
    }

    #[test]
    fn prefill_budget_defaults_when_unreported() {
        use crate::plugins::worker_selection::inputs::DEFAULT_MAX_BATCHED_TOKENS;
        assert_eq!(
            WorkerCapacity::default().prefill_budget(),
            DEFAULT_MAX_BATCHED_TOKENS
        );
        assert_eq!(WorkerCapacity::new(Some(64), None).prefill_budget(), 64);
        assert_eq!(WorkerCapacity::default().max_num_batched_tokens(), None);
        // Every reported value round-trips; there is no sentinel.
        let max = WorkerCapacity::new(Some(u64::MAX), Some(u64::MAX));
        assert_eq!(max.max_num_batched_tokens(), Some(u64::MAX));
        assert_eq!(max.total_kv_blocks(), Some(u64::MAX));
    }

    #[test]
    fn unobserved_rank_row_is_unavailable_and_idle() {
        let seen = WorkerWithDpRank::new(1, 0);
        let unseen = WorkerWithDpRank::new(1, 1);
        let mut loads = FxHashMap::default();
        loads.insert(
            seen,
            WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        let unobserved = || WorkerLoadProjection {
            additional_active_blocks: 3,
            ..Default::default()
        };

        let row = WorkerLoadInput::for_worker(&loads, seen, unobserved);
        assert!(row.is_available());
        assert_eq!(row.active_prefill_tokens(), 7);

        let row = WorkerLoadInput::for_worker(&loads, unseen, unobserved);
        assert!(!row.is_available());
        assert_eq!(row.active_prefill_tokens(), 0);
        assert_eq!(row.decode_cost_blocks(), 3.0);
    }

    #[test]
    fn worker_is_full_only_when_every_rank_is_full() {
        let limits = BusyThresholds {
            prefill_busy_threshold: Some(10),
            ..Default::default()
        };
        let config = crate::test_utils::SimpleWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..Default::default()
        };
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 11,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();

        loads.insert(WorkerWithDpRank::new(5, 2), busy);
        // Rank 3 has no projection and counts as idle.
        assert!(!limits.worker_is_full(5, &config, &loads));

        loads.insert(WorkerWithDpRank::new(5, 3), busy);
        assert!(limits.worker_is_full(5, &config, &loads));

        // A rank outside the worker's range does not stand in for one inside it.
        loads.remove(&WorkerWithDpRank::new(5, 3));
        loads.insert(WorkerWithDpRank::new(5, 0), busy);
        assert!(!limits.worker_is_full(5, &config, &loads));
    }

    #[test]
    fn zero_capacity_is_unreported() {
        let zero = WorkerCapacity::new(Some(0), Some(0));
        assert_eq!(zero.max_num_batched_tokens(), None);
        assert_eq!(zero.total_kv_blocks(), None);
        assert_eq!(zero, WorkerCapacity::default());

        // vLLM under the Ray data-parallel backend registers `total_kv_blocks = 0` as a
        // placeholder; the decode limit skips such a rank instead of calling it full.
        let limits = BusyThresholds {
            decode_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        let held = WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_decode_blocks: 1_000,
            ..Default::default()
        });
        assert!(limits.has_room(&held, WorkerCapacity::new(None, Some(0))));
        assert!(!limits.has_room(&held, WorkerCapacity::new(None, Some(1_000))));

        // A zero prefill budget falls back to the default budget instead of being full at once.
        let limits = BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        assert!(limits.has_room(&prefill_load(1), WorkerCapacity::new(Some(0), None)));
    }
}
