// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;
use thiserror::Error;

use crate::plugins::request_classifier::RawRequestClassifierConfig;
use crate::plugins::worker_selection::{WorkerCapacity, WorkerCapacityExt};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::WorkerLoadProjection;
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::request_classifier::RequestClassifierConfig;
use crate::plugins::worker_selection::RawWorkerSelectionConfig;
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::worker_selection::{WorkerSelectionConfig, WorkerSelectionInstance};

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

/// Supplies the host's live reject thresholds. Selection skips a worker whose every rank is
/// past them, and rejects the request when no eligible worker remains.
pub use dynamo_scheduler::types::RejectThresholdsProvider;

pub use dynamo_slot_tracker::busy::BusyRanksExt;
pub use dynamo_slot_tracker::busy::{BusyThresholds, FullRankCounts};

pub use super::admission_config::{
    AdmissionConfigError, FamilyBucketTable, PolicyClassConfig, PolicyProfile, RouterQueuePolicy,
    SYNTHETIC_POLICY_CLASS,
};

impl From<AdmissionConfigError> for RouterPolicyConfigError {
    fn from(error: AdmissionConfigError) -> Self {
        match error {
            AdmissionConfigError::Validation(message) => Self::Validation(message),
        }
    }
}

/// Judges a whole worker against [`BusyThresholds`] from its runtime config. Lives beside
/// [`WorkerConfigLike`] because the slot tracker does not know the config trait.
pub trait BusyThresholdsExt {
    /// Whether no configured rank of `worker_id` has room, judged on the `LOAD` rows in `loads`.
    fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool;
}

impl BusyThresholdsExt for BusyThresholds {
    fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        self.worker_is_full_in_ranks(worker_id, start..end, WorkerCapacity::of(config), loads)
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
    default_policy_family: Option<String>,
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    uncached_isl_buckets: Option<Vec<RawUncachedIslBucket>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    worker_selection: Option<RawWorkerSelectionConfig>,
    request_classifier: Option<RawRequestClassifierConfig>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        if let Some(router) = &self.router {
            router
                .apply(&mut super::config::KvRouterConfig::default())
                .map_err(RouterPolicyConfigError::Validation)?;
        }
        let root = match (
            self.default_policy_family,
            self.policy_classes,
            self.uncached_isl_buckets,
        ) {
            (None, None, None) => None,
            (Some(default_policy_family), Some(policy_classes), Some(uncached_isl_buckets)) => {
                Some(resolve_profile(
                    RawPolicyProfile {
                        default_policy_family,
                        policy_classes,
                        uncached_isl_buckets,
                    },
                    "root",
                )?)
            }
            _ => {
                return Err(RouterPolicyConfigError::Validation(
                    "root profile must specify default_policy_family, uncached_isl_buckets, and policy_classes when any root profile field is present".to_string(),
                ));
            }
        };

        let mut models = HashMap::with_capacity(self.models.len());
        for (model_name, profile) in self.models {
            if model_name.is_empty() {
                return Err(RouterPolicyConfigError::Validation(
                    "model profile name must not be empty".to_string(),
                ));
            }
            let resolved = resolve_profile(profile, &format!("model {model_name:?}"))?;
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
    default_policy_family: String,
    policy_classes: Vec<RawPolicyClassConfig>,
    uncached_isl_buckets: Vec<RawUncachedIslBucket>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawUncachedIslBucket {
    min_tokens: usize,
    bucket: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    policy_family: Option<String>,
    cache_bucket: Option<String>,
    #[serde(default)]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    prefill_busy_threshold: Option<usize>,
    prefill_busy_threshold_frac: Option<f64>,
    decode_busy_threshold_frac: Option<f64>,
    request_queue_limit_per_worker: Option<usize>,
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    cached_token_queue_limit_per_worker: Option<usize>,
}

fn resolve_profile(
    profile: RawPolicyProfile,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    validate_identifier(&profile.default_policy_family, "policy family", location)?;
    if profile.policy_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let resolved_buckets = resolve_uncached_isl_buckets(profile.uncached_isl_buckets, location)?;
    let mut names = HashSet::with_capacity(profile.policy_classes.len());
    let mut classes = Vec::with_capacity(profile.policy_classes.len());
    let mut bindings = Vec::with_capacity(profile.policy_classes.len());
    for raw in profile.policy_classes {
        let resolved = resolve_policy_class(raw, &resolved_buckets.indices, location)?;
        if !names.insert(resolved.config.name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                resolved.config.name
            )));
        }
        classes.push(resolved.config);
        bindings.push(resolved.binding);
    }

    let mut family_names = Vec::new();
    let mut family_indices = HashMap::new();
    for binding in &bindings {
        let ClassBinding::FamilyBucket { policy_family, .. } = binding else {
            continue;
        };
        if !family_indices.contains_key(policy_family) {
            let family_index = family_names.len();
            family_names.push(policy_family.clone());
            family_indices.insert(policy_family.clone(), family_index);
        }
    }

    let Some(default_family_index) = family_indices.get(&profile.default_policy_family).copied()
    else {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_family {:?} does not name a configured family",
            profile.default_policy_family
        )));
    };

    let mut explicit_class_indices = HashMap::new();
    let mut class_by_family_bucket = vec![
        None;
        family_names
            .len()
            .saturating_mul(resolved_buckets.buckets.len())
    ];
    for (class_index, binding) in bindings.into_iter().enumerate() {
        match binding {
            ClassBinding::Explicit => {
                let class_name = &classes[class_index].name;
                if family_indices.contains_key(class_name) {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} explicit policy class {class_name:?} collides with a policy family"
                    )));
                }
                explicit_class_indices.insert(class_name.clone(), class_index);
            }
            ClassBinding::FamilyBucket {
                policy_family,
                bucket_index,
            } => {
                let family_index = family_indices[&policy_family];
                let table_index = family_index * resolved_buckets.buckets.len() + bucket_index;
                if class_by_family_bucket[table_index]
                    .replace(class_index)
                    .is_some()
                {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} contains duplicate policy classes for family {policy_family:?} and bucket {:?}",
                        resolved_buckets.names[bucket_index]
                    )));
                }
            }
        }
    }

    for (family_index, family_name) in family_names.iter().enumerate() {
        for (bucket_index, bucket_name) in resolved_buckets.names.iter().enumerate() {
            if class_by_family_bucket[family_index * resolved_buckets.buckets.len() + bucket_index]
                .is_none()
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} is missing a policy class for family {family_name:?} and bucket {bucket_name:?}"
                )));
            }
        }
    }

    Ok(PolicyProfile::family_buckets(FamilyBucketTable {
        classes,
        default_family_index,
        family_indices,
        explicit_class_indices,
        bucket_min_tokens: resolved_buckets.buckets,
        class_by_family_bucket: class_by_family_bucket
            .into_iter()
            .map(|class_index| class_index.expect("validated complete policy matrix"))
            .collect(),
    })?)
}

struct ResolvedPolicyClass {
    config: PolicyClassConfig,
    binding: ClassBinding,
}

enum ClassBinding {
    Explicit,
    FamilyBucket {
        policy_family: String,
        bucket_index: usize,
    },
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    bucket_indices: &HashMap<String, usize>,
    location: &str,
) -> Result<ResolvedPolicyClass, RouterPolicyConfigError> {
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    if raw.queue_policy == RouterQueuePolicy::Lcfs {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} queue_policy must be fcfs or wspt",
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

    let binding = match (raw.policy_family.as_deref(), raw.cache_bucket.as_deref()) {
        (None, None) => ClassBinding::Explicit,
        (Some(policy_family), Some(cache_bucket)) => {
            validate_identifier(policy_family, "policy family", location)?;
            validate_identifier(cache_bucket, "cache bucket", location)?;
            let Some(bucket_index) = bucket_indices.get(cache_bucket).copied() else {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} policy class {:?} references unknown cache bucket {:?}",
                    raw.name, cache_bucket
                )));
            };
            ClassBinding::FamilyBucket {
                policy_family: policy_family.to_string(),
                bucket_index,
            }
        }
        _ => {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} must specify both policy_family and cache_bucket or neither for an explicit class",
                raw.name
            )));
        }
    };
    if raw.prefill_busy_threshold.is_some() {
        tracing::warn!(
            policy_class = %raw.name,
            "prefill_busy_threshold is deprecated and will be removed in v1.7; use prefill_busy_threshold_frac"
        );
    }
    Ok(ResolvedPolicyClass {
        config: PolicyClassConfig {
            name: raw.name,
            queue_policy: raw.queue_policy,
            quantum: raw.quantum,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold: raw.prefill_busy_threshold,
                prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
                decode_busy_threshold_frac: raw.decode_busy_threshold_frac,
            },
            request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
            raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
            cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
        },
        binding,
    })
}

struct ResolvedBuckets {
    /// Each bucket's lower bound in uncached tokens, in order.
    buckets: Vec<usize>,
    names: Vec<String>,
    indices: HashMap<String, usize>,
}

fn resolve_uncached_isl_buckets(
    raw_buckets: Vec<RawUncachedIslBucket>,
    location: &str,
) -> Result<ResolvedBuckets, RouterPolicyConfigError> {
    if raw_buckets.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must not be empty"
        )));
    }
    if raw_buckets[0].min_tokens != 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must start at min_tokens 0"
        )));
    }
    for window in raw_buckets.windows(2) {
        if window[1].min_tokens <= window[0].min_tokens {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} uncached_isl_buckets min_tokens must be strictly increasing"
            )));
        }
    }

    let mut bucket_names = Vec::with_capacity(raw_buckets.len());
    let mut bucket_indices = HashMap::with_capacity(raw_buckets.len());
    let mut buckets = Vec::with_capacity(raw_buckets.len());
    for raw in raw_buckets {
        validate_identifier(&raw.bucket, "cache bucket", location)?;
        let bucket_index = bucket_names.len();
        if bucket_indices
            .insert(raw.bucket.clone(), bucket_index)
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate cache bucket {:?}",
                raw.bucket
            )));
        }
        bucket_names.push(raw.bucket);
        buckets.push(raw.min_tokens);
    }

    Ok(ResolvedBuckets {
        buckets,
        names: bucket_names,
        indices: bucket_indices,
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
    use crate::plugins::worker_selection::{WorkerLoadInput, WorkerLoadInputExt};
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
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root-default
    policy_family: standard
    cache_bucket: all
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_family: latency
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: cached
      - min_tokens: 32
        bucket: uncached
    policy_classes:
      - name: model-cached
        policy_family: latency
        cache_bucket: cached
        quantum: 2
        request_queue_limit_per_worker: 0
      - name: model-uncached
        policy_family: latency
        cache_bucket: uncached
        quantum: 4
        prefill_busy_threshold_frac: 0.0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-cached");
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
                .class(exact.resolve_class_index(None, usize::MAX))
                .queueing_enabled()
        );
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Fcfs);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact
                .class(exact.resolve_class_index(Some("unknown"), usize::MAX))
                .name,
            "model-uncached",
            "unknown policies must use the model's default family and observed bucket"
        );

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
        let half_default = (dynamo_router_protocol::load::DEFAULT_MAX_BATCHED_TOKENS / 2) as usize;
        assert!(limits.has_room(&prefill_load(half_default), WorkerCapacity::default()));
        assert!(!limits.has_room(&prefill_load(half_default + 1), WorkerCapacity::default()));
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
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
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
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
    default_policy_family: standard
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: absolute
        policy_family: standard
        cache_bucket: all
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
    }

    #[test]
    fn rejects_interacting_profile_errors() {
        for yaml in [
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: first
    policy_family: standard
    cache_bucket: cached
    quantum: 1
  - name: second
    policy_family: standard
    cache_bucket: cached
    quantum: 2
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: invalid-family
    policy_family: invalid/family
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: missing-bucket
    policy_family: standard
    cache_bucket: absent
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: partial
    policy_family: standard
    quantum: 1
"#,
            r#"
default_policy_family: priority
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: priority
    quantum: 1
  - name: paired
    policy_family: priority
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 1
    bucket: cached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: cached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 64
    bucket: uncached
  - min_tokens: 32
    bucket: large
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: zero
    policy_family: standard
    cache_bucket: cached
    quantum: 0
"#,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
policy_classes:
  - name: lcfs
    policy_family: standard
    cache_bucket: cached
    queue_policy: lcfs
    quantum: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn documented_sample_exercises_root_model_and_unknown_class_semantics() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        assert_eq!(root.classes().len(), 5);
        assert_eq!(root.default_class().name, "cached");
        assert_eq!(
            root.class(root.resolve_class_index(Some("latency"), 0))
                .name,
            "latency_cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("latency"), usize::MAX))
                .name,
            "latency_uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("unknown"), 0))
                .name,
            "cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, 3071)).name,
            "cached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, 3072)).name,
            "uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(None, usize::MAX)).name,
            "uncached"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("cached"), usize::MAX))
                .name,
            "uncached",
            "ordinary physical class names must not bypass family and bucket classification"
        );
        assert_eq!(
            root.class(root.resolve_class_index(Some("custom_priority"), usize::MAX))
                .name,
            "custom_priority",
            "explicit classes intentionally bypass cache classification"
        );
        assert_eq!(
            root.default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(16.0)
        );

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Fcfs,
        );
        assert_eq!(model.classes().len(), 4);
        assert_eq!(model.default_class().name, "latency_cached");
        assert_eq!(
            model
                .class(model.resolve_class_index(Some("unknown"), usize::MAX))
                .name,
            "latency_uncached",
            "unknown policies must use the model's default family and bucket mapping"
        );
        assert_eq!(
            model
                .class(model.resolve_class_index(Some("batch"), 0))
                .name,
            "batch_cached"
        );
        assert!(
            model
                .classes()
                .iter()
                .all(|class| class.name != "custom_priority"),
            "model profiles must completely replace root classes"
        );
    }

    #[test]
    fn prefill_budget_defaults_when_unreported() {
        use dynamo_router_protocol::load::DEFAULT_MAX_BATCHED_TOKENS;
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
        let capacity = WorkerCapacity::of(&config);
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 11,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();

        loads.insert(WorkerWithDpRank::new(5, 2), busy);
        // Rank 3 has no projection and counts as idle.
        assert!(limits.rank_is_full(WorkerWithDpRank::new(5, 2), capacity, &loads));
        assert!(!limits.rank_is_full(WorkerWithDpRank::new(5, 3), capacity, &loads));
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
