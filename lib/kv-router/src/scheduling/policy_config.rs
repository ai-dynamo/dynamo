// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use super::config::RouterQueuePolicy;
use super::worker_selection_config::RawWorkerSelectionConfig;
pub use super::worker_selection_config::{WorkerSelectionConfig, WorkerSelectionInstance};

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
    pub prefill_busy_threshold: Option<usize>,
    pub prefill_busy_threshold_frac: Option<f64>,
    /// Share of *eligible workers* that must be prefill-busy before the pool
    /// counts as busy for queue admission, in `(0.0, 1.0]`.
    ///
    /// This is a pool-level ratio and is deliberately NOT named `*_frac`: the
    /// sibling `prefill_busy_threshold_frac` is a multiplier on a single
    /// worker's `max_num_batched_tokens` (values such as `16.0`), whereas this
    /// knob is a true fraction of the worker count.
    ///
    /// `None` (the default) preserves the historical all-workers-busy rule.
    pub busy_worker_ratio: Option<f64>,
    pub request_queue_limit_per_worker: Option<usize>,
    pub raw_isl_token_queue_limit_per_worker: Option<usize>,
    pub cached_token_queue_limit_per_worker: Option<usize>,
}

impl PolicyClassConfig {
    pub fn queueing_enabled(&self) -> bool {
        self.prefill_busy_threshold.is_some() || self.prefill_busy_threshold_frac.is_some()
    }

    pub fn worker_is_busy(&self, active_tokens: usize, max_batched_tokens: u64) -> bool {
        let absolute_busy = self
            .prefill_busy_threshold
            .is_some_and(|threshold| active_tokens > threshold);
        let fractional_busy = self.prefill_busy_threshold_frac.is_some_and(|threshold| {
            (active_tokens as f64) > threshold * (max_batched_tokens as f64)
        });
        absolute_busy || fractional_busy
    }

    /// Whether an eligible worker pool counts as busy for queue admission.
    ///
    /// Default (`busy_worker_ratio == None`) keeps the historical rule: the
    /// pool is busy only when *every* eligible worker is busy.
    ///
    /// When set, the pool is busy once the busy share reaches the ratio. A high
    /// ratio degrades to the historical rule on small pools, which is why no
    /// separate minimum-pool-size knob is needed: with 2 eligible workers and a
    /// ratio of `0.95`, `2 >= 1.9` still requires both workers to be busy.
    ///
    /// Returns `false` when no eligible workers were inspected so the request
    /// falls through to `schedule`, which reports a proper `NoEndpoints` error.
    pub fn pool_is_busy(&self, busy_workers: usize, eligible_workers: usize) -> bool {
        if eligible_workers == 0 {
            return false;
        }
        match self.busy_worker_ratio {
            Some(ratio) => (busy_workers as f64) >= ratio * (eligible_workers as f64),
            None => busy_workers == eligible_workers,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    classifier: PolicyClassifier,
}

#[derive(Debug, Clone, PartialEq)]
enum PolicyClassifier {
    SyntheticSingle { class_index: usize },
    FamilyBucket(FamilyBucketClassifier),
}

#[derive(Debug, Clone, PartialEq)]
struct FamilyBucketClassifier {
    default_family_index: usize,
    family_indices: HashMap<String, usize>,
    explicit_class_indices: HashMap<String, usize>,
    buckets: Vec<UncachedIslBucket>,
    class_by_family_bucket: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UncachedIslBucket {
    min_tokens: usize,
}

impl FamilyBucketClassifier {
    /// Returns only selections that do not require a cache snapshot.
    fn direct_class_index(&self, requested: Option<&str>) -> Option<usize> {
        requested.and_then(|name| self.explicit_class_indices.get(name).copied())
    }

    /// Combines a recognized family (or the default) with the observed bucket.
    fn class_index(&self, requested: Option<&str>, uncached_tokens: usize) -> usize {
        if let Some(class_index) = self.direct_class_index(requested) {
            return class_index;
        }

        let family_index = requested
            .and_then(|name| self.family_indices.get(name).copied())
            .unwrap_or(self.default_family_index);
        let bucket_index = self
            .buckets
            .partition_point(|bucket| bucket.min_tokens <= uncached_tokens)
            .saturating_sub(1);
        self.class_by_family_bucket[family_index * self.buckets.len() + bucket_index]
    }
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
            prefill_busy_threshold: None,
            prefill_busy_threshold_frac: router_queue_threshold,
            busy_worker_ratio: None,
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        };
        Self {
            classes: vec![class],
            classifier: PolicyClassifier::SyntheticSingle { class_index: 0 },
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    pub fn default_class(&self) -> &PolicyClassConfig {
        &self.classes[self.resolve_class_index(None, 0)]
    }

    /// Resolves synthetic and explicit requests without observing cache state.
    pub fn direct_class_index(&self, requested: Option<&str>) -> Option<usize> {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => Some(*class_index),
            PolicyClassifier::FamilyBucket(classifier) => classifier.direct_class_index(requested),
        }
    }

    /// Resolves a requested family and exact uncached ISL to a physical queue.
    pub fn resolve_class_index(&self, requested: Option<&str>, uncached_tokens: usize) -> usize {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => *class_index,
            PolicyClassifier::FamilyBucket(classifier) => {
                // TODO: Add bounded observability for unknown requested policy values.
                classifier.class_index(requested, uncached_tokens)
            }
        }
    }

    pub fn class(&self, index: usize) -> &PolicyClassConfig {
        &self.classes[index]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicyConfig {
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
    worker_selection: Option<WorkerSelectionConfig>,
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

    /// Returns the process-wide worker-selection policy configuration, if present.
    pub fn worker_selection(&self) -> Option<&WorkerSelectionConfig> {
        self.worker_selection.as_ref()
    }

    /// Whether this document configures queue policy profiles.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouterPolicyConfig {
    #[serde(default)]
    default_policy_family: Option<String>,
    #[serde(default)]
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    uncached_isl_buckets: Option<Vec<RawUncachedIslBucket>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    #[serde(default)]
    worker_selection: Option<RawWorkerSelectionConfig>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
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

        if root.is_none() && models.is_empty() && worker_selection.is_none() {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define a root profile, at least one model profile, or worker_selection".to_string(),
            ));
        }

        Ok(RouterPolicyConfig {
            root,
            models,
            worker_selection,
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
    #[serde(default)]
    policy_family: Option<String>,
    #[serde(default)]
    cache_bucket: Option<String>,
    #[serde(default)]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    #[serde(default)]
    prefill_busy_threshold: Option<usize>,
    #[serde(default)]
    prefill_busy_threshold_frac: Option<f64>,
    #[serde(default)]
    busy_worker_ratio: Option<f64>,
    #[serde(default)]
    request_queue_limit_per_worker: Option<usize>,
    #[serde(default)]
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    #[serde(default)]
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

    Ok(PolicyProfile {
        classes,
        classifier: PolicyClassifier::FamilyBucket(FamilyBucketClassifier {
            default_family_index,
            family_indices,
            explicit_class_indices,
            buckets: resolved_buckets.buckets,
            class_by_family_bucket: class_by_family_bucket
                .into_iter()
                .map(|class_index| class_index.expect("validated complete policy matrix"))
                .collect(),
        }),
    })
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
    if raw
        .prefill_busy_threshold_frac
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} prefill_busy_threshold_frac must be finite and non-negative",
            raw.name
        )));
    }
    // Range is deliberately stricter than `prefill_busy_threshold_frac`, which
    // is a multiplier on `max_num_batched_tokens` and routinely exceeds 1.0.
    // Rejecting out-of-range values here stops that scale being pasted into
    // this knob, where it would silently never trigger.
    if raw
        .busy_worker_ratio
        .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} busy_worker_ratio must be a finite fraction in (0.0, 1.0]",
            raw.name
        )));
    }
    // `busy_worker_ratio` only rescales how many busy workers are needed; it
    // cannot define what "busy" means. Without a per-worker threshold every
    // worker reports idle, so the ratio would never fire and queueing would
    // stay disabled. Fail loudly instead of silently doing nothing.
    if raw.busy_worker_ratio.is_some()
        && raw.prefill_busy_threshold.is_none()
        && raw.prefill_busy_threshold_frac.is_none()
    {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} sets busy_worker_ratio but no prefill_busy_threshold or prefill_busy_threshold_frac, so no worker can ever be considered busy",
            raw.name
        )));
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
    Ok(ResolvedPolicyClass {
        config: PolicyClassConfig {
            name: raw.name,
            queue_policy: raw.queue_policy,
            quantum: raw.quantum,
            prefill_busy_threshold: raw.prefill_busy_threshold,
            prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
            busy_worker_ratio: raw.busy_worker_ratio,
            request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
            raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
            cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
        },
        binding,
    })
}

struct ResolvedBuckets {
    buckets: Vec<UncachedIslBucket>,
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
        buckets.push(UncachedIslBucket {
            min_tokens: raw.min_tokens,
        });
    }

    Ok(ResolvedBuckets {
        buckets,
        names: bucket_names,
        indices: bucket_indices,
    })
}

pub(super) fn validate_identifier(
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
        assert_eq!(exact.default_class().prefill_busy_threshold_frac, None);
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
        assert_eq!(unmatched.default_class().prefill_busy_threshold, Some(100));
        assert_eq!(unmatched.default_class().prefill_busy_threshold_frac, None);
    }

    fn class_with_busy_worker_ratio(ratio: Option<f64>) -> PolicyClassConfig {
        PolicyClassConfig {
            name: "test".to_string(),
            queue_policy: RouterQueuePolicy::Fcfs,
            admission: None,
            quantum: 1,
            prefill_busy_threshold: Some(0),
            prefill_busy_threshold_frac: None,
            busy_worker_ratio: ratio,
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        }
    }

    #[test]
    fn pool_is_busy_defaults_to_every_worker_busy() {
        let class = class_with_busy_worker_ratio(None);
        assert!(!class.pool_is_busy(127, 128));
        assert!(class.pool_is_busy(128, 128));
    }

    #[test]
    fn pool_is_busy_triggers_once_the_ratio_is_reached() {
        let class = class_with_busy_worker_ratio(Some(0.95));
        // 0.95 * 128 = 121.6, so 122 busy workers trip the gate while 121 do not.
        assert!(!class.pool_is_busy(121, 128));
        assert!(class.pool_is_busy(122, 128));
    }

    #[test]
    fn pool_is_busy_high_ratio_keeps_small_pools_on_the_all_busy_rule() {
        // Why no separate minimum-pool-size knob is needed: on a 2-worker pool
        // a 0.95 ratio still requires 2 >= 1.9, i.e. every worker busy.
        let class = class_with_busy_worker_ratio(Some(0.95));
        assert!(!class.pool_is_busy(1, 2));
        assert!(class.pool_is_busy(2, 2));
    }

    #[test]
    fn pool_is_busy_is_false_without_eligible_workers() {
        // Must stay false so the request falls through to `schedule` and gets a
        // proper NoEndpoints error instead of being queued forever.
        assert!(!class_with_busy_worker_ratio(None).pool_is_busy(0, 0));
        assert!(!class_with_busy_worker_ratio(Some(0.5)).pool_is_busy(0, 0));
    }

    fn busy_ratio_yaml(class_fields: &str) -> String {
        format!(
            "default_policy_family: standard\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: ratioed\n    policy_family: standard\n    cache_bucket: all\n    quantum: 4\n{class_fields}"
        )
    }

    #[test]
    fn busy_worker_ratio_rejects_the_prefill_threshold_frac_scale() {
        // `prefill_busy_threshold_frac` is a multiplier and is routinely 16.0.
        // Pasting that scale here must fail loudly rather than never trigger.
        let err = RouterPolicyConfig::from_yaml(&busy_ratio_yaml(
            "    prefill_busy_threshold: 10\n    busy_worker_ratio: 16.0\n",
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("busy_worker_ratio"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn busy_worker_ratio_requires_a_per_worker_busy_threshold() {
        // Without a per-worker threshold no worker is ever busy, so the ratio
        // would silently never fire.
        let err = RouterPolicyConfig::from_yaml(&busy_ratio_yaml("    busy_worker_ratio: 0.9\n"))
            .unwrap_err();
        assert!(
            err.to_string().contains("no prefill_busy_threshold"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn busy_worker_ratio_parses_and_defaults_to_none() {
        let config = RouterPolicyConfig::from_yaml(&busy_ratio_yaml(
            "    prefill_busy_threshold: 10\n    busy_worker_ratio: 0.9\n",
        ))
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        assert_eq!(profile.default_class().busy_worker_ratio, Some(0.9));

        assert_eq!(
            PolicyProfile::synthetic(Some(16.0), RouterQueuePolicy::Fcfs)
                .default_class()
                .busy_worker_ratio,
            None
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
        assert!(exact.default_class().worker_is_busy(11, 10_000_000));
        assert!(exact.default_class().worker_is_busy(6, 10));
        assert!(!exact.default_class().worker_is_busy(5, 10));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback.default_class().prefill_busy_threshold_frac,
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
        assert_eq!(root.default_class().prefill_busy_threshold_frac, Some(16.0));

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
}
