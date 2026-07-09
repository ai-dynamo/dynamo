// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use super::{QueueAdmissionConfig, config::RouterQueuePolicy};

const DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC: f64 = 16.0;
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
    pub queue_admission: Option<QueueAdmissionConfig>,
    pub quantum: usize,
    pub prefill_busy_threshold: Option<usize>,
    pub prefill_busy_threshold_frac: Option<f64>,
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
            queue_admission: None,
            quantum: 1,
            prefill_busy_threshold: None,
            prefill_busy_threshold_frac: router_queue_threshold,
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

        if root.is_none() && models.is_empty() {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define a root profile or at least one model profile"
                    .to_string(),
            ));
        }

        Ok(RouterPolicyConfig { root, models })
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
    #[serde(default)]
    queue_admission: Option<QueueAdmissionConfig>,
    quantum: usize,
    #[serde(default)]
    prefill_busy_threshold: Option<usize>,
    #[serde(default)]
    prefill_busy_threshold_frac: Option<f64>,
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
    let mut has_queue_admission = false;
    for raw in profile.policy_classes {
        let resolved = resolve_policy_class(raw, &resolved_buckets.indices, location)?;
        if !names.insert(resolved.config.name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                resolved.config.name
            )));
        }
        if resolved.config.queue_admission.is_some() {
            if has_queue_admission {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} must contain at most one capacity-reserving queue_admission class"
                )));
            }
            has_queue_admission = true;
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
    if raw.queue_admission.is_some() && raw.queue_policy != RouterQueuePolicy::Fcfs {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} queue_admission requires queue_policy fcfs",
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

    let (prefill_busy_threshold, prefill_busy_threshold_frac) =
        match (raw.prefill_busy_threshold, raw.prefill_busy_threshold_frac) {
            (None, None) => (None, Some(DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC)),
            thresholds => thresholds,
        };

    Ok(ResolvedPolicyClass {
        config: PolicyClassConfig {
            name: raw.name,
            queue_policy: raw.queue_policy,
            queue_admission: raw.queue_admission,
            quantum: raw.quantum,
            prefill_busy_threshold,
            prefill_busy_threshold_frac,
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

fn validate_identifier(
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
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-cached");
        assert_eq!(
            exact.default_class().prefill_busy_threshold_frac,
            Some(DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC)
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
    fn accepts_session_aware_admission_with_fcfs() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents
    queue_policy: fcfs
    queue_admission:
      type: session_aware
    quantum: 1
"#,
        )
        .unwrap();

        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let agents = profile.class(profile.resolve_class_index(Some("agents"), 0));
        assert!(matches!(
            agents.queue_admission,
            Some(QueueAdmissionConfig::SessionAware {})
        ));
    }

    #[test]
    fn synthetic_profile_has_no_queue_admission() {
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        assert!(profile.default_class().queue_admission.is_none());
    }

    #[test]
    fn rejects_duplicate_or_invalid_queue_admission() {
        for (yaml, expected) in [
            (
                r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: standard
    cache_bucket: all
    queue_policy: wspt
    queue_admission:
      type: session_aware
    quantum: 1
"#,
                "queue_admission requires queue_policy fcfs",
            ),
            (
                r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents-a
    queue_admission:
      type: session_aware
    quantum: 1
  - name: agents-b
    queue_admission:
      type: session_aware
    quantum: 1
"#,
                "at most one capacity-reserving queue_admission class",
            ),
            (
                r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: standard
    cache_bucket: all
    queue_admission:
      type: session_aware
      pause_threshold: 0.7
    quantum: 1
"#,
                "unknown field",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(yaml).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
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
            root.default_class().prefill_busy_threshold_frac,
            Some(DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC)
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
}
