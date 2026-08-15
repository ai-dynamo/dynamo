// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;
use tokio::time::Instant;

use super::config::RouterQueuePolicy;
use super::worker_selection_config::RawWorkerSelectionConfig;
pub use super::worker_selection_config::{WorkerSelectionConfig, WorkerSelectionInstance};

const FALLBACK_POLICY_CLASS: &str = "default";

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

/// How one policy class orders its single runnable queue.
///
/// Both shapes use the same one min-max heap per class; they differ only in the
/// comparison key, and every request in a class uses that class's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyClassOrdering {
    /// A configured flat policy class. Each request gets one absolute deadline
    /// of `router arrival + slo`, the queue key is exactly
    /// `(deadline, enqueue sequence)`, and work past its deadline is rejected
    /// rather than dispatched.
    Deadline { slo: Duration },
    /// The fallback profile used when no `router_policy_config` defines
    /// classes. It keeps the pre-existing `--router-queue-policy` ordering and
    /// its strict-priority tier so deployments without a policy config are
    /// unaffected. There is no configured SLO, so no deadline and no expiry.
    Legacy { queue_policy: RouterQueuePolicy },
}

impl PolicyClassOrdering {
    pub fn slo(self) -> Option<Duration> {
        match self {
            Self::Deadline { slo } => Some(slo),
            Self::Legacy { .. } => None,
        }
    }
}

impl fmt::Display for PolicyClassOrdering {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deadline { slo } => write!(formatter, "deadline(slo={}ms)", slo.as_millis()),
            Self::Legacy { queue_policy } => write!(formatter, "legacy({queue_policy})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub ordering: PolicyClassOrdering,
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

    /// This class's fixed SLO, or `None` for the fallback profile.
    pub fn slo(&self) -> Option<Duration> {
        self.ordering.slo()
    }

    /// Absolute deadline for a request that reached the router at `arrival`.
    ///
    /// Computed once per request from the single captured arrival instant and
    /// never recomputed from a later clock read, so deferral, wake-up, and a
    /// long queue wait cannot extend a request's budget.
    pub fn deadline(&self, arrival: Instant) -> Option<Instant> {
        self.slo().map(|slo| arrival + slo)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    classifier: PolicyClassifier,
}

#[derive(Debug, Clone, PartialEq)]
enum PolicyClassifier {
    /// No configured classes. The profile holds one synthetic class and the
    /// policy-class name in request metadata is not a configured value, so it
    /// is ignored rather than rejected.
    Fallback,
    /// Configured flat classes. An exact name selects its class, an absent name
    /// selects `default_policy_class`, and any other name is unknown.
    Flat {
        default_class_index: usize,
        class_indices: HashMap<String, usize>,
    },
}

impl PolicyProfile {
    pub fn synthetic(
        router_queue_threshold: Option<f64>,
        router_queue_policy: RouterQueuePolicy,
    ) -> Self {
        let class = PolicyClassConfig {
            name: FALLBACK_POLICY_CLASS.to_string(),
            ordering: PolicyClassOrdering::Legacy {
                queue_policy: router_queue_policy,
            },
            quantum: 1,
            prefill_busy_threshold: None,
            prefill_busy_threshold_frac: router_queue_threshold,
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        };
        Self {
            classes: vec![class],
            classifier: PolicyClassifier::Fallback,
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    pub fn default_class_index(&self) -> usize {
        match &self.classifier {
            PolicyClassifier::Fallback => 0,
            PolicyClassifier::Flat {
                default_class_index,
                ..
            } => *default_class_index,
        }
    }

    pub fn default_class(&self) -> &PolicyClassConfig {
        &self.classes[self.default_class_index()]
    }

    /// Resolve the policy class named in request metadata.
    ///
    /// `requested` must already have the empty name normalized to `None`: an
    /// absent name means "no preference" and selects `default_policy_class`.
    /// Every other value is matched exactly, so a padded or differently cased
    /// name is unknown rather than normalized onto a class it does not spell.
    /// A name that no configured class carries returns `None` so the caller can
    /// reject the request; silently serving a typo under the default class
    /// would hide a misconfigured client behind another class's SLO and
    /// quantum.
    pub fn resolve_class_index(&self, requested: Option<&str>) -> Option<usize> {
        match &self.classifier {
            PolicyClassifier::Fallback => Some(0),
            PolicyClassifier::Flat {
                default_class_index,
                class_indices,
            } => match requested {
                None => Some(*default_class_index),
                Some(name) => class_indices.get(name).copied(),
            },
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
    default_policy_class: Option<String>,
    #[serde(default)]
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    #[serde(default)]
    worker_selection: Option<RawWorkerSelectionConfig>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        let root = match (self.default_policy_class, self.policy_classes) {
            (None, None) => None,
            (Some(default_policy_class), Some(policy_classes)) => Some(resolve_profile(
                RawPolicyProfile {
                    default_policy_class,
                    policy_classes,
                },
                "root",
            )?),
            _ => {
                return Err(RouterPolicyConfigError::Validation(
                    "root profile must specify both default_policy_class and policy_classes when either is present".to_string(),
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
    default_policy_class: String,
    policy_classes: Vec<RawPolicyClassConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    slo_ms: u64,
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
    validate_identifier(&profile.default_policy_class, "policy class", location)?;
    if profile.policy_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let mut class_indices = HashMap::with_capacity(profile.policy_classes.len());
    let mut classes = Vec::with_capacity(profile.policy_classes.len());
    for raw in profile.policy_classes {
        let resolved = resolve_policy_class(raw, location)?;
        if class_indices
            .insert(resolved.name.clone(), classes.len())
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                resolved.name
            )));
        }
        classes.push(resolved);
    }

    let Some(default_class_index) = class_indices.get(&profile.default_policy_class).copied()
    else {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_class {:?} does not name a configured policy class",
            profile.default_policy_class
        )));
    };

    Ok(PolicyProfile {
        classes,
        classifier: PolicyClassifier::Flat {
            default_class_index,
            class_indices,
        },
    })
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    location: &str,
) -> Result<PolicyClassConfig, RouterPolicyConfigError> {
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    if raw.slo_ms == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} slo_ms must be greater than zero",
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

    Ok(PolicyClassConfig {
        name: raw.name,
        ordering: PolicyClassOrdering::Deadline {
            slo: Duration::from_millis(raw.slo_ms),
        },
        quantum: raw.quantum,
        prefill_busy_threshold: raw.prefill_busy_threshold,
        prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
        request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
        raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
        cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
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
  default: example
  instances:
    - name: example
      type: example-policy
      parameters:
        score_weight: 1.0
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.default_instance(), Some("example"));
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
                .ordering,
            PolicyClassOrdering::Legacy {
                queue_policy: RouterQueuePolicy::Wspt
            },
            "a profile with no configured classes keeps --router-queue-policy ordering"
        );
    }

    #[test]
    fn rejects_invalid_worker_selection_config() {
        for yaml in [
            r#"
worker_selection: {}
"#,
            r#"
worker_selection:
  default: missing
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
    fn flat_classes_resolve_by_exact_name_with_an_explicit_default() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: regular
policy_classes:
  - name: premium
    slo_ms: 500
    quantum: 4096
    prefill_busy_threshold_frac: 16.0
  - name: regular
    slo_ms: 5000
    quantum: 512
    prefill_busy_threshold: 100
"#,
        )
        .unwrap();

        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        assert_eq!(profile.classes().len(), 2);
        assert_eq!(profile.default_class().name, "regular");
        assert_eq!(
            profile.default_class().slo(),
            Some(Duration::from_millis(5_000))
        );
        assert_eq!(
            profile
                .class(profile.resolve_class_index(Some("premium")).unwrap())
                .name,
            "premium"
        );
        assert_eq!(
            profile
                .class(profile.resolve_class_index(None).unwrap())
                .name,
            "regular",
            "an absent policy class selects default_policy_class"
        );
        assert_eq!(
            profile.resolve_class_index(Some("premiun")),
            None,
            "a typo must not silently inherit the default class"
        );
        assert_eq!(
            profile.class(0).slo(),
            Some(Duration::from_millis(500)),
            "each class keeps its own fixed SLO"
        );
    }

    #[test]
    fn fallback_profile_ignores_requested_class_names() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  instances:
    - name: alpha
      type: alpha
"#,
        )
        .unwrap();

        let profile = config.resolve_profile(None, Some(4.0), RouterQueuePolicy::Wspt);
        assert_eq!(profile.classes().len(), 1);
        assert_eq!(profile.default_class().name, FALLBACK_POLICY_CLASS);
        assert_eq!(profile.default_class().slo(), None);
        assert_eq!(profile.resolve_class_index(None), Some(0));
        assert_eq!(
            profile.resolve_class_index(Some("anything")),
            Some(0),
            "an unconfigured profile has no class namespace to validate against"
        );
        assert_eq!(profile.default_class().deadline(Instant::now()), None);
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: root-default
policy_classes:
  - name: root-default
    slo_ms: 2000
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_class: model-cached
    policy_classes:
      - name: model-cached
        slo_ms: 250
        quantum: 2
        request_queue_limit_per_worker: 0
      - name: model-uncached
        slo_ms: 30000
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
                .class(exact.resolve_class_index(Some("model-uncached")).unwrap())
                .queueing_enabled()
        );
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact.resolve_class_index(Some("root-default")),
            None,
            "model profiles must completely replace root classes"
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
    default_policy_class: absolute
    policy_classes:
      - name: absolute
        slo_ms: 1000
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
        assert_eq!(fallback.default_class().name, FALLBACK_POLICY_CLASS);
        assert_eq!(
            fallback.default_class().prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(fallback.default_class().slo(), None);
    }

    #[test]
    fn deadline_is_arrival_plus_the_configured_slo() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: only
policy_classes:
  - name: only
    slo_ms: 1500
    quantum: 1
"#,
        )
        .unwrap();

        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let arrival = Instant::now();
        assert_eq!(
            profile.default_class().deadline(arrival),
            Some(arrival + Duration::from_millis(1_500))
        );
    }

    #[test]
    fn rejects_interacting_profile_errors() {
        for yaml in [
            // default_policy_class does not name a configured class
            r#"
default_policy_class: missing
policy_classes:
  - name: present
    slo_ms: 1000
    quantum: 1
"#,
            // duplicate class names
            r#"
default_policy_class: repeated
policy_classes:
  - name: repeated
    slo_ms: 1000
    quantum: 1
  - name: repeated
    slo_ms: 2000
    quantum: 2
"#,
            // empty policy_classes
            r#"
default_policy_class: any
policy_classes: []
"#,
            // policy_classes without default_policy_class
            r#"
policy_classes:
  - name: only
    slo_ms: 1000
    quantum: 1
"#,
            // default_policy_class without policy_classes
            r#"
default_policy_class: only
"#,
            // zero quantum
            r#"
default_policy_class: zero
policy_classes:
  - name: zero
    slo_ms: 1000
    quantum: 0
"#,
            // missing slo_ms
            r#"
default_policy_class: no-slo
policy_classes:
  - name: no-slo
    quantum: 1
"#,
            // zero slo_ms
            r#"
default_policy_class: zero-slo
policy_classes:
  - name: zero-slo
    slo_ms: 0
    quantum: 1
"#,
            // invalid class identifier
            r#"
default_policy_class: bad/name
policy_classes:
  - name: bad/name
    slo_ms: 1000
    quantum: 1
"#,
            // negative busy fraction
            r#"
default_policy_class: negative
policy_classes:
  - name: negative
    slo_ms: 1000
    quantum: 1
    prefill_busy_threshold_frac: -1.0
"#,
            // removed hierarchical schema
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: all
    quantum: 1
"#,
            // per-class queue_policy moved out of Stage 0 class configuration
            r#"
default_policy_class: scored
policy_classes:
  - name: scored
    slo_ms: 1000
    quantum: 1
    queue_policy: wspt
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn documented_sample_exercises_root_model_and_default_class_semantics() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        assert_eq!(root.classes().len(), 5);
        assert_eq!(root.default_class().name, "cached");
        for name in [
            "cached",
            "uncached",
            "latency_cached",
            "latency_uncached",
            "custom_priority",
        ] {
            assert_eq!(
                root.class(root.resolve_class_index(Some(name)).unwrap())
                    .name,
                name
            );
        }
        assert_eq!(
            root.class(root.resolve_class_index(None).unwrap()).name,
            "cached",
            "a request that names no class uses default_policy_class"
        );
        assert_eq!(
            root.resolve_class_index(Some("unknown")),
            None,
            "an unconfigured class name is rejected, not defaulted"
        );
        assert_eq!(root.default_class().prefill_busy_threshold_frac, Some(16.0));
        assert_eq!(root.default_class().quantum, 2048);
        assert_eq!(
            root.class(root.resolve_class_index(Some("uncached")).unwrap())
                .raw_isl_token_queue_limit_per_worker,
            Some(1_048_576)
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
                .class(model.resolve_class_index(Some("batch_uncached")).unwrap())
                .quantum,
            1024
        );
        assert_eq!(
            model.resolve_class_index(Some("custom_priority")),
            None,
            "model profiles must completely replace root classes"
        );
    }
}
