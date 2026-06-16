// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use super::config::RouterQueuePolicy;

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
    pub quantum: usize,
    pub prefill_busy_threshold: Option<usize>,
    pub prefill_busy_threshold_frac: Option<f64>,
    pub request_queue_limit: Option<usize>,
    pub token_queue_limit: Option<usize>,
    pub cached_token_queue_limit: Option<usize>,
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
    default_policy_class: usize,
    classes: Vec<PolicyClassConfig>,
    class_indices: HashMap<String, usize>,
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
            request_queue_limit: None,
            token_queue_limit: None,
            cached_token_queue_limit: None,
        };
        let class_indices = HashMap::from([(class.name.clone(), 0)]);
        Self {
            default_policy_class: 0,
            classes: vec![class],
            class_indices,
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    pub fn default_class(&self) -> &PolicyClassConfig {
        &self.classes[self.default_policy_class]
    }

    pub fn resolve_class_index(&self, requested: Option<&str>) -> usize {
        // TODO: Add bounded observability for unknown policy-class values.
        requested
            .and_then(|name| self.class_indices.get(name).copied())
            .unwrap_or(self.default_policy_class)
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
    default_policy_class: Option<String>,
    #[serde(default)]
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
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
                    "root profile must specify both default_policy_class and policy_classes"
                        .to_string(),
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
    default_policy_class: String,
    policy_classes: Vec<RawPolicyClassConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    #[serde(default)]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    #[serde(default)]
    prefill_busy_threshold: Option<usize>,
    #[serde(default)]
    prefill_busy_threshold_frac: Option<f64>,
    #[serde(default)]
    request_queue_limit: Option<usize>,
    #[serde(default)]
    token_queue_limit: Option<usize>,
    #[serde(default)]
    cached_token_queue_limit: Option<usize>,
}

fn resolve_profile(
    profile: RawPolicyProfile,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    if profile.policy_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let mut names = HashSet::with_capacity(profile.policy_classes.len());
    let mut classes = Vec::with_capacity(profile.policy_classes.len());
    for raw in profile.policy_classes {
        validate_class_name(&raw.name, location)?;
        if !names.insert(raw.name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                raw.name
            )));
        }
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

        let (prefill_busy_threshold, prefill_busy_threshold_frac) =
            match (raw.prefill_busy_threshold, raw.prefill_busy_threshold_frac) {
                (None, None) => (None, Some(DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC)),
                thresholds => thresholds,
            };
        classes.push(PolicyClassConfig {
            name: raw.name,
            queue_policy: raw.queue_policy,
            quantum: raw.quantum,
            prefill_busy_threshold,
            prefill_busy_threshold_frac,
            request_queue_limit: raw.request_queue_limit,
            token_queue_limit: raw.token_queue_limit,
            cached_token_queue_limit: raw.cached_token_queue_limit,
        });
    }

    let class_indices: HashMap<_, _> = classes
        .iter()
        .enumerate()
        .map(|(index, class)| (class.name.clone(), index))
        .collect();
    let Some(default_policy_class) = class_indices.get(&profile.default_policy_class).copied()
    else {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_class {:?} does not name a configured class",
            profile.default_policy_class
        )));
    };

    Ok(PolicyProfile {
        default_policy_class,
        classes,
        class_indices,
    })
}

fn validate_class_name(name: &str, location: &str) -> Result<(), RouterPolicyConfigError> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Ok(());
    }

    Err(RouterPolicyConfigError::Validation(format!(
        "{location} policy class name {name:?} must match [A-Za-z0-9_.-]+"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: root-default
policy_classes:
  - name: root-default
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_class: model-default
    policy_classes:
      - name: model-default
        quantum: 2
        request_queue_limit: 0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 1);
        assert_eq!(exact.default_class().name, "model-default");
        assert_eq!(
            exact.default_class().prefill_busy_threshold_frac,
            Some(DEFAULT_PREFILL_BUSY_THRESHOLD_FRAC)
        );
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Fcfs);
        assert_eq!(exact.default_class().request_queue_limit, Some(0));

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
default_policy_class: missing
policy_classes:
  - name: valid
    quantum: 1
"#,
            r#"
default_policy_class: duplicate
policy_classes:
  - name: duplicate
    quantum: 1
  - name: duplicate
    quantum: 2
"#,
            r#"
default_policy_class: invalid/name
policy_classes:
  - name: invalid/name
    quantum: 1
"#,
            r#"
default_policy_class: zero
policy_classes:
  - name: zero
    quantum: 0
"#,
            r#"
default_policy_class: lcfs
policy_classes:
  - name: lcfs
    queue_policy: lcfs
    quantum: 1
"#,
            r#"
default_policy_class: valid
policy_classes:
  - name: valid
    quantum: 1
    typo_limit: 3
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
        assert_eq!(root.classes().len(), 2);
        assert_eq!(root.default_class().name, "uncached");
        assert_eq!(root.resolve_class_index(Some("cached")), 1);
        assert_eq!(
            root.resolve_class_index(Some("unknown")),
            root.resolve_class_index(None)
        );
        assert!(root.class(0).worker_is_busy(32_769, 1_000_000));
        assert!(root.class(0).worker_is_busy(1001, 1000));
        assert!(!root.class(0).worker_is_busy(1000, 1000));

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Fcfs,
        );
        assert_eq!(model.classes().len(), 2);
        assert_eq!(model.default_class().name, "latency");
        assert!(
            model.classes().iter().all(|class| class.name != "uncached"),
            "model profiles must completely replace the root profile"
        );
    }
}
