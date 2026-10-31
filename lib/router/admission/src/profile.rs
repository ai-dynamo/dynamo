// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission stage's resolved configuration: the queue policy of a class, the per-class
//! limits, and a profile's class table. The raw YAML forms, the file reader, and the
//! identifier rules are the router's (`policy_config`); this module validates the table they
//! hand it and never sees a file.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use dynamo_router_protocol::load::BusyThresholds;

/// The one class of a profile built from the flat router config alone.
pub const SYNTHETIC_POLICY_CLASS: &str = "default";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterQueuePolicy {
    #[default]
    Fcfs,
    Lcfs,
    Wspt,
}

impl fmt::Display for RouterQueuePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcfs => f.write_str("fcfs"),
            Self::Lcfs => f.write_str("lcfs"),
            Self::Wspt => f.write_str("wspt"),
        }
    }
}

impl FromStr for RouterQueuePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fcfs" => Ok(Self::Fcfs),
            "lcfs" => Ok(Self::Lcfs),
            "wspt" => Ok(Self::Wspt),
            _ => Err(format!(
                "unknown queue policy: {s:?}, expected 'fcfs', 'lcfs', or 'wspt'"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub queue_policy: RouterQueuePolicy,
    pub quantum: usize,
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
}

/// Why a class table is not a valid profile.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AdmissionConfigError {
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

/// A profile's resolved class table: every policy family crossed with every uncached-ISL
/// bucket names one class, and explicit classes are reached by name. Indices refer to
/// `classes`; the table is family-major with `bucket_min_tokens.len()` cells per family.
#[derive(Debug, Clone, PartialEq)]
pub struct FamilyBucketTable {
    pub classes: Vec<PolicyClassConfig>,
    pub default_family_index: usize,
    pub family_indices: HashMap<String, usize>,
    pub explicit_class_indices: HashMap<String, usize>,
    /// Each bucket's lower bound in uncached tokens: starts at zero, strictly increasing.
    pub bucket_min_tokens: Vec<usize>,
    pub class_by_family_bucket: Vec<usize>,
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
        self.class_index_for_family(family_index, uncached_tokens)
    }

    fn strict_class_index(&self, requested: &str, uncached_tokens: usize) -> Option<usize> {
        self.explicit_class_indices
            .get(requested)
            .copied()
            .or_else(|| {
                self.family_indices
                    .get(requested)
                    .map(|family_index| self.class_index_for_family(*family_index, uncached_tokens))
            })
    }

    fn class_index_for_family(&self, family_index: usize, uncached_tokens: usize) -> usize {
        let bucket_index = self
            .buckets
            .partition_point(|bucket| bucket.min_tokens <= uncached_tokens)
            .saturating_sub(1);
        self.class_by_family_bucket[family_index * self.buckets.len() + bucket_index]
    }
}

impl PolicyProfile {
    /// A profile from a resolved class table, refused unless every index is in range and
    /// every (family, bucket) cell names a class.
    pub fn family_buckets(table: FamilyBucketTable) -> Result<Self, AdmissionConfigError> {
        let FamilyBucketTable {
            classes,
            default_family_index,
            family_indices,
            explicit_class_indices,
            bucket_min_tokens,
            class_by_family_bucket,
        } = table;
        let invalid = |message: String| Err(AdmissionConfigError::Validation(message));
        if classes.is_empty() {
            return invalid("profile has no policy classes".to_string());
        }
        if bucket_min_tokens.first() != Some(&0) {
            return invalid("uncached_isl_buckets must start at min_tokens 0".to_string());
        }
        if bucket_min_tokens.windows(2).any(|pair| pair[1] <= pair[0]) {
            return invalid(
                "uncached_isl_buckets min_tokens must be strictly increasing".to_string(),
            );
        }
        let families = family_indices.len();
        if default_family_index >= families {
            return invalid(format!(
                "default family index {default_family_index} is out of range for {families} families"
            ));
        }
        if family_indices.values().any(|&index| index >= families) {
            return invalid("policy family index is out of range".to_string());
        }
        if explicit_class_indices
            .values()
            .any(|&index| index >= classes.len())
        {
            return invalid("explicit policy class index is out of range".to_string());
        }
        if class_by_family_bucket.len() != families * bucket_min_tokens.len() {
            return invalid(format!(
                "class table has {} cells for {families} families and {} buckets",
                class_by_family_bucket.len(),
                bucket_min_tokens.len()
            ));
        }
        if class_by_family_bucket
            .iter()
            .any(|&index| index >= classes.len())
        {
            return invalid("class table names a policy class index out of range".to_string());
        }
        Ok(Self {
            classes,
            classifier: PolicyClassifier::FamilyBucket(FamilyBucketClassifier {
                default_family_index,
                family_indices,
                explicit_class_indices,
                buckets: bucket_min_tokens
                    .into_iter()
                    .map(|min_tokens| UncachedIslBucket { min_tokens })
                    .collect(),
                class_by_family_bucket,
            }),
        })
    }

    pub fn synthetic(
        router_queue_threshold: Option<f64>,
        router_queue_policy: RouterQueuePolicy,
    ) -> Self {
        let class = PolicyClassConfig {
            name: SYNTHETIC_POLICY_CLASS.to_string(),
            queue_policy: router_queue_policy,
            quantum: 1,
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

    pub fn resolve_class_index_strict(
        &self,
        requested: &str,
        uncached_tokens: usize,
    ) -> Option<usize> {
        match &self.classifier {
            PolicyClassifier::SyntheticSingle { class_index } => {
                (self.classes[*class_index].name == requested).then_some(*class_index)
            }
            PolicyClassifier::FamilyBucket(classifier) => {
                classifier.strict_class_index(requested, uncached_tokens)
            }
        }
    }
}
