// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The admission stage's resolved configuration: the queue policy of a class, the per-class
//! limits and budgets, and a profile's flat class table. The raw YAML forms, the file reader,
//! and the identifier rules are the router's (`policy_config`); this module validates the
//! classes they hand it and never sees a file.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU32;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use dynamo_slot_tracker::BusyThresholds;

/// The one class of a profile built from the flat router config alone.
pub const SYNTHETIC_POLICY_CLASS: &str = "default";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterQueuePolicy {
    /// Earliest due time first, then enqueue order. A request's due time is its explicit
    /// deadline, else its router ingress plus its class's `slo_ttft_ms`; a request with
    /// neither sorts after every request with one. Strict priority and priority jumps do
    /// not reorder it. Policy-class YAML defaults to it.
    Edf,
    #[default]
    Fcfs,
    Lcfs,
    Wspt,
}

impl fmt::Display for RouterQueuePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Edf => f.write_str("edf"),
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
            "edf" => Ok(Self::Edf),
            "fcfs" => Ok(Self::Fcfs),
            "lcfs" => Ok(Self::Lcfs),
            "wspt" => Ok(Self::Wspt),
            _ => Err(format!(
                "unknown queue policy: {s:?}, expected 'edf', 'fcfs', 'lcfs', or 'wspt'"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub queue_policy: RouterQueuePolicy,
    pub quantum: usize,
    /// Time-to-first-token budget in whole milliseconds, at most `u32::MAX` (about 49.7 days),
    /// or `None` when the class sets none. An EDF class orders a request without an explicit
    /// deadline by its router ingress plus this budget. Other orders neither sort nor expire
    /// by it.
    pub slo_ttft_ms: Option<NonZeroU32>,
    /// Controlled delay, EDF classes only: the online router rejects a request still waiting
    /// for admission when its due time passes. That is its explicit deadline, which wins even
    /// when later, else router ingress plus `slo_ttft_ms`. Off by default, when the class
    /// budget only orders the class. Explicit deadlines expire either way.
    pub controlled_delay: bool,
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

    /// Whether an online queue expires this class's budget: controlled delay in an EDF class,
    /// the only order in which every lane root, blocked lanes included, carries the earliest
    /// due time of its lane.
    pub fn expires_class_budget(&self) -> bool {
        self.controlled_delay && self.queue_policy == RouterQueuePolicy::Edf
    }

    /// Any zero per-worker limit means `queue_rejection` rejects every queued
    /// entry (`current >= limit` holds at zero usage), so the class can never
    /// hold a queued request and must keep the direct admission path.
    pub fn never_queues(&self) -> bool {
        self.request_queue_limit_per_worker == Some(0)
            || self.raw_isl_token_queue_limit_per_worker == Some(0)
            || self.cached_token_queue_limit_per_worker == Some(0)
    }

    /// When this class's TTFT budget runs out for a request that entered the router at
    /// `ingress_at`, or `None` when the class sets no budget.
    pub fn slo_ttft_due_at(&self, ingress_at: Instant) -> Option<Instant> {
        // The `u32` millisecond budget keeps the sum within the monotonic clock's range.
        self.slo_ttft_ms
            .map(|slo_ttft_ms| ingress_at + Duration::from_millis(u64::from(slo_ttft_ms.get())))
    }
}

/// Why a class table is not a valid profile.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AdmissionConfigError {
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

/// A requested policy class name that matches no configured class. The kernel refuses the
/// request at queue entry with it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown policy class {0:?}")]
pub struct UnknownPolicyClassError(pub String);

/// One flat set of policy classes, each selected directly by its name.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    class_indices: HashMap<String, usize>,
    default_class_index: usize,
}

impl PolicyProfile {
    /// A profile from its resolved classes, refused when a name repeats, `default_policy_class`
    /// names no class, or more than one class is configured without a default.
    pub fn from_classes(
        classes: Vec<PolicyClassConfig>,
        default_policy_class: Option<&str>,
    ) -> Result<Self, AdmissionConfigError> {
        let invalid = |message: String| Err(AdmissionConfigError::Validation(message));
        if classes.is_empty() {
            return invalid("profile has no policy classes".to_string());
        }
        let mut class_indices = HashMap::with_capacity(classes.len());
        for (index, class) in classes.iter().enumerate() {
            if class_indices.insert(class.name.clone(), index).is_some() {
                return invalid(format!("contains duplicate policy class {:?}", class.name));
            }
        }
        let default_class_index = match default_policy_class {
            Some(name) => match class_indices.get(name) {
                Some(&index) => index,
                None => {
                    return invalid(format!(
                        "default_policy_class {name:?} does not name a configured policy class"
                    ));
                }
            },
            None if classes.len() == 1 => 0,
            None => {
                return invalid(
                    "default_policy_class is required with more than one policy class".to_string(),
                );
            }
        };
        Ok(Self {
            classes,
            class_indices,
            default_class_index,
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
            slo_ttft_ms: None,
            controlled_delay: false,
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

    /// Resolves a requested class name to its queue. An omitted name selects the default
    /// class; a name that matches no configured class is an error.
    pub fn resolve_class_index(
        &self,
        requested: Option<&str>,
    ) -> Result<usize, UnknownPolicyClassError> {
        match requested {
            None => Ok(self.default_class_index),
            Some(name) => self
                .resolve_class_index_strict(name)
                .ok_or_else(|| UnknownPolicyClassError(name.to_string())),
        }
    }

    pub fn class(&self, index: usize) -> &PolicyClassConfig {
        &self.classes[index]
    }

    /// Resolves a configured class name, or `None` for any other name.
    pub fn resolve_class_index_strict(&self, requested: &str) -> Option<usize> {
        self.class_indices.get(requested).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(name: &str) -> PolicyClassConfig {
        PolicyClassConfig {
            name: name.to_string(),
            queue_policy: RouterQueuePolicy::Edf,
            quantum: 1,
            slo_ttft_ms: None,
            controlled_delay: false,
            busy_thresholds: BusyThresholds::default(),
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        }
    }

    #[test]
    fn flat_profile_resolves_names_and_refuses_unknown_ones() {
        let profile = PolicyProfile::from_classes(vec![class("a"), class("b")], Some("b")).unwrap();
        assert_eq!(profile.default_class().name, "b");
        assert_eq!(profile.resolve_class_index(None), Ok(1));
        assert_eq!(profile.resolve_class_index(Some("a")), Ok(0));
        assert_eq!(
            profile.resolve_class_index(Some("missing")),
            Err(UnknownPolicyClassError("missing".to_string()))
        );
        assert_eq!(profile.resolve_class_index_strict("missing"), None);
        assert_eq!(
            UnknownPolicyClassError("missing".to_string()).to_string(),
            "unknown policy class \"missing\""
        );
    }

    #[test]
    fn flat_profile_validates_its_class_table() {
        let message = |result: Result<PolicyProfile, AdmissionConfigError>| match result {
            Err(AdmissionConfigError::Validation(message)) => message,
            Ok(profile) => panic!("accepted {profile:?}"),
        };
        assert_eq!(
            message(PolicyProfile::from_classes(vec![], None)),
            "profile has no policy classes"
        );
        assert_eq!(
            message(PolicyProfile::from_classes(
                vec![class("same"), class("same")],
                None
            )),
            "contains duplicate policy class \"same\""
        );
        assert_eq!(
            message(PolicyProfile::from_classes(
                vec![class("a"), class("b")],
                Some("c")
            )),
            "default_policy_class \"c\" does not name a configured policy class"
        );
        assert_eq!(
            message(PolicyProfile::from_classes(
                vec![class("a"), class("b")],
                None
            )),
            "default_policy_class is required with more than one policy class"
        );
        // One class needs no default.
        assert_eq!(
            PolicyProfile::from_classes(vec![class("only")], None)
                .unwrap()
                .default_class()
                .name,
            "only"
        );
    }

    #[test]
    fn class_budget_expires_only_with_controlled_delay_under_edf() {
        let mut budgeted = class("budgeted");
        budgeted.slo_ttft_ms = NonZeroU32::new(250);
        let ingress_at = Instant::now();
        assert_eq!(
            budgeted.slo_ttft_due_at(ingress_at),
            Some(ingress_at + Duration::from_millis(250))
        );
        assert_eq!(class("unbudgeted").slo_ttft_due_at(ingress_at), None);
        assert!(!budgeted.expires_class_budget());
        budgeted.controlled_delay = true;
        assert!(budgeted.expires_class_budget());
        budgeted.queue_policy = RouterQueuePolicy::Fcfs;
        assert!(!budgeted.expires_class_budget());
    }

    #[test]
    fn queue_policy_round_trips_every_spelling() {
        assert_eq!(RouterQueuePolicy::default(), RouterQueuePolicy::Fcfs);
        for (spelling, policy) in [
            ("edf", RouterQueuePolicy::Edf),
            ("fcfs", RouterQueuePolicy::Fcfs),
            ("lcfs", RouterQueuePolicy::Lcfs),
            ("wspt", RouterQueuePolicy::Wspt),
        ] {
            assert_eq!(spelling.parse::<RouterQueuePolicy>().unwrap(), policy);
            assert_eq!(policy.to_string(), spelling);
        }
        assert!(
            "random"
                .parse::<RouterQueuePolicy>()
                .unwrap_err()
                .contains("expected 'edf', 'fcfs', 'lcfs', or 'wspt'")
        );
    }
}
