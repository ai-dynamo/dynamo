// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker roles for the standalone EPP's disaggregated topology.
//!
//! In `DYN_EPP_TOPOLOGY_MODE=disaggregated` every pod selected by the
//! `InferencePool` must carry a role label whose value names the stage that pod
//! serves. The EPP splits the pool on that label into a prefill catalog and a
//! decode catalog, each backing its own embedded `SelectionService`.
//!
//! In `aggregated` mode the label is never read and every eligible pod is
//! [`WorkerRole::Aggregated`].

use std::fmt;
use std::str::FromStr;

use dynamo_kv_router::WorkerType;

/// Default pod label key naming a worker's role, per DEP #11661's environment
/// contract. Overridable with `DYN_EPP_WORKER_ROLE_LABEL`.
///
/// Deliberately not the operator's `nvidia.com/dynamo-component-type`: that key
/// is the operator's own workload-selector contract and admits the value
/// `worker`, which names no disaggregated stage. The standalone path fronts
/// user-managed raw engine Deployments that carry no `nvidia.com/*` labels at
/// all, so the operator's key would match nothing anyway.
pub const DEFAULT_WORKER_ROLE_LABEL: &str = "nvidia.com/role";

/// The stage a discovered worker serves.
///
/// `PartialEq`/`Eq` are load-bearing, not conveniences: `RawWorker` and
/// `WorkerEntry` derive them, and the pod watch loop decides whether the derived
/// index changed by comparing whole entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerRole {
    /// Serves prefill and decode in one process. The only role in `aggregated`
    /// topology, and never produced in `disaggregated` topology.
    Aggregated,
    /// Computes prompt KV and hands it off. A selection input only — never a
    /// gateway destination.
    Prefill,
    /// Receives transferred KV and generates tokens. The gateway destination in
    /// `disaggregated` topology.
    Decode,
}

impl WorkerRole {
    /// Canonical lowercase form, for logs and error messages.
    ///
    /// This is **not** the inverse of [`WorkerRole::from_pod_label`].
    /// `Aggregated.as_str()` is `"aggregated"`, but that value is rejected as a
    /// pod label: it describes an EPP-wide topology, not a stage a pod can
    /// claim. Only `prefill` and `decode` round-trip.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Aggregated => "aggregated",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }

    /// Resolve a role from a pod label value.
    ///
    /// Delegates to [`WorkerType::from_str`] — which trims and lowercases — so
    /// the accepted vocabulary cannot drift from the operator's component-type
    /// enum, then narrows to the two stages a disaggregated pool may contain.
    /// `encode` and `aggregated` parse as worker types but are not roles here.
    pub fn from_pod_label(value: &str) -> Result<Self, RoleLabelError> {
        match WorkerType::from_str(value) {
            Ok(WorkerType::Prefill) => Ok(Self::Prefill),
            Ok(WorkerType::Decode) => Ok(Self::Decode),
            Ok(_) | Err(_) => Err(RoleLabelError::Invalid {
                token: value.to_string(),
            }),
        }
    }
}

impl fmt::Display for WorkerRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why an otherwise-eligible pod could not be assigned a role.
///
/// Only reachable in `disaggregated` topology: `aggregated` never reads the
/// label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoleLabelError {
    /// The pod carries no value for the configured role-label key.
    Missing,
    /// The pod carries a value that names no disaggregated stage.
    Invalid { token: String },
}

impl RoleLabelError {
    /// Bounded, pod-independent discriminant for structured logs.
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Missing => "role_label_missing",
            Self::Invalid { .. } => "role_label_invalid",
        }
    }
}

impl fmt::Display for RoleLabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => f.write_str("role label is missing"),
            Self::Invalid { token } => {
                write!(f, "role label value {token:?} names no worker role")
            }
        }
    }
}

/// Ready-worker count per role, maintained alongside the worker index.
///
/// Exists so the per-request emptiness check stays O(1) after the index gains a
/// role dimension; scanning the map instead would put an O(workers) pass on
/// every pick.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RoleCounts {
    pub aggregated: usize,
    pub prefill: usize,
    pub decode: usize,
}

impl RoleCounts {
    pub const fn get(self, role: WorkerRole) -> usize {
        match role {
            WorkerRole::Aggregated => self.aggregated,
            WorkerRole::Prefill => self.prefill,
            WorkerRole::Decode => self.decode,
        }
    }

    pub fn add(&mut self, role: WorkerRole) {
        *self.slot(role) += 1;
    }

    /// Saturating so a bookkeeping slip cannot underflow into a huge count and
    /// make an empty catalog look occupied.
    pub fn remove(&mut self, role: WorkerRole) {
        let slot = self.slot(role);
        *slot = slot.saturating_sub(1);
    }

    fn slot(&mut self, role: WorkerRole) -> &mut usize {
        match role {
            WorkerRole::Aggregated => &mut self.aggregated,
            WorkerRole::Prefill => &mut self.prefill,
            WorkerRole::Decode => &mut self.decode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_two_disaggregated_stages() {
        assert_eq!(
            WorkerRole::from_pod_label("prefill").unwrap(),
            WorkerRole::Prefill
        );
        assert_eq!(
            WorkerRole::from_pod_label("decode").unwrap(),
            WorkerRole::Decode
        );
    }

    #[test]
    fn parsing_trims_and_lowercases_via_worker_type() {
        // Kubernetes label values cannot actually carry surrounding whitespace,
        // but the vocabulary is delegated to `WorkerType::from_str`, so this
        // pins the leniency we inherit rather than leaving it undiscovered.
        assert_eq!(
            WorkerRole::from_pod_label(" Decode ").unwrap(),
            WorkerRole::Decode
        );
        assert_eq!(
            WorkerRole::from_pod_label("PREFILL").unwrap(),
            WorkerRole::Prefill
        );
    }

    #[test]
    fn rejects_values_that_name_no_disaggregated_stage() {
        // `encode` and `aggregated` are valid WorkerTypes; the narrowing is what
        // rejects them. `worker` is the operator's component-type value.
        for value in ["encode", "aggregated", "worker", "", "gibberish"] {
            let error = WorkerRole::from_pod_label(value)
                .expect_err("value should not resolve to a disaggregated role");
            assert_eq!(error.reason(), "role_label_invalid");
            assert_eq!(
                error,
                RoleLabelError::Invalid {
                    token: value.into()
                }
            );
        }
    }

    #[test]
    fn invalid_token_preserves_the_original_spelling() {
        // The operator needs to see what they actually wrote, not a normalized
        // form, to spot a typo or a stray character.
        let error = WorkerRole::from_pod_label(" Prefil ").unwrap_err();
        assert_eq!(
            error,
            RoleLabelError::Invalid {
                token: " Prefil ".to_string()
            }
        );
    }

    #[test]
    fn aggregated_is_display_only_and_does_not_round_trip() {
        // `as_str` feeds logs and the RoleCatalogEmpty message; it is not a
        // label serializer. Guarding this stops a future round-trip test or a
        // label built from `as_str()` from silently accepting `aggregated`.
        assert_eq!(WorkerRole::Aggregated.as_str(), "aggregated");
        assert!(WorkerRole::from_pod_label(WorkerRole::Aggregated.as_str()).is_err());

        for role in [WorkerRole::Prefill, WorkerRole::Decode] {
            assert_eq!(WorkerRole::from_pod_label(role.as_str()).unwrap(), role);
        }
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(WorkerRole::Prefill.to_string(), "prefill");
        assert_eq!(WorkerRole::Decode.to_string(), "decode");
    }

    #[test]
    fn missing_and_invalid_have_distinct_reasons() {
        assert_eq!(RoleLabelError::Missing.reason(), "role_label_missing");
        assert_eq!(
            RoleLabelError::Invalid {
                token: "x".to_string()
            }
            .reason(),
            "role_label_invalid"
        );
    }

    #[test]
    fn counts_track_each_role_independently() {
        let mut counts = RoleCounts::default();
        counts.add(WorkerRole::Prefill);
        counts.add(WorkerRole::Prefill);
        counts.add(WorkerRole::Decode);

        assert_eq!(counts.get(WorkerRole::Prefill), 2);
        assert_eq!(counts.get(WorkerRole::Decode), 1);
        assert_eq!(counts.get(WorkerRole::Aggregated), 0);

        counts.remove(WorkerRole::Prefill);
        assert_eq!(counts.get(WorkerRole::Prefill), 1);
        assert_eq!(counts.get(WorkerRole::Decode), 1);
    }

    #[test]
    fn removing_below_zero_saturates() {
        let mut counts = RoleCounts::default();
        counts.remove(WorkerRole::Decode);
        assert_eq!(counts.get(WorkerRole::Decode), 0);
    }
}
