// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker roles for the standalone EPP's disaggregated topology.
//!
//! In `aggregated` topology the label is never read and every eligible pod is
//! [`WorkerRole::Aggregated`].

use std::fmt;
use std::str::FromStr;

use dynamo_kv_router::WorkerType;

/// Default pod label key naming a worker's role; `DYN_EPP_WORKER_ROLE_LABEL`
/// overrides it, for example to reuse the operator's component-type label.
///
/// Not the operator's `nvidia.com/dynamo-component-type` by default: that key
/// admits the value `worker`, which names no disaggregated stage, and the raw
/// engine Deployments this path fronts do not carry it.
pub const DEFAULT_WORKER_ROLE_LABEL: &str = "nvidia.com/dynamo-worker-role";

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

    /// Delegates to [`WorkerType::from_str`] so the accepted vocabulary cannot
    /// drift from the operator's component-type enum.
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
    fn from_pod_label_narrows_worker_type_to_the_two_stages() {
        // Label values cannot carry whitespace, but the vocabulary is delegated
        // to `WorkerType::from_str`, so pin the trim/lowercase leniency inherited
        // from it. `encode` and `aggregated` are valid WorkerTypes; the
        // narrowing is what rejects them.
        let invalid = |token: &str| -> Result<WorkerRole, RoleLabelError> {
            Err(RoleLabelError::Invalid {
                token: token.to_string(),
            })
        };
        let cases = [
            ("prefill", Ok(WorkerRole::Prefill)),
            ("decode", Ok(WorkerRole::Decode)),
            (" Decode ", Ok(WorkerRole::Decode)),
            ("PREFILL", Ok(WorkerRole::Prefill)),
            ("encode", invalid("encode")),
            ("aggregated", invalid("aggregated")),
            ("worker", invalid("worker")),
            ("", invalid("")),
            ("gibberish", invalid("gibberish")),
            // The token keeps the original spelling, not the normalized form.
            (" Prefil ", invalid(" Prefil ")),
        ];
        for (input, want) in cases {
            let got = WorkerRole::from_pod_label(input);
            assert_eq!(got, want, "{input:?}");
            if let Err(error) = got {
                assert_eq!(error.reason(), "role_label_invalid", "{input:?}");
            }
        }
    }

    #[test]
    fn display_matches_as_str_and_only_stages_round_trip() {
        let cases = [
            (WorkerRole::Aggregated, "aggregated", false),
            (WorkerRole::Prefill, "prefill", true),
            (WorkerRole::Decode, "decode", true),
        ];
        for (role, text, round_trips) in cases {
            assert_eq!(role.as_str(), text, "{text}");
            assert_eq!(role.to_string(), text, "{text}");
            assert_eq!(
                WorkerRole::from_pod_label(text).ok(),
                round_trips.then_some(role),
                "{text}"
            );
        }

        assert_eq!(RoleLabelError::Missing.reason(), "role_label_missing");
        assert_ne!(
            RoleLabelError::Missing.reason(),
            RoleLabelError::Invalid {
                token: "x".to_string()
            }
            .reason()
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

        counts.remove(WorkerRole::Decode);
        counts.remove(WorkerRole::Decode);
        assert_eq!(
            counts.get(WorkerRole::Decode),
            0,
            "remove saturates at zero"
        );
    }
}
