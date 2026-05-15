// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dev` module protocol: tier reset.

use serde::{Deserialize, Serialize};

use crate::control::ControlError;

// ---------------------------------------------------------------------------
// Tier
// ---------------------------------------------------------------------------

/// Logical block-manager tier identifier.
///
/// Add variants here as new tiers come online. Wire format mirror
/// (`"g2"` / `"g3"`) is held by `#[serde(rename_all = "lowercase")]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    G2,
    G3,
}

impl Tier {
    /// Iteration order used by `reset` when honoring "all" — outer
    /// tiers (closer to GPU) first.
    pub const ORDERED: &'static [Tier] = &[Tier::G2, Tier::G3];
}

// ---------------------------------------------------------------------------
// Velo handler name
// ---------------------------------------------------------------------------

/// Velo handler name for the leader reset operation.
///
/// Kept at its original `kvbm.connector.leader.*` value (even though the
/// handler now lives in the engine control plane) so the kvbm-hub HTTP→velo
/// proxy needs no change. Renaming to `kvbm.leader.control.*` is a flagged
/// follow-up.
pub const RESET_HANDLER: &str = "kvbm.connector.leader.reset";

// ---------------------------------------------------------------------------
// Reset request/response
// ---------------------------------------------------------------------------

/// Reset request payload.
///
/// Semantics:
/// - `tiers: None` (or an empty `tiers` field in JSON) — reset every
///   tier currently configured on this leader. Missing tiers are
///   silently skipped and reported in [`ResetResponse::skipped_unconfigured`].
/// - `tiers: Some(list)` — reset exactly the listed tiers. If any
///   requested tier is not configured on this leader, the request
///   fails with [`ControlError::TierNotConfigured`] **before any tier
///   is reset**. Per-tier reset failures populate
///   [`ResetResponse::failed`] without aborting the rest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResetRequest {
    #[serde(default)]
    pub tiers: Option<Vec<Tier>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetResponse {
    pub reset: Vec<Tier>,
    pub failed: Vec<TierError>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_unconfigured: Vec<Tier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierError {
    pub tier: Tier,
    pub message: String,
}

// ---------------------------------------------------------------------------
// plan_reset (precondition logic)
// ---------------------------------------------------------------------------

/// Pre-validate a [`ResetRequest`] against the set of tiers the leader
/// has configured. Returns `(tiers_to_reset, skipped_unconfigured)`.
///
/// Free function so transports/tests can run the precondition logic
/// without holding a leader.
pub fn plan_reset(
    req: &ResetRequest,
    available: &std::collections::HashSet<Tier>,
) -> Result<(Vec<Tier>, Vec<Tier>), ControlError> {
    match req.tiers.as_deref() {
        Some(list) => {
            // Explicit list — reject atomically if any named tier is missing.
            for t in list {
                if !available.contains(t) {
                    return Err(ControlError::TierNotConfigured(*t));
                }
            }
            // Preserve caller order, dedupe.
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::with_capacity(list.len());
            for t in list {
                if seen.insert(*t) {
                    out.push(*t);
                }
            }
            Ok((out, Vec::new()))
        }
        None => {
            let mut to_reset = Vec::new();
            let mut skipped = Vec::new();
            for t in Tier::ORDERED.iter().copied() {
                if available.contains(&t) {
                    to_reset.push(t);
                } else {
                    skipped.push(t);
                }
            }
            Ok((to_reset, skipped))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn avail(tiers: &[Tier]) -> HashSet<Tier> {
        tiers.iter().copied().collect()
    }

    #[test]
    fn reset_request_default_serde() {
        let parsed: ResetRequest = serde_json::from_str("{}").unwrap();
        assert!(parsed.tiers.is_none());
    }

    #[test]
    fn reset_response_skipped_omitted_when_empty() {
        let r = ResetResponse {
            reset: vec![Tier::G2],
            failed: vec![],
            skipped_unconfigured: vec![],
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(!s.contains("skipped_unconfigured"));
    }

    #[test]
    fn plan_all_with_g2_only() {
        let (r, s) = plan_reset(&ResetRequest::default(), &avail(&[Tier::G2])).unwrap();
        assert_eq!(r, vec![Tier::G2]);
        assert_eq!(s, vec![Tier::G3]);
    }

    #[test]
    fn plan_all_with_both() {
        let (r, s) = plan_reset(&ResetRequest::default(), &avail(&[Tier::G2, Tier::G3])).unwrap();
        assert_eq!(r, vec![Tier::G2, Tier::G3]);
        assert!(s.is_empty());
    }

    #[test]
    fn plan_explicit_present() {
        let req = ResetRequest {
            tiers: Some(vec![Tier::G2]),
        };
        let (r, s) = plan_reset(&req, &avail(&[Tier::G2, Tier::G3])).unwrap();
        assert_eq!(r, vec![Tier::G2]);
        assert!(s.is_empty());
    }

    #[test]
    fn plan_explicit_missing_fails_atomically() {
        let req = ResetRequest {
            tiers: Some(vec![Tier::G3]),
        };
        let err = plan_reset(&req, &avail(&[Tier::G2])).unwrap_err();
        assert_eq!(err, ControlError::TierNotConfigured(Tier::G3));
    }

    #[test]
    fn plan_explicit_mix_fails_before_any_reset() {
        let req = ResetRequest {
            tiers: Some(vec![Tier::G2, Tier::G3]),
        };
        let err = plan_reset(&req, &avail(&[Tier::G2])).unwrap_err();
        assert_eq!(err, ControlError::TierNotConfigured(Tier::G3));
    }

    #[test]
    fn plan_explicit_dedupes() {
        let req = ResetRequest {
            tiers: Some(vec![Tier::G2, Tier::G2, Tier::G3]),
        };
        let (r, _) = plan_reset(&req, &avail(&[Tier::G2, Tier::G3])).unwrap();
        assert_eq!(r, vec![Tier::G2, Tier::G3]);
    }
}
