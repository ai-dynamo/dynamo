// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared protocol for the KVBM connector-leader control plane.
//!
//! These types are wire-format-stable and travel three paths with the
//! same semantics:
//! 1. The connector's local axum HTTP server (when enabled).
//! 2. The connector's velo handlers
//!    (`kvbm.connector.leader.{reset,register_leader}`).
//! 3. The hub's HTTP→velo proxy.
//!
//! Lifting them out of `kvbm-connector` lets `kvbm-hub` map velo error
//! responses to HTTP status codes without taking a cargo dep on
//! `kvbm-connector` (which would create a cycle).

use serde::{Deserialize, Serialize};
use thiserror::Error;
use velo_common::InstanceId;

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
// ControlError
// ---------------------------------------------------------------------------

/// Connector-control errors. Both transports map this enum to their
/// own status / error encoding via [`ControlError::http_status`].
#[derive(Debug, Error, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum ControlError {
    /// Caller asked for an explicit list that names a tier this leader
    /// does not have configured. The reset is rejected atomically; no
    /// tier is touched.
    #[error("tier {0:?} is not configured on this leader")]
    TierNotConfigured(Tier),

    /// Leader's `InstanceLeader` hasn't been built yet (workers not
    /// initialized).
    #[error("InstanceLeader is not yet initialized")]
    NotInitialized,

    /// `discover_and_register_peer` failed for a leader registration.
    #[error("could not discover peer {instance_id}: {reason}")]
    PeerNotFound {
        instance_id: InstanceId,
        reason: String,
    },

    /// Generic internal error from the underlying engine.
    #[error("internal: {0}")]
    Internal(String),
}

impl ControlError {
    /// HTTP status code semantically equivalent to this error.
    ///
    /// Used by both the connector's local axum shim and the hub's
    /// HTTP→velo proxy so the operator sees the same status code
    /// regardless of which transport reached the leader.
    pub fn http_status(&self) -> u16 {
        match self {
            ControlError::TierNotConfigured(_) => 400,
            ControlError::NotInitialized => 503,
            ControlError::PeerNotFound { .. } => 404,
            ControlError::Internal(_) => 500,
        }
    }

    /// Stable kind discriminant for error envelopes (string in JSON).
    pub fn kind(&self) -> &'static str {
        match self {
            ControlError::TierNotConfigured(_) => "tier_not_configured",
            ControlError::NotInitialized => "not_initialized",
            ControlError::PeerNotFound { .. } => "peer_not_found",
            ControlError::Internal(_) => "internal",
        }
    }
}

impl From<anyhow::Error> for ControlError {
    fn from(e: anyhow::Error) -> Self {
        Self::Internal(format!("{e:#}"))
    }
}

// ---------------------------------------------------------------------------
// ControlReply envelope
// ---------------------------------------------------------------------------

/// Wire envelope for control-handler responses sent over velo.
///
/// Velo `Handler::typed_unary_async` requires the handler to return
/// `Result<Reply, _>` where the `Err` half is reserved for transport
/// failures. Application-level success vs. failure is carried inside
/// `Reply` itself via this enum so both halves serialize as a single
/// JSON shape:
/// `{"status":"ok", "Ok": <inner>}` or `{"status":"err", "Err": <ControlError>}`.
///
/// The hub's HTTP→velo proxy can deserialize the bytes as
/// `ControlReply<serde_json::Value>` to introspect status without
/// knowing the inner shape, then map to an HTTP status code via
/// [`ControlError::http_status`].
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ControlReply<T> {
    Ok(T),
    Err(ControlError),
}

impl<T> From<Result<T, ControlError>> for ControlReply<T> {
    fn from(r: Result<T, ControlError>) -> Self {
        match r {
            Ok(v) => ControlReply::Ok(v),
            Err(e) => ControlReply::Err(e),
        }
    }
}

impl<T> ControlReply<T> {
    pub fn into_result(self) -> Result<T, ControlError> {
        match self {
            ControlReply::Ok(v) => Ok(v),
            ControlReply::Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Reset
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
// Register-leader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterLeaderRequest {
    pub instance_id: InstanceId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterLeaderResponse {
    pub status: RegisterLeaderStatus,
    pub remote_leaders: Vec<InstanceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterLeaderStatus {
    Registered,
    AlreadyRegistered,
}

// ---------------------------------------------------------------------------
// Velo handler names
// ---------------------------------------------------------------------------

/// Velo handler name for the connector-leader reset operation.
pub const RESET_HANDLER: &str = "kvbm.connector.leader.reset";

/// Velo handler name for the connector-leader register-leader operation.
pub const REGISTER_LEADER_HANDLER: &str = "kvbm.connector.leader.register_leader";

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
    fn http_status_mapping() {
        assert_eq!(ControlError::TierNotConfigured(Tier::G3).http_status(), 400);
        assert_eq!(ControlError::NotInitialized.http_status(), 503);
        assert_eq!(
            ControlError::PeerNotFound {
                instance_id: InstanceId::new_v4(),
                reason: "x".into(),
            }
            .http_status(),
            404
        );
        assert_eq!(ControlError::Internal("x".into()).http_status(), 500);
    }

    #[test]
    fn control_reply_ok_struct_roundtrip() {
        let ok: ControlReply<ResetResponse> = ControlReply::Ok(ResetResponse {
            reset: vec![Tier::G2],
            failed: vec![],
            skipped_unconfigured: vec![Tier::G3],
        });
        let s = serde_json::to_string(&ok).unwrap();
        assert!(s.contains(r#""status":"ok""#));
        let back: ControlReply<ResetResponse> = serde_json::from_str(&s).unwrap();
        match back.into_result() {
            Ok(r) => {
                assert_eq!(r.reset, vec![Tier::G2]);
                assert_eq!(r.skipped_unconfigured, vec![Tier::G3]);
            }
            Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn control_reply_err_roundtrip() {
        let err: ControlReply<ResetResponse> =
            ControlReply::Err(ControlError::TierNotConfigured(Tier::G3));
        let s = serde_json::to_string(&err).unwrap();
        assert!(s.contains(r#""status":"err""#));
        let back: ControlReply<ResetResponse> = serde_json::from_str(&s).unwrap();
        assert!(matches!(
            back.into_result(),
            Err(ControlError::TierNotConfigured(Tier::G3))
        ));
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
