// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public KVBM leader control plane.
//!
//! These types are wire-format-stable and travel two paths with the same
//! semantics:
//! 1. The leader's velo handlers (registered by the engine `ControlPlane`).
//! 2. The hub's HTTP→velo proxy.
//!
//! Living here (rather than in `kvbm-connector`) lets `kvbm-hub` map velo
//! replies to HTTP status codes — and, behind the `client` feature, *call* the
//! control plane — without a cargo dep on `kvbm-connector` or `kvbm-engine`.
//!
//! ## Module layout
//!
//! - [`reset`], [`leader`] — the always-on `core` module's request/response
//!   types (re-exported at this level for wire-compatible imports).
//! - [`modules`] — per-plugin protocol slices (`tests`, `transfer`).
//! - `client` (feature `client`) — the [`client::LeaderControlClient`] that
//!   speaks all of the above.

pub mod layout_compat;
pub mod modules;

#[cfg(feature = "client")]
pub mod client;

// Re-export the public client surface at `control::*` so consumers can write
// `use kvbm_protocols::control::LeaderControlClient` without reaching into
// the `client` submodule.
#[cfg(feature = "client")]
pub use client::{CoreClient, DevClient, LeaderControlClient};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-export the `core` + `dev` modules' protocol surface at `control::*` so
// callers migrating from the old `kvbm-control-protocol` crate keep flat
// import paths.
pub use layout_compat::{LayoutCompatPayload, check_layout_compat};
pub use modules::core::{
    DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest, DisaggRole, HostInfo, InstanceDescription,
    LayerRange, LayoutConfigDescription, LayoutDescription, ParallelismDescription,
    StorageKindDescription, TierCapacity, TierKind, WorkerInfo,
};
pub use modules::dev::{RESET_HANDLER, ResetRequest, ResetResponse, Tier, TierError, plan_reset};
pub use modules::metrics::{
    MetricsSnapshotRequest, MetricsSnapshotResponse, PoolBreakdown, SNAPSHOT_HANDLER,
};
pub use modules::transfer::{
    CLOSE_SESSION_HANDLER, CloseTransferSessionRequest, CloseTransferSessionResponse, FindMode,
    MatchBreakdown, OPEN_SESSION_HANDLER, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PULL_FROM_SESSION_HANDLER, PullFromSessionRequest, PullFromSessionResponse,
    SEARCH_PREFIX_HANDLER, SEARCH_SCATTER_HANDLER, SearchMode, SearchRequest, SearchResponse,
    TierSelection, TransferSessionCapability,
};

// ---------------------------------------------------------------------------
// ModuleId — the plugin registry
// ---------------------------------------------------------------------------

/// Stable identifier for a control-plane module (plugin).
///
/// A "module" is a togglable bundle of {protocol types + client + service
/// impl}. The protocol/client halves live under [`modules`]; the service-impl
/// half lives in `kvbm-engine::leader::control`. `ModuleId` is the shared
/// identity that ties the halves together and is what
/// [`ListModulesResponse`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModuleId {
    /// Always-on: `describe_instance`.
    Core,
    /// Opt-in operator/debug tooling: `reset`. Safe in production.
    Dev,
    /// Always-on: G2 search → disagg-session creation (and, later,
    /// `transfer_to` / `transfer_from`).
    Transfer,
    /// Opt-in: on-demand runtime snapshot (`snapshot`) — per-pool block
    /// populations + in-flight session count, sourced from the leader's
    /// Prometheus registry. Registered when the leader was built with
    /// observability available; for dev/test eyeballing.
    Metrics,
}

impl ModuleId {
    /// Stable string discriminant (matches the snake_case wire form).
    pub fn as_str(&self) -> &'static str {
        match self {
            ModuleId::Core => "core",
            ModuleId::Dev => "dev",
            ModuleId::Transfer => "transfer",
            ModuleId::Metrics => "metrics",
        }
    }
}

// ---------------------------------------------------------------------------
// list_modules — "which plugins are enabled" query
// ---------------------------------------------------------------------------

/// Velo handler name for the control-plane module-introspection query.
pub const LIST_MODULES_HANDLER: &str = "kvbm.leader.control.list_modules";

/// Request for [`LIST_MODULES_HANDLER`]. Empty — the target instance is
/// addressed by the velo call itself.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListModulesRequest {}

/// Response for [`LIST_MODULES_HANDLER`]: the modules enabled on the queried
/// leader instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModulesResponse {
    pub modules: Vec<ModuleId>,
}

// ---------------------------------------------------------------------------
// ControlError
// ---------------------------------------------------------------------------

/// Control-plane errors. Both transports map this enum to their own
/// status / error encoding via [`ControlError::http_status`].
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
        instance_id: velo_ext::InstanceId,
        reason: String,
    },

    /// Caller invoked a handler whose module is not enabled on this leader.
    #[error("control module {0:?} is not enabled on this leader")]
    ModuleNotEnabled(ModuleId),

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
            ControlError::ModuleNotEnabled(_) => 404,
            ControlError::Internal(_) => 500,
        }
    }

    /// Stable kind discriminant for error envelopes (string in JSON).
    pub fn kind(&self) -> &'static str {
        match self {
            ControlError::TierNotConfigured(_) => "tier_not_configured",
            ControlError::NotInitialized => "not_initialized",
            ControlError::PeerNotFound { .. } => "peer_not_found",
            ControlError::ModuleNotEnabled(_) => "module_not_enabled",
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

#[cfg(test)]
mod tests {
    use super::*;
    use velo_ext::InstanceId;

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
        assert_eq!(
            ControlError::ModuleNotEnabled(ModuleId::Dev).http_status(),
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
    fn module_id_round_trips_json() {
        for id in [
            ModuleId::Core,
            ModuleId::Dev,
            ModuleId::Transfer,
            ModuleId::Metrics,
        ] {
            let s = serde_json::to_string(&id).unwrap();
            assert!(s.contains(id.as_str()));
            let back: ModuleId = serde_json::from_str(&s).unwrap();
            assert_eq!(back, id);
        }
    }
}
