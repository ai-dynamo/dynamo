// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `transfer` module protocol: multi-tier search → disagg-session creation,
//! puller-side pull driver, explicit close.
//!
//! Three primary handlers form the v1 surface:
//!
//! - [`OPEN_SESSION_HANDLER`] — dispatched at the *holder*. Searches the
//!   holder's tiers per [`OpenTransferSessionRequest::tiers`], opens a
//!   disagg session, commits matches, and (eventually) makes their G2-tier
//!   blocks available. Returns the attach triple in
//!   [`OpenTransferSessionResponse`].
//! - [`PULL_FROM_SESSION_HANDLER`] — dispatched at the *puller*. Attaches
//!   to a session living on a holder, drains its `commits()` /
//!   `availability()` streams, and pulls into the puller's local G2 pool.
//!   Long-poll unary: the RPC returns when the pull is complete.
//! - [`CLOSE_SESSION_HANDLER`] — dispatched at the *holder*. Idempotent
//!   teardown of a session by id.
//!
//! Two legacy handlers ([`SEARCH_PREFIX_HANDLER`], [`SEARCH_SCATTER_HANDLER`])
//! are retained as thin shims for back-compat with the hub's existing HTTP
//! routes; they delegate to `open_session` with `find_mode = Sync` and the
//! corresponding [`SearchMode`].

use std::time::Duration;

use kvbm_common::SequenceHash;
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

use crate::disagg::{SessionEndpoint, SessionId};

// ---------------------------------------------------------------------------
// Handler names
// ---------------------------------------------------------------------------

/// Velo handler name for the unified open-session call.
pub const OPEN_SESSION_HANDLER: &str = "kvbm.leader.control.open_session";

/// Velo handler name for the puller-side attach-and-pull call.
pub const PULL_FROM_SESSION_HANDLER: &str = "kvbm.leader.control.pull_from_session";

/// Velo handler name for explicit session teardown.
pub const CLOSE_SESSION_HANDLER: &str = "kvbm.leader.control.close_session";

/// Legacy handler: contiguous-prefix G2 search, kept as a shim over
/// `open_session` with `find_mode = Sync`, `tiers = default`, and
/// `search_mode = Prefix`.
pub const SEARCH_PREFIX_HANDLER: &str = "kvbm.leader.control.search_prefix";

/// Legacy handler: scatter (gather-all) G2 search, kept as a shim over
/// `open_session` with `find_mode = Sync`, `tiers = default`, and
/// `search_mode = Scatter`.
pub const SEARCH_SCATTER_HANDLER: &str = "kvbm.leader.control.search_scatter";

// ---------------------------------------------------------------------------
// Option enums
// ---------------------------------------------------------------------------

/// How matched hashes are gathered from a tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Contiguous prefix — stop at the first miss. Maps to
    /// `BlockManager::match_blocks`. The right choice for LLM
    /// prompt-prefix KV reuse.
    #[default]
    Prefix,
    /// Gather every hash present, ignoring gaps. Maps to
    /// `BlockManager::scan_matches` with `touch = false` (does not
    /// perturb G2 LRU). For arbitrary-subset reuse and cross-session
    /// block sharing.
    Scatter,
}

/// Whether the open call awaits the multi-tier find before returning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindMode {
    /// Return immediately; the populator runs find + stage in the
    /// background. The puller learns of matches via the disagg
    /// `commits()` stream.
    #[default]
    Async,
    /// Await the cross-tier find phase; return matched hashes inline
    /// in [`OpenTransferSessionResponse::Sync`]. Staging (e.g. G3→G2)
    /// still runs in the background. Useful for orchestrators that
    /// fan out opens across several holders and compare matched sets
    /// before committing to a puller.
    Sync,
}

/// Tiers eligible for matching beyond G2 (G2 is always on).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TierSelection {
    /// Match in G3; staged G3→G2 in the background before
    /// `make_available`. v1 ships this off by default to preserve
    /// existing G2-only behavior — callers opt in.
    #[serde(default)]
    pub g3: bool,
    /// (v1.1) Match in G4 (object); same staging path. Currently
    /// ignored.
    #[serde(default)]
    pub g4: bool,
}

// ---------------------------------------------------------------------------
// MatchBreakdown — per-tier hit counts (telemetry)
// ---------------------------------------------------------------------------

/// Per-tier breakdown of where committed hashes were found on the
/// holder. Field names match the engine's internal
/// `leader::types::MatchBreakdown` so the engine-side conversion is
/// trivial.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MatchBreakdown {
    /// G2 (host) matches.
    #[serde(default)]
    pub host_blocks: usize,
    /// G3 (disk) matches.
    #[serde(default)]
    pub disk_blocks: usize,
    /// G4 (object) matches.
    #[serde(default)]
    pub object_blocks: usize,
}

// ---------------------------------------------------------------------------
// Open session
// ---------------------------------------------------------------------------

/// The bearer triple. Serialize + Send + Clone. Possession is
/// sufficient to attach to the session. The caller can ship this to
/// another instance (e.g. via the hub) and that instance can drive a
/// pull against the same holder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferSessionCapability {
    pub session_id: SessionId,
    pub instance_id: InstanceId,
    pub endpoint: SessionEndpoint,
}

/// Request for [`OPEN_SESSION_HANDLER`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenTransferSessionRequest {
    /// Hashes to look for, in the order the caller cares about (for
    /// [`SearchMode::Prefix`] this is also the prefix order).
    pub sequence_hashes: Vec<SequenceHash>,

    #[serde(default)]
    pub search_mode: SearchMode,

    #[serde(default)]
    pub find_mode: FindMode,

    #[serde(default)]
    pub tiers: TierSelection,

    /// Per-session watchdog override (milliseconds). `None` → the
    /// `SessionManager`'s default. Carried as `u64` ms rather than
    /// `Duration` to keep the JSON shape unambiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watchdog_ms: Option<u64>,
}

impl OpenTransferSessionRequest {
    /// Convenience: convert `watchdog_ms` to a `Duration`.
    pub fn watchdog(&self) -> Option<Duration> {
        self.watchdog_ms.map(Duration::from_millis)
    }
}

/// Response for [`OPEN_SESSION_HANDLER`].
///
/// `find_mode = Sync` and zero matches → [`Self::NoBlocksFound`] (no
/// session was opened). `find_mode = Async` always opens a session;
/// the puller observes a zero-match outcome via `CommitsClosed` with
/// an empty cumulative set.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum OpenTransferSessionResponse {
    /// `find_mode = Sync` only. No matches across the selected tiers;
    /// no session was created.
    NoBlocksFound,
    /// `find_mode = Async`. Puller learns matched set via the disagg
    /// `commits()` stream.
    Async {
        capability: TransferSessionCapability,
    },
    /// `find_mode = Sync` with at least one match. Matched set is
    /// inline.
    Sync {
        capability: TransferSessionCapability,
        committed: Vec<SequenceHash>,
        breakdown: MatchBreakdown,
    },
}

impl OpenTransferSessionResponse {
    /// Capability triple, if a session was opened.
    pub fn capability(&self) -> Option<&TransferSessionCapability> {
        match self {
            OpenTransferSessionResponse::NoBlocksFound => None,
            OpenTransferSessionResponse::Async { capability } => Some(capability),
            OpenTransferSessionResponse::Sync { capability, .. } => Some(capability),
        }
    }
}

// ---------------------------------------------------------------------------
// Pull from session
// ---------------------------------------------------------------------------

/// Request for [`PULL_FROM_SESSION_HANDLER`].
///
/// Dispatched at the puller. The puller attaches to the holder's
/// session and drains the requested blocks into its own G2 pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullFromSessionRequest {
    pub session_id: SessionId,
    /// The holder instance that owns the session.
    pub source_instance_id: InstanceId,
    /// Holder's disagg `SessionEndpoint`. The hub-orchestrated
    /// workflow passes this through from the open response. `None`
    /// requires the puller to look it up via the hub peer registry;
    /// v1 errors with [`crate::control::ControlError::Internal`]
    /// (`endpoint_required`) until that path is wired (v1.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<SessionEndpoint>,
    /// `None` → pull every committed hash (waits for stragglers).
    /// `Some(h)` → pull the intersection of committed with `h`;
    /// fails if any hash in `h` is not committed after the holder
    /// closes its commits stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<Vec<SequenceHash>>,
}

/// Response for [`PULL_FROM_SESSION_HANDLER`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PullFromSessionResponse {
    /// Sequence hashes successfully pulled into the puller's local
    /// G2 pool, in pull order.
    pub pulled: Vec<SequenceHash>,
    /// Per-tier breakdown of where the pulled blocks lived on the
    /// holder before the pull. Telemetry.
    #[serde(default)]
    pub breakdown: MatchBreakdown,
}

// ---------------------------------------------------------------------------
// Close session
// ---------------------------------------------------------------------------

/// Request for [`CLOSE_SESSION_HANDLER`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseTransferSessionRequest {
    pub session_id: SessionId,
    /// Optional reason propagated to the disagg `Session::close` call
    /// and to logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Response for [`CLOSE_SESSION_HANDLER`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloseTransferSessionResponse {
    /// `true` if the session was present in the registry and
    /// removed; `false` if it had already been evicted (by its
    /// watchdog, lifecycle stream, or a prior close). The call is
    /// successful either way — both indicate the session is gone.
    pub was_present: bool,
}

// ---------------------------------------------------------------------------
// Legacy search request/response (kept for hub back-compat)
// ---------------------------------------------------------------------------

/// Request for [`SEARCH_PREFIX_HANDLER`] / [`SEARCH_SCATTER_HANDLER`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

/// Response for the legacy search handlers. Either no matches (no
/// session was opened) or the id of a freshly-opened disagg session
/// pre-populated with the matched G2 blocks. The endpoint is resolved
/// out-of-band (e.g. via the hub peer registry).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum SearchResponse {
    NoBlocksFound,
    Session { session_id: SessionId },
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[cfg(feature = "client")]
pub use client::TransferClient;

#[cfg(feature = "client")]
mod client {
    use super::*;
    use crate::control::ControlError;
    use crate::control::client::ControlChannel;

    /// Client for the `transfer` control module.
    ///
    /// `TransferClient` is bound to one target `InstanceId` (the one
    /// the owning `LeaderControlClient` was constructed for). For the
    /// hub-orchestrated workflow (open on A, pull on B) the caller
    /// holds two `LeaderControlClient`s and uses each one's
    /// `.transfer()` sub-client.
    #[derive(Clone)]
    pub struct TransferClient {
        chan: ControlChannel,
    }

    impl TransferClient {
        pub(crate) fn new(chan: ControlChannel) -> Self {
            Self { chan }
        }

        /// HOLDER-SIDE. Open a transfer session populated by a
        /// multi-tier search. See [`OpenTransferSessionRequest`].
        pub async fn open_session(
            &self,
            req: OpenTransferSessionRequest,
        ) -> Result<OpenTransferSessionResponse, ControlError> {
            self.chan.call(OPEN_SESSION_HANDLER, &req).await
        }

        /// PULLER-SIDE. Tell the targeted instance to attach to a
        /// session on `req.source_instance_id` and pull blocks into
        /// the targeted instance's local G2 pool. Long-poll —
        /// returns when the pull is complete.
        pub async fn pull_from_session(
            &self,
            req: PullFromSessionRequest,
        ) -> Result<PullFromSessionResponse, ControlError> {
            self.chan.call(PULL_FROM_SESSION_HANDLER, &req).await
        }

        /// HOLDER-SIDE. Close a session by id. Idempotent.
        pub async fn close_session(
            &self,
            req: CloseTransferSessionRequest,
        ) -> Result<CloseTransferSessionResponse, ControlError> {
            self.chan.call(CLOSE_SESSION_HANDLER, &req).await
        }

        /// Legacy: contiguous-prefix G2 search. Shim over
        /// `open_session(find_mode = Sync, search_mode = Prefix)`.
        pub async fn search_prefix(
            &self,
            req: SearchRequest,
        ) -> Result<SearchResponse, ControlError> {
            self.chan.call(SEARCH_PREFIX_HANDLER, &req).await
        }

        /// Legacy: scatter (gather-all) G2 search. Shim over
        /// `open_session(find_mode = Sync, search_mode = Scatter)`.
        pub async fn search_scatter(
            &self,
            req: SearchRequest,
        ) -> Result<SearchResponse, ControlError> {
            self.chan.call(SEARCH_SCATTER_HANDLER, &req).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_mode_defaults_to_prefix() {
        let m: SearchMode = Default::default();
        assert!(matches!(m, SearchMode::Prefix));
    }

    #[test]
    fn find_mode_defaults_to_async() {
        let m: FindMode = Default::default();
        assert!(matches!(m, FindMode::Async));
    }

    #[test]
    fn tier_selection_defaults_to_g2_only() {
        let t: TierSelection = Default::default();
        assert!(!t.g3);
        assert!(!t.g4);
    }

    #[test]
    fn open_request_serde_round_trip_minimal() {
        let req = OpenTransferSessionRequest::default();
        let s = serde_json::to_string(&req).unwrap();
        let back: OpenTransferSessionRequest = serde_json::from_str(&s).unwrap();
        assert!(back.sequence_hashes.is_empty());
        assert!(matches!(back.find_mode, FindMode::Async));
    }

    #[test]
    fn open_response_no_blocks_found_round_trip() {
        let resp = OpenTransferSessionResponse::NoBlocksFound;
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains(r#""result":"no_blocks_found""#));
        let back: OpenTransferSessionResponse = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, OpenTransferSessionResponse::NoBlocksFound));
        assert!(back.capability().is_none());
    }

    #[test]
    fn open_response_async_round_trip() {
        let cap = TransferSessionCapability {
            session_id: uuid::Uuid::new_v4(),
            instance_id: InstanceId::new_v4(),
            endpoint: SessionEndpoint {
                kind: "kvbm_cd_session_v2".into(),
                payload: serde_json::Value::Null,
            },
        };
        let resp = OpenTransferSessionResponse::Async {
            capability: cap.clone(),
        };
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains(r#""result":"async""#));
        let back: OpenTransferSessionResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(back.capability(), Some(&cap));
    }

    #[test]
    fn pull_request_round_trip() {
        let req = PullFromSessionRequest {
            session_id: uuid::Uuid::new_v4(),
            source_instance_id: InstanceId::new_v4(),
            endpoint: None,
            selector: None,
        };
        let s = serde_json::to_string(&req).unwrap();
        let back: PullFromSessionRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.session_id, req.session_id);
        assert_eq!(back.source_instance_id, req.source_instance_id);
        assert!(back.endpoint.is_none());
        assert!(back.selector.is_none());
    }

    #[test]
    fn watchdog_helper_converts_ms_to_duration() {
        let req = OpenTransferSessionRequest {
            watchdog_ms: Some(7500),
            ..Default::default()
        };
        assert_eq!(req.watchdog(), Some(Duration::from_millis(7500)));
    }
}
