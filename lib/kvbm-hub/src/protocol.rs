// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP wire protocol shared between [`HubClient`](crate::HubClient) and
//! [`HubServer`](crate::HubServer).
//!
//! The hub exposes two axum listeners:
//! - **Port 1337** — peer discovery HTTP endpoints (see [`paths`]).
//! - **Port 8337** — control-plane endpoints (registration, heartbeat, health).
//!
//! All request/response bodies are JSON.

use kvbm_common::BlockLayoutMode;
use kvbm_protocols::control::MetricsSnapshotResponse;
pub use kvbm_protocols::control::layout_compat::LayoutCompatPayload;
/// Remote-prefill request payload carried by the hub's CD queue.
///
/// The payload shape is owned by `kvbm-protocols (disagg)`; the hub owns only
/// the queue transport, queue name, and feature registration surface.
pub use kvbm_protocols::disagg::{DISAGG_PROTOCOL_VERSION, RemotePrefillRequest as PrefillRequest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use velo_ext::{InstanceId, PeerInfo, WorkerId};

/// Default HTTP port for peer-discovery lookups (the `PeerDiscovery` surface).
pub const DEFAULT_DISCOVERY_PORT: u16 = 1337;

/// Default HTTP port for the control plane (registration, heartbeat).
pub const DEFAULT_CONTROL_PORT: u16 = 8337;

/// URL path fragments for the HTTP API.
pub mod paths {
    /// Peer-discovery lookup by `InstanceId`.
    ///
    /// `GET /v1/peers/instance/{instance_id}` → [`super::PeerLookupResponse`]
    pub const PEERS_BY_INSTANCE: &str = "/v1/peers/instance/{instance_id}";

    /// Peer-discovery lookup by `WorkerId`.
    ///
    /// `GET /v1/peers/worker/{worker_id}` → [`super::PeerLookupResponse`]
    pub const PEERS_BY_WORKER: &str = "/v1/peers/worker/{worker_id}";

    /// Register an instance.
    ///
    /// `POST /v1/instances` with body [`super::RegisterRequest`]
    /// → [`super::RegisterResponse`]
    pub const INSTANCES: &str = "/v1/instances";

    /// Unregister an instance.
    ///
    /// `DELETE /v1/instances/{instance_id}`
    pub const INSTANCE_BY_ID: &str = "/v1/instances/{instance_id}";

    /// Liveness heartbeat.
    ///
    /// `POST /v1/instances/{instance_id}/heartbeat` → [`super::HeartbeatResponse`]
    pub const INSTANCE_HEARTBEAT: &str = "/v1/instances/{instance_id}/heartbeat";

    /// Hub-initiated velo probe.
    ///
    /// `POST /v1/instances/{instance_id}/probe` → [`super::ProbeResponse`]
    pub const INSTANCE_PROBE: &str = "/v1/instances/{instance_id}/probe";

    /// Hub health probe.
    ///
    /// `GET /health` → `200 OK`
    pub const HEALTH: &str = "/health";

    /// Aggregate hub configuration — the single source of truth a connector
    /// (or `kvbmctl`) pulls to learn the hub's shared `primary` config and its
    /// enabled feature set.
    ///
    /// `GET /v1/config` → [`super::HubConfigResponse`]
    pub const HUB_CONFIG: &str = "/v1/config";

    /// `GET` connector health (one-shot velo probe + last-heartbeat info).
    pub const CONNECTOR_HEALTH: &str = "/v1/instances/{instance_id}/health";

    // -----------------------------------------------------------------------
    // Typed control-plane routes (Phase D)
    //
    // Canonical `/control/<module>/<handler>` namespace, one POST per velo
    // handler exposed by the leader. The hub acts as a typed client of the
    // leader; HTTP status comes from `ControlError::http_status`. Module-
    // gated routes return `404 module_not_enabled` for `has_module ==
    // Some(false)` *without* dispatching to velo.
    //
    // Routing ownership: the `core`/`dev`/`test`/`metrics` routes plus
    // `/modules` + `/describe` are served by `ControlPlaneManager`
    // (infrastructure, always attached). The `/control/transfer/*` routes are
    // served by `P2pManager` — block-copy session management is a P2P concern,
    // so those routes only exist when the P2P feature is enabled on the hub.
    // -----------------------------------------------------------------------

    /// `POST` pull a fresh [`InstanceDescription`] from the leader's velo
    /// handler. Always-on. Updates the hub's describe cache as a side effect
    /// — shares the same code path as `GET /describe?force=true`.
    pub const CONTROL_CORE_DESCRIBE_INSTANCE: &str =
        "/v1/instances/{instance_id}/control/core/describe_instance";

    /// `POST` reset the inactive pools of the requested tiers. Module-gated
    /// on [`kvbm_protocols::control::ModuleId::Dev`]. Body: optional
    /// [`kvbm_protocols::control::ResetRequest`] (defaults to "all tiers").
    pub const CONTROL_DEV_RESET: &str = "/v1/instances/{instance_id}/control/dev/reset";

    /// `POST` contiguous-prefix search of the leader's G2 block manager.
    /// Served by `P2pManager` (requires the P2P feature). Body:
    /// [`kvbm_protocols::control::SearchRequest`].
    pub const CONTROL_TRANSFER_SEARCH_PREFIX: &str =
        "/v1/instances/{instance_id}/control/transfer/search_prefix";

    /// `POST` scatter (gather-all) search of the leader's G2 block manager.
    /// Served by `P2pManager` (requires the P2P feature). Body:
    /// [`kvbm_protocols::control::SearchRequest`].
    pub const CONTROL_TRANSFER_SEARCH_SCATTER: &str =
        "/v1/instances/{instance_id}/control/transfer/search_scatter";

    /// `POST` open a transfer session on the targeted (holder) leader.
    /// Served by `P2pManager` (requires the P2P feature). Body:
    /// [`kvbm_protocols::control::OpenTransferSessionRequest`].
    /// Returns [`kvbm_protocols::control::OpenTransferSessionResponse`].
    pub const CONTROL_TRANSFER_OPEN_SESSION: &str =
        "/v1/instances/{instance_id}/control/transfer/open_session";

    /// `POST` drive a pull on the targeted (puller) leader against a session
    /// living on `request.source_instance_id`. Long-poll: returns when the
    /// pull is complete. Served by `P2pManager` (requires the P2P feature).
    /// Body: [`kvbm_protocols::control::PullFromSessionRequest`]. Returns
    /// [`kvbm_protocols::control::PullFromSessionResponse`].
    pub const CONTROL_TRANSFER_PULL_FROM_SESSION: &str =
        "/v1/instances/{instance_id}/control/transfer/pull_from_session";

    /// `POST` close a transfer session on the targeted (holder) leader.
    /// Idempotent. Served by `P2pManager` (requires the P2P feature). Body:
    /// [`kvbm_protocols::control::CloseTransferSessionRequest`].
    pub const CONTROL_TRANSFER_CLOSE_SESSION: &str =
        "/v1/instances/{instance_id}/control/transfer/close_session";

    /// `POST` on-demand runtime snapshot of the leader. Module-gated on
    /// [`kvbm_protocols::control::ModuleId::Metrics`]. Empty body is
    /// equivalent to [`kvbm_protocols::control::MetricsSnapshotRequest::default`].
    /// Returns [`kvbm_protocols::control::MetricsSnapshotResponse`] as JSON.
    pub const CONTROL_METRICS_SNAPSHOT: &str =
        "/v1/instances/{instance_id}/control/metrics/snapshot";

    /// `GET` fanout helper: collect a snapshot from every registered leader
    /// that has the metrics module enabled, in parallel, and return per-
    /// instance entries keyed by stringified `instance_id`. Slow / failing
    /// leaders surface as `{ "error": "<msg>" }` for their entry rather than
    /// failing the whole response. See [`super::MetricsFanoutResponse`].
    pub const METRICS_FANOUT: &str = "/v1/metrics";

    /// `GET` the set of control-plane modules enabled on this instance.
    /// Body: `{ "modules": [..], "cached": bool, "age_secs": u64 }`. Cache is
    /// populated on register and refreshed every 60s; `?force=true` triggers
    /// an inline re-fetch.
    pub const INSTANCE_MODULES: &str = "/v1/instances/{instance_id}/modules";

    /// Describe an instance.
    ///
    /// - `GET`: returns the cached `InstanceDescription`. `503 describe_pending`
    ///   if the leader has not yet pushed and `?force=true` was not set.
    ///   `?force=true` triggers a fallback pull via the velo
    ///   `describe_instance` handler.
    /// - `POST`: accepts an [`InstanceDescription`] body (push from the leader);
    ///   the hub stores it in its describe cache.
    pub const INSTANCE_DESCRIBE: &str = "/v1/instances/{instance_id}/describe";
}

/// Velo-queue name used by the ConditionalDisagg feature to move prefill
/// requests from Decode workers to Prefill workers. The hub owns the queue;
/// Decode clients enqueue and Prefill clients dequeue via velo active
/// messaging (the queue is backed by [`velo::queue`]).
pub const CD_PREFILL_QUEUE: &str = "kvbm.cd.prefill_requests";

/// Request body for `POST /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    /// Full peer information (instance id + opaque worker address).
    pub peer_info: PeerInfo,
    /// Optional feature declarations. Each entry is dispatched to the
    /// corresponding [`crate::features::FeatureManager`] on the hub after
    /// base registration completes. Empty by default for
    /// backward-compatibility with clients that predate the feature surface.
    #[serde(default)]
    pub features: Vec<Feature>,
    /// Optional must-match runtime config summary. When present, the hub
    /// validates each field required by a declared feature against its
    /// [`PrimaryConfig`] and rejects the registration on mismatch. May be
    /// omitted by legacy clients (P2P / CD), but features that mandate it
    /// (e.g. KV-index) reject a registration without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeConfigSummary>,
}

/// Feature participation declared by a client at registration time.
///
/// Non-exhaustive so new variants can be added without breaking downstream
/// clients that only serialize variants they know about.
///
/// **Feature dependencies (c2):** `ConditionalDisagg` is a specialisation
/// of `P2P` — any register containing `Feature::ConditionalDisagg` MUST
/// also contain `Feature::P2P` in the same request. The server enforces
/// this pre-dispatch in [`crate::server`]. Do not duplicate the check
/// inside individual managers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Feature {
    /// Peer-to-peer block transfers between leaders. Carries the
    /// `layout_compat` payload that the hub gates against the baseline
    /// established by the first P2P registration. This is the *only*
    /// place `LayoutCompatPayload` lives on the wire.
    #[serde(rename = "p2p")]
    P2P(P2pConfig),
    /// The client participates in the disagg feature.
    /// Requires `Feature::P2P` to also be present in the same register
    /// request (CD is a specialisation of P2P). The Rust variant keeps its
    /// `ConditionalDisagg` name; the wire tag + feature label are `"disagg"`.
    #[serde(rename = "disagg")]
    ConditionalDisagg(ConditionalDisaggConfig),
    /// The client participates in the hub-side KV block index: it publishes
    /// block create/remove events to the ZMQ ingest endpoint advertised in
    /// the aggregate config, and registers here so the hub sweeps its index
    /// entries on unregister (TTL or explicit `DELETE`). Carries no payload
    /// today — block-size consistency is validated via [`RuntimeConfigSummary`]
    /// at registration, not a per-feature config.
    Indexer(IndexerFeatureConfig),
}

/// Stable discriminant for [`Feature`] — lets managers match by kind
/// without exhaustively matching the enum's variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FeatureKey {
    /// Matches [`Feature::P2P`].
    P2P,
    /// Matches [`Feature::ConditionalDisagg`].
    ConditionalDisagg,
    /// Connector-control HTTP→velo proxy. No client-side `Feature`
    /// payload — the manager only contributes axum routes, so this key
    /// never matches an incoming registration.
    ConnectorControl,
    /// Hub-side KV block index. The client publishes block events to the ZMQ
    /// ingest endpoint (advertised in the aggregate config) *and* declares
    /// [`Feature::Indexer`] at registration so the hub reclaims its index
    /// entries on unregister. The manager also contributes axum routes and the
    /// ingest loop.
    Indexer,
}

impl FeatureKey {
    /// Stable wire/label string for this key. Used in the aggregate config
    /// response, `--features` CLI parsing, and the connector's feature-subset
    /// selection. Keep these in sync with the connector's capability list.
    pub fn as_str(&self) -> &'static str {
        match self {
            FeatureKey::P2P => "p2p",
            FeatureKey::ConditionalDisagg => "disagg",
            FeatureKey::ConnectorControl => "connector_control",
            FeatureKey::Indexer => "indexer",
        }
    }

    /// Parse a [`FeatureKey`] from its [`as_str`](Self::as_str) label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "p2p" => Some(FeatureKey::P2P),
            "disagg" => Some(FeatureKey::ConditionalDisagg),
            "connector_control" => Some(FeatureKey::ConnectorControl),
            "indexer" => Some(FeatureKey::Indexer),
            _ => None,
        }
    }
}

impl std::fmt::Display for FeatureKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for FeatureKey {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FeatureKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        FeatureKey::from_label(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown feature key {s:?}")))
    }
}

/// Hub-wide shared configuration ("primary"). Split into two classes:
///
/// - **Must-match** (`block_size`, `block_layout`): the hub validates every
///   registrant's [`RuntimeConfigSummary`] against these and rejects on
///   mismatch — they define cross-instance compatibility.
/// - **Free / advisory** (`max_seq_len`, `g2_memory_gib`, `g2_blocks`,
///   `advertise_host`): the hub publishes these as defaults for config
///   generation (`kvbmctl`) but does not enforce them. `max_seq_len` seeds the
///   KV-index's initial capacity and is grown by registrants reporting a larger
///   value (see [`IndexerFeatureConfig`]); it is never a hard match.
///
/// All sizing fields are `Option` so the library / tests can build a hub that
/// is not authoritative for a given field (must-match validation is skipped
/// when `None`). The `kvbm-hub` binary enforces the required subset
/// (`block_size`, at least one of `g2_*`) at startup; `max_seq_len` is optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PrimaryConfig {
    /// Block size (tokens per block). Must-match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// Maximum sequence length (tokens). Advisory: seeds the KV-index initial
    /// capacity and feeds `kvbmctl`'s rendered `--max-model-len`; grown by
    /// registrants, never a hard match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<usize>,
    /// Cross-leader block-layout compatibility mode. Must-match.
    #[serde(default)]
    pub block_layout: BlockLayoutMode,
    /// Advisory G2 (host) cache size in GiB. Populated into generated connector
    /// config; not validated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g2_memory_gib: Option<f64>,
    /// Advisory G2 (host) cache size in blocks. Populated into generated
    /// connector config; not validated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g2_blocks: Option<usize>,
    /// Host advertised to publishers (e.g. KV-index ZMQ endpoint). Advisory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advertise_host: Option<String>,
}

/// Must-match runtime values a client reports at registration so the hub can
/// validate consistency against its [`PrimaryConfig`]. Sent in
/// [`RegisterRequest::runtime`].
///
/// Absent → the hub skips must-match validation, *unless* a declared feature
/// mandates the summary (`FeatureManager::requires_runtime_summary`, e.g.
/// KV-index), in which case registration is rejected. Present → each field
/// required by a declared feature must be `Some` and equal the hub's
/// authoritative value.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeConfigSummary {
    /// The client's effective block size (page size). Must equal
    /// `primary.block_size` when the hub is authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// The client's block-layout mode. Must equal `primary.block_layout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_layout: Option<BlockLayoutMode>,
}

/// Response body for `GET /v1/config` — the aggregate the connector and
/// `kvbmctl` consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubConfigResponse {
    /// The hub's shared `primary` config.
    pub primary: PrimaryConfig,
    /// One entry per attached feature manager (the hub's advertised
    /// capability set).
    pub features: Vec<FeatureDescriptor>,
    /// Operator-supplied default connector config: a sparse
    /// `kv_connector_extra_config`-shaped object (the `default`/`leader`/`worker`
    /// figment profiles) built from the hub's `--kvbm` / `--kvbm-config` flags
    /// and validated at startup. Sits alongside `primary` (which stays the
    /// source of truth for must-match fields); a consumer merges this as a base
    /// layer beneath its own role profile + local overrides. `{}` when no
    /// overrides were given.
    #[serde(default)]
    pub base_config: serde_json::Value,
}

/// Which [`PrimaryConfig`] must-match fields a feature requires a registrant to
/// declare (via [`RuntimeConfigSummary`]) and that the hub validates for
/// consistency. All `false` by default. Surfaced in [`FeatureDescriptor`] so a
/// client can pre-check compatibility before registering (and, in best-effort
/// mode, drop an incompatible feature instead of being rejected at register).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureConfigRequirements {
    /// Requires `block_size` to be declared and to match the hub's primary.
    #[serde(default)]
    pub block_size: bool,
    /// Requires `block_layout` to be declared and to match the hub's primary.
    #[serde(default)]
    pub block_layout: bool,
}

/// One feature's contribution to [`HubConfigResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDescriptor {
    /// Stable feature key (e.g. `"indexer"`).
    pub key: FeatureKey,
    /// Feature keys this feature depends on (must also be enabled / declared).
    pub dependencies: Vec<FeatureKey>,
    /// Must-match fields this feature requires of a registrant.
    #[serde(default)]
    pub config_requirements: FeatureConfigRequirements,
    /// Feature-specific config view (e.g. KV-index `zmq_endpoint`). `null` when
    /// the feature exposes nothing extra beyond its key.
    pub config: serde_json::Value,
}

impl Feature {
    /// Return the stable discriminant for this feature.
    pub fn key(&self) -> FeatureKey {
        match self {
            Feature::P2P(_) => FeatureKey::P2P,
            Feature::ConditionalDisagg(_) => FeatureKey::ConditionalDisagg,
            Feature::Indexer(_) => FeatureKey::Indexer,
        }
    }
}

/// Configuration payload for [`Feature::Indexer`].
///
/// The connector opts into hub-side indexing (the hub records the registration
/// so it can reclaim the instance's index entries on unregister) and reports
/// its `max_seq_len` so the hub can **grow** the index capacity to fit it.
/// `max_seq_len` is advisory — it never shrinks the index and is not a
/// must-match field; block-size consistency is enforced via
/// [`RuntimeConfigSummary`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexerFeatureConfig {
    /// The connector's max sequence length (from vLLM `max_model_len`), if
    /// known. When larger than the hub's current index capacity, the hub grows
    /// the index to fit it. `None` leaves capacity unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<usize>,
}

/// Configuration payload for the P2P feature. Carries the
/// `layout_compat` payload that the hub validates against the baseline
/// established by the first P2P registration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct P2pConfig {
    /// Block-layout compatibility payload — mandatory for every P2P
    /// registration. The hub runs `validate_self` on the first payload
    /// and `check_layout_compat` against the baseline on subsequent
    /// payloads.
    pub layout_compat: LayoutCompatPayload,
}

/// Configuration payload for the ConditionalDisagg feature.
///
/// `layout_compat` no longer lives here — c2 moved it to
/// [`P2pConfig`]. CD register requests must include `Feature::P2P`
/// in the same payload; the server rejects otherwise.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConditionalDisaggConfig {
    /// The role this instance is taking inside the ConditionalDisagg split.
    pub role: ConditionalDisaggRole,
    /// Optional vLLM HTTP frontend endpoint advertised by a Prefill role
    /// so the hub's load-aware dispatcher can route requests directly
    /// without a separate operator-managed URL list. Decode registrations
    /// leave this `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm_http: Option<VllmHttpEndpoint>,
}

/// HTTP frontend endpoint a prefill worker advertises at registration so
/// the hub can POST `/v1/completions` requests to it directly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VllmHttpEndpoint {
    /// Base URL of the vLLM frontend (e.g. `http://10.0.0.5:8000`).
    /// Trailing slashes are not stripped on the wire — consumers should
    /// trim before appending paths.
    pub base_url: String,
    /// Model name to pass in dispatched POST bodies. Must match the vLLM
    /// process's `--model` argument.
    pub model: String,
}

/// Role a ConditionalDisagg participant takes on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalDisaggRole {
    /// Prefill instance.
    Prefill,
    /// Decode instance.
    Decode,
}

/// Response body for `POST /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    /// The registered instance id, echoed back.
    pub instance_id: InstanceId,
    /// The hub's own velo `InstanceId`, when the hub runs with a velo
    /// participant. Clients can resolve the hub's `PeerInfo` via
    /// `GET /v1/peers/instance/{id}` and wire it into their own Velo for
    /// bidirectional active messaging. `None` when the hub has no velo.
    #[serde(default)]
    pub hub_instance_id: Option<InstanceId>,
}

/// Response body for peer-discovery lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerLookupResponse {
    /// Full peer information for the requested id.
    pub peer_info: PeerInfo,
}

/// Response body for `POST /v1/instances/{instance_id}/heartbeat`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeartbeatResponse {
    /// Whether the hub considers this instance registered.
    pub acknowledged: bool,
}

/// Response body for `POST /v1/instances/{instance_id}/probe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResponse {
    /// Echoed sequence number from the velo heartbeat.
    pub seq: u64,
    /// Whether the instance reported itself healthy.
    pub ok: bool,
}

/// Response body for `GET /v1/instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListInstancesResponse {
    /// All currently registered instances.
    pub instances: Vec<PeerInfo>,
}

/// Response body for `GET /v1/metrics` — fanout across every leader that has
/// the `metrics` module enabled.
///
/// Entries are always present for every leader the hub queried; a slow or
/// failing leader surfaces as an entry with `snapshot = None` and a non-empty
/// `error` string rather than failing the whole response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsFanoutResponse {
    /// Hub-side wall-clock at the moment the fanout was initiated, in
    /// milliseconds since the Unix epoch. Per-leader timestamps live on each
    /// [`MetricsSnapshotResponse`] under [`MetricsInstanceEntry::snapshot`].
    pub gathered_at_unix_ms: u64,
    /// Per-instance entries keyed by stringified [`InstanceId`]. `BTreeMap`
    /// rather than `HashMap` so the JSON ordering is deterministic — handy
    /// for tests and for the UI's "diff vs last refresh" view.
    pub instances: BTreeMap<String, MetricsInstanceEntry>,
}

/// One leader's contribution to [`MetricsFanoutResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsInstanceEntry {
    /// The leader's snapshot, `Some` on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<MetricsSnapshotResponse>,
    /// Error string from the velo call or module-gate check. `Some` iff
    /// `snapshot` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Typed error body returned by the hub on non-2xx responses.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct ErrorBody {
    /// Stable machine-readable error code.
    pub code: ErrorCode,
    /// Human-readable description.
    pub message: String,
}

/// Stable error codes returned by the hub API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Requested instance or worker id is not registered.
    NotFound,
    /// Registration conflicts with an existing registration.
    Conflict,
    /// Request payload was malformed.
    BadRequest,
    /// Unexpected server-side failure.
    Internal,
}

/// Path helper: format [`paths::PEERS_BY_INSTANCE`] with a concrete id.
pub fn peers_by_instance(id: InstanceId) -> String {
    format!("/v1/peers/instance/{id}")
}

/// Path helper: format [`paths::PEERS_BY_WORKER`] with a concrete id.
pub fn peers_by_worker(id: WorkerId) -> String {
    format!("/v1/peers/worker/{}", id.as_u64())
}

/// Path helper: format [`paths::INSTANCE_BY_ID`] with a concrete id.
pub fn instance_by_id(id: InstanceId) -> String {
    format!("/v1/instances/{id}")
}

/// Path helper: format [`paths::INSTANCE_HEARTBEAT`] with a concrete id.
pub fn instance_heartbeat(id: InstanceId) -> String {
    format!("/v1/instances/{id}/heartbeat")
}

/// Path helper: format [`paths::INSTANCE_PROBE`] with a concrete id.
pub fn instance_probe(id: InstanceId) -> String {
    format!("/v1/instances/{id}/probe")
}

/// Path helper: format [`paths::INSTANCE_DESCRIBE`] with a concrete id.
pub fn instance_describe(id: InstanceId) -> String {
    format!("/v1/instances/{id}/describe")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_common::SequenceHash;
    use velo_ext::WorkerAddress;

    fn make_peer_info() -> PeerInfo {
        let id = InstanceId::new_v4();
        PeerInfo::new(id, WorkerAddress::from_encoded(b"test".to_vec()))
    }

    #[test]
    fn heartbeat_response_serde_round_trip() {
        for acknowledged in [true, false] {
            let orig = HeartbeatResponse { acknowledged };
            let json = serde_json::to_string(&orig).unwrap();
            let back: HeartbeatResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(back.acknowledged, acknowledged);
        }
    }

    #[test]
    fn heartbeat_response_default_is_not_acknowledged() {
        assert!(!HeartbeatResponse::default().acknowledged);
    }

    #[test]
    fn register_request_serde_round_trip() {
        let peer_info = make_peer_info();
        let orig = RegisterRequest {
            peer_info: peer_info.clone(),
            features: vec![Feature::ConditionalDisagg(ConditionalDisaggConfig {
                role: ConditionalDisaggRole::Prefill,
                vllm_http: None,
            })],
            runtime: None,
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: RegisterRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.peer_info.instance_id(), peer_info.instance_id());
        assert_eq!(back.features, orig.features);
    }

    #[test]
    fn register_request_accepts_legacy_payload_without_features() {
        let peer_info = make_peer_info();
        let legacy_json = format!(
            r#"{{"peer_info":{}}}"#,
            serde_json::to_string(&peer_info).unwrap()
        );
        let back: RegisterRequest = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(back.peer_info.instance_id(), peer_info.instance_id());
        assert!(back.features.is_empty());
    }

    #[test]
    fn feature_cd_serde_round_trip() {
        let f = Feature::ConditionalDisagg(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Decode,
            vllm_http: None,
        });
        let json = serde_json::to_string(&f).unwrap();
        // Adjacently-tagged: {"kind":"disagg","config":{"role":"decode"}}
        assert!(json.contains("\"kind\":\"disagg\""));
        assert!(json.contains("\"role\":\"decode\""));
        let back: Feature = serde_json::from_str(&json).unwrap();
        assert_eq!(back, f);
        assert_eq!(back.key(), FeatureKey::ConditionalDisagg);
    }

    #[test]
    fn feature_p2p_serde_round_trip() {
        use crate::protocol::LayoutCompatPayload;
        use kvbm_common::shape::CanonicalBlockShape;
        use kvbm_common::{BlockLayoutMode, KvBlockLayout};
        use kvbm_protocols::control::LayoutConfigDescription;

        let payload = LayoutCompatPayload {
            mode: BlockLayoutMode::Operational,
            canonical: Some(CanonicalBlockShape {
                num_layers_total: 4,
                outer_dim: 2,
                page_size: 16,
                num_heads_total: 8,
                head_dim: 64,
                dtype_width_bytes: 2,
            }),
            per_worker_layout: KvBlockLayout::OperationalNHD,
            per_worker_config: LayoutConfigDescription {
                num_blocks: 16,
                num_layers: 4,
                outer_dim: 2,
                page_size: 16,
                inner_dim: 8 * 64,
                alignment: 256,
                dtype_width_bytes: 2,
                num_heads: Some(8),
            },
            tp_size: 1,
            pp_size: 1,
        };
        let f = Feature::P2P(P2pConfig {
            layout_compat: payload,
        });
        let json = serde_json::to_string(&f).unwrap();
        // Renamed variant (default snake_case would mangle the acronym).
        assert!(
            json.contains("\"kind\":\"p2p\""),
            "expected kind=p2p, got: {json}"
        );
        let back: Feature = serde_json::from_str(&json).unwrap();
        assert_eq!(back, f);
        assert_eq!(back.key(), FeatureKey::P2P);
    }

    #[test]
    fn cd_role_serde_lowercase() {
        let prefill = serde_json::to_string(&ConditionalDisaggRole::Prefill).unwrap();
        let decode = serde_json::to_string(&ConditionalDisaggRole::Decode).unwrap();
        assert_eq!(prefill, "\"prefill\"");
        assert_eq!(decode, "\"decode\"");
    }

    #[test]
    fn prefill_request_serde_round_trip() {
        let orig = PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "req-123".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            sequence_hashes: vec![
                SequenceHash::new(11, None, 11),
                SequenceHash::new(12, None, 12),
            ],
            token_ids: vec![1, 2, 3],
            num_computed_tokens: 16,
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: PrefillRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, orig);
    }

    #[test]
    fn cd_prefill_queue_name_is_stable() {
        assert_eq!(CD_PREFILL_QUEUE, "kvbm.cd.prefill_requests");
    }

    #[test]
    fn register_response_serde_round_trip() {
        let instance_id = InstanceId::new_v4();
        let hub_instance_id = Some(InstanceId::new_v4());
        let orig = RegisterResponse {
            instance_id,
            hub_instance_id,
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: RegisterResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.instance_id, instance_id);
        assert_eq!(back.hub_instance_id, hub_instance_id);
    }

    #[test]
    fn register_response_accepts_legacy_payload_without_hub_instance_id() {
        let instance_id = InstanceId::new_v4();
        let legacy_json = format!("{{\"instance_id\":\"{instance_id}\"}}");
        let back: RegisterResponse = serde_json::from_str(&legacy_json).unwrap();
        assert_eq!(back.instance_id, instance_id);
        assert!(back.hub_instance_id.is_none());
    }

    #[test]
    fn peer_lookup_response_serde_round_trip() {
        let peer_info = make_peer_info();
        let orig = PeerLookupResponse {
            peer_info: peer_info.clone(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: PeerLookupResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.peer_info.instance_id(), peer_info.instance_id());
    }

    #[test]
    fn error_code_serde_all_variants() {
        for code in [
            ErrorCode::NotFound,
            ErrorCode::Conflict,
            ErrorCode::BadRequest,
            ErrorCode::Internal,
        ] {
            let json = serde_json::to_string(&code).unwrap();
            let back: ErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, code);
        }
    }

    #[test]
    fn error_body_serde_round_trip() {
        let orig = ErrorBody {
            code: ErrorCode::NotFound,
            message: "instance abc not found".to_string(),
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: ErrorBody = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code, ErrorCode::NotFound);
        assert_eq!(back.message, "instance abc not found");
        assert_eq!(back.to_string(), "instance abc not found");
    }

    #[test]
    fn path_helpers_contain_id() {
        let id = InstanceId::new_v4();
        let worker_id = id.worker_id();
        let id_str = id.to_string();

        assert!(peers_by_instance(id).contains(&id_str));
        assert!(peers_by_worker(worker_id).contains(&worker_id.as_u64().to_string()));
        assert!(instance_by_id(id).contains(&id_str));
        assert!(instance_heartbeat(id).contains(&id_str));
    }

    #[test]
    fn path_helpers_have_expected_prefixes() {
        let id = InstanceId::new_v4();
        let worker_id = id.worker_id();

        assert!(peers_by_instance(id).starts_with("/v1/peers/instance/"));
        assert!(peers_by_worker(worker_id).starts_with("/v1/peers/worker/"));
        assert!(instance_by_id(id).starts_with("/v1/instances/"));
        assert!(instance_heartbeat(id).starts_with("/v1/instances/"));
        assert!(instance_heartbeat(id).ends_with("/heartbeat"));
    }
}
