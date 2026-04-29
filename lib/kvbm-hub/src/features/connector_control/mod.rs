// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub HTTP→velo proxy for the connector-leader control plane.
//!
//! Operators hit a stable hub URL like
//! `PUT /v1/instances/{id}/reset`; the hub looks up `{id}` in the
//! registry, forwards the JSON body via velo to the connector's
//! `kvbm.connector.leader.reset` handler, and proxies the response.
//!
//! Velo uses `serde_json` for typed unary payloads, so HTTP request and
//! response bytes pass through unchanged via [`UnaryBuilder::raw_payload`]
//! — the hub does not need to know the schema and therefore does not
//! need a Cargo dependency on `kvbm-connector`.

mod manager;

pub use manager::ConnectorControlManager;

/// Velo handler name for connector-leader reset.
///
/// Mirrored from `kvbm_connector::connector::leader::velo_control::RESET_HANDLER`
/// — kept as a string here to avoid a cargo dep cycle (kvbm-connector already
/// depends on kvbm-hub).
pub const RESET_HANDLER: &str = "kvbm.connector.leader.reset";

/// Velo handler name for connector-leader register-leader.
pub const REGISTER_LEADER_HANDLER: &str = "kvbm.connector.leader.register_leader";
