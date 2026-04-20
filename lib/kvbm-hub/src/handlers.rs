// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo active-message handlers installed on a client by
//! [`HubClient::register_handlers`](crate::HubClient::register_handlers).
//!
//! These are the control-plane messages the hub sends to its clients over
//! velo (not HTTP). HTTP is reserved for discovery + bootstrap registration;
//! velo is used once both sides know about each other.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use velo::Handler;

use crate::client::HubClient;

/// Velo handler name for the hub → client heartbeat probe.
pub const HEARTBEAT_HANDLER: &str = "_kvbm_hub_heartbeat";

/// Payload sent by the hub on each heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeartbeatRequest {
    /// Monotonic sequence from the hub — echoed back for jitter tracking.
    pub seq: u64,
}

/// Response returned by the client to acknowledge a heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeartbeatAck {
    /// Echoed `seq` from the request.
    pub seq: u64,
    /// Always `true` for now — reserved for future status flags.
    pub ok: bool,
}

/// Build the heartbeat velo handler for this client.
///
/// Installed by [`HubClient::register_handlers`](crate::HubClient::register_handlers).
/// Stub: always acks. Real implementations will consult local state
/// (connector health, engine readiness, ...) via the captured [`HubClient`].
pub fn create_heartbeat_handler(_client: Arc<HubClient>) -> Handler {
    Handler::typed_unary_async::<HeartbeatRequest, HeartbeatAck, _, _>(
        HEARTBEAT_HANDLER,
        |ctx| async move {
            Ok(HeartbeatAck {
                seq: ctx.input.seq,
                ok: true,
            })
        },
    )
    .build()
}
