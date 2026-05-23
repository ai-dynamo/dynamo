// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo active-message handlers installed on a client by
//! [`HubClient::register_handlers`](crate::HubClient::register_handlers).
//!
//! These are the control-plane messages the hub sends to its clients over
//! velo (not HTTP). HTTP is reserved for discovery + bootstrap registration;
//! velo is used once both sides know about each other.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use velo::Handler;

use crate::client::HubClient;

/// Velo handler name for the hub → client heartbeat probe.
pub const HEARTBEAT_HANDLER: &str = "kvbm_hub_heartbeat";

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

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the heartbeat velo handler for this client.
///
/// Installed by [`HubClient::register_handlers`](crate::HubClient::register_handlers).
/// Records the latest hub-heartbeat `seq` and wall-clock arrival time on the
/// captured [`HubClient`] so downstream consumers can observe liveness.
pub fn create_heartbeat_handler(client: Arc<HubClient>) -> Handler {
    Handler::typed_unary_async::<HeartbeatRequest, HeartbeatAck, _, _>(
        HEARTBEAT_HANDLER,
        move |ctx| {
            let client = Arc::clone(&client);
            async move {
                client
                    .last_heartbeat_seq
                    .store(ctx.input.seq, Ordering::Relaxed);
                client
                    .last_heartbeat_at_ms
                    .store(now_unix_ms(), Ordering::Relaxed);
                Ok(HeartbeatAck {
                    seq: ctx.input.seq,
                    ok: true,
                })
            }
        },
    )
    .build()
}
