// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Roundhouse core: the session state machine and the routing decision layer.
//!
//! The design rests on one structure: an append-only event log per session with
//! a monotonic sequence number. That single log serves three otherwise separate
//! needs — OpenAI Responses streaming resumption (`starting_after`), reconnect
//! replay for the bidirectional transports, and the routing audit trail.
//!
//! Nothing here knows how to execute a turn. Execution lives behind
//! [`routing::RoutingPolicy`] and the fleet traits in `roundhouse-fleet`, so
//! the state machine stays testable without a GPU, a network, or a provider
//! account.

pub mod context;
pub mod event;
pub mod ids;
pub mod item;
pub mod routing;
pub mod session;
pub mod store;

pub use event::{SessionEvent, SessionEventKind};
pub use ids::{ResponseId, SessionId, TurnId};
pub use item::{Item, ItemContent, Role};
pub use session::{Session, SessionError};
pub use store::{Lease, SessionStore, StoreError};

/// Milliseconds since the Unix epoch.
///
/// Wall-clock rather than monotonic: these timestamps are persisted, compared
/// across processes after a failover, and fed to the frontier cache model,
/// which reasons in provider TTL terms.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
