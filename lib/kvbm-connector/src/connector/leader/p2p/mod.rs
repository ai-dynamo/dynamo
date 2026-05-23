// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P feature foundation for the connector leader.
//!
//! P2P is a standalone feature: a leader that registers `Feature::P2P` becomes
//! a hub-discoverable, remote-controllable peer that can serve block copies via
//! its control-plane `transfer` module (search / open_session / pull). This
//! module owns the connector-side P2P bits:
//!
//! - [`transport`] — the block-transfer seam (local G2→G1 copy) and leader
//!   shims used by transfer flows.
//! - [`peer_resolver`] — [`HubPeerResolver`](peer_resolver::HubPeerResolver),
//!   which resolves a remote `InstanceId` to its `PeerInfo` and registers it in
//!   velo's streaming registry (needed for session attach/pull).
//!
//! ConditionalDisagg (`super::disagg`) builds its decode/prefill flows **on
//! top of** this foundation rather than embedding it.

pub mod peer_resolver;
pub mod transport;
pub mod wire;
