// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side wrapper around [`HubClient`] for the P2P feature.
//!
//! Thin parity wrapper used by leaders that register `Feature::P2P` without
//! also requesting `Feature::ConditionalDisagg`. CD callers register both
//! features through [`crate::features::disagg::ConditionalDisaggClient`],
//! which carries the same `LayoutCompatPayload` under the hood.

use std::sync::Arc;

use anyhow::Result;
use velo_ext::{InstanceId, PeerInfo};

use crate::client::HubClient;
use crate::protocol::{Feature, LayoutCompatPayload, P2pConfig};

/// Register a leader under the P2P feature alone (no CD role).
pub struct P2pClient {
    hub: Arc<HubClient>,
}

impl P2pClient {
    /// Wrap a [`HubClient`].
    pub fn new(hub: Arc<HubClient>) -> Arc<Self> {
        Arc::new(Self { hub })
    }

    /// Register the leader with the hub, declaring only the P2P feature.
    pub async fn register(
        &self,
        peer_info: PeerInfo,
        layout_compat: LayoutCompatPayload,
    ) -> Result<Option<InstanceId>> {
        self.hub
            .register_instance_with_features(
                peer_info,
                vec![Feature::P2P(P2pConfig { layout_compat })],
            )
            .await
    }
}
