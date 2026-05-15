// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `tests` module protocol: test-only block helpers.
//!
//! The matching service impl (`kvbm-engine::leader::control::modules::tests`)
//! is feature-gated on the engine's `testing` feature, so a leader built
//! without it simply will not list [`super::super::ModuleId::Tests`] in its
//! `list_modules` response.

use kvbm_common::SequenceHash;
use serde::{Deserialize, Serialize};

/// Velo handler name for `register_test_blocks`.
pub const REGISTER_TEST_BLOCKS_HANDLER: &str = "kvbm.leader.control.register_test_blocks";

/// Request: allocate one G2 block per sequence hash, stamp the hash onto it,
/// register it, then release it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegisterTestBlocksRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

/// Response: the number of blocks successfully allocated+registered.
///
/// G2 allocation is all-or-nothing, so this is either
/// `sequence_hashes.len()` or `0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterTestBlocksResponse {
    pub allocated: usize,
}

#[cfg(feature = "client")]
pub use client::TestClient;

#[cfg(feature = "client")]
mod client {
    use super::*;
    use crate::control::ControlError;
    use crate::control::client::ControlChannel;

    /// Client for the `tests` control module.
    #[derive(Clone)]
    pub struct TestClient {
        chan: ControlChannel,
    }

    impl TestClient {
        pub(crate) fn new(chan: ControlChannel) -> Self {
            Self { chan }
        }

        /// Ask the leader to allocate, hash-stamp, register and release one G2
        /// block per requested sequence hash.
        pub async fn register_test_blocks(
            &self,
            req: RegisterTestBlocksRequest,
        ) -> Result<RegisterTestBlocksResponse, ControlError> {
            self.chan.call(REGISTER_TEST_BLOCKS_HANDLER, &req).await
        }
    }
}
