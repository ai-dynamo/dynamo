// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `transfer` module protocol: G2 search → disagg-session creation.
//!
//! Both search handlers take the same [`SearchRequest`] and return the same
//! [`SearchResponse`]; they differ only in how the leader's G2 block manager
//! is searched — `search_prefix` is a contiguous prefix match, `search_scatter`
//! gathers every hash present regardless of gaps.

use kvbm_common::SequenceHash;
use serde::{Deserialize, Serialize};

use crate::disagg::SessionId;

/// Velo handler name for the contiguous-prefix search.
pub const SEARCH_PREFIX_HANDLER: &str = "kvbm.leader.control.search_prefix";

/// Velo handler name for the scatter (gather-all) search.
pub const SEARCH_SCATTER_HANDLER: &str = "kvbm.leader.control.search_scatter";

/// Request: the sequence hashes to look for in the leader's G2 block manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

/// Response: either no matches (no session created) or the id of a freshly
/// opened disagg session pre-populated with the matched G2 blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum SearchResponse {
    /// No requested hash matched; no session was created.
    NoBlocksFound,
    /// Matches were found; a disagg session was opened and populated.
    ///
    /// The caller resolves the session's `SessionEndpoint` out-of-band (e.g.
    /// via the hub peer registry) in order to attach.
    Session { session_id: SessionId },
}

#[cfg(feature = "client")]
pub use client::TransferClient;

#[cfg(feature = "client")]
mod client {
    use super::*;
    use crate::control::ControlError;
    use crate::control::client::ControlChannel;

    /// Client for the `transfer` control module.
    #[derive(Clone)]
    pub struct TransferClient {
        chan: ControlChannel,
    }

    impl TransferClient {
        pub(crate) fn new(chan: ControlChannel) -> Self {
            Self { chan }
        }

        /// Contiguous-prefix search of the leader's G2 block manager.
        pub async fn search_prefix(
            &self,
            req: SearchRequest,
        ) -> Result<SearchResponse, ControlError> {
            self.chan.call(SEARCH_PREFIX_HANDLER, &req).await
        }

        /// Scatter (gather-all) search of the leader's G2 block manager.
        pub async fn search_scatter(
            &self,
            req: SearchRequest,
        ) -> Result<SearchResponse, ControlError> {
            self.chan.call(SEARCH_SCATTER_HANDLER, &req).await
        }
    }
}
