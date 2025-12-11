// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod coordination;
pub mod data;
pub mod init;
pub mod messages;
pub mod runtime;
pub mod service;

#[cfg(feature = "console")]
pub mod console;

#[cfg(feature = "console")]
pub mod events;

// #[cfg(test)]
// pub mod testing;

pub use data::{
    Blocks, CachedRequestData, FinishedStatus, KVConnectorOutput, MatchResult, NewRequestData,
    Request, SchedulerOutput,
};

use super::{G1, G2, G3};

pub use coordination::{TransferCoordHandle, TransferKind, spawn_coordination_task};
pub use init::InitializedState;
pub use runtime::ConnectorLeader;
pub use service::WorkerClient;

#[cfg(feature = "console")]
pub use console::InstrumentedLeader;

pub use anyhow::Result;

use crate::{
    integrations::connector::leader::data::BlocksView, v2::integrations::IntegrationsConfig,
};

// pub fn build_connector_leader(
//     engine_id: &str,
//     config: IntegrationsConfig,
// ) -> Result<Box<dyn LeaderRuntime>> {
//     #[cfg(feature = "console")]
//     {
//         if console::is_enabled() {
//             Ok(Box::new(InstrumentedLeader::new(ConnectorLeader::new(
//                 engine_id, config,
//             ))))
//         } else {
//             Ok(Box::new(ConnectorLeader::new(engine_id, config)))
//         }
//     }

//     #[cfg(not(feature = "console"))]
//     {
//         Ok(Box::new(ConnectorLeader::new(engine_id, config)))
//     }
// }

/// Trait for a leader runtime implementation.
pub trait LeaderRuntime {
    // ================================
    // vLLM v0.11 Methods
    // ================================

    /// Get the number of new matched tokens for the given request.
    ///
    /// Returns: (Option<Count>, AsyncLoading)
    /// - Option<Count>: The number of new matched tokens for the given request. If the connector needs more time to process the request,
    ///   this will be `None` and the scheduler ask again in the future. Allowed return values are: (None, False), (Some(0), False), (Some(>0), True).
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<MatchResult>;

    /// Called during the prefill phase to provide the leader with Blocks that should be used for prefill.
    ///
    /// If [`LeaderRuntime::get_num_new_matched_tokens`] returns [`MatchResult::Matched`], the scheduler will call this method twice:
    /// - First, to provide the Leader with the GPU/G1 Blocks that should be used for Onboarding.
    /// - Second, to provide the Leader with the GPU/G1 Blocks that will be used for the remainder of the prefill. We expect that on
    ///   a second call into this method, the `num_computed_tokens` will be 0.
    ///
    /// If [`LeaderRuntime::get_num_new_matched_tokens`] returns [`MatchResult::NoMatches`], the scheduler will not call this method.
    fn update_state_after_alloc(
        &mut self,
        request_id: &str,
        block_ids: BlocksView<G1>,
        num_external_tokens: usize,
    ) -> Result<()>;

    /// Notify the leader that a request has finished.
    ///
    /// Returns:
    /// - [`FinishedStatus::Finished`] if there are no outstanding operations on the request.
    /// - [`FinishedStatus::Pending`] if there are outstanding operations on the request making it unsafe to free the blocks.
    ///
    /// If a [`FinishedStatus::Pending`] is returned, the worker(s) will be notified to wait for the request to finish, and it
    /// becomes the responsibility of the worker(s) to report when they have no more outstanding operations for this request.
    fn request_finished(
        &mut self,
        request_id: &str,
        block_ids: Blocks<G1>,
    ) -> Result<FinishedStatus>;

    /// Called once per forward pass to build the connector metadata.
    fn build_connector_metadata(&mut self, output: &SchedulerOutput) -> Result<Vec<u8>>;

    /// Called either once per forward pass or only when KVConnectorOutput is non-empty.
    /// New in v0.11 so it's behavior is not fully quantified.
    fn update_connector_output(&mut self, connector_output: KVConnectorOutput) -> Result<()>;

    // ================================
    // KVBM Specific Methods
    // ================================

    /// The unique identifier for the engine.
    fn engine_id(&self) -> &str;

    /// Check if the leader is ready to accept requests.
    fn is_ready(&self) -> bool;

    /// Wait for the leader to be ready to accept requests.
    fn wait_ready(&self) -> Result<()>;

    /// Check if the leader has a slot for the given request.
    fn has_slot(&self, request_id: &str) -> bool;

    /// Create a slot for the given request.
    fn create_slot(&mut self, request: Request, all_token_ids: Vec<Vec<i64>>) -> Result<()>;
}
