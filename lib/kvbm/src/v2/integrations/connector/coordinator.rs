// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer coordination for the connector.
//!
//! This module provides the `TransferCoordinator` trait for coordinating
//! transfer execution across workers.
//!
//! # Design
//!
//! The connector leader needs to:
//! 1. Trigger transfers on workers (onboarding, offloading)
//! 2. Track when transfers complete
//! 3. Notify workers when operations finish
//!
//! The `TransferCoordinator` trait abstracts this coordination, allowing:
//! - `MockCoordinator` for testing without real workers
//! - `SpmdCoordinator` for production with ReplicatedWorker (future)
//!
//! # Leader-Driven Design
//!
//! The coordinator follows the leader-driven principle:
//! - Leader decides when to start transfers
//! - Leader polls for completion
//! - Leader notifies workers when done
//! - Workers just execute and report state

use anyhow::Result;

use crate::BlockId;
use crate::physical::transfer::context::TransferCompleteNotification;

/// Coordinates transfer execution across workers.
///
/// The connector leader uses this trait to orchestrate data movement
/// between storage tiers (G1, G2, G3).
pub trait TransferCoordinator: Send + Sync {
    /// Execute onboard transfer: move data into G1 (GPU).
    ///
    /// Depending on configuration, this may:
    /// - G2→G1: Host memory to GPU
    /// - G3→G2→G1: Disk to host to GPU (staged)
    ///
    /// # Arguments
    /// * `request_id` - Request identifier for tracking
    /// * `g2_blocks` - Blocks in G2 (host) to transfer
    /// * `g3_blocks` - Optional blocks in G3 (disk) to stage through G2
    /// * `g1_blocks` - Destination blocks in G1 (GPU)
    ///
    /// # Returns
    /// A notification handle to poll for completion.
    fn execute_onboard(
        &self,
        request_id: &str,
        g2_blocks: &[BlockId],
        g3_blocks: Option<&[BlockId]>,
        g1_blocks: &[BlockId],
    ) -> Result<TransferCompleteNotification>;

    /// Notify all workers that onboarding is complete for a request.
    ///
    /// Called by the leader after `execute_onboard` completes. Workers will
    /// add the request_id to their `finished_onboarding` state.
    fn notify_onboard_complete(&self, request_id: &str) -> Result<()>;

    /// Execute offload transfer: move data from G1 (GPU) to host.
    ///
    /// # Arguments
    /// * `request_id` - Request identifier for tracking
    /// * `g1_blocks` - Source blocks in G1 (GPU)
    /// * `g2_blocks` - Destination blocks in G2 (host)
    ///
    /// # Returns
    /// A notification handle to poll for completion.
    fn execute_offload(
        &self,
        request_id: &str,
        g1_blocks: &[BlockId],
        g2_blocks: &[BlockId],
    ) -> Result<TransferCompleteNotification>;

    /// Notify all workers that offloading is complete for a request.
    ///
    /// Called by the leader after `execute_offload` completes.
    fn notify_offload_complete(&self, request_id: &str) -> Result<()>;
}

/// Mock coordinator for testing without real workers.
///
/// All operations complete immediately with success. Useful for:
/// - Unit testing connector logic
/// - Testing slot state machine
/// - Development without CUDA/NIXL
#[derive(Default, Debug)]
pub struct MockCoordinator;

impl MockCoordinator {
    /// Create a new MockCoordinator.
    pub fn new() -> Self {
        Self
    }
}

impl TransferCoordinator for MockCoordinator {
    fn execute_onboard(
        &self,
        request_id: &str,
        g2_blocks: &[BlockId],
        g3_blocks: Option<&[BlockId]>,
        g1_blocks: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        tracing::debug!(
            request_id,
            g2_block_count = g2_blocks.len(),
            g3_block_count = g3_blocks.map(|b| b.len()),
            g1_block_count = g1_blocks.len(),
            "MockCoordinator: execute_onboard (immediate complete)"
        );
        Ok(TransferCompleteNotification::completed())
    }

    fn notify_onboard_complete(&self, request_id: &str) -> Result<()> {
        tracing::debug!(
            request_id,
            "MockCoordinator: notify_onboard_complete (no-op)"
        );
        Ok(())
    }

    fn execute_offload(
        &self,
        request_id: &str,
        g1_blocks: &[BlockId],
        g2_blocks: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        tracing::debug!(
            request_id,
            g1_block_count = g1_blocks.len(),
            g2_block_count = g2_blocks.len(),
            "MockCoordinator: execute_offload (immediate complete)"
        );
        Ok(TransferCompleteNotification::completed())
    }

    fn notify_offload_complete(&self, request_id: &str) -> Result<()> {
        tracing::debug!(
            request_id,
            "MockCoordinator: notify_offload_complete (no-op)"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_coordinator_onboard() {
        let coord = MockCoordinator::new();

        let g2_blocks: Vec<BlockId> = vec![0, 1];
        let g1_blocks: Vec<BlockId> = vec![10, 11];

        let notification = coord
            .execute_onboard("test-req-1", &g2_blocks, None, &g1_blocks)
            .expect("onboard should succeed");

        // MockCoordinator returns immediately-complete notifications (no yielding)
        assert!(!notification.could_yield());
    }

    #[test]
    fn test_mock_coordinator_offload() {
        let coord = MockCoordinator::new();

        let g1_blocks: Vec<BlockId> = vec![0];
        let g2_blocks: Vec<BlockId> = vec![100];

        let notification = coord
            .execute_offload("test-req-2", &g1_blocks, &g2_blocks)
            .expect("offload should succeed");

        // MockCoordinator returns immediately-complete notifications (no yielding)
        assert!(!notification.could_yield());
    }

    #[test]
    fn test_mock_coordinator_notify() {
        let coord = MockCoordinator::new();

        coord
            .notify_onboard_complete("test-req-3")
            .expect("notify should succeed");

        coord
            .notify_offload_complete("test-req-4")
            .expect("notify should succeed");
    }
}
