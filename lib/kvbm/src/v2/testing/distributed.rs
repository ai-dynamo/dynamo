// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed leader testing utilities.

use anyhow::Result;
use std::sync::Arc;

use crate::{
    logical::{blocks::BlockMetadata, manager::BlockManager},
    v2::{
        InstanceId,
        distributed::leader::InstanceLeader,
        integrations::{G2, G3},
        logical::pools::SequenceHash,
    },
};

use super::{managers, nova};

/// Container for a test InstanceLeader with its managers.
pub struct TestInstanceLeader {
    pub instance_id: InstanceId,
    pub leader: InstanceLeader,
    pub g2_manager: Arc<BlockManager<G2>>,
    pub g3_manager: Option<Arc<BlockManager<G3>>>,
}

/// Container for a pair of connected InstanceLeaders.
pub struct InstanceLeaderPair {
    pub leader_a: TestInstanceLeader,
    pub leader_b: TestInstanceLeader,
}

/// Create a pair of InstanceLeaders connected via Nova for integration testing.
///
/// Setup:
/// - Two Nova instances with TCP transport
/// - Bidirectional peer registration
/// - G2 BlockManagers for each leader
/// - Handlers registered for distributed communication
///
/// # Arguments
/// * `block_count` - Number of blocks in each G2 manager
/// * `block_size` - Tokens per block
///
/// # Returns
/// InstanceLeaderPair with both leaders ready for testing
///
/// # Example
/// ```ignore
/// let pair = create_instance_leader_pair(100, 16).await?;
///
/// // Populate leader A with blocks
/// let (_, hashes) = populate_leader_with_blocks(&pair.leader_a, 32, 16, 0)?;
///
/// // Leader B can search leader A
/// let result = pair.leader_b.leader.find_matches(&hashes)?;
/// ```
pub async fn create_instance_leader_pair(
    block_count: usize,
    block_size: usize,
) -> Result<InstanceLeaderPair> {
    // Create Nova pair
    let nova::NovaPair { nova_a, nova_b } = nova::create_nova_pair_tcp().await?;

    // Create G2 managers
    let g2_manager_a = Arc::new(managers::create_test_manager::<G2>(block_count, block_size));
    let g2_manager_b = Arc::new(managers::create_test_manager::<G2>(block_count, block_size));

    // Build InstanceLeader A
    let leader_a = InstanceLeader::builder()
        .instance_id(nova_a.instance_id())
        .nova(nova_a.clone())
        .g2_manager(g2_manager_a.clone())
        .workers(vec![]) // No workers for now (no transfers)
        .remote_leaders(vec![nova_b.instance_id()])
        .build()?;

    // Register handlers for A
    leader_a.register_handlers()?;

    // Build InstanceLeader B
    let leader_b = InstanceLeader::builder()
        .instance_id(nova_b.instance_id())
        .nova(nova_b.clone())
        .g2_manager(g2_manager_b.clone())
        .workers(vec![]) // No workers for now
        .remote_leaders(vec![nova_a.instance_id()])
        .build()?;

    // Register handlers for B
    leader_b.register_handlers()?;

    Ok(InstanceLeaderPair {
        leader_a: TestInstanceLeader {
            instance_id: nova_a.instance_id(),
            leader: leader_a,
            g2_manager: g2_manager_a,
            g3_manager: None,
        },
        leader_b: TestInstanceLeader {
            instance_id: nova_b.instance_id(),
            leader: leader_b,
            g2_manager: g2_manager_b,
            g3_manager: None,
        },
    })
}

/// Populate a leader's G2 manager with token blocks.
///
/// # Arguments
/// * `leader` - The test leader instance
/// * `num_blocks` - Number of blocks to create
/// * `block_size` - Tokens per block
/// * `start_token` - Starting token value
///
/// # Returns
/// (BlockManager, Vec<SequenceHash>) - Manager and sequence hashes of populated blocks
///
/// # Example
/// ```ignore
/// let pair = create_instance_leader_pair(100, 4).await?;
/// let (manager, hashes) = populate_leader_with_blocks(&pair.leader_a, 32, 4, 0)?;
/// assert_eq!(hashes.len(), 32);
/// ```
pub fn populate_leader_with_blocks(
    leader: &TestInstanceLeader,
    num_blocks: usize,
    block_size: usize,
    start_token: u32,
) -> Result<(Arc<BlockManager<G2>>, Vec<SequenceHash>)> {
    let token_sequence = super::token_blocks::create_token_sequence(num_blocks, block_size, start_token);
    let seq_hashes = managers::populate_manager_with_blocks(&leader.g2_manager, token_sequence.blocks())?;

    Ok((leader.g2_manager.clone(), seq_hashes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_instance_leader_pair() {
        let pair = create_instance_leader_pair(100, 16)
            .await
            .expect("Should create leader pair");

        // Verify different instance IDs
        assert_ne!(pair.leader_a.instance_id, pair.leader_b.instance_id);

        // Verify managers are configured correctly
        assert_eq!(pair.leader_a.g2_manager.total_blocks(), 100);
        assert_eq!(pair.leader_a.g2_manager.block_size(), 16);
        assert_eq!(pair.leader_b.g2_manager.total_blocks(), 100);
        assert_eq!(pair.leader_b.g2_manager.block_size(), 16);
    }

    #[tokio::test]
    async fn test_populate_leader_with_blocks() {
        let pair = create_instance_leader_pair(50, 4)
            .await
            .expect("Should create pair");

        let (manager, hashes) = populate_leader_with_blocks(&pair.leader_a, 10, 4, 0)
            .expect("Should populate");

        assert_eq!(hashes.len(), 10);
        assert_eq!(manager.available_blocks(), 50); // All blocks available (10 in inactive)

        // Verify blocks can be matched
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 10);
    }
}
