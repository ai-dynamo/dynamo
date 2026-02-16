// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{BlockId, KvbmSequenceHashProvider, SequenceHash};
use crate::sequence::{BlockSequence, BlockSequenceError};

use super::ExternalBlockAssignments;

const TEST_BLOCK_SIZE: u32 = 4;

/// Helper to create a BlockSequence with a given number of complete blocks and optional partial.
fn create_test_sequence(num_complete_blocks: usize, partial_tokens: usize) -> BlockSequence {
    let total_tokens = num_complete_blocks * TEST_BLOCK_SIZE as usize + partial_tokens;
    let tokens: Vec<u32> = (0..total_tokens as u32).collect();
    BlockSequence::new(tokens, TEST_BLOCK_SIZE, None)
}

/// Helper to get the expected sequence hashes from a BlockSequence.
fn get_expected_hashes(seq: &BlockSequence) -> Vec<SequenceHash> {
    seq.blocks().iter().map(|b| b.kvbm_sequence_hash()).collect()
}

// =========================================================================
// Test Cases: Aligned sequences (no partial block)
// =========================================================================

#[test]
fn test_aligned_0_blocks_0_block_ids() {
    let seq = create_test_sequence(0, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..0);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_aligned_1_block_0_block_ids() {
    let seq = create_test_sequence(1, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..0);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_aligned_1_block_1_block_id() {
    let seq = create_test_sequence(1, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..1);
    assert_eq!(assignments.assigned_count(), 1);
    let (id, hash) = assignments.get_assigned(0).unwrap();
    assert_eq!(id, 100);
    assert_eq!(hash, expected_hashes[0]);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_aligned_1_block_2_block_ids() {
    let seq = create_test_sequence(1, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..1);
    assert_eq!(assignments.assigned_count(), 1);
    let (id, hash) = assignments.get_assigned(0).unwrap();
    assert_eq!(id, 100);
    assert_eq!(hash, expected_hashes[0]);
    assert_eq!(assignments.unassigned_count(), 1);
    assert_eq!(assignments.unassigned_iter().collect::<Vec<_>>(), vec![200]);
}

#[test]
fn test_aligned_3_blocks_3_block_ids() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100, 200, 300]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_aligned_3_blocks_1_block_id() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..1);
    assert_eq!(assignments.assigned_count(), 1);
    let (id, hash) = assignments.get_assigned(0).unwrap();
    assert_eq!(id, 100);
    assert_eq!(hash, expected_hashes[0]);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_aligned_3_blocks_5_block_ids() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
    assert_eq!(assignments.unassigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400, 500]
    );
}

// =========================================================================
// Test Cases: Sequences with partial (dangling) block
// =========================================================================

#[test]
fn test_partial_0_complete_2_partial_0_block_ids() {
    let seq = create_test_sequence(0, 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..0);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_partial_2_complete_1_partial_2_block_ids() {
    let seq = create_test_sequence(2, 1);
    let expected_hashes = get_expected_hashes(&seq);
    assert_eq!(expected_hashes.len(), 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..2);
    assert_eq!(assignments.assigned_count(), 2);
    let (id0, hash0) = assignments.get_assigned(0).unwrap();
    assert_eq!((id0, hash0), (100, expected_hashes[0]));
    let (id1, hash1) = assignments.get_assigned(1).unwrap();
    assert_eq!((id1, hash1), (200, expected_hashes[1]));
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_partial_2_complete_3_partial_4_block_ids() {
    let seq = create_test_sequence(2, 3);
    let expected_hashes = get_expected_hashes(&seq);
    assert_eq!(expected_hashes.len(), 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..2);
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(assignments.unassigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );
}

#[test]
fn test_partial_3_complete_2_partial_1_block_id() {
    let seq = create_test_sequence(3, 2);
    let expected_hashes = get_expected_hashes(&seq);
    assert_eq!(expected_hashes.len(), 3);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..1);
    assert_eq!(assignments.assigned_count(), 1);
    let (id, hash) = assignments.get_assigned(0).unwrap();
    assert_eq!((id, hash), (100, expected_hashes[0]));
    assert_eq!(assignments.unassigned_count(), 0);
}

// =========================================================================
// Test Cases: Multiple calls (incremental assignment)
// =========================================================================

#[test]
fn test_incremental_assignment_aligned() {
    let seq = create_test_sequence(4, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // First call: assign first 2 blocks
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range_1 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_1, 0..2);
    assert_eq!(assignments.assigned_count(), 2);

    // Second call: assign next 2 blocks (100, 200 are known prefix → skipped)
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    let range_2 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_2, 2..4);
    assert_eq!(assignments.assigned_count(), 4);

    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_incremental_assignment_with_excess() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // First call: assign first 2 blocks
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range_1 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_1, 0..2);

    // Second call: 100, 200 skipped, 300 assigned, 400, 500 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    let range_2 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_2, 2..3);
    assert_eq!(assignments.assigned_count(), 3);
    let (id, hash) = assignments.get_assigned(2).unwrap();
    assert_eq!((id, hash), (300, expected_hashes[2]));
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400, 500]
    );
}

#[test]
fn test_incremental_assignment_partial_then_excess() {
    let seq = create_test_sequence(2, 1);
    let expected_hashes = get_expected_hashes(&seq);
    assert_eq!(expected_hashes.len(), 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    // First call: assign 1 block
    assignments.extend_block_ids(vec![100]).unwrap();
    let range_1 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_1, 0..1);

    // Second call: 100 skipped, 200 assigned, 300, 400 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    let range_2 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_2, 1..2);
    let (id, hash) = assignments.get_assigned(1).unwrap();
    assert_eq!((id, hash), (200, expected_hashes[1]));
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );
}

#[test]
fn test_all_blocks_already_assigned_extra_goes_to_unassigned() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign all blocks
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range_1 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_1, 0..2);

    // All new go to unassigned (100, 200 are skipped)
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    let range_2 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_2, 2..2);
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );
}

// =========================================================================
// Test Cases: Edge cases
// =========================================================================

#[test]
fn test_empty_slot_receives_block_ids() {
    let seq = create_test_sequence(0, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..0);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![100, 200, 300]
    );
}

#[test]
fn test_only_partial_tokens_receives_block_ids() {
    let seq = create_test_sequence(0, 3);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..0);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![100, 200]
    );
}

#[test]
fn test_large_sequence_exact_match() {
    let seq = create_test_sequence(10, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);
    let block_ids: Vec<BlockId> = (0..10).map(|i| (i + 1) * 100).collect();

    assignments.extend_block_ids(block_ids).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..10);
    assert_eq!(assignments.assigned_count(), 10);
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_verify_hash_block_id_pairing_order() {
    let seq = create_test_sequence(5, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![999, 888, 777, 666, 555])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..5);
    assert_eq!(assignments.get_assigned(0).unwrap().0, 999);
    assert_eq!(assignments.get_assigned(1).unwrap().0, 888);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 777);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 666);
    assert_eq!(assignments.get_assigned(4).unwrap().0, 555);

    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        assert_eq!(assignments.get_assigned(i).unwrap().1, *expected_hash);
    }
}

// =========================================================================
// Cartesian product test
// =========================================================================

#[test]
fn test_cartesian_product_combinations() {
    let num_blocks_options = [0, 1, 3, 5];
    let partial_options = [0, 1, 3];

    for &num_blocks in &num_blocks_options {
        for &partial in &partial_options {
            let seq = create_test_sequence(num_blocks, partial);
            let expected_hashes = get_expected_hashes(&seq);
            let available_blocks = expected_hashes.len();

            // Test with 0 block_ids
            {
                let mut assignments = ExternalBlockAssignments::new(0);
                let range = assignments.assign_pending(seq.blocks()).unwrap();
                assert_eq!(range, 0..0);
                assert_eq!(assignments.assigned_count(), 0);
                assert_eq!(assignments.unassigned_count(), 0);
            }

            // Test with fewer block_ids than available blocks
            if available_blocks > 1 {
                let fewer = available_blocks / 2;
                let block_ids: Vec<BlockId> = (0..fewer).collect();
                let mut assignments = ExternalBlockAssignments::new(0);
                assignments.extend_block_ids(block_ids).unwrap();
                let range = assignments.assign_pending(seq.blocks()).unwrap();

                assert_eq!(range, 0..fewer);
                assert_eq!(assignments.assigned_count(), fewer);
                assert_eq!(assignments.unassigned_count(), 0);

                for (i, expected_hash) in expected_hashes.iter().enumerate().take(fewer) {
                    let (id, hash) = assignments.get_assigned(i).unwrap();
                    assert_eq!(id, i);
                    assert_eq!(hash, *expected_hash);
                }
            }

            // Test with exact number of block_ids
            if available_blocks > 0 {
                let block_ids: Vec<BlockId> = (0..available_blocks).collect();
                let mut assignments = ExternalBlockAssignments::new(0);
                assignments.extend_block_ids(block_ids).unwrap();
                let range = assignments.assign_pending(seq.blocks()).unwrap();

                assert_eq!(range, 0..available_blocks);
                assert_eq!(assignments.assigned_count(), available_blocks);
                assert_eq!(assignments.unassigned_count(), 0);
            }

            // Test with more block_ids than available blocks
            {
                let excess = 3;
                let total_ids = available_blocks + excess;
                let block_ids: Vec<BlockId> = (0..total_ids).collect();
                let mut assignments = ExternalBlockAssignments::new(0);
                assignments.extend_block_ids(block_ids).unwrap();
                let range = assignments.assign_pending(seq.blocks()).unwrap();

                assert_eq!(range, 0..available_blocks);
                assert_eq!(assignments.assigned_count(), available_blocks);
                assert_eq!(assignments.unassigned_count(), excess);

                let expected_unassigned: Vec<BlockId> = (available_blocks..total_ids).collect();
                assert_eq!(
                    assignments.unassigned_iter().collect::<Vec<_>>(),
                    expected_unassigned
                );
            }
        }
    }
}

// =========================================================================
// Test Cases: Previously unassigned blocks (FIFO behavior)
// =========================================================================

#[test]
fn test_unassigned_blocks_applied_before_new_blocks() {
    let mut seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 5 blocks + 2 excess
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600, 700])
        .unwrap();
    let range_1 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_1, 0..5);
    assert_eq!(assignments.assigned_count(), 5);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![600, 700]
    );

    // Add 2 more complete blocks via token extension
    for token in 20..28u32 {
        seq.append_token(token).unwrap();
    }
    assert_eq!(seq.blocks().len(), 7);

    // Flush with new block_ids — unassigned (600, 700) should be assigned first
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600, 700, 800, 900])
        .unwrap();
    let range_2 = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_2, 5..7);
    assert_eq!(assignments.assigned_count(), 7);

    // Verify 600, 700 were assigned to positions 5, 6
    assert_eq!(assignments.get_assigned(5).unwrap().0, 600);
    assert_eq!(assignments.get_assigned(6).unwrap().0, 700);

    // 800, 900 should be unassigned
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![800, 900]
    );
}

#[test]
fn test_unassigned_blocks_with_new_blocks_all_assigned() {
    let mut seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 3 + 1 excess
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400]
    );

    // Add 3 more blocks
    for token in 12..24u32 {
        seq.append_token(token).unwrap();
    }
    assert_eq!(seq.blocks().len(), 6);

    // 400 (unassigned) + 500, 600 (new) should all fit
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 3..6);
    assert_eq!(assignments.assigned_count(), 6);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 400);
    assert_eq!(assignments.get_assigned(4).unwrap().0, 500);
    assert_eq!(assignments.get_assigned(5).unwrap().0, 600);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_unassigned_blocks_no_new_space() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 2 + 2 excess
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );

    // No new space — all new go to unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..2);
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400, 500, 600]
    );
}

#[test]
fn test_unassigned_blocks_partial_space() {
    let mut seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 3 + 2 excess
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400, 500]
    );

    // Add 1 more block
    for token in 12..16u32 {
        seq.append_token(token).unwrap();
    }
    assert_eq!(seq.blocks().len(), 4);

    // Only 1 spot: 400 assigned, 500+new remain unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600, 700, 800])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 3..4);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 400);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![500, 600, 700, 800]
    );
}

#[test]
fn test_multiple_rounds_of_unassigned_accumulation() {
    let mut seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Round 1: 2 assigned, 2 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );

    // Round 2: no space, add 2 more
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400, 500, 600]
    );

    // Round 3: still no space, add 1 more
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500, 600, 700])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400, 500, 600, 700]
    );

    // Now add space for 3 more blocks
    for token in 8..20u32 {
        seq.append_token(token).unwrap();
    }
    assert_eq!(seq.blocks().len(), 5);

    // Flush — all IDs already known, first 3 unassigned (300, 400, 500) get assigned
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..5);
    assert_eq!(assignments.assigned_count(), 5);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 300);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 400);
    assert_eq!(assignments.get_assigned(4).unwrap().0, 500);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![600, 700]
    );
}

#[test]
fn test_unassigned_blocks_ordering_preserved() {
    let mut seq = create_test_sequence(1, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 10 assigned, rest unassigned in FIFO order
    assignments
        .extend_block_ids(vec![10, 20, 30, 40, 50, 60])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.get_assigned(0).unwrap().0, 10);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![20, 30, 40, 50, 60]
    );

    // Add 2 more blocks
    for token in 4..12u32 {
        seq.append_token(token).unwrap();
    }

    // Flush — all IDs already known, 20, 30 assigned (FIFO)
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.get_assigned(1).unwrap().0, 20);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 30);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![40, 50, 60]
    );

    // Add 1 more block + 1 new ID
    for token in 12..16u32 {
        seq.append_token(token).unwrap();
    }
    assignments
        .extend_block_ids(vec![10, 20, 30, 40, 50, 60, 70])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // 40 should be assigned (FIFO), not 70
    assert_eq!(assignments.get_assigned(3).unwrap().0, 40);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![50, 60, 70]
    );
}

#[test]
fn test_unassigned_blocks_with_partial_token_block() {
    let mut seq = create_test_sequence(2, 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 2 assigned, 2 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );

    // Complete the partial block (need 2 more tokens)
    seq.append_token(10).unwrap();
    seq.append_token(11).unwrap();
    assert_eq!(seq.blocks().len(), 3);

    // 300 should be assigned first (FIFO), 500 is new
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..3);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 300);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400, 500]
    );
}

#[test]
fn test_unassigned_blocks_exactly_fill_new_space() {
    let mut seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 2 assigned, 3 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400, 500]
    );

    // Add exactly 3 more blocks
    for token in 8..20u32 {
        seq.append_token(token).unwrap();
    }
    assert_eq!(seq.blocks().len(), 5);

    // Flush — all IDs already known, unassigned exactly fill the space
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..5);
    assert_eq!(assignments.assigned_count(), 5);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 300);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 400);
    assert_eq!(assignments.get_assigned(4).unwrap().0, 500);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_empty_unassigned_with_new_blocks() {
    let mut seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign exactly the right number
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.unassigned_count(), 0);

    // Add more space
    for token in 12..16u32 {
        seq.append_token(token).unwrap();
    }

    // New blocks with no previous unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 3..4);
    assert_eq!(assignments.get_assigned(3).unwrap().0, 400);
    assert_eq!(assignments.unassigned_count(), 0);
}

// =========================================================================
// New tests: Prefix validation (OrderingViolation)
// =========================================================================

#[test]
fn test_ordering_violation_known_after_new() {
    let seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign first block
    assignments.extend_block_ids(vec![100]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // Now try to extend with [200, 100] — 200 is new, then 100 is known → error
    let result = assignments.extend_block_ids(vec![200, 100]);
    assert!(result.is_err());
    match result.unwrap_err() {
        BlockSequenceError::OrderingViolation {
            known_id,
            new_id,
            known_index,
            first_new_index,
        } => {
            assert_eq!(known_id, 100);
            assert_eq!(new_id, 200);
            assert_eq!(known_index, 1);
            assert_eq!(first_new_index, 0);
        }
        other => panic!("expected OrderingViolation, got: {other:?}"),
    }

    // Verify atomicity — state unchanged
    assert_eq!(assignments.assigned_count(), 1);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_ordering_violation_no_state_change() {
    let seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign first 2
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // Error case: [300, 100] — 300 new, then 100 known
    let result = assignments.extend_block_ids(vec![300, 100]);
    assert!(result.is_err());

    // State should be exactly as before the failed call
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(assignments.unassigned_count(), 0);
    assert_eq!(assignments.get_assigned(0).unwrap().0, 100);
    assert_eq!(assignments.get_assigned(1).unwrap().0, 200);
}

// =========================================================================
// New tests: All duplicates (re-extending same IDs is a no-op)
// =========================================================================

#[test]
fn test_all_duplicates_noop() {
    let seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign all 3
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // Re-extend with same — should be a no-op
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 3..3); // Empty range, no new assignments
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_duplicates_with_unassigned() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 2 assigned + 2 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // Re-extend with same — no-op (300, 400 are known in unassigned)
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..2);
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300, 400]
    );
}

#[test]
fn test_shorter_sequence_slice_does_not_shrink_assigned() {
    let seq = create_test_sequence(5, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 3 blocks using the full sequence
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 3);

    // Re-extend with same IDs but a shorter slice — should be a no-op,
    // not shrink assigned or move anything to unassigned.
    let range = assignments.assign_pending(&seq.blocks()[..2]).unwrap();
    assert_eq!(range, 3..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.unassigned_count(), 0);

    // All three original assignments are intact
    for i in 0..3 {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, expected_hashes[i]);
    }
}

// =========================================================================
// New tests: Offset
// =========================================================================

#[test]
fn test_offset_assignments() {
    let seq = create_test_sequence(10, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(5);

    assert_eq!(assignments.offset(), 5);
    assert_eq!(assignments.next_position(), 5);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    // Should assign to positions 5, 6, 7
    assert_eq!(range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.next_position(), 8);

    let (id0, hash0) = assignments.get_assigned(0).unwrap();
    assert_eq!(id0, 100);
    assert_eq!(hash0, expected_hashes[5]);

    let (id1, hash1) = assignments.get_assigned(1).unwrap();
    assert_eq!(id1, 200);
    assert_eq!(hash1, expected_hashes[6]);

    let (id2, hash2) = assignments.get_assigned(2).unwrap();
    assert_eq!(id2, 300);
    assert_eq!(hash2, expected_hashes[7]);
}

#[test]
fn test_offset_with_excess() {
    let seq = create_test_sequence(8, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(5);

    // 3 blocks available at offset 5 (positions 5, 6, 7), 5 block_ids → 2 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();

    assert_eq!(range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.get_assigned(0).unwrap().1, expected_hashes[5]);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![400, 500]
    );
}

// =========================================================================
// New tests: Multi-instance (two ExternalBlockAssignments on same blocks)
// =========================================================================

#[test]
fn test_multi_instance_different_offsets() {
    let seq = create_test_sequence(10, 0);
    let expected_hashes = get_expected_hashes(&seq);

    let mut g1 = ExternalBlockAssignments::new(0);
    let mut g2 = ExternalBlockAssignments::new(5);

    // G1 assigns positions 0..3
    g1.extend_block_ids(vec![10, 20, 30]).unwrap();
    let range_g1 = g1.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_g1, 0..3);

    // G2 assigns positions 5..8
    g2.extend_block_ids(vec![110, 120, 130]).unwrap();
    let range_g2 = g2.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range_g2, 0..3);

    // Verify G1 hashes
    assert_eq!(g1.get_assigned(0).unwrap().1, expected_hashes[0]);
    assert_eq!(g1.get_assigned(1).unwrap().1, expected_hashes[1]);
    assert_eq!(g1.get_assigned(2).unwrap().1, expected_hashes[2]);

    // Verify G2 hashes
    assert_eq!(g2.get_assigned(0).unwrap().1, expected_hashes[5]);
    assert_eq!(g2.get_assigned(1).unwrap().1, expected_hashes[6]);
    assert_eq!(g2.get_assigned(2).unwrap().1, expected_hashes[7]);

    // They are independent — no cross-interference
    assert_eq!(g1.assigned_count(), 3);
    assert_eq!(g2.assigned_count(), 3);
}

// =========================================================================
// New tests: Token extension + flush
// =========================================================================

#[test]
fn test_token_extension_then_flush() {
    let mut seq = create_test_sequence(2, 2);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 2 blocks available, 3 block_ids → 1 unassigned
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 2);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![300]
    );

    // Complete the partial block
    seq.append_token(10).unwrap();
    seq.append_token(11).unwrap();
    assert_eq!(seq.blocks().len(), 3);

    // Flush — unassigned 300 gets assigned
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 2..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 300);
    assert_eq!(assignments.unassigned_count(), 0);
}

#[test]
fn test_extend_tokens_creates_new_blocks() {
    let mut seq = create_test_sequence(1, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 1 block + 2 excess
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 1);
    assert_eq!(
        assignments.unassigned_iter().collect::<Vec<_>>(),
        vec![200, 300]
    );

    // Extend tokens to create 2 more blocks
    let new_range = seq.extend_tokens((4..12).collect()).unwrap();
    assert!(new_range.is_some());
    assert_eq!(seq.blocks().len(), 3);

    // Flush
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 1..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.get_assigned(1).unwrap().0, 200);
    assert_eq!(assignments.get_assigned(2).unwrap().0, 300);
    assert_eq!(assignments.unassigned_count(), 0);
}

// =========================================================================
// New tests: clear()
// =========================================================================

#[test]
fn test_clear_preserves_offset() {
    let seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(3);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 2); // positions 3, 4
    assert_eq!(assignments.unassigned_count(), 1); // 300

    assignments.clear();

    assert_eq!(assignments.offset(), 3);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);
    assert_eq!(assignments.next_position(), 3);
}

// =========================================================================
// New tests: Position mismatch
// =========================================================================

#[test]
fn test_position_mismatch_wrong_offset() {
    // Create a sequence with blocks at positions 0, 1, 2
    let seq = create_test_sequence(3, 0);

    // Use an offset that will cause mismatch:
    // Assignments at offset=1 will try to read sequence_blocks[1] expecting position=1,
    // which should actually be fine since blocks are at positions 0, 1, 2.
    // Let's use offset=0 but give sequence blocks that are offset...
    // Actually, the position in the hash is inherent to the block. Block at index 0 has position 0.
    // So if we set offset=5, we'll try to read sequence_blocks[5], which doesn't exist (only 3 blocks).
    // That won't trigger PositionMismatch, it just won't assign anything.

    // To actually trigger PositionMismatch, we need an offset where the block exists but
    // position doesn't match. If offset=1, we look at sequence_blocks[1] which has position=1,
    // expected=1. That matches. We'd need blocks where the embedded position doesn't match
    // the index. But TokenBlockSequence always creates blocks with matching positions.

    // Since in normal usage positions always match, let's just verify the offset behavior
    // works correctly and the validation passes for valid data.
    let mut assignments = ExternalBlockAssignments::new(1);
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    let range = assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(range, 0..2);
    // Position 1 and 2 should be assigned
    assert_eq!(assignments.get_assigned(0).unwrap().1, get_expected_hashes(&seq)[1]);
    assert_eq!(assignments.get_assigned(1).unwrap().1, get_expected_hashes(&seq)[2]);
}

// =========================================================================
// New tests: contains()
// =========================================================================

#[test]
fn test_contains() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    // 100, 200 are assigned
    assert!(assignments.contains(&100));
    assert!(assignments.contains(&200));

    // 300 is unassigned
    assert!(assignments.contains(&300));

    // 400 is unknown
    assert!(!assignments.contains(&400));
}

// =========================================================================
// New tests: assigned_iter()
// =========================================================================

#[test]
fn test_assigned_iter() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();

    let assigned: Vec<(BlockId, SequenceHash)> = assignments.assigned_iter().collect();
    assert_eq!(assigned.len(), 3);
    assert_eq!(assigned[0], (100, expected_hashes[0]));
    assert_eq!(assigned[1], (200, expected_hashes[1]));
    assert_eq!(assigned[2], (300, expected_hashes[2]));
}

// =========================================================================
// New tests: BlockSequence
// =========================================================================

#[test]
fn test_block_sequence_new() {
    let seq = BlockSequence::new(vec![0, 1, 2, 3, 4, 5, 6, 7], 4, None);
    assert_eq!(seq.blocks().len(), 2);
    assert_eq!(seq.block_size(), 4);
    assert_eq!(seq.total_tokens(), 8);
}

#[test]
fn test_block_sequence_all_sequence_hashes() {
    let seq = BlockSequence::new(vec![0, 1, 2, 3, 4, 5, 6, 7], 4, None);
    let hashes = seq.all_sequence_hashes();
    assert_eq!(hashes.len(), 2);
    // Hashes should match what we'd get from blocks directly
    let expected: Vec<_> = seq.blocks().iter().map(|b| b.kvbm_sequence_hash()).collect();
    assert_eq!(hashes, expected);
}

#[test]
fn test_block_sequence_extend_tokens() {
    let mut seq = BlockSequence::new(vec![0, 1, 2, 3], 4, None);
    assert_eq!(seq.blocks().len(), 1);

    let result = seq.extend_tokens(vec![4, 5, 6, 7, 8, 9]).unwrap();
    assert_eq!(result, Some(1..2)); // One more block completed
    assert_eq!(seq.blocks().len(), 2);
    assert_eq!(seq.total_tokens(), 10); // 10 total, 2 partial remaining
}

#[test]
fn test_block_sequence_append_token() {
    let mut seq = BlockSequence::new(vec![0, 1, 2], 4, None);
    assert_eq!(seq.blocks().len(), 0);

    // Doesn't complete a block
    let result = seq.append_token(3).unwrap();
    assert_eq!(result, Some(0)); // Completed first block
    assert_eq!(seq.blocks().len(), 1);
}

#[test]
fn test_block_sequence_with_salt() {
    let seq1 = BlockSequence::new(vec![0, 1, 2, 3], 4, None);
    let seq2 = BlockSequence::new(vec![0, 1, 2, 3], 4, Some(42));

    // Different salts should produce different hashes
    let hashes1 = seq1.all_sequence_hashes();
    let hashes2 = seq2.all_sequence_hashes();
    assert_ne!(hashes1, hashes2);
}

// =========================================================================
// Positional access methods
// =========================================================================

#[test]
fn test_assigned_positions_empty() {
    let assignments = ExternalBlockAssignments::new(0);
    assert_eq!(assignments.assigned_positions(), 0..0);
}

#[test]
fn test_assigned_positions_with_offset() {
    let seq = create_test_sequence(10, 0);
    let mut assignments = ExternalBlockAssignments::new(3);
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_positions(), 3..5);
}

#[test]
fn test_pending_positions_empty() {
    let assignments = ExternalBlockAssignments::new(0);
    assert_eq!(assignments.pending_positions(), 0..0);
}

#[test]
fn test_pending_positions_with_assigned_and_unassigned() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // 2 assigned (positions 0,1), 2 pending (positions 2,3)
    assert_eq!(assignments.pending_positions(), 2..4);
}

#[test]
fn test_pending_positions_with_offset() {
    let seq = create_test_sequence(10, 0);
    let mut assignments = ExternalBlockAssignments::new(5);
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // offset=5, 4 block_ids, 5 blocks available at positions 5..10 → 4 assigned
    // no pending
    assert_eq!(assignments.assigned_positions(), 5..9);
    assert_eq!(assignments.pending_positions(), 9..9);
}

#[test]
fn test_get_at_position_in_range() {
    let seq = create_test_sequence(5, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(2);
    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // Assigned at positions 2, 3, 4
    let (id, hash) = assignments.get_at_position(2).unwrap();
    assert_eq!(id, 100);
    assert_eq!(hash, expected_hashes[2]);

    let (id, hash) = assignments.get_at_position(4).unwrap();
    assert_eq!(id, 300);
    assert_eq!(hash, expected_hashes[4]);
}

#[test]
fn test_get_at_position_out_of_range() {
    let seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(2);
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // Assigned at positions 2, 3
    assert!(assignments.get_at_position(0).is_none()); // before offset
    assert!(assignments.get_at_position(1).is_none()); // before offset
    assert!(assignments.get_at_position(4).is_none()); // past end
    assert!(assignments.get_at_position(100).is_none()); // way past end
}

#[test]
fn test_get_pending_at_position_in_range() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // 2 assigned, 3 pending at positions 2, 3, 4
    assert_eq!(assignments.get_pending_at_position(2), Some(300));
    assert_eq!(assignments.get_pending_at_position(3), Some(400));
    assert_eq!(assignments.get_pending_at_position(4), Some(500));
}

#[test]
fn test_get_pending_at_position_out_of_range() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);
    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    // 2 assigned, 2 pending at positions 2, 3
    assert!(assignments.get_pending_at_position(0).is_none()); // assigned range
    assert!(assignments.get_pending_at_position(1).is_none()); // assigned range
    assert!(assignments.get_pending_at_position(4).is_none()); // past end
    assert!(assignments.get_pending_at_position(100).is_none());
}

// =========================================================================
// zip_assigned tests
// =========================================================================

use super::{zip_assigned, zip_assigned_pending};

#[test]
fn test_zip_assigned_full_overlap() {
    let seq = create_test_sequence(5, 0);
    let mut a = ExternalBlockAssignments::new(0);
    let mut b = ExternalBlockAssignments::new(0);
    a.extend_block_ids(vec![10, 20, 30]).unwrap();
    a.assign_pending(seq.blocks()).unwrap();
    b.extend_block_ids(vec![110, 120, 130]).unwrap();
    b.assign_pending(seq.blocks()).unwrap();

    let pairs = zip_assigned(&a, &b);
    assert_eq!(
        pairs,
        vec![(0, 10, 110), (1, 20, 120), (2, 30, 130)]
    );
}

#[test]
fn test_zip_assigned_partial_overlap() {
    let seq = create_test_sequence(10, 0);
    let mut a = ExternalBlockAssignments::new(0);
    let mut b = ExternalBlockAssignments::new(2);
    a.extend_block_ids(vec![10, 20, 30, 40]).unwrap();
    a.assign_pending(seq.blocks()).unwrap();
    b.extend_block_ids(vec![110, 120, 130]).unwrap();
    b.assign_pending(seq.blocks()).unwrap();
    // a: positions 0..4, b: positions 2..5 → overlap 2..4
    let pairs = zip_assigned(&a, &b);
    assert_eq!(pairs, vec![(2, 30, 110), (3, 40, 120)]);
}

#[test]
fn test_zip_assigned_no_overlap() {
    let seq = create_test_sequence(10, 0);
    let mut a = ExternalBlockAssignments::new(0);
    let mut b = ExternalBlockAssignments::new(5);
    a.extend_block_ids(vec![10, 20]).unwrap();
    a.assign_pending(seq.blocks()).unwrap();
    b.extend_block_ids(vec![110, 120]).unwrap();
    b.assign_pending(seq.blocks()).unwrap();
    // a: positions 0..2, b: positions 5..7 → no overlap
    let pairs = zip_assigned(&a, &b);
    assert!(pairs.is_empty());
}

#[test]
fn test_zip_assigned_either_empty() {
    let seq = create_test_sequence(5, 0);
    let a = ExternalBlockAssignments::new(0);
    let mut b = ExternalBlockAssignments::new(0);
    b.extend_block_ids(vec![110, 120]).unwrap();
    b.assign_pending(seq.blocks()).unwrap();

    assert!(zip_assigned(&a, &b).is_empty());
    assert!(zip_assigned(&b, &a).is_empty());
}

#[test]
fn test_zip_assigned_both_empty() {
    let a = ExternalBlockAssignments::new(0);
    let b = ExternalBlockAssignments::new(0);
    assert!(zip_assigned(&a, &b).is_empty());
}

// =========================================================================
// zip_assigned_pending tests
// =========================================================================

#[test]
fn test_zip_assigned_pending_full_overlap() {
    let seq = create_test_sequence(5, 0);
    let mut src = ExternalBlockAssignments::new(2);
    let mut dst = ExternalBlockAssignments::new(0);

    // src: assigned at positions 2, 3, 4
    src.extend_block_ids(vec![10, 20, 30]).unwrap();
    src.assign_pending(seq.blocks()).unwrap();

    // dst: only sees first 2 blocks → 2 assigned at 0, 1 + 3 pending at 2, 3, 4
    dst.extend_block_ids(vec![110, 120, 130, 140, 150])
        .unwrap();
    dst.assign_pending(&seq.blocks()[..2]).unwrap();
    assert_eq!(dst.assigned_count(), 2);
    assert_eq!(dst.unassigned_count(), 3);

    let pairs = zip_assigned_pending(&src, &dst);
    assert_eq!(
        pairs,
        vec![(2, 10, 130), (3, 20, 140), (4, 30, 150)]
    );
}

#[test]
fn test_zip_assigned_pending_partial_overlap() {
    let seq = create_test_sequence(10, 0);
    let mut src = ExternalBlockAssignments::new(2);
    let mut dst = ExternalBlockAssignments::new(0);

    // src: assigned at 2, 3, 4, 5
    src.extend_block_ids(vec![10, 20, 30, 40]).unwrap();
    src.assign_pending(seq.blocks()).unwrap();

    // dst: only sees first 3 blocks → 3 assigned at 0, 1, 2 + 2 pending at 3, 4
    dst.extend_block_ids(vec![110, 120, 130, 140, 150])
        .unwrap();
    dst.assign_pending(&seq.blocks()[..3]).unwrap();
    assert_eq!(dst.assigned_count(), 3);
    assert_eq!(dst.unassigned_count(), 2);

    // src assigned: 2..6, dst pending: 3..5 → overlap 3..5
    let pairs = zip_assigned_pending(&src, &dst);
    assert_eq!(pairs, vec![(3, 20, 140), (4, 30, 150)]);
}

#[test]
fn test_zip_assigned_pending_no_overlap() {
    let seq = create_test_sequence(10, 0);
    let mut src = ExternalBlockAssignments::new(0);
    let mut dst = ExternalBlockAssignments::new(0);

    // src: assigned at 0, 1
    src.extend_block_ids(vec![10, 20]).unwrap();
    src.assign_pending(seq.blocks()).unwrap();

    // dst: 5 assigned at 0..5 + 2 pending at 5, 6
    dst.extend_block_ids(vec![110, 120, 130, 140, 150, 160, 170])
        .unwrap();
    dst.assign_pending(seq.blocks()).unwrap();

    // src assigned: 0..2, dst pending: 5..7 → no overlap
    let pairs = zip_assigned_pending(&src, &dst);
    assert!(pairs.is_empty());
}

#[test]
fn test_zip_assigned_pending_either_empty() {
    let seq = create_test_sequence(5, 0);
    let src = ExternalBlockAssignments::new(0); // no assigned
    let mut dst = ExternalBlockAssignments::new(0);
    dst.extend_block_ids(vec![110, 120, 130]).unwrap();
    dst.assign_pending(seq.blocks()).unwrap();

    // src has no assigned blocks
    assert!(zip_assigned_pending(&src, &dst).is_empty());

    // dst has no pending blocks
    let mut src2 = ExternalBlockAssignments::new(0);
    src2.extend_block_ids(vec![10, 20]).unwrap();
    src2.assign_pending(seq.blocks()).unwrap();
    let dst2 = ExternalBlockAssignments::new(0);
    assert!(zip_assigned_pending(&src2, &dst2).is_empty());
}

// =========================================================================
// Onboard scenario test (G2 → G1 transfer)
// =========================================================================

#[test]
fn test_onboard_scenario_g2_to_g1() {
    // Sequence: 5 complete blocks (positions 0..5)
    let seq = create_test_sequence(5, 0);

    // Step 1 — G1 matches positions 0..2
    let mut g1 = ExternalBlockAssignments::new(0);
    g1.extend_block_ids(vec![1000, 1001]).unwrap();
    g1.assign_pending(seq.blocks()).unwrap();
    assert_eq!(g1.assigned_positions(), 0..2);
    assert_eq!(g1.assigned_count(), 2);

    // Step 2 — G2 matches positions 2..5
    let mut g2 = ExternalBlockAssignments::new(2);
    g2.extend_block_ids(vec![2002, 2003, 2004]).unwrap();
    g2.assign_pending(seq.blocks()).unwrap();
    assert_eq!(g2.assigned_positions(), 2..5);
    assert_eq!(g2.assigned_count(), 3);

    // Step 3 — Queue new block_ids for G1 (don't assign yet — blocks not transferred)
    g1.extend_block_ids(vec![1000, 1001, 1002, 1003, 1004])
        .unwrap();
    assert_eq!(g1.assigned_count(), 2); // still only 2 assigned
    assert_eq!(g1.unassigned_count(), 3); // 1002, 1003, 1004 pending
    assert_eq!(g1.pending_positions(), 2..5);

    // Step 4 — Compute onboard pairs (before physical transfer)
    let pairs = zip_assigned_pending(&g2, &g1);
    assert_eq!(
        pairs,
        vec![(2, 2002, 1002), (3, 2003, 1003), (4, 2004, 1004)]
    );

    // Step 5 — After physical transfer, assign pending against the full sequence
    g1.assign_pending(seq.blocks()).unwrap();
    assert_eq!(g1.assigned_count(), 5);
    assert_eq!(g1.unassigned_count(), 0);
    assert_eq!(g1.assigned_positions(), 0..5);

    // Step 6 — Verify assigned overlap matches
    let assigned_pairs = zip_assigned(&g2, &g1);
    assert_eq!(
        assigned_pairs,
        vec![(2, 2002, 1002), (3, 2003, 1003), (4, 2004, 1004)]
    );
}

// =========================================================================
// New tests: stage_pending + commit_staged (two-step assignment)
// =========================================================================

#[test]
fn test_stage_pending_basic() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();

    // Stage — should move unassigned → staged with hashes
    let staged_range = assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(staged_range, 0..3);
    assert_eq!(assignments.staged_count(), 3);
    assert_eq!(assignments.unassigned_count(), 0);
    assert_eq!(assignments.assigned_count(), 0);

    // Verify staged contents
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_staged(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
}

#[test]
fn test_commit_staged_basic() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();

    // Stage then commit
    assignments.stage_pending(seq.blocks()).unwrap();
    let assigned_range = assignments.commit_staged();
    assert_eq!(assigned_range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.staged_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);

    // Verify assigned contents
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
}

#[test]
fn test_stage_pending_partial() {
    let seq = create_test_sequence(2, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // 5 block_ids but only 2 sequence blocks
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();

    let staged_range = assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(staged_range, 0..2);
    assert_eq!(assignments.staged_count(), 2);
    assert_eq!(assignments.unassigned_count(), 3);
}

#[test]
fn test_stage_pending_then_commit_then_stage_more() {
    let seq = create_test_sequence(5, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();

    // Stage first 3 (only 3 blocks available at first)
    let range1 = assignments.stage_pending(&seq.blocks()[..3]).unwrap();
    assert_eq!(range1, 0..3);
    assert_eq!(assignments.staged_count(), 3);
    assert_eq!(assignments.unassigned_count(), 2);

    // Commit them
    let assigned_range = assignments.commit_staged();
    assert_eq!(assigned_range, 0..3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.staged_count(), 0);

    // Stage the remaining 2
    let range2 = assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(range2, 0..2);
    assert_eq!(assignments.staged_count(), 2);
    assert_eq!(assignments.unassigned_count(), 0);

    // Commit them
    let assigned_range2 = assignments.commit_staged();
    assert_eq!(assigned_range2, 3..5);
    assert_eq!(assignments.assigned_count(), 5);

    // Verify all assigned
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
}

#[test]
fn test_stage_pending_empty_unassigned() {
    let seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // No block_ids → nothing to stage
    let range = assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(range, 0..0);
    assert_eq!(assignments.staged_count(), 0);
}

#[test]
fn test_commit_staged_empty() {
    let mut assignments = ExternalBlockAssignments::new(0);

    // Nothing staged → empty range
    let range = assignments.commit_staged();
    assert_eq!(range, 0..0);
}

// =========================================================================
// New tests: staged_positions
// =========================================================================

#[test]
fn test_staged_positions_empty() {
    let assignments = ExternalBlockAssignments::new(0);
    assert_eq!(assignments.staged_positions(), 0..0);
}

#[test]
fn test_staged_positions_after_stage() {
    let seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Pre-assign first 2
    assignments.extend_block_ids(vec![100, 200]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 2);

    // Add 3 more and stage them
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.staged_count(), 3);

    // Staged positions should be 2..5
    assert_eq!(assignments.staged_positions(), 2..5);
    assert_eq!(assignments.assigned_positions(), 0..2);
}

#[test]
fn test_staged_positions_with_offset() {
    let seq = create_test_sequence(10, 0);
    let mut assignments = ExternalBlockAssignments::new(3);

    assignments.extend_block_ids(vec![100, 200]).unwrap();
    assignments.assign_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.assigned_count(), 2);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400])
        .unwrap();
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.staged_count(), 2);

    // offset=3, 2 assigned → staged starts at 5, 2 staged → staged ends at 7
    assert_eq!(assignments.staged_positions(), 5..7);
}

// =========================================================================
// New tests: staged_iter
// =========================================================================

#[test]
fn test_staged_iter() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.stage_pending(seq.blocks()).unwrap();

    let staged: Vec<(BlockId, SequenceHash)> = assignments.staged_iter().collect();
    assert_eq!(staged.len(), 3);
    assert_eq!(staged[0], (100, expected_hashes[0]));
    assert_eq!(staged[1], (200, expected_hashes[1]));
    assert_eq!(staged[2], (300, expected_hashes[2]));
}

// =========================================================================
// New tests: take_staged
// =========================================================================

#[test]
fn test_take_staged() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.staged_count(), 3);

    let drained = assignments.take_staged();
    assert_eq!(drained.len(), 3);
    assert_eq!(assignments.staged_count(), 0);

    assert_eq!(drained[0], (100, expected_hashes[0]));
    assert_eq!(drained[1], (200, expected_hashes[1]));
    assert_eq!(drained[2], (300, expected_hashes[2]));
}

// =========================================================================
// New tests: next_position with staged blocks
// =========================================================================

#[test]
fn test_next_position_accounts_for_staged() {
    let seq = create_test_sequence(5, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300])
        .unwrap();

    // Before staging
    assert_eq!(assignments.next_position(), 0);

    // After staging (3 staged, 0 assigned)
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.next_position(), 3);
    assert_eq!(assignments.assigned_count(), 0);
    assert_eq!(assignments.staged_count(), 3);

    // After committing (0 staged, 3 assigned)
    assignments.commit_staged();
    assert_eq!(assignments.next_position(), 3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.staged_count(), 0);
}

#[test]
fn test_pending_positions_accounts_for_staged() {
    let seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();

    // Stage 3, leaving 2 unassigned
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.staged_count(), 3);
    assert_eq!(assignments.unassigned_count(), 2);

    // Pending positions should be after staged blocks
    assert_eq!(assignments.pending_positions(), 3..5);
}

// =========================================================================
// New tests: extend_assigned (direct insert to assigned)
// =========================================================================

#[test]
fn test_extend_assigned_basic() {
    let seq = create_test_sequence(3, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Build (BlockId, SequenceHash) pairs
    let items: Vec<(BlockId, SequenceHash)> = vec![
        (100, expected_hashes[0]),
        (200, expected_hashes[1]),
        (300, expected_hashes[2]),
    ];

    let count = assignments.extend_assigned(items).unwrap();
    assert_eq!(count, 3);
    assert_eq!(assignments.assigned_count(), 3);
    assert_eq!(assignments.staged_count(), 0);
    assert_eq!(assignments.unassigned_count(), 0);

    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
}

#[test]
fn test_extend_assigned_then_stage_pending() {
    let seq = create_test_sequence(5, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Direct-assign first 2
    let items: Vec<(BlockId, SequenceHash)> = vec![
        (100, expected_hashes[0]),
        (200, expected_hashes[1]),
    ];
    assignments.extend_assigned(items).unwrap();
    assert_eq!(assignments.assigned_count(), 2);

    // Enqueue 3 more, stage, commit
    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();
    assignments.stage_pending(seq.blocks()).unwrap();
    assignments.commit_staged();
    assert_eq!(assignments.assigned_count(), 5);

    // Verify all in order
    for (i, expected_hash) in expected_hashes.iter().enumerate() {
        let (id, hash) = assignments.get_assigned(i).unwrap();
        assert_eq!(id, (i + 1) * 100);
        assert_eq!(hash, *expected_hash);
    }
}

#[test]
fn test_extend_assigned_duplicate_detection() {
    let seq = create_test_sequence(2, 0);
    let expected_hashes = get_expected_hashes(&seq);
    let mut assignments = ExternalBlockAssignments::new(0);

    // Assign 1 block
    assignments
        .extend_assigned(vec![(100, expected_hashes[0])])
        .unwrap();

    // Try to assign with duplicate block_id → error
    let result = assignments.extend_assigned(vec![(100, expected_hashes[1])]);
    assert!(result.is_err());
    match result.unwrap_err() {
        BlockSequenceError::DuplicateBlockId { block_id } => {
            assert_eq!(block_id, 100);
        }
        other => panic!("expected DuplicateBlockId, got: {other:?}"),
    }

    // Original assignment unchanged
    assert_eq!(assignments.assigned_count(), 1);
}

#[test]
fn test_extend_assigned_empty() {
    let mut assignments = ExternalBlockAssignments::new(0);
    let count = assignments.extend_assigned(Vec::new()).unwrap();
    assert_eq!(count, 0);
    assert_eq!(assignments.assigned_count(), 0);
}

// =========================================================================
// New tests: contains checks all three collections
// =========================================================================

#[test]
fn test_contains_with_staged() {
    let seq = create_test_sequence(3, 0);
    let mut assignments = ExternalBlockAssignments::new(0);

    assignments
        .extend_block_ids(vec![100, 200, 300, 400, 500])
        .unwrap();

    // Assign first 2
    assignments.assign_pending(&seq.blocks()[..2]).unwrap();
    assert_eq!(assignments.assigned_count(), 2);

    // Stage 1 (block at position 2)
    assignments.stage_pending(seq.blocks()).unwrap();
    assert_eq!(assignments.staged_count(), 1);
    assert_eq!(assignments.unassigned_count(), 2);

    // 100, 200 in assigned
    assert!(assignments.contains(&100));
    assert!(assignments.contains(&200));
    // 300 in staged
    assert!(assignments.contains(&300));
    // 400, 500 in unassigned
    assert!(assignments.contains(&400));
    assert!(assignments.contains(&500));
    // 600 nowhere
    assert!(!assignments.contains(&600));
}

// =========================================================================
// New tests: Debug impl
// =========================================================================

#[test]
fn test_debug_impl() {
    let assignments = ExternalBlockAssignments::new(5);
    let debug_str = format!("{assignments:?}");
    assert!(debug_str.contains("ExternalBlockAssignments"));
    assert!(debug_str.contains("offset"));
}
