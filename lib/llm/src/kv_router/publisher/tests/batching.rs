// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(test)]
mod batching_state_tests {
    use super::*;

    #[test]
    fn test_batching_state_default() {
        let state = BatchingState::new();
        assert!(!state.has_pending(), "Default state should have no pending");
        assert!(
            state.pending_removed.is_none(),
            "Default pending_removed should be None"
        );
        assert!(
            state.pending_stored.is_none(),
            "Default pending_stored should be None"
        );
    }

    #[test]
    fn test_batching_state_new() {
        let state = BatchingState::new();
        // last_flush_time should be set to approximately now
        let elapsed = state.last_flush_time.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "new() should create state with flush time set to approximately now"
        );
    }

    #[test]
    fn test_batching_state_pending_removed() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_removed = Some(KvCacheRemoveData {
            block_hashes: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_removed"
        );
    }

    #[test]
    fn test_batching_state_pending_stored() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_stored = Some(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_stored"
        );
    }

    #[test]
    fn test_batching_state_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining_before = state.remaining_timeout(10);
        assert!(
            remaining_before.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test zero timeout returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero.as_millis(),
            0,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_record_flush_time() {
        let mut state = BatchingState::new();

        let initial_time = state.last_flush_time;

        state.record_flush_time();

        assert!(
            state.last_flush_time >= initial_time,
            "record_flush_time should update the time"
        );
    }

    #[test]
    fn test_batching_state_remaining_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining = state.remaining_timeout(10);
        assert!(
            remaining.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test that with 0 timeout, returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero,
            Duration::ZERO,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_accumulate_removed() {
        let mut state = BatchingState::new();

        let first = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
        };

        state.pending_removed = Some(first);

        if let Some(ref mut pending) = state.pending_removed {
            pending
                .block_hashes
                .extend(vec![ExternalSequenceBlockHash(3)]);
        }

        let pending = state.pending_removed.as_ref().unwrap();
        assert_eq!(
            pending.block_hashes.len(),
            3,
            "Should have accumulated 3 block hashes"
        );
    }

    #[test]
    fn test_batching_state_accumulate_stored() {
        let mut state = BatchingState::new();

        let block1 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(1),
            tokens_hash: LocalBlockHash(100),
            mm_extra_info: None,
        };
        let first = KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(0)),
            start_position: None,
            blocks: vec![block1],
        };

        state.pending_stored = Some(first);

        let block2 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(2),
            tokens_hash: LocalBlockHash(200),
            mm_extra_info: None,
        };

        if let Some(ref mut pending) = state.pending_stored {
            pending.blocks.extend(vec![block2]);
        }

        let pending = state.pending_stored.as_ref().unwrap();
        assert_eq!(pending.blocks.len(), 2, "Should have accumulated 2 blocks");
    }
}
