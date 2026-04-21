// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub(super) struct BlockAcquire {
    pub(super) rc: Arc<()>,
    pub(super) became_present_on_worker: bool,
}

#[derive(Debug, Default)]
pub(super) struct BlockTracker {
    pub(super) unique_blocks: FxHashMap<SequenceHash, Weak<()>>,
    pub(super) fractional_blocks: FxHashMap<SequenceHash, f64>,
}

impl BlockTracker {
    pub(super) fn touch_block(&mut self, block: &SequenceHash) -> BlockAcquire {
        if let Some(weak) = self.unique_blocks.get(block)
            && let Some(rc) = weak.upgrade()
        {
            return BlockAcquire {
                rc,
                became_present_on_worker: false,
            };
        }

        let rc = Arc::new(());
        self.unique_blocks.insert(*block, Arc::downgrade(&rc));
        BlockAcquire {
            rc,
            became_present_on_worker: true,
        }
    }

    pub(super) fn try_remove_block(&mut self, block: &SequenceHash) -> bool {
        if let Some(weak) = self.unique_blocks.get(block)
            && weak.strong_count() == 0
        {
            self.unique_blocks.remove(block);
            self.fractional_blocks.remove(block);
            return true;
        }

        false
    }

    pub(super) fn active_blocks(&self) -> usize {
        let mut count = self.unique_blocks.len() as f64;
        for (hash, frac) in &self.fractional_blocks {
            if self.unique_blocks.contains_key(hash) {
                count = count - 1.0 + frac;
            }
        }
        count.round() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_touch_and_last_remove_report_presence_transitions() {
        let mut tracker = BlockTracker::default();

        let first = tracker.touch_block(&1);
        let second = tracker.touch_block(&1);

        assert!(first.became_present_on_worker);
        assert!(!second.became_present_on_worker);
        assert_eq!(tracker.active_blocks(), 1);

        drop(first.rc);
        assert!(!tracker.try_remove_block(&1));
        assert_eq!(tracker.active_blocks(), 1);

        drop(second.rc);
        assert!(tracker.try_remove_block(&1));
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn fractional_blocks_adjust_active_block_count() {
        let mut tracker = BlockTracker::default();
        let first = tracker.touch_block(&1);
        let second = tracker.touch_block(&2);

        tracker.fractional_blocks.insert(1, 0.5);
        tracker.fractional_blocks.insert(2, 0.5);
        assert_eq!(tracker.active_blocks(), 1);

        drop(first.rc);
        assert!(tracker.try_remove_block(&1));
        assert!(!tracker.fractional_blocks.contains_key(&1));
        assert_eq!(tracker.active_blocks(), 1);

        drop(second.rc);
        assert!(tracker.try_remove_block(&2));
        assert!(tracker.fractional_blocks.is_empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn shared_block_counts_once_until_last_reference_drops() {
        let mut tracker = BlockTracker::default();
        let first = tracker.touch_block(&7);
        let second = tracker.touch_block(&7);
        let third = tracker.touch_block(&7);

        assert_eq!(tracker.active_blocks(), 1);

        drop(first.rc);
        drop(second.rc);
        assert!(!tracker.try_remove_block(&7));
        assert_eq!(tracker.active_blocks(), 1);

        drop(third.rc);
        assert!(tracker.try_remove_block(&7));
        assert_eq!(tracker.active_blocks(), 0);
    }
}
