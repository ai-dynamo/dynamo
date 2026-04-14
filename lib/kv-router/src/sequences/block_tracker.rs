// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub(super) struct BlockAcquire {
    pub(super) rc: Arc<()>,
    pub(super) became_present_on_worker: bool,
}

#[derive(Debug, Default)]
pub(super) struct BlockTracker {
    pub(super) unique_blocks: HashMap<SequenceHash, Weak<()>>,
    pub(super) fractional_blocks: HashMap<SequenceHash, f64>,
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
