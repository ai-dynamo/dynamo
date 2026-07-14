// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;

use super::unified_prompt_tracker::{UnifiedPromptTracker, UnifiedRequestHandle};
use crate::protocols::WorkerWithDpRank;

#[derive(Debug, Default)]
#[must_use = "request block chains must be retained by a request and released through BlockTracker"]
pub(super) struct RequestBlockChain {
    prompt: UnifiedRequestHandle,
    output_hashes: Vec<SequenceHash>,
}

/// Per-worker facade for request-local output state. Prompt ownership lives
/// exclusively in the shared [`UnifiedPromptTracker`].
#[derive(Debug)]
pub(super) struct BlockTracker {
    worker: WorkerWithDpRank,
    prompts: Arc<UnifiedPromptTracker>,
    output_blocks: FxHashMap<SequenceHash, f64>,
}

impl BlockTracker {
    pub(super) fn new(worker: WorkerWithDpRank, prompts: Arc<UnifiedPromptTracker>) -> Self {
        prompts.ensure_worker(worker);
        Self {
            worker,
            prompts,
            output_blocks: FxHashMap::default(),
        }
    }

    pub(super) fn acquire_prompt(&mut self, sequence: &[SequenceHash]) -> RequestBlockChain {
        RequestBlockChain {
            prompt: self.prompts.acquire(self.worker, sequence),
            output_hashes: Vec::new(),
        }
    }

    pub(super) fn append_output(&mut self, chain: &mut RequestBlockChain, hash: SequenceHash) {
        assert!(
            self.output_blocks.insert(hash, 1.0).is_none(),
            "output block hash unexpectedly became live twice"
        );
        chain.output_hashes.push(hash);
    }

    pub(super) fn release(&mut self, chain: RequestBlockChain) {
        for hash in chain.output_hashes {
            self.output_blocks
                .remove(&hash)
                .expect("request output hash is missing from active bookkeeping");
        }
        self.prompts.release(self.worker, chain.prompt);
    }

    pub(super) fn set_unique_suffix_fractional(
        &mut self,
        chain: &RequestBlockChain,
        fraction: f64,
    ) {
        for hash in &chain.output_hashes {
            *self
                .output_blocks
                .get_mut(hash)
                .expect("request output hash is missing from active bookkeeping") = fraction;
        }
        self.prompts
            .set_unique_suffix_fractional(self.worker, &chain.prompt, fraction);
    }

    pub(super) fn active_blocks(&self) -> usize {
        let prompt_blocks = self.prompts.active_block_weight(self.worker);
        (prompt_blocks + self.output_blocks.values().sum::<f64>()).round() as usize
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent<'a>(
        &self,
        chains: impl IntoIterator<Item = &'a RequestBlockChain>,
    ) {
        use rustc_hash::FxHashSet;

        self.prompts.assert_consistent();
        let expected_outputs = chains
            .into_iter()
            .flat_map(|chain| chain.output_hashes.iter().copied())
            .collect::<FxHashSet<_>>();
        assert_eq!(
            expected_outputs,
            self.output_blocks.keys().copied().collect(),
            "output bookkeeping differs from request ownership"
        );
    }

    #[cfg(test)]
    pub(super) fn active_hashes(&self) -> impl Iterator<Item = SequenceHash> {
        let mut hashes = self.prompts.worker_hashes(self.worker);
        hashes.extend(self.output_blocks.keys().copied());
        hashes.into_iter()
    }

    #[cfg(test)]
    pub(super) fn prompt_hashes(
        &self,
        chain: &RequestBlockChain,
    ) -> impl Iterator<Item = SequenceHash> {
        self.prompts.prompt_hashes(&chain.prompt).into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn facade_delegates_prompt_lifecycle_and_keeps_outputs_local() {
        let prompts = Arc::new(UnifiedPromptTracker::default());
        let mut tracker = BlockTracker::new(worker(1), prompts.clone());
        let mut chain = tracker.acquire_prompt(&[1, 2, 3]);
        tracker.append_output(&mut chain, 42);

        assert_eq!(prompts.active_blocks(worker(1)), 3);
        assert_eq!(tracker.active_blocks(), 4);
        tracker.assert_consistent([&chain]);
        tracker.release(chain);
        assert_eq!(tracker.active_blocks(), 0);
        assert!(prompts.is_empty());
    }
}
