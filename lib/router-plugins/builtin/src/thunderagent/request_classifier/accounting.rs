// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical prefix ownership, scoped to a worker/rank and the router's hash domain.
//! Hash equality is not evidence of physical KV residency. Private tails and future
//! generation remain separately charged; missing hashes receive no discount.

use std::collections::HashMap;

use dynamo_kv_router::protocols::WorkerWithDpRank;

#[derive(Clone, Default)]
pub(super) struct WorkerPrefixes {
    references: HashMap<(usize, u64), usize>,
    duplicate_blocks: usize,
}

impl WorkerPrefixes {
    pub(super) fn add(&mut self, hashes: &[u64]) {
        for (position, hash) in hashes.iter().enumerate() {
            let count = self.references.entry((position, *hash)).or_default();
            self.duplicate_blocks += usize::from(*count > 0);
            *count += 1;
        }
    }

    pub(super) fn remove(&mut self, hashes: &[u64]) {
        for (position, hash) in hashes.iter().enumerate() {
            let key = (position, *hash);
            let count = self
                .references
                .get_mut(&key)
                .expect("registered prefix owner");
            self.duplicate_blocks -= usize::from(*count > 1);
            *count -= 1;
            if *count == 0 {
                self.references.remove(&key);
            }
        }
    }

    pub(super) fn savings(&self, block_size: usize) -> usize {
        self.duplicate_blocks.saturating_mul(block_size)
    }

    pub(super) fn incremental_cost(
        &self,
        hashes: &[u64],
        full_cost: usize,
        block_size: usize,
    ) -> usize {
        let reused = hashes
            .iter()
            .enumerate()
            .filter(|(position, hash)| self.references.contains_key(&(*position, **hash)))
            .count();
        full_cost.saturating_sub(reused.saturating_mul(block_size))
    }
}

#[derive(Default)]
pub(super) struct PrefixAccounting {
    workers: HashMap<WorkerWithDpRank, WorkerPrefixes>,
    pub(super) block_size: usize,
}

impl PrefixAccounting {
    pub(super) fn add(&mut self, worker: WorkerWithDpRank, hashes: &[u64]) {
        if !hashes.is_empty() {
            self.workers.entry(worker).or_default().add(hashes);
        }
    }

    pub(super) fn remove(&mut self, worker: WorkerWithDpRank, hashes: &[u64]) {
        if hashes.is_empty() {
            return;
        }
        let prefixes = self
            .workers
            .get_mut(&worker)
            .expect("registered worker prefixes");
        prefixes.remove(hashes);
        if prefixes.references.is_empty() {
            self.workers.remove(&worker);
        }
    }

    pub(super) fn savings(&self, worker: WorkerWithDpRank) -> usize {
        self.workers
            .get(&worker)
            .map_or(0, |p| p.savings(self.block_size))
    }

    pub(super) fn incremental_cost(
        &self,
        worker: WorkerWithDpRank,
        hashes: &[u64],
        full_cost: usize,
    ) -> usize {
        self.workers.get(&worker).map_or(full_cost, |p| {
            p.incremental_cost(hashes, full_cost, self.block_size)
        })
    }

    pub(super) fn snapshot(&self, worker: WorkerWithDpRank) -> WorkerPrefixes {
        self.workers.get(&worker).cloned().unwrap_or_default()
    }
}
