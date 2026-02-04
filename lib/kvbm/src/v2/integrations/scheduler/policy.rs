// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduling policies for request prioritization.

use super::request::SchedulerRequest;

/// Trait for scheduling policies.
///
/// A scheduling policy determines which request to schedule next from the
/// waiting queue and which request to preempt when memory pressure occurs.
pub trait SchedulingPolicy: Send + Sync {
    /// Select the next request to schedule from the waiting queue.
    ///
    /// Returns the index of the request to schedule, or None if no request
    /// should be scheduled (e.g., due to resource constraints).
    ///
    /// # Arguments
    /// * `waiting` - Slice of waiting requests to choose from
    /// * `num_running` - Current number of running requests
    /// * `available_blocks` - Number of free blocks available
    /// * `block_size` - Size of each block in tokens
    fn select_next(
        &self,
        waiting: &[&SchedulerRequest],
        num_running: usize,
        available_blocks: usize,
        block_size: usize,
    ) -> Option<usize>;

    /// Select a request to preempt when memory pressure occurs.
    ///
    /// Returns the request ID of the request to preempt, or None if no
    /// preemption should occur.
    ///
    /// # Arguments
    /// * `running` - Slice of running requests to choose from
    /// * `blocks_needed` - Number of blocks needed to relieve memory pressure
    fn select_victim<'a>(
        &self,
        running: &[&'a SchedulerRequest],
        blocks_needed: usize,
    ) -> Option<&'a str>;
}

/// First-Come-First-Served (FCFS) scheduling policy.
///
/// This is the simplest scheduling policy:
/// - Schedules requests in the order they arrive
/// - Preempts the most recently scheduled request (LIFO) when under memory pressure
#[derive(Debug, Default, Clone)]
pub struct FCFSPolicy {
    /// Maximum number of sequences that can run concurrently.
    pub max_num_seqs: usize,
}

impl FCFSPolicy {
    /// Create a new FCFS policy with the given maximum sequences.
    pub fn new(max_num_seqs: usize) -> Self {
        Self { max_num_seqs }
    }
}

impl SchedulingPolicy for FCFSPolicy {
    fn select_next(
        &self,
        waiting: &[&SchedulerRequest],
        num_running: usize,
        available_blocks: usize,
        block_size: usize,
    ) -> Option<usize> {
        // Check if we've hit the max sequences limit
        if num_running >= self.max_num_seqs {
            return None;
        }

        // FCFS: try to schedule the first request in the queue
        if let Some(request) = waiting.first() {
            // Check if we have enough blocks for at least one token
            let blocks_needed = request.num_new_blocks_needed(block_size);
            if blocks_needed == 0 || available_blocks >= blocks_needed {
                return Some(0);
            }
        }

        None
    }

    fn select_victim<'a>(
        &self,
        running: &[&'a SchedulerRequest],
        _blocks_needed: usize,
    ) -> Option<&'a str> {
        // LIFO: preempt the most recently added request
        // In a simple implementation, we preempt the one with the fewest computed tokens
        // (likely the most recent one)
        running
            .iter()
            .min_by_key(|r| r.num_computed_tokens)
            .map(|r| r.request_id())
    }
}
