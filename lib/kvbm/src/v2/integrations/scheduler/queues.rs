// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request queues for the scheduler.

use super::request::{RequestStatus, SchedulerRequest};
use std::collections::{HashMap, VecDeque};

/// Queue of requests waiting to be scheduled.
///
/// Requests are stored in FIFO order by default. Preempted requests
/// are added to the front to be rescheduled first.
#[derive(Debug, Default)]
pub struct WaitingQueue {
    /// Requests waiting to be scheduled, in priority order.
    requests: VecDeque<SchedulerRequest>,
}

impl WaitingQueue {
    /// Create a new empty waiting queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new request to the back of the queue.
    pub fn push_back(&mut self, request: SchedulerRequest) {
        debug_assert!(request.status.can_schedule());
        self.requests.push_back(request);
    }

    /// Add a preempted request to the front of the queue (priority).
    pub fn push_front(&mut self, request: SchedulerRequest) {
        debug_assert!(request.status.can_schedule());
        self.requests.push_front(request);
    }

    /// Pop a request from the front of the queue.
    pub fn pop_front(&mut self) -> Option<SchedulerRequest> {
        self.requests.pop_front()
    }

    /// Peek at the front request without removing it.
    pub fn peek(&self) -> Option<&SchedulerRequest> {
        self.requests.front()
    }

    /// Get the number of waiting requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Iterate over waiting requests.
    pub fn iter(&self) -> impl Iterator<Item = &SchedulerRequest> {
        self.requests.iter()
    }

    /// Iterate over waiting requests mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SchedulerRequest> {
        self.requests.iter_mut()
    }

    /// Drain all requests from the queue.
    pub fn drain(&mut self) -> impl Iterator<Item = SchedulerRequest> + '_ {
        self.requests.drain(..)
    }

    /// Remove a request by ID.
    pub fn remove(&mut self, request_id: &str) -> Option<SchedulerRequest> {
        let pos = self
            .requests
            .iter()
            .position(|r| r.request_id() == request_id)?;
        self.requests.remove(pos)
    }
}

/// Map of currently running requests.
#[derive(Debug, Default)]
pub struct RunningRequests {
    /// Requests currently running, keyed by request ID.
    requests: HashMap<String, SchedulerRequest>,
}

impl RunningRequests {
    /// Create a new empty running requests map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a request to the running set.
    pub fn insert(&mut self, mut request: SchedulerRequest) {
        request.status = RequestStatus::Running;
        self.requests
            .insert(request.request_id().to_string(), request);
    }

    /// Remove a request from the running set.
    pub fn remove(&mut self, request_id: &str) -> Option<SchedulerRequest> {
        self.requests.remove(request_id)
    }

    /// Get a reference to a running request.
    pub fn get(&self, request_id: &str) -> Option<&SchedulerRequest> {
        self.requests.get(request_id)
    }

    /// Get a mutable reference to a running request.
    pub fn get_mut(&mut self, request_id: &str) -> Option<&mut SchedulerRequest> {
        self.requests.get_mut(request_id)
    }

    /// Check if a request is running.
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get the number of running requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if there are no running requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Iterate over running requests.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &SchedulerRequest)> {
        self.requests.iter()
    }

    /// Iterate over running requests mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut SchedulerRequest)> {
        self.requests.iter_mut()
    }

    /// Drain all running requests.
    pub fn drain(&mut self) -> impl Iterator<Item = (String, SchedulerRequest)> + '_ {
        self.requests.drain()
    }

    /// Get the total number of tokens scheduled for running requests.
    pub fn total_tokens(&self) -> usize {
        self.requests.values().map(|r| r.total_known_tokens()).sum()
    }

    /// Get request IDs of all running requests.
    pub fn request_ids(&self) -> impl Iterator<Item = &String> {
        self.requests.keys()
    }
}

/// Collection of paused requests that hold blocks but are not scheduled.
///
/// Paused requests can progressively release blocks that are already in G2
/// (or that we're willing to recompute) to other requests, then reclaim them
/// when resuming.
///
/// # Resume Order
///
/// Requests are resumed in LIFO order (last paused, first resumed) because:
/// - Recently paused requests are likely to have more blocks still in G1
/// - They can resume with less onboarding overhead
/// - This naturally load-balances the pause pool
#[derive(Debug, Default)]
pub struct PausedRequests {
    /// Paused requests, keyed by request ID.
    requests: HashMap<String, SchedulerRequest>,

    /// Order in which requests were paused (for LIFO resume).
    pause_order: VecDeque<String>,

    /// Blocks that have been lent from each paused request.
    /// When a request resumes, it must reclaim these blocks.
    lent_blocks: HashMap<String, Vec<crate::v2::BlockId>>,
}

impl PausedRequests {
    /// Create a new empty paused requests collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pause a request, moving it from running to paused.
    ///
    /// The request keeps its blocks but is no longer scheduled.
    pub fn pause(&mut self, mut request: SchedulerRequest) {
        debug_assert!(request.status.can_pause());
        request.pause();
        let request_id = request.request_id().to_string();
        self.pause_order.push_back(request_id.clone());
        self.requests.insert(request_id, request);
    }

    /// Get the next request to resume (LIFO order).
    ///
    /// Returns the most recently paused request that can resume.
    /// The caller is responsible for ensuring blocks can be reclaimed.
    pub fn resume_next(&mut self) -> Option<SchedulerRequest> {
        while let Some(request_id) = self.pause_order.pop_back() {
            if let Some(mut request) = self.requests.remove(&request_id) {
                // Clear lent blocks tracking (should have been reclaimed already)
                self.lent_blocks.remove(&request_id);
                request.resume_from_pause();
                return Some(request);
            }
        }
        None
    }

    /// Resume a specific request by ID.
    pub fn resume_by_id(&mut self, request_id: &str) -> Option<SchedulerRequest> {
        if let Some(mut request) = self.requests.remove(request_id) {
            // Remove from pause order
            self.pause_order.retain(|id| id != request_id);
            // Clear lent blocks tracking
            self.lent_blocks.remove(request_id);
            request.resume_from_pause();
            return Some(request);
        }
        None
    }

    /// Get a reference to a paused request.
    pub fn get(&self, request_id: &str) -> Option<&SchedulerRequest> {
        self.requests.get(request_id)
    }

    /// Get a mutable reference to a paused request.
    pub fn get_mut(&mut self, request_id: &str) -> Option<&mut SchedulerRequest> {
        self.requests.get_mut(request_id)
    }

    /// Check if a request is paused.
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Record that blocks have been lent from a paused request.
    ///
    /// These blocks must be reclaimed before the request can resume.
    pub fn record_lent_blocks(&mut self, request_id: &str, block_ids: Vec<crate::v2::BlockId>) {
        self.lent_blocks
            .entry(request_id.to_string())
            .or_default()
            .extend(block_ids);
    }

    /// Get the blocks that have been lent from a request.
    pub fn get_lent_blocks(&self, request_id: &str) -> &[crate::v2::BlockId] {
        self.lent_blocks
            .get(request_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the number of blocks lent from a request.
    pub fn num_lent_blocks(&self, request_id: &str) -> usize {
        self.lent_blocks.get(request_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Remove a request from the paused collection (for eviction).
    ///
    /// Unlike `resume_next`, this doesn't transition the request to Running.
    /// The caller is responsible for handling the request's state.
    pub fn remove(&mut self, request_id: &str) -> Option<SchedulerRequest> {
        self.pause_order.retain(|id| id != request_id);
        self.lent_blocks.remove(request_id);
        self.requests.remove(request_id)
    }

    /// Get the number of paused requests.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if there are no paused requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Iterate over paused requests.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &SchedulerRequest)> {
        self.requests.iter()
    }

    /// Iterate over paused requests mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut SchedulerRequest)> {
        self.requests.iter_mut()
    }

    /// Get the total number of blocks held by paused requests.
    pub fn total_held_blocks(&self) -> usize {
        self.requests
            .values()
            .map(|r| r.block_state.total_blocks())
            .sum()
    }

    /// Get the total number of blocks currently lent to other requests.
    pub fn total_lent_blocks(&self) -> usize {
        self.lent_blocks.values().map(|v| v.len()).sum()
    }

    /// Get request IDs in pause order (oldest first).
    pub fn request_ids_by_pause_order(&self) -> impl Iterator<Item = &String> {
        self.pause_order.iter()
    }
}
