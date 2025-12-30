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
        self.requests.values().map(|r| r.total_tokens()).sum()
    }

    /// Get request IDs of all running requests.
    pub fn request_ids(&self) -> impl Iterator<Item = &String> {
        self.requests.keys()
    }
}


