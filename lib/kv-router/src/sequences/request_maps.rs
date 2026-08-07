// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::{DashMap, mapref::entry::Entry};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tokio::time::Instant;

use super::single::RequestId;
use crate::protocols::{SCHEDULER_EXPIRATION_TIMEOUT, WorkerWithDpRank};

const SCHEDULER_TOMBSTONE_RETENTION: Duration = Duration::from_secs(600);

#[derive(Debug)]
struct SchedulerRequests {
    deadline: Instant,
    requests: HashSet<RequestId>,
    expired: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SchedulerEvent {
    Untracked,
    Track,
    Reject,
}

impl SchedulerRequests {
    fn new(now: Instant) -> Self {
        Self {
            deadline: now + SCHEDULER_EXPIRATION_TIMEOUT,
            requests: HashSet::new(),
            expired: false,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct RequestIndex {
    request_to_worker: DashMap<RequestId, WorkerWithDpRank>,
    request_to_lora: DashMap<RequestId, String>,
    request_to_scheduler: DashMap<RequestId, u64>,
    scheduler_requests: DashMap<u64, SchedulerRequests>,
}

impl RequestIndex {
    pub(super) fn try_insert_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) -> Result<(), WorkerWithDpRank> {
        match self.request_to_worker.entry(request_id.clone()) {
            Entry::Occupied(entry) => Err(*entry.get()),
            Entry::Vacant(entry) => {
                entry.insert(worker);
                if let Some(lora_name) = lora_name {
                    self.request_to_lora.insert(request_id, lora_name);
                }
                Ok(())
            }
        }
    }

    #[cfg(test)]
    pub(super) fn set_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) {
        self.set_request_metadata(request_id, worker, lora_name);
    }

    pub(super) fn track_scheduler_request<T>(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
        scheduler_id: u64,
        apply: impl FnOnce() -> T,
    ) -> Option<T> {
        let mut scheduler = self.scheduler_requests.get_mut(&scheduler_id)?;
        if scheduler.expired {
            return None;
        }
        scheduler.requests.insert(request_id.clone());
        self.request_to_scheduler
            .insert(request_id.clone(), scheduler_id);
        self.request_to_worker.insert(request_id.clone(), worker);
        self.set_lora(request_id, lora_name);
        Some(apply())
    }

    pub(super) fn set_request_metadata(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) {
        self.remove_scheduler_request(&request_id);
        self.request_to_worker.insert(request_id.clone(), worker);
        self.set_lora(request_id, lora_name);
    }

    fn set_lora(&self, request_id: RequestId, lora_name: Option<String>) {
        if let Some(lora_name) = lora_name {
            self.request_to_lora.insert(request_id.clone(), lora_name);
        } else {
            self.request_to_lora.remove(&request_id);
        }
    }

    pub(super) fn worker_for(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        self.request_to_worker.get(request_id).map(|entry| *entry)
    }

    pub(super) fn lora_for(&self, request_id: &RequestId) -> Option<String> {
        self.request_to_lora
            .get(request_id)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn remove_request(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        let worker = self
            .request_to_worker
            .remove(request_id)
            .map(|(_request_id, worker)| worker);
        self.request_to_lora.remove(request_id);
        self.remove_scheduler_request(request_id);
        worker
    }

    fn remove_scheduler_request(&self, request_id: &RequestId) {
        if let Some((_request_id, scheduler_id)) = self.request_to_scheduler.remove(request_id)
            && let Some(mut scheduler) = self.scheduler_requests.get_mut(&scheduler_id)
        {
            scheduler.requests.remove(request_id);
        }
    }

    pub(super) fn scheduler_heartbeat(&self, scheduler_id: u64, now: Instant) {
        let mut scheduler = self
            .scheduler_requests
            .entry(scheduler_id)
            .or_insert_with(|| SchedulerRequests::new(now));
        scheduler.expired = false;
        scheduler.deadline = now + SCHEDULER_EXPIRATION_TIMEOUT;
    }

    pub(super) fn scheduler_event(&self, scheduler_id: u64, now: Instant) -> SchedulerEvent {
        let Some(mut scheduler) = self.scheduler_requests.get_mut(&scheduler_id) else {
            return SchedulerEvent::Untracked;
        };
        if scheduler.expired {
            return SchedulerEvent::Reject;
        }
        scheduler.deadline = now + SCHEDULER_EXPIRATION_TIMEOUT;
        SchedulerEvent::Track
    }

    pub(super) fn expire_schedulers(&self, now: Instant) -> Vec<(u64, Vec<RequestId>)> {
        let mut expired = Vec::new();
        for mut scheduler in self.scheduler_requests.iter_mut() {
            if !scheduler.expired && scheduler.deadline <= now {
                scheduler.expired = true;
                scheduler.deadline = now + SCHEDULER_TOMBSTONE_RETENTION;
                expired.push((*scheduler.key(), scheduler.requests.drain().collect()));
            }
        }
        expired
    }

    pub(super) fn prune_scheduler_tombstones(&self, now: Instant) {
        self.scheduler_requests
            .retain(|_, scheduler| !scheduler.expired || scheduler.deadline > now);
    }

    pub(super) fn remove_requests<'a>(&self, request_ids: impl IntoIterator<Item = &'a RequestId>) {
        for request_id in request_ids {
            self.remove_request(request_id);
        }
    }

    pub(super) fn remove_worker_requests(&self, worker: WorkerWithDpRank) -> Vec<RequestId> {
        let request_ids: Vec<_> = self
            .request_to_worker
            .iter()
            .filter(|entry| *entry.value() == worker)
            .map(|entry| entry.key().clone())
            .collect();
        self.remove_requests(request_ids.iter());
        request_ids
    }

    pub(super) fn active_lora_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.request_to_lora.iter() {
            let lora_name = entry.value().clone();
            *counts.entry(lora_name).or_insert(0) += 1;
        }
        counts
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        self.request_to_worker.is_empty()
            && self.request_to_lora.is_empty()
            && self.request_to_scheduler.is_empty()
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn worker_len(&self) -> usize {
        self.request_to_worker.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_insert_returns_existing_worker() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);

        index
            .try_insert_request("req-1".to_string(), worker, Some("adapter".to_string()))
            .unwrap();
        assert_eq!(
            index.try_insert_request("req-1".to_string(), WorkerWithDpRank::new(2, 0), None),
            Err(worker)
        );
        assert_eq!(index.worker_for(&"req-1".to_string()), Some(worker));
        assert_eq!(
            index.lora_for(&"req-1".to_string()),
            Some("adapter".to_string())
        );
    }

    #[test]
    fn remove_request_is_idempotent() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);
        let request_id = "req-1".to_string();

        index.set_request(request_id.clone(), worker, Some("adapter".to_string()));
        assert_eq!(index.remove_request(&request_id), Some(worker));
        assert_eq!(index.remove_request(&request_id), None);
        assert!(index.is_empty());
    }

    #[test]
    fn set_request_without_lora_clears_stale_lora_mapping() {
        let index = RequestIndex::default();
        let request_id = "req-1".to_string();

        index.set_request(
            request_id.clone(),
            WorkerWithDpRank::new(1, 0),
            Some("adapter".to_string()),
        );
        index.set_request(request_id.clone(), WorkerWithDpRank::new(2, 0), None);

        assert_eq!(
            index.worker_for(&request_id),
            Some(WorkerWithDpRank::new(2, 0))
        );
        assert_eq!(index.lora_for(&request_id), None);
    }

    #[test]
    fn remove_worker_requests_clears_both_maps() {
        let index = RequestIndex::default();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        index.set_request("req-a".to_string(), worker_a, Some("adapter-a".to_string()));
        index.set_request("req-b".to_string(), worker_b, Some("adapter-b".to_string()));
        index.set_request("req-c".to_string(), worker_a, None);

        let mut removed = index.remove_worker_requests(worker_a);
        removed.sort();
        assert_eq!(removed, vec!["req-a".to_string(), "req-c".to_string()]);
        assert_eq!(index.worker_for(&"req-b".to_string()), Some(worker_b));
        assert_eq!(
            index.active_lora_counts(),
            HashMap::from([("adapter-b".to_string(), 1)])
        );
    }
}
