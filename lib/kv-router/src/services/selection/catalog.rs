// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::identity::RoutingPartitionId;
use crate::protocols::{WorkerId, WorkerWithDpRank};

use super::types::{SelectionWorkerConfig, WorkerCatalogRecord, WorkerLifecycle};

#[derive(Debug, Default)]
pub(super) struct WorkerCatalog {
    workers: RwLock<HashMap<WorkerId, WorkerCatalogRecord>>,
}

impl WorkerCatalog {
    pub(super) fn replace(&self, record: WorkerCatalogRecord) {
        self.workers.write().insert(record.worker_id, record);
    }

    pub(super) fn get(&self, worker_id: WorkerId) -> Option<WorkerCatalogRecord> {
        self.workers.read().get(&worker_id).cloned()
    }

    pub(super) fn set_lifecycle(
        &self,
        worker_id: WorkerId,
        lifecycle: WorkerLifecycle,
        reasons: Vec<String>,
    ) -> Option<WorkerCatalogRecord> {
        let mut workers = self.workers.write();
        let record = workers.get_mut(&worker_id)?;
        record.lifecycle = lifecycle;
        record.not_schedulable_reasons = reasons;
        Some(record.clone())
    }

    pub(super) fn list(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        let mut records: Vec<_> = self
            .workers
            .read()
            .values()
            .filter(|record| {
                model_name.is_none_or(|model_name| record.model_name == model_name)
                    && routing_group
                        .is_none_or(|routing_group| record.routing_group == routing_group)
            })
            .cloned()
            .collect();
        records.sort_by_key(|record| {
            (
                record.model_name.clone(),
                record.routing_group.clone(),
                record.worker_id,
            )
        });
        records
    }

    pub(super) fn has_schedulable_for_key(&self, key: &RoutingPartitionId) -> bool {
        self.workers
            .read()
            .values()
            .any(|record| schedulable_in(record, key))
    }

    /// `total_kv_blocks` published by a schedulable worker in `key`'s partition.
    pub(super) fn total_kv_blocks(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Option<u64> {
        let workers = self.workers.read();
        let record = workers.get(&worker_id)?;
        (record.lifecycle == WorkerLifecycle::Schedulable
            && record.model_name == key.model_name
            && record.routing_group == key.routing_group)
            .then_some(record.total_kv_blocks)
            .flatten()
    }

    pub(super) fn remove(&self, worker_id: WorkerId) -> Option<WorkerCatalogRecord> {
        self.workers.write().remove(&worker_id)
    }

    /// The schedulable workers in `key`'s partition, keyed by worker id.
    pub(super) fn scheduler_configs_for_key(
        &self,
        key: &RoutingPartitionId,
    ) -> HashMap<WorkerId, SelectionWorkerConfig> {
        self.workers
            .read()
            .values()
            .filter(|record| schedulable_in(record, key))
            .filter_map(|record| {
                record
                    .scheduler_config()
                    .map(|config| (record.worker_id, config))
            })
            .collect()
    }

    pub(super) fn schedulable_count(&self) -> usize {
        self.workers
            .read()
            .values()
            .filter(|record| record.lifecycle == WorkerLifecycle::Schedulable)
            .count()
    }

    pub(super) fn is_schedulable(&self, worker_id: WorkerId, key: &RoutingPartitionId) -> bool {
        self.workers
            .read()
            .get(&worker_id)
            .is_some_and(|record| schedulable_in(record, key))
    }

    pub(super) fn schedulable_endpoint(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker_id)?;
        if !schedulable_in(record, key) {
            return None;
        }
        record.endpoint.clone()
    }

    pub(super) fn schedulable_worker_endpoint(
        &self,
        worker: WorkerWithDpRank,
        key: &RoutingPartitionId,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker.worker_id)?;
        if !schedulable_in(record, key) || !record.dp_ranks().any(|rank| rank == worker.dp_rank) {
            return None;
        }
        record.endpoint.clone()
    }
}

fn schedulable_in(record: &WorkerCatalogRecord, key: &RoutingPartitionId) -> bool {
    record.lifecycle == WorkerLifecycle::Schedulable
        && record.model_name == key.model_name
        && record.routing_group == key.routing_group
}
