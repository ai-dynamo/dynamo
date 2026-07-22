// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::protocols::{WorkerId, WorkerWithDpRank};

use super::error::SelectionError;
use super::types::{
    SelectionKey, SelectionWorkerConfig, WorkerCatalogRecord, WorkerLifecycle, WorkerPatchRequest,
    WorkerRequest,
};

#[derive(Debug, Default)]
pub(super) struct WorkerCatalog {
    workers: RwLock<HashMap<WorkerId, WorkerCatalogRecord>>,
}

impl WorkerCatalog {
    pub(super) fn upsert(
        &self,
        req: WorkerRequest,
    ) -> (Option<WorkerCatalogRecord>, WorkerCatalogRecord) {
        let mut workers = self.workers.write();
        let previous = workers.get(&req.worker_id).cloned();
        let record = WorkerCatalogRecord::new(req);
        workers.insert(record.worker_id, record.clone());
        (previous, record)
    }

    pub(super) fn patch(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<(WorkerCatalogRecord, WorkerCatalogRecord), SelectionError> {
        let mut workers = self.workers.write();
        let Some(record) = workers.get_mut(&worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let previous = record.clone();
        record.apply_patch(patch);
        record.lifecycle = WorkerLifecycle::Incomplete;
        record.not_schedulable_reasons.clear();
        Ok((previous, record.clone()))
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

    pub(super) fn has_schedulable_for_key(&self, key: &SelectionKey) -> bool {
        self.workers.read().values().any(|record| {
            record.lifecycle == WorkerLifecycle::Schedulable
                && record.model_name == key.model_name
                && record.routing_group == key.routing_group
        })
    }

    pub(super) fn scheduler_configs_for_key(
        &self,
        key: &SelectionKey,
    ) -> HashMap<WorkerId, SelectionWorkerConfig> {
        self.workers
            .read()
            .values()
            .filter(|record| {
                record.lifecycle == WorkerLifecycle::Schedulable
                    && record.model_name == key.model_name
                    && record.routing_group == key.routing_group
            })
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

    pub(super) fn schedulable_endpoint(
        &self,
        worker_id: WorkerId,
        key: &SelectionKey,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker_id)?;
        if record.lifecycle != WorkerLifecycle::Schedulable
            || record.model_name != key.model_name
            || record.routing_group != key.routing_group
        {
            return None;
        }
        record.endpoint.clone()
    }

    pub(super) fn schedulable_worker_endpoint(
        &self,
        worker: WorkerWithDpRank,
        key: &SelectionKey,
    ) -> Option<String> {
        let workers = self.workers.read();
        let record = workers.get(&worker.worker_id)?;
        if record.lifecycle != WorkerLifecycle::Schedulable
            || record.model_name != key.model_name
            || record.routing_group != key.routing_group
            || !record.dp_ranks().any(|rank| rank == worker.dp_rank)
        {
            return None;
        }
        record.endpoint.clone()
    }
}
