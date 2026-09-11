// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_runtime::protocols::EndpointId;

use crate::local_model::runtime_config::ModelRuntimeConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkerGroupState {
    Ready,
    Pending,
    MaterializationFailed,
    CommitBlocked,
    Removed,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkerGroupObservation {
    pub(crate) model: String,
    pub(crate) endpoint: EndpointId,
    pub(crate) worker_type: &'static str,
    pub(crate) workers: HashMap<u64, ModelRuntimeConfig>,
    pub(crate) committed: HashSet<u64>,
    pub(crate) checksum_mismatches: HashSet<u64>,
    pub(crate) state: WorkerGroupState,
}

/// Discovery inventory survives catalog withdrawal so a rejected or empty group remains observable.
/// Empty group descriptors survive removal so counts can explicitly report zero workers.
#[derive(Default)]
pub(crate) struct WorkerInventory {
    groups: parking_lot::Mutex<HashMap<String, WorkerGroupObservation>>,
}

impl WorkerInventory {
    pub(crate) fn publish(&self, group_id: String, observation: Option<WorkerGroupObservation>) {
        let mut groups = self.groups.lock();
        let next = match observation {
            Some(observation) => observation,
            None => {
                let Some(previous) = groups.get(&group_id) else {
                    return;
                };
                WorkerGroupObservation {
                    model: previous.model.clone(),
                    endpoint: previous.endpoint.clone(),
                    worker_type: previous.worker_type,
                    workers: HashMap::new(),
                    committed: HashSet::new(),
                    checksum_mismatches: HashSet::new(),
                    state: WorkerGroupState::Removed,
                }
            }
        };
        groups.insert(group_id, next);
    }

    pub(crate) fn snapshot(&self) -> Vec<(String, WorkerGroupObservation)> {
        self.groups
            .lock()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}
