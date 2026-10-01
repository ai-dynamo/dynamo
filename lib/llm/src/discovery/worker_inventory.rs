// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_runtime::protocols::EndpointId;

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_type::ModelType;
use crate::protocols::common::timing::WORKER_TYPE_DECODE;
use crate::worker_type::WorkerType;

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
    pub(crate) model_type: ModelType,
    pub(crate) workers: HashMap<u64, ModelRuntimeConfig>,
    pub(crate) committed: HashSet<u64>,
    pub(crate) checksum_mismatches: HashSet<u64>,
    pub(crate) state: WorkerGroupState,
}

impl WorkerGroupObservation {
    pub(crate) fn timing_worker_type(&self) -> &'static str {
        // Encode workers can serve the public token-generation path, whose timing
        // attribution is decode even though its topology role remains encode.
        if self.worker_type == WorkerType::Encode.as_str()
            && self
                .model_type
                .intersects(ModelType::Chat | ModelType::Completions)
        {
            WORKER_TYPE_DECODE
        } else {
            self.worker_type
        }
    }
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
                    model_type: previous.model_type,
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
