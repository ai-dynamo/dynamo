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
    pub(crate) fn metric_worker_type(&self) -> &'static str {
        // Encode workers can serve the public token-generation path, whose load and timing
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

/// Tracks discovered groups independently of successful catalog admission.
#[derive(Default)]
pub(crate) struct WorkerInventory {
    groups: parking_lot::Mutex<HashMap<String, WorkerGroupObservation>>,
}

impl WorkerInventory {
    pub(crate) fn publish(&self, group_id: String, observation: Option<WorkerGroupObservation>) {
        let mut groups = self.groups.lock();
        if let Some(observation) = observation {
            groups.insert(group_id, observation);
        } else {
            groups.remove(&group_id);
        }
    }

    pub(crate) fn snapshot(&self) -> Vec<(String, WorkerGroupObservation)> {
        self.groups
            .lock()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}
