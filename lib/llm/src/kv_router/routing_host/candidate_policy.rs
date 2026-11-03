// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dynamo_kv_router::{
    scheduling::SchedulingRequest,
    selector::{WorkerSelectionInput, WorkerSelector},
};
use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::Error,
};

use super::BuiltinWorkerSelector;
use crate::{
    discovery::RuntimeConfigWatch,
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
};

pub(crate) type AttemptLease = Box<dyn Send + Sync>;

pub(crate) struct CandidateSelection {
    pub target: u64,
    pub allowed_fallback: HashSet<u64>,
    pub lease: Option<AttemptLease>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraRoutingTable, LoraStateTracker};
    use dynamo_runtime::pipeline::RouterMode;

    #[test]
    fn base_requests_do_not_read_candidates_or_update_load() {
        let estimator = Arc::new(LoadEstimator::new());
        let policy = LegacyLoraPolicy::hosted(
            Arc::new(LoraFilter::new(
                LoraRoutingTable::new(),
                LoraStateTracker::new(),
            )),
            estimator.clone(),
            BuiltinWorkerSelector::new(RouterMode::RoundRobin).unwrap(),
        );
        let request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        assert!(
            policy
                .select(&request, &|| panic!("unexpected discovery"), &|| panic!(
                    "unexpected capacity read"
                ))
                .unwrap()
                .is_none()
        );
        assert!(estimator.get_current_load().is_empty());
    }
}

pub(crate) trait CandidatePolicy: Send + Sync {
    fn select(
        &self,
        request: &PreprocessedRequest,
        live: &dyn Fn() -> Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> Result<Option<CandidateSelection>, Error>;
    fn refresh_admission(
        &self,
        request: &SchedulingRequest,
        available: Option<Arc<HashSet<u64>>>,
        workers: &RuntimeConfigWatch,
    ) -> Option<Arc<HashSet<u64>>>;
}

pub(crate) struct LegacyLoraPolicy {
    filter: Arc<LoraFilter>,
    hosted: Option<(Arc<LoadEstimator>, BuiltinWorkerSelector)>,
}

impl LegacyLoraPolicy {
    pub(crate) fn hosted(
        filter: Arc<LoraFilter>,
        estimator: Arc<LoadEstimator>,
        selector: BuiltinWorkerSelector,
    ) -> Arc<dyn CandidatePolicy> {
        Arc::new(Self {
            filter,
            hosted: Some((estimator, selector)),
        })
    }

    pub(crate) fn kv(filter: Arc<LoraFilter>) -> Arc<dyn CandidatePolicy> {
        Arc::new(Self {
            filter,
            hosted: None,
        })
    }
}

struct LoadLease {
    estimator: Arc<LoadEstimator>,
    name: String,
}

impl Drop for LoadLease {
    fn drop(&mut self) {
        self.estimator.decrement_load(&self.name);
    }
}

impl CandidatePolicy for LegacyLoraPolicy {
    fn select(
        &self,
        request: &PreprocessedRequest,
        live: &dyn Fn() -> Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> Result<Option<CandidateSelection>, Error> {
        let Some(name) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.as_deref())
        else {
            return Ok(None);
        };
        let Some((estimator, selector)) = &self.hosted else {
            anyhow::bail!("hosted selection requires a hosted policy")
        };
        estimator.increment_load(name);
        let lease: AttemptLease = Box::new(LoadLease {
            estimator: estimator.clone(),
            name: name.to_string(),
        });
        let routable = live();
        let candidates = self
            .filter
            .filter_worker_ids_for_lora(Some(name), &routable);
        if candidates.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={name})");
        }
        let free = free().into_iter().collect::<HashSet<_>>();
        let candidates = candidates
            .into_iter()
            .filter(|worker| free.contains(worker))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::ResourceExhausted)
                    .message(format!(
                        "All eligible LoRA workers are overloaded (lora={name})"
                    ))
                    .build()
            ));
        }
        let target = selector
            .select_worker(WorkerSelectionInput::hosted(&candidates, None))?
            .worker
            .worker_id;
        tracing::debug!(lora = %name, worker_id = target, candidates = candidates.len(), routable = routable.len(), free = free.len(), "LoRA-filtered router selected worker");
        Ok(Some(CandidateSelection {
            target,
            allowed_fallback: candidates.into_iter().collect(),
            lease: Some(lease),
        }))
    }

    fn refresh_admission(
        &self,
        request: &SchedulingRequest,
        available: Option<Arc<HashSet<u64>>>,
        workers: &RuntimeConfigWatch,
    ) -> Option<Arc<HashSet<u64>>> {
        let Some(name) = request.lora_name.as_deref() else {
            return available;
        };
        let candidates = match (&request.allowed_worker_ids, available.as_ref()) {
            (Some(allowed), _) => allowed
                .iter()
                .filter(|id| available.as_ref().is_none_or(|live| live.contains(*id)))
                .copied()
                .collect::<Vec<_>>(),
            (None, Some(available)) => available.iter().copied().collect(),
            (None, None) => workers.borrow().keys().copied().collect(),
        };
        Some(Arc::new(
            self.filter
                .filter_worker_ids_for_lora_with_pin(
                    Some(name),
                    &candidates,
                    request.pinned_worker.map(|worker| worker.worker_id),
                )
                .into_iter()
                .collect(),
        ))
    }
}
