// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dynamo_kv_router::{scheduling::SchedulingRequest, selector::HostedWorkerPicker};
use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::Error,
};

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::hosted_request,
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    routing_policy::{AttemptLease, CandidateSelection, HostedCandidatePolicy, KvAdmissionPolicy},
};

pub(crate) struct LegacyLoraPolicy {
    filter: Arc<LoraFilter>,
    estimator: Arc<LoadEstimator>,
    picker: HostedWorkerPicker,
}

impl LegacyLoraPolicy {
    pub(crate) fn hosted(
        filter: Arc<LoraFilter>,
        estimator: Arc<LoadEstimator>,
        picker: HostedWorkerPicker,
    ) -> Arc<dyn HostedCandidatePolicy> {
        Arc::new(Self {
            filter,
            estimator,
            picker,
        })
    }
}

pub(crate) struct LegacyLoraAdmission {
    filter: Arc<LoraFilter>,
}

impl LegacyLoraAdmission {
    pub(crate) fn kv(filter: Arc<LoraFilter>) -> Arc<dyn KvAdmissionPolicy> {
        Arc::new(Self { filter })
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

impl HostedCandidatePolicy for LegacyLoraPolicy {
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
        self.estimator.increment_load(name);
        let lease: AttemptLease = Box::new(LoadLease {
            estimator: self.estimator.clone(),
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
        let target = self
            .picker
            .select(&candidates, None, &hosted_request(request))?;
        tracing::debug!(
            lora = %name,
            worker_id = target,
            candidates = candidates.len(),
            routable = routable.len(),
            free = free.len(),
            "LoRA-filtered router selected worker"
        );
        Ok(Some(CandidateSelection {
            target,
            allowed_fallback: candidates.into_iter().collect(),
            lease: Some(lease),
        }))
    }
}

impl KvAdmissionPolicy for LegacyLoraAdmission {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraRoutingTable, LoraStateTracker};

    #[test]
    fn base_requests_do_not_read_candidates_or_update_load() {
        let estimator = Arc::new(LoadEstimator::new());
        let policy = LegacyLoraPolicy::hosted(
            Arc::new(LoraFilter::new(
                LoraRoutingTable::new(),
                LoraStateTracker::new(),
            )),
            estimator.clone(),
            crate::kv_router::hosted_picker(dynamo_runtime::pipeline::RouterMode::RoundRobin)
                .unwrap(),
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

    #[test]
    fn kv_admission_intersects_live_and_allowed_workers_on_each_refresh() {
        use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::ScheduleMode};
        let policy = LegacyLoraAdmission::kv(Arc::new(LoraFilter::new(
            LoraRoutingTable::new(),
            LoraStateTracker::new(),
        )));
        let (_tx, workers) = tokio::sync::watch::channel(Default::default());
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 0,
            lora_name: Some("adapter".into()),
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: Some(WorkerWithDpRank::new(2, 0)),
            allowed_worker_ids: Some(HashSet::from([1, 2])),
            routing_constraints: Default::default(),
            router_config_override: None,
            track_prefill_tokens: false,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: Default::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        let available = Some(Arc::new(HashSet::from([2, 3])));
        assert_eq!(
            *policy
                .refresh_admission(&request, available.clone(), &workers)
                .unwrap(),
            HashSet::from([2])
        );
        request.lora_name = None;
        assert!(Arc::ptr_eq(
            policy
                .refresh_admission(&request, available.clone(), &workers)
                .as_ref()
                .unwrap(),
            available.as_ref().unwrap()
        ));
        request.lora_name = Some("adapter".into());
        assert!(
            policy
                .refresh_admission(&request, Some(Arc::new(HashSet::new())), &workers)
                .unwrap()
                .is_empty()
        );
    }
}
