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
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    routing_policy::{AttemptLease, CandidatePolicy, CandidateSelection},
};

pub(crate) struct LegacyLoraPolicy {
    filter: Arc<LoraFilter>,
    hosted: Option<(Arc<LoadEstimator>, HostedWorkerPicker)>,
}

impl LegacyLoraPolicy {
    pub(crate) fn hosted(
        filter: Arc<LoraFilter>,
        estimator: Arc<LoadEstimator>,
        selector: HostedWorkerPicker,
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
        let target = selector.select(
            &candidates,
            None,
            &dynamo_kv_router::selector::HostedSelectionRequest::new(request.token_ids.len()),
        )?;
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
    fn hosted_policy_releases_load_after_selection_and_on_error() {
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
        let mut request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.routing = Some(crate::protocols::common::preprocessor::RoutingHints {
            lora_name: Some("adapter".into()),
            ..Default::default()
        });
        let selection = policy
            .select(&request, &|| vec![1, 2], &|| vec![2])
            .unwrap()
            .unwrap();
        assert_eq!(selection.target, 2);
        assert_eq!(selection.allowed_fallback, HashSet::from([2]));
        assert_eq!(estimator.get_inflight_counts().get("adapter"), Some(&1));
        drop(selection);
        assert_eq!(
            estimator
                .get_inflight_counts()
                .get("adapter")
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(estimator.get_raw_arrival_counts().get("adapter"), Some(&1));
        assert!(policy.select(&request, &|| vec![1], &Vec::new).is_err());
        assert_eq!(
            estimator
                .get_inflight_counts()
                .get("adapter")
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(estimator.get_raw_arrival_counts().get("adapter"), Some(&2));
    }

    #[test]
    fn kv_admission_intersects_live_and_allowed_workers_on_each_refresh() {
        use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::ScheduleMode};
        let policy = LegacyLoraPolicy::kv(Arc::new(LoraFilter::new(
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
            accounting_attribution: Default::default(),
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
