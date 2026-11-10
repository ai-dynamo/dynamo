// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dynamo_lora::policy::{LoadGuard, RequestPolicy, Stage};
use dynamo_runtime::pipeline::{Error, RouterMode};
use dynamo_scheduler::host::{KvSchedulerError, RoutingConstraints, SchedulingRequest};
use dynamo_worker_selection::host::HostedWorkerPicker;

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::hosted_request,
    local_model::runtime_config::ModelRuntimeConfig,
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    routing_policy::{
        CandidateSelection, ExecutionRequirements, ExecutionStage, HostedCandidatePolicy,
        KvAdmissionPolicy,
    },
};

fn policy(request: &PreprocessedRequest) -> RequestPolicy {
    let version = request.routing.as_ref().and_then(|routing| {
        routing.lora_resolution_version.or_else(|| {
            (routing.base_model_name.is_some() || routing.lora_source_uri.is_some()).then_some(0)
        })
    });
    RequestPolicy::new(version)
}

pub(crate) fn execution_requirements(request: &PreprocessedRequest) -> ExecutionRequirements {
    let policy = policy(request);
    ExecutionRequirements {
        allows_replay: policy.allows_replay(),
        policy_required: policy.requires_candidate_policy(),
        accounting_attribution: if policy.tracks_legacy_load() {
            dynamo_scheduler::host::AccountingAttribution::UseRoutingName
        } else {
            dynamo_scheduler::host::AccountingAttribution::Exclude
        },
    }
}

pub(crate) fn check_candidate_policy(
    request: &PreprocessedRequest,
    installed: bool,
) -> Result<(), Error> {
    if let Some(routing) = request.routing.as_ref()
        && policy(request).requires_candidate_policy()
    {
        let valid = match (
            routing.base_model_name.as_deref(),
            routing.lora_source_uri.as_ref(),
            routing.lora_name.as_deref(),
        ) {
            (Some(base), Some(source), Some(key)) => {
                routing.lora_resolution_version
                    == Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION)
                    && crate::lora::runtime::validate_runtime_lora_identity(
                        &request.model,
                        base,
                        source.as_str(),
                        key,
                    )
                    .is_ok()
            }
            _ => false,
        };
        if !valid {
            return Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::InvalidArgument)
                .message("invalid_lora_model_id")
                .build()
                .into());
        }
    }
    policy(request).check_capability(
        installed,
        request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.as_deref()),
    )?;
    Ok(())
}

pub(crate) fn check_stage(
    request: &PreprocessedRequest,
    stage: ExecutionStage,
) -> Result<(), Error> {
    policy(request).check_stage(match stage {
        ExecutionStage::Prefill => Stage::Prefill,
        ExecutionStage::Encoder => Stage::Encoder,
    })?;
    Ok(())
}

pub(crate) struct LoraHostedPolicy {
    filter: Arc<LoraFilter>,
    estimator: Arc<LoadEstimator>,
    picker: HostedWorkerPicker,
    runtime_configs: Option<RuntimeConfigWatch>,
}

pub(crate) struct LoraKvPolicy {
    filter: Arc<LoraFilter>,
    runtime_enabled: bool,
}

impl LoraKvPolicy {
    pub(crate) fn kv(filter: Arc<LoraFilter>) -> Arc<dyn KvAdmissionPolicy> {
        Arc::new(Self {
            filter,
            runtime_enabled: dynamo_runtime::config::env_is_truthy(
                dynamo_runtime::config::environment_names::llm::DYN_LORA_RUNTIME_LOAD_ENABLED,
            ),
        })
    }
}

impl LoraHostedPolicy {
    pub(crate) fn hosted(
        filter: Arc<LoraFilter>,
        estimator: Arc<LoadEstimator>,
        mode: RouterMode,
        runtime_configs: Option<RuntimeConfigWatch>,
    ) -> Result<Arc<dyn HostedCandidatePolicy>, Error> {
        if !matches!(mode, RouterMode::RoundRobin | RouterMode::Random) {
            anyhow::bail!("LoRA filtering is unsupported with {mode:?} routing");
        }
        let picker = crate::kv_router::hosted_picker(mode)?;
        Ok(Arc::new(Self {
            filter,
            estimator,
            picker,
            runtime_configs,
        }))
    }
}

impl HostedCandidatePolicy for LoraHostedPolicy {
    fn select(
        &self,
        request: &PreprocessedRequest,
        pin: Option<u64>,
        live: &dyn Fn() -> Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> Result<Option<CandidateSelection>, Error> {
        let policy = policy(request);
        let routing = request.routing.as_ref();
        let name = routing.and_then(|routing| routing.lora_name.as_deref());
        policy.check_capability(self.runtime_configs.is_some(), name)?;
        let Some(name) = name else {
            return Ok(None);
        };
        let lease = policy.tracks_legacy_load().then(|| {
            Box::new(LoadGuard::new(self.estimator.clone(), name.into()))
                as crate::routing_policy::AttemptLease
        });
        let live = live();
        let configs = self
            .runtime_configs
            .as_ref()
            .filter(|_| policy.requires_candidate_policy())
            .map(|watch| watch.borrow());
        let default_constraints = RoutingConstraints::default();
        let constraints = routing
            .and_then(|routing| routing.routing_constraints.as_ref())
            .unwrap_or(&default_constraints);
        let pin = pin.filter(|_| policy.requires_candidate_policy());
        let candidates: Vec<u64> = if policy.requires_candidate_policy() {
            policy
                .restrict_workers(
                    &self.filter,
                    Some(name),
                    configs
                        .as_deref()
                        .ok_or_else(RequestPolicy::capability_error)?,
                    &live.into_iter().collect(),
                    routing.and_then(|routing| routing.allowed_worker_ids.clone()),
                    constraints,
                    pin.as_slice(),
                )?
                .unwrap_or_default()
                .into_iter()
                .collect()
        } else {
            policy.candidates(
                &self.filter,
                name,
                live,
                None,
                configs.as_deref(),
                constraints,
            )?
        };
        drop(configs);
        if pin.is_some_and(|worker| !candidates.contains(&worker)) {
            return Err(RequestPolicy::affinity_error().into());
        }
        let candidates = RequestPolicy::free_candidates(name, candidates, free)?;
        let worker = match pin {
            Some(worker) if candidates.contains(&worker) => worker,
            Some(_) => {
                return Err(dynamo_runtime::error::DynamoError::builder()
                    .error_type(dynamo_runtime::error::ErrorType::ResourceExhausted)
                    .message(format!("Pinned LoRA worker is overloaded (lora={name})"))
                    .build()
                    .into());
            }
            None => self
                .picker
                .select(&candidates, None, &hosted_request(request))
                .map_err(KvSchedulerError::from)?,
        };
        Ok(Some(CandidateSelection {
            worker,
            fallback: match pin {
                Some(_) => HashSet::from([worker]),
                None => candidates.into_iter().collect(),
            },
            lease,
        }))
    }

    fn check_selected(
        &self,
        request: &PreprocessedRequest,
        worker: u64,
        pin: Option<u64>,
        live: &dyn Fn() -> Vec<u64>,
    ) -> Result<(), Error> {
        let policy = policy(request);
        policy.check_affinity(worker, pin)?;
        if policy.requires_candidate_policy() {
            let routing = request
                .routing
                .as_ref()
                .ok_or_else(RequestPolicy::capability_error)?;
            let live = live();
            let configs = self.runtime_configs.as_ref().map(|watch| watch.borrow());
            let default_constraints = RoutingConstraints::default();
            policy.check_worker_eligible(
                worker,
                configs.as_deref(),
                &live.into_iter().collect(),
                routing.allowed_worker_ids.clone(),
                routing
                    .routing_constraints
                    .as_ref()
                    .unwrap_or(&default_constraints),
            )?;
        }
        Ok(())
    }
}

impl KvAdmissionPolicy for LoraKvPolicy {
    fn restrict(
        &self,
        request: &PreprocessedRequest,
        workers: &HashMap<u64, ModelRuntimeConfig>,
        live: &HashSet<u64>,
        allowed: Option<HashSet<u64>>,
        constraints: &RoutingConstraints,
        pins: &[u64],
    ) -> Result<Option<HashSet<u64>>, Error> {
        policy(request).check_capability(
            self.runtime_enabled,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.as_deref()),
        )?;
        Ok(policy(request).restrict_workers(
            &self.filter,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.as_deref()),
            workers,
            live,
            allowed,
            constraints,
            pins,
        )?)
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

    fn check_current(
        &self,
        request: &PreprocessedRequest,
        worker: u64,
        workers: &HashMap<u64, ModelRuntimeConfig>,
        live: &HashSet<u64>,
        allowed: Option<HashSet<u64>>,
        constraints: &RoutingConstraints,
    ) -> Result<(), Error> {
        policy(request).check_capability(
            self.runtime_enabled,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.as_deref()),
        )?;
        policy(request).check_worker_eligible(worker, Some(workers), live, allowed, constraints)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraRoutingTable, LoraStateTracker};

    #[test]
    fn kv_policy_rejects_runtime_requests_when_runtime_loading_is_disabled() {
        use crate::protocols::common::preprocessor::RoutingHints;

        temp_env::with_vars(
            [
                ("DYN_LORA_ENABLED", Some("true")),
                ("DYN_LORA_RUNTIME_LOAD_ENABLED", Some("false")),
            ],
            || {
                let policy = LoraKvPolicy::kv(Arc::new(LoraFilter::new(
                    LoraRoutingTable::new(),
                    LoraStateTracker::new(),
                )));
                let mut request = PreprocessedRequest::builder()
                    .model("base".to_string())
                    .token_ids(vec![1])
                    .stop_conditions(Default::default())
                    .sampling_options(Default::default())
                    .output_options(Default::default())
                    .build()
                    .unwrap();
                request.routing = Some(RoutingHints {
                    lora_name: Some(crate::lora::runtime::adapter_key(
                        "base",
                        "custom:///opaque",
                    )),
                    base_model_name: Some("base".into()),
                    lora_source_uri: Some(crate::lora::runtime::RuntimeLoraSourceUri::new(
                        "custom:///opaque".into(),
                    )),
                    lora_resolution_version: Some(2),
                    ..Default::default()
                });
                let mut config = ModelRuntimeConfig::default();
                config.taints.insert("dynamo.runtime-lora/v2".into());
                config
                    .set_engine_specific("supports_runtime_lora_resolution", true)
                    .unwrap();
                config
                    .set_engine_specific("runtime_lora_protocol_versions", vec![2_u32])
                    .unwrap();
                config
                    .set_engine_specific("runtime_lora_schemes", vec!["custom"])
                    .unwrap();
                let error = policy
                    .restrict(
                        &request,
                        &HashMap::from([(1, config)]),
                        &HashSet::from([1]),
                        None,
                        &Default::default(),
                        &[],
                    )
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("runtime_lora_capability_unavailable")
                );
                request.routing = Some(RoutingHints {
                    lora_name: Some("legacy".into()),
                    ..Default::default()
                });
                assert!(
                    policy
                        .restrict(
                            &request,
                            &HashMap::new(),
                            &HashSet::new(),
                            None,
                            &Default::default(),
                            &[]
                        )
                        .is_ok()
                );
            },
        );
    }

    #[test]
    fn accounting_uses_typed_runtime_metadata_not_adapter_name() {
        use crate::protocols::common::preprocessor::RoutingHints;
        use dynamo_scheduler::host::AccountingAttribution;

        let mut request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.routing = Some(RoutingHints {
            lora_name: Some("dyn-lora-legacy-alias".into()),
            ..Default::default()
        });
        assert_eq!(
            execution_requirements(&request).accounting_attribution,
            AccountingAttribution::UseRoutingName
        );
        request.routing = Some(RoutingHints {
            lora_name: Some("runtime-key".into()),
            base_model_name: Some("base".into()),
            lora_source_uri: Some(crate::lora::runtime::RuntimeLoraSourceUri::new(
                "custom:///opaque|reference".into(),
            )),
            lora_resolution_version: Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION),
            ..Default::default()
        });
        assert_eq!(
            execution_requirements(&request).accounting_attribution,
            AccountingAttribution::Exclude
        );
    }

    #[test]
    fn partial_runtime_metadata_is_rejected_and_never_replayed() {
        use crate::protocols::common::preprocessor::RoutingHints;
        use dynamo_scheduler::host::AccountingAttribution;

        let mut request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let complete = RoutingHints {
            lora_name: Some(crate::lora::runtime::adapter_key(
                "base",
                "custom:///opaque",
            )),
            base_model_name: Some("base".into()),
            lora_source_uri: Some(crate::lora::runtime::RuntimeLoraSourceUri::new(
                "custom:///opaque".into(),
            )),
            lora_resolution_version: Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION),
            ..Default::default()
        };
        for missing in 0..3 {
            let mut routing = complete.clone();
            match missing {
                0 => routing.base_model_name = None,
                1 => routing.lora_source_uri = None,
                _ => routing.lora_resolution_version = None,
            }
            request.routing = Some(routing);
            let requirements = execution_requirements(&request);
            assert!(!requirements.allows_replay);
            assert_eq!(
                requirements.accounting_attribution,
                AccountingAttribution::Exclude
            );
            assert!(check_candidate_policy(&request, true).is_err());
            assert!(check_stage(&request, ExecutionStage::Prefill).is_err());
            assert!(check_stage(&request, ExecutionStage::Encoder).is_err());
        }
        request.routing = Some(complete.clone());
        assert!(check_candidate_policy(&request, true).is_ok());
        assert!(dynamo_runtime::error::match_error_chain(
            check_candidate_policy(&request, false)
                .unwrap_err()
                .as_ref(),
            &[dynamo_runtime::error::ErrorType::Unavailable],
            &[]
        ));
        request.routing.as_mut().unwrap().lora_name = None;
        for installed in [true, false] {
            assert!(dynamo_runtime::error::match_error_chain(
                check_candidate_policy(&request, installed)
                    .unwrap_err()
                    .as_ref(),
                &[dynamo_runtime::error::ErrorType::InvalidArgument],
                &[]
            ));
        }
        request.routing = Some(complete.clone());
        request.routing.as_mut().unwrap().lora_name = Some("wrong-key".into());
        assert!(check_candidate_policy(&request, true).is_err());
        request.routing = Some(complete.clone());
        request.model = "wrong-base".into();
        assert!(check_candidate_policy(&request, true).is_err());
        request.model = "base".into();
        request.routing.as_mut().unwrap().lora_resolution_version = Some(u16::MAX);
        assert!(check_candidate_policy(&request, true).is_err());
        request.routing = Some(RoutingHints {
            lora_name: Some("legacy-adapter".into()),
            ..Default::default()
        });
        assert!(execution_requirements(&request).allows_replay);
        assert!(check_candidate_policy(&request, true).is_ok());
    }

    #[test]
    fn base_request_does_not_read_candidate_or_capacity_state() {
        let policy = LoraHostedPolicy::hosted(
            Arc::new(LoraFilter::new(
                LoraRoutingTable::new(),
                LoraStateTracker::new(),
            )),
            Arc::new(LoadEstimator::new()),
            RouterMode::RoundRobin,
            None,
        )
        .unwrap();
        let request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let result = policy
            .select(
                &request,
                None,
                &|| panic!("base request must not collect live workers"),
                &|| panic!("base request must not read free capacity"),
            )
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn selected_runtime_worker_remains_eligible_after_scale_up() {
        use crate::protocols::common::preprocessor::RoutingHints;

        let mut config = ModelRuntimeConfig::default();
        config.taints.insert("dynamo.runtime-lora/v2".into());
        config
            .set_engine_specific("supports_runtime_lora_resolution", true)
            .unwrap();
        config
            .set_engine_specific("runtime_lora_protocol_versions", vec![2_u32])
            .unwrap();
        config
            .set_engine_specific("runtime_lora_schemes", vec!["custom"])
            .unwrap();
        let (tx, rx) = tokio::sync::watch::channel(HashMap::from([(1, config.clone())]));
        let filter = Arc::new(LoraFilter::new(
            LoraRoutingTable::new(),
            LoraStateTracker::new(),
        ));
        let policy = LoraHostedPolicy::hosted(
            filter.clone(),
            Arc::new(LoadEstimator::new()),
            RouterMode::RoundRobin,
            Some(rx),
        )
        .unwrap();
        let mut request = PreprocessedRequest::builder()
            .model("base".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let name = (0..1000)
            .map(|n| format!("adapter-{n}"))
            .find(|name| filter.filter_worker_ids_for_runtime_lora(name, &[1, 2]) == [2])
            .unwrap();
        request.routing = Some(RoutingHints {
            lora_name: Some(name),
            base_model_name: Some("base".into()),
            lora_source_uri: Some(crate::lora::runtime::RuntimeLoraSourceUri::new(
                "custom:///opaque".into(),
            )),
            lora_resolution_version: Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION),
            ..Default::default()
        });
        let selection = policy
            .select(&request, None, &|| vec![1], &|| vec![1])
            .unwrap()
            .unwrap();
        assert_eq!(selection.worker, 1);
        let workers = HashMap::from([(1, config.clone()), (2, config.clone())]);
        tx.send(workers.clone()).unwrap();
        assert!(
            policy
                .check_selected(&request, 1, None, &|| vec![1, 2])
                .is_ok()
        );
        let kv_policy = LoraKvPolicy {
            filter,
            runtime_enabled: true,
        };
        assert!(
            kv_policy
                .check_current(
                    &request,
                    1,
                    &workers,
                    &HashSet::from([1, 2]),
                    None,
                    &Default::default()
                )
                .is_ok()
        );
        assert!(
            kv_policy
                .check_current(
                    &request,
                    1,
                    &workers,
                    &HashSet::from([2]),
                    None,
                    &Default::default()
                )
                .is_err()
        );
        assert!(
            kv_policy
                .check_current(
                    &request,
                    1,
                    &workers,
                    &HashSet::from([1, 2]),
                    Some(HashSet::from([2])),
                    &Default::default()
                )
                .is_err()
        );
        let mut revoked = config;
        revoked.taints.clear();
        tx.send(HashMap::from([(1, revoked), (2, workers[&2].clone())]))
            .unwrap();
        assert!(
            policy
                .check_selected(&request, 1, None, &|| vec![1, 2])
                .is_err()
        );
    }

    #[test]
    fn kv_admission_intersects_live_and_allowed_workers_on_each_refresh() {
        use dynamo_router_protocol::worker::WorkerWithDpRank;
        use dynamo_scheduler::host::ScheduleMode;
        let policy = LoraKvPolicy::kv(Arc::new(LoraFilter::new(
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
