// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_scheduler::host::{RoutingConstraints, WorkerConfigLike};

use crate::{
    LoadEstimator, LoraFilter,
    runtime::{RuntimeLoraWorkerFacts, constrain_runtime_lora_workers},
};
use dynamo_runtime::pipeline::RouterMode;

pub fn validate_router_mode_for_lora(
    router_mode: RouterMode,
    lora_enabled: bool,
    session_affinity_enabled: bool,
) -> anyhow::Result<()> {
    if !lora_enabled {
        return Ok(());
    }
    if session_affinity_enabled
        && matches!(router_mode, RouterMode::Random | RouterMode::RoundRobin)
    {
        anyhow::bail!(
            "session affinity is unsupported with DYN_LORA_ENABLED and {router_mode:?} routing"
        );
    }
    match router_mode {
        RouterMode::KV | RouterMode::Random | RouterMode::RoundRobin => Ok(()),
        RouterMode::Direct
        | RouterMode::PowerOfTwoChoices
        | RouterMode::LeastLoaded
        | RouterMode::DeviceAwareWeighted => anyhow::bail!(
            "LoRA serving (DYN_LORA_ENABLED) is not supported with router mode {router_mode:?}; \
             use KV, Random, or RoundRobin for LoRA-aware routing, or disable LoRA serving."
        ),
    }
}

#[derive(Clone, Copy)]
pub struct RequestPolicy {
    runtime: bool,
}

#[derive(Clone, Copy)]
pub enum Stage {
    Prefill,
    Encoder,
    SpeculativePrefill,
}

impl RequestPolicy {
    pub fn candidates<C: RuntimeLoraWorkerFacts + WorkerConfigLike>(
        self,
        filter: &LoraFilter,
        name: &str,
        live: Vec<u64>,
        allowed: Option<&HashSet<u64>>,
        configs: Option<&HashMap<u64, C>>,
        constraints: &RoutingConstraints,
    ) -> Result<Vec<u64>, DynamoError> {
        if self.runtime {
            let eligible = Self::runtime_eligible(
                allowed.cloned(),
                &live.into_iter().collect(),
                configs,
                constraints,
            )?;
            Ok(filter.filter_worker_ids_for_runtime_lora(
                name,
                &eligible.into_iter().collect::<Vec<_>>(),
            ))
        } else {
            let eligible = live
                .into_iter()
                .filter(|id| allowed.is_none_or(|allowed| allowed.contains(id)))
                .collect::<Vec<_>>();
            Ok(filter.filter_worker_ids_for_lora(Some(name), &eligible))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restrict_workers<C: RuntimeLoraWorkerFacts + WorkerConfigLike>(
        self,
        filter: &LoraFilter,
        name: Option<&str>,
        configs: &HashMap<u64, C>,
        live: &HashSet<u64>,
        allowed: Option<HashSet<u64>>,
        constraints: &RoutingConstraints,
        pins: &[u64],
    ) -> Result<Option<HashSet<u64>>, DynamoError> {
        if !self.runtime {
            return Ok(allowed);
        }
        let name = name.ok_or_else(Self::capability_error)?;
        let eligible = Self::runtime_eligible(allowed, live, Some(configs), constraints)?;
        Self::pin_workers(filter, name, &eligible, pins.iter().copied()).map(Some)
    }

    fn runtime_eligible<C: RuntimeLoraWorkerFacts + WorkerConfigLike>(
        allowed: Option<HashSet<u64>>,
        live: &HashSet<u64>,
        configs: Option<&HashMap<u64, C>>,
        constraints: &RoutingConstraints,
    ) -> Result<HashSet<u64>, DynamoError> {
        let configs = configs.ok_or_else(Self::capability_error)?;
        let eligible = constrain_runtime_lora_workers(true, allowed, live, configs, constraints)
            .unwrap_or_default();
        if eligible.is_empty() {
            return Err(Self::capability_error());
        }
        Ok(eligible)
    }

    pub fn pin_workers(
        filter: &LoraFilter,
        name: &str,
        eligible: &HashSet<u64>,
        pins: impl IntoIterator<Item = u64>,
    ) -> Result<HashSet<u64>, DynamoError> {
        let pins = pins.into_iter().collect::<HashSet<_>>();
        if pins.len() > 1 || !pins.is_subset(eligible) {
            return Err(Self::affinity_error());
        }
        if !pins.is_empty() {
            return Ok(pins);
        }
        Ok(filter
            .filter_worker_ids_for_runtime_lora(name, &eligible.iter().copied().collect::<Vec<_>>())
            .into_iter()
            .collect())
    }

    pub fn free_candidates(
        name: &str,
        candidates: Vec<u64>,
        free: &dyn Fn() -> Vec<u64>,
    ) -> anyhow::Result<Vec<u64>> {
        if candidates.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={name})");
        }
        let free = free().into_iter().collect::<HashSet<_>>();
        let candidates = candidates
            .into_iter()
            .filter(|id| free.contains(id))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(DynamoError::builder()
                .error_type(ErrorType::ResourceExhausted)
                .message(format!(
                    "All eligible LoRA workers are overloaded (lora={name})"
                ))
                .build()
                .into());
        }
        Ok(candidates)
    }

    pub fn new(resolution_version: Option<u16>) -> Self {
        Self {
            runtime: resolution_version.is_some(),
        }
    }

    pub fn requires_candidate_policy(self) -> bool {
        self.runtime
    }

    pub fn allows_replay(self) -> bool {
        !self.runtime
    }

    pub fn tracks_legacy_load(self) -> bool {
        !self.runtime
    }

    pub fn check_capability(self, installed: bool, name: Option<&str>) -> Result<(), DynamoError> {
        if self.runtime && (!installed || name.is_none()) {
            return Err(Self::capability_error());
        }
        Ok(())
    }

    pub fn capability_error() -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::Unavailable)
            .message("runtime_lora_capability_unavailable")
            .build()
    }

    pub fn check_affinity(self, target: u64, requested: Option<u64>) -> Result<(), DynamoError> {
        if self.runtime && requested.is_some_and(|worker| worker != target) {
            return Err(Self::affinity_error());
        }
        Ok(())
    }

    pub fn affinity_error() -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message("runtime_lora_affinity_conflict")
            .build()
    }

    pub fn check_stage(self, stage: Stage) -> Result<(), DynamoError> {
        if !self.runtime {
            return Ok(());
        }
        let message = match stage {
            Stage::Prefill => "runtime_lora_disaggregated_unsupported",
            Stage::Encoder => "runtime_lora_encoder_unsupported",
            Stage::SpeculativePrefill => "runtime_lora_speculative_prefill_unsupported",
        };
        Err(DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message(message)
            .build())
    }
}

pub struct LoadGuard {
    estimator: Arc<LoadEstimator>,
    name: String,
}

impl LoadGuard {
    pub fn new(estimator: Arc<LoadEstimator>, name: String) -> Self {
        estimator.increment_load(&name);
        Self { estimator, name }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.estimator.decrement_load(&self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_placement_precedes_capacity_read_and_busy_targets_do_not_widen() {
        let error = RequestPolicy::free_candidates("adapter", vec![], &|| {
            panic!("empty placement must not read capacity")
        })
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("No workers available after LoRA filtering")
        );
        let error = RequestPolicy::free_candidates("adapter", vec![1], &|| vec![2]).unwrap_err();
        assert_eq!(
            error.downcast_ref::<DynamoError>().unwrap().error_type(),
            ErrorType::ResourceExhausted
        );
        assert_eq!(
            RequestPolicy::free_candidates("adapter", vec![1, 2], &|| vec![2, 3]).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn runtime_policy_denies_replay_and_unsupported_stages() {
        let runtime = RequestPolicy::new(Some(2));
        assert!(!runtime.allows_replay());
        assert!(!runtime.tracks_legacy_load());
        assert!(runtime.check_capability(false, Some("adapter")).is_err());
        assert!(runtime.check_capability(true, None).is_err());
        assert!(runtime.check_affinity(1, Some(2)).is_err());
        for stage in [Stage::Prefill, Stage::Encoder, Stage::SpeculativePrefill] {
            assert!(runtime.check_stage(stage).is_err());
            assert!(RequestPolicy::new(None).check_stage(stage).is_ok());
        }
    }

    #[test]
    fn eligible_explicit_pin_overrides_default_placement_without_widening() {
        let filter = LoraFilter::new(
            crate::routing::table::LoraRoutingTable::new(),
            crate::state_tracker::LoraStateTracker::new(),
        );
        let eligible = HashSet::from([1, 2, 3]);
        let default = RequestPolicy::pin_workers(&filter, "adapter", &eligible, []).unwrap();
        assert_eq!(default.len(), 1);
        let pin = *eligible.difference(&default).next().unwrap();
        assert_eq!(
            RequestPolicy::pin_workers(&filter, "adapter", &eligible, [pin, pin]).unwrap(),
            HashSet::from([pin])
        );
        assert!(RequestPolicy::pin_workers(&filter, "adapter", &eligible, [4]).is_err());
        assert!(RequestPolicy::pin_workers(&filter, "adapter", &eligible, [1, 2]).is_err());
    }

    #[test]
    fn load_guard_releases_exactly_once() {
        let estimator = Arc::new(LoadEstimator::new());
        let guard = LoadGuard::new(estimator.clone(), "adapter".into());
        assert_eq!(estimator.get_inflight_counts().get("adapter"), Some(&1));
        drop(guard);
        assert!(!estimator.get_inflight_counts().contains_key("adapter"));
    }
}
