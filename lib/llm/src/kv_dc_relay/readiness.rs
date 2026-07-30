// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_kv_router::protocols::WorkerId;

use super::discovery::{AdapterMembership, DomainWorkerTopology};
use super::identity::{CanonicalModelId, CanonicalModelRegistration, ModelTarget};
use crate::worker_type::WorkerType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServingReadinessState {
    Ready,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServingReadinessEntry {
    pub(crate) producer: ProducerIdentity,
    pub(crate) model: CanonicalModelId,
    pub(crate) target: ModelTarget,
    pub(crate) state: ServingReadinessState,
    pub(crate) present_roles: Vec<String>,
    pub(crate) missing_roles: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServingReadinessSnapshot {
    pub(crate) revision: u64,
    pub(crate) entries: Vec<ServingReadinessEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EndpointServingFacts {
    pub(crate) worker_topology: HashMap<WorkerId, DomainWorkerTopology>,
    pub(crate) adapters: HashMap<CanonicalModelId, AdapterMembership>,
    /// None means that endpoint availability has not produced an authoritative snapshot.
    pub(crate) live_workers: Option<HashSet<WorkerId>>,
}

struct TopologyEvaluation {
    state: ServingReadinessState,
    present_roles: Vec<String>,
    missing_roles: Vec<String>,
}

pub(crate) fn derive_endpoint_readiness(
    producer: ProducerIdentity,
    registrations: &[CanonicalModelRegistration],
    facts: &EndpointServingFacts,
) -> Vec<ServingReadinessEntry> {
    registrations
        .iter()
        .map(|registration| {
            let evaluation = evaluate_registration(registration.target(), facts);
            ServingReadinessEntry {
                producer,
                model: registration.model().clone(),
                target: registration.target().clone(),
                state: evaluation.state,
                present_roles: evaluation.present_roles,
                missing_roles: evaluation.missing_roles,
            }
        })
        .collect()
}

fn evaluate_registration(target: &ModelTarget, facts: &EndpointServingFacts) -> TopologyEvaluation {
    if facts.worker_topology.is_empty() {
        return unknown();
    }
    let Some(live_workers) = facts.live_workers.as_ref() else {
        return unknown();
    };
    let eligible_workers = match target {
        ModelTarget::Base { .. } => None,
        ModelTarget::Lora {
            base_model,
            adapter,
        } => {
            let Some(membership) = facts.adapters.get(adapter) else {
                return unknown();
            };
            if &membership.base_model != base_model {
                return unknown();
            }
            Some(membership.workers.keys().copied().collect::<HashSet<_>>())
        }
    };

    let mut typed_workers = 0usize;
    let mut legacy_workers = 0usize;
    let mut present = HashSet::new();
    let mut declared = HashSet::new();
    let mut has_live_eligible_worker = false;
    for (&worker_id, topology) in &facts.worker_topology {
        let eligible = eligible_workers
            .as_ref()
            .is_none_or(|workers| workers.contains(&worker_id));
        let live = eligible && live_workers.contains(&worker_id);
        has_live_eligible_worker |= live;
        match topology.worker_type {
            Some(worker_type) => {
                typed_workers += 1;
                declared.insert(worker_type);
                if live {
                    present.insert(worker_type);
                }
            }
            None => legacy_workers += 1,
        }
    }

    if typed_workers != 0 && legacy_workers != 0 {
        return unknown();
    }
    if typed_workers == 0 {
        return TopologyEvaluation {
            state: if has_live_eligible_worker {
                ServingReadinessState::Ready
            } else {
                ServingReadinessState::Unavailable
            },
            present_roles: Vec::new(),
            missing_roles: Vec::new(),
        };
    }

    let mut missing = declared
        .difference(&present)
        .copied()
        .collect::<HashSet<_>>();
    for (&worker_id, topology) in &facts.worker_topology {
        let eligible = eligible_workers
            .as_ref()
            .is_none_or(|workers| workers.contains(&worker_id));
        if !eligible || !live_workers.contains(&worker_id) || topology.needs.is_empty() {
            continue;
        }
        let satisfied = topology
            .needs
            .iter()
            .any(|alternative| alternative.iter().all(|needed| present.contains(needed)));
        if !satisfied {
            for needed in topology.needs.iter().flatten() {
                if !present.contains(needed) {
                    missing.insert(*needed);
                }
            }
        }
    }

    TopologyEvaluation {
        state: if has_live_eligible_worker && missing.is_empty() {
            ServingReadinessState::Ready
        } else {
            ServingReadinessState::Unavailable
        },
        present_roles: sorted_roles(present),
        missing_roles: sorted_roles(missing),
    }
}

fn unknown() -> TopologyEvaluation {
    TopologyEvaluation {
        state: ServingReadinessState::Unknown,
        present_roles: Vec::new(),
        missing_roles: Vec::new(),
    }
}

fn sorted_roles(roles: HashSet<WorkerType>) -> Vec<String> {
    let mut roles = roles
        .into_iter()
        .map(|role| role.as_str().to_string())
        .collect::<Vec<_>>();
    roles.sort_unstable();
    roles
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};

    use super::super::discovery::AdapterWorkerMembership;
    use super::*;

    fn producer() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn registration(model: &str) -> CanonicalModelRegistration {
        CanonicalModelRegistration::new(CanonicalModelId::new(model).unwrap(), Vec::new())
    }

    fn topology(
        worker_type: Option<WorkerType>,
        needs: Vec<Vec<WorkerType>>,
    ) -> DomainWorkerTopology {
        DomainWorkerTopology { worker_type, needs }
    }

    #[test]
    fn authoritative_legacy_endpoint_with_one_live_worker_is_ready() {
        let facts = EndpointServingFacts {
            worker_topology: HashMap::from([(1, topology(None, Vec::new()))]),
            live_workers: Some(HashSet::from([1])),
            ..EndpointServingFacts::default()
        };
        let entries = derive_endpoint_readiness(producer(), &[registration("llama")], &facts);
        assert_eq!(entries[0].state, ServingReadinessState::Ready);
    }

    #[test]
    fn missing_availability_is_unknown_and_fresh_empty_is_unavailable() {
        let mut facts = EndpointServingFacts {
            worker_topology: HashMap::from([(
                1,
                topology(Some(WorkerType::Aggregated), Vec::new()),
            )]),
            ..EndpointServingFacts::default()
        };
        let registration = registration("llama");
        assert_eq!(
            derive_endpoint_readiness(producer(), std::slice::from_ref(&registration), &facts)[0]
                .state,
            ServingReadinessState::Unknown
        );
        facts.live_workers = Some(HashSet::new());
        assert_eq!(
            derive_endpoint_readiness(producer(), &[registration], &facts)[0].state,
            ServingReadinessState::Unavailable
        );
    }

    #[test]
    fn disaggregated_topology_requires_all_live_roles() {
        let facts = EndpointServingFacts {
            worker_topology: HashMap::from([
                (
                    1,
                    topology(Some(WorkerType::Prefill), vec![vec![WorkerType::Decode]]),
                ),
                (
                    2,
                    topology(Some(WorkerType::Decode), vec![vec![WorkerType::Prefill]]),
                ),
            ]),
            live_workers: Some(HashSet::from([1])),
            ..EndpointServingFacts::default()
        };
        let entries = derive_endpoint_readiness(producer(), &[registration("llama")], &facts);
        assert_eq!(entries[0].state, ServingReadinessState::Unavailable);
        assert_eq!(entries[0].missing_roles, ["decode"]);
    }

    #[test]
    fn base_can_be_ready_while_lora_is_unavailable() {
        let base_model = CanonicalModelId::new("llama").unwrap();
        let adapter = CanonicalModelId::new("tenant-a").unwrap();
        let registrations = vec![
            CanonicalModelRegistration::new(base_model.clone(), Vec::new()),
            CanonicalModelRegistration::with_target(
                adapter.clone(),
                ModelTarget::Lora {
                    base_model: base_model.clone(),
                    adapter: adapter.clone(),
                },
                Vec::new(),
            ),
        ];
        let facts = EndpointServingFacts {
            worker_topology: HashMap::from([
                (
                    1,
                    topology(Some(WorkerType::Prefill), vec![vec![WorkerType::Decode]]),
                ),
                (
                    2,
                    topology(Some(WorkerType::Decode), vec![vec![WorkerType::Prefill]]),
                ),
            ]),
            adapters: HashMap::from([(
                adapter,
                AdapterMembership {
                    base_model,
                    workers: HashMap::from([(
                        1,
                        AdapterWorkerMembership {
                            max_gpu_lora_count: Some(4),
                        },
                    )]),
                },
            )]),
            live_workers: Some(HashSet::from([1, 2])),
        };
        let entries = derive_endpoint_readiness(producer(), &registrations, &facts);
        let base = entries
            .iter()
            .find(|entry| matches!(entry.target, ModelTarget::Base { .. }))
            .unwrap();
        let lora = entries
            .iter()
            .find(|entry| matches!(entry.target, ModelTarget::Lora { .. }))
            .unwrap();
        assert_eq!(base.state, ServingReadinessState::Ready);
        assert_eq!(lora.state, ServingReadinessState::Unavailable);
        assert_eq!(lora.missing_roles, ["decode"]);
    }
}
