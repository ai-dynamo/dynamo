// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's policies as router core plugins. The core makes one
//! picker per model.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{RoutingPartitionRef, WorkerType};
use dynamo_router_policies::router_modes;

use super::directory::PoolDirectory;
use super::{LeastInFlightConfig, SelectionCorePolicy};
use crate::frontend_signals::FrontendSignals;
use crate::placement::{self, Placed};
use crate::selector::in_flight_load;

impl SelectionCorePolicy {
    fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => router_modes::ROUND_ROBIN,
            Self::LeastInFlight(_) => "global-least-in-flight",
        }
    }
}

/// The router's builtin plugins plus `policy` when it is the Global Router's
/// own; least-in-flight reads `counts` through `directory`.
pub(super) fn registry(
    policy: &SelectionCorePolicy,
    directory: Arc<PoolDirectory>,
    counts: Arc<FrontendSignals>,
) -> Result<RouterPluginRegistry, WorkerSelectionPolicyRegistryError> {
    let mut registry = dynamo_router_policies::default_registry();
    let provider = match policy {
        SelectionCorePolicy::RoundRobin => return Ok(registry),
        SelectionCorePolicy::LeastInFlight(config) => {
            let config = config.clone();
            provider(move || LeastInFlightPicker {
                config: config.clone(),
                counts: counts.clone(),
                directory: directory.clone(),
                ties: AtomicUsize::new(0),
            })
        }
    };
    registry.register_worker_selection(policy.name(), provider)?;
    Ok(registry)
}

/// The policy YAML selecting `policy` for aggregated workers.
pub(super) fn policy_config(policy: &SelectionCorePolicy) -> String {
    let name = policy.name();
    format!(
        "worker_selection:\n  aggregated: {name}\n  instances:\n    - name: {name}\n      type: {name}\n"
    )
}

/// One picker per partition, no scorers, no parameters.
fn provider<P, F>(picker: F) -> WorkerSelectionPolicyProvider
where
    P: WorkerPicker + 'static,
    F: Fn() -> P + Send + Sync + 'static,
{
    let picker = Arc::new(picker);
    Arc::new(
        move |_parameters: &WorkerSelectionPolicyParameters| -> Result<
            WorkerSelectionPolicyFactory,
            WorkerSelectionPolicyProviderError,
        > {
            let picker = picker.clone();
            Ok(Arc::new(
                move |config: &KvRouterConfig, worker_type: WorkerType, _: RoutingPartitionRef<'_>| {
                    WorkerSelectionPolicy::new(
                        config.clone(),
                        worker_type.as_str(),
                        Vec::new(),
                        Box::new(picker()),
                    )
                },
            ))
        },
    )
}

/// [`crate::LeastInFlightSelector`]'s rule over the frontend's counts. The
/// pick stays pending on its pool until the request takes it over. A preview
/// advances nothing.
struct LeastInFlightPicker {
    config: LeastInFlightConfig,
    counts: Arc<FrontendSignals>,
    directory: Arc<PoolDirectory>,
    ties: AtomicUsize,
}

impl WorkerPicker for LeastInFlightPicker {
    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let pools = self.directory.snapshot();
        let signals = &self.config.signals;
        let round_trips = signals.region_round_trips();
        // Read each count once: requests start and finish while this runs.
        let mut placed = candidates
            .iter()
            .enumerate()
            .map(|(row, candidate)| {
                let worker = candidate.worker().worker_id;
                match pools.get(&worker) {
                    Some(pool) => {
                        let in_flight = self.counts.in_flight(&pool.id, worker);
                        Placed {
                            item: row,
                            load: in_flight_load(in_flight, signals.capacity(&pool.id)),
                            distance: placement::distance(
                                &pool.region,
                                &self.config.region,
                                &round_trips,
                            ),
                            in_flight,
                        }
                    }
                    None => Placed {
                        item: row,
                        load: None,
                        distance: None,
                        in_flight: u64::MAX,
                    },
                }
            })
            .collect::<Vec<_>>();
        placement::order(&mut placed);
        let tied = placement::leading_ties(&placed);
        if tied == 0 {
            return Err(WorkerSelectionPolicyError::failed("no candidates"));
        }
        let default = if context.is_preview() {
            self.ties.load(Ordering::Relaxed)
        } else {
            self.ties.fetch_add(1, Ordering::Relaxed)
        } % tied;
        let row = placed[placement::headroom(&placed, default, self.config.headroom_factor)].item;
        if !context.is_preview() {
            self.counts.add_pending(candidates[row].worker().worker_id);
        }
        Ok(row)
    }
}
