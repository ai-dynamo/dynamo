// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's policies as router core plugins.
//!
//! Each policy is one [`WorkerPicker`] and no scorers, registered in a
//! [`RouterPluginRegistry`] under its name, which [`policy_config`] then
//! selects for the aggregated worker pool. The core creates one picker per
//! partition, so a picker's cursor is already per model.
//!
//! The core's built-in selector weighs KV overlap and projected load; it has
//! no plain "fewest active requests" picker, so [`LeastRequestsPicker`] is
//! written here.

use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{RoutingPartitionRef, WorkerId, WorkerType};

use super::CorePolicy;

impl CorePolicy {
    /// The name the policy is registered and selected under.
    pub fn name(self) -> &'static str {
        match self {
            Self::RoundRobin => "global-round-robin",
            Self::LeastRequests => "global-least-requests",
        }
    }
}

/// A registry with every policy of [`CorePolicy`].
pub fn registry() -> Result<RouterPluginRegistry, WorkerSelectionPolicyRegistryError> {
    let mut registry = RouterPluginRegistry::default();
    registry.register_worker_selection(
        CorePolicy::RoundRobin.name(),
        provider(RoundRobinPicker::default),
    )?;
    registry.register_worker_selection(
        CorePolicy::LeastRequests.name(),
        provider(LeastRequestsPicker::default),
    )?;
    Ok(registry)
}

/// The router policy configuration that selects `policy` for the
/// aggregated pool, as the core reads it from YAML.
pub fn policy_config(policy: CorePolicy) -> String {
    let name = policy.name();
    format!(
        "worker_selection:\n  aggregated: {name}\n  instances:\n    - name: {name}\n      type: {name}\n"
    )
}

/// A provider for a policy of one picker and no scorers, with a fresh
/// picker for each partition. The policies take no parameters.
fn provider<P>(picker: fn() -> P) -> WorkerSelectionPolicyProvider
where
    P: WorkerPicker + 'static,
{
    Arc::new(
        move |_parameters: &WorkerSelectionPolicyParameters| -> Result<
            WorkerSelectionPolicyFactory,
            WorkerSelectionPolicyProviderError,
        > {
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

/// Picks the candidates in turn, in ascending worker id, so each gets an
/// equal share. When the candidates change, the cursor carries on over the
/// new list, as [`crate::RoundRobinSelector`]'s does.
#[derive(Debug, Default)]
pub struct RoundRobinPicker {
    turn: usize,
}

impl WorkerPicker for RoundRobinPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let rows = input.candidates();
        let turn = self.turn;
        self.turn = self.turn.wrapping_add(1);
        nth_by_worker(rows.iter().enumerate(), turn)
    }
}

/// Picks the candidate with the fewest active requests, counting the core's
/// own bookings, as [`crate::LeastInFlightSelector`] counts this process's
/// requests. Ties rotate among the tied candidates.
#[derive(Debug, Default)]
pub struct LeastRequestsPicker {
    ties: usize,
}

impl WorkerPicker for LeastRequestsPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let rows = input.candidates();
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load inputs were not supplied"))?;
        let requests = |row: usize| loads.get(row).map_or(0, |load| load.active_requests());
        let fewest = (0..rows.len())
            .map(requests)
            .min()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))?;
        let tied = rows
            .iter()
            .enumerate()
            .filter(|(row, _)| requests(*row) == fewest);
        let turn = self.ties;
        self.ties = self.ties.wrapping_add(1);
        nth_by_worker(tied, turn)
    }
}

/// The row that is `turn`th, modulo their number, once `rows` are in
/// ascending worker id: a stable order whatever order the core lists them
/// in.
fn nth_by_worker<'a>(
    rows: impl IntoIterator<Item = (usize, &'a ScoredWorkerCandidate)>,
    turn: usize,
) -> Result<usize, WorkerSelectionPolicyError> {
    let mut rows: Vec<(usize, WorkerId)> = rows
        .into_iter()
        .map(|(row, candidate)| (row, candidate.worker().worker_id))
        .collect();
    if rows.is_empty() {
        return Err(WorkerSelectionPolicyError::failed("no candidates"));
    }
    rows.sort_unstable_by_key(|(_, worker)| *worker);
    let (row, _) = rows[turn % rows.len()];
    Ok(row)
}
