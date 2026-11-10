// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the router does for one role, worked out once from its plugins' declarations.
//!
//! [`resolve_role`] reads every configured plugin's declarations when the router is built for a
//! role, checks that they fit together, and records the result in a [`RouterResponsibilities`].
//! Each part of the router reads one field instead of deriving its piece from the plugins.

use super::hint_policy::RoleHintPolicies;
use super::hint_policy::runtime::HintPolicySlot;
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::plugins::worker_selection::{WorkerInputs, WorkerSelectionPolicy};

/// What the router does for one role, worked out once from its plugins' declarations.
pub struct RouterResponsibilities {
    /// Union of the worker-selection policy's inputs and those of the configured hint policies.
    /// Decides which resources start: the KV index and its event subscription, the shared cache,
    /// load tracking. Default hint policies add nothing.
    pub(crate) prepare_inputs: WorkerInputs,
    /// Some scorer reads [`WorkerInputs::KV_TRANSFER`]: compute a copy offer for every candidate
    /// and keep the request's block chain, for previews as well as bookings.
    pub(crate) commit_kv_copy: bool,
    /// Compute the booked worker's copy source at booking.
    pub(crate) compute_booked_copy_source: bool,
    /// The role's hint policies in run order: the one that sends committed copies first, then
    /// configured order.
    pub(crate) run_hint_policies: Vec<HintPolicySlot>,
}

impl RouterResponsibilities {
    /// The worker inputs whose resources the router starts for this role.
    pub fn prepare_inputs(&self) -> WorkerInputs {
        self.prepare_inputs
    }

    /// Whether every selection computes copy offers for its candidates.
    pub fn commit_kv_copy(&self) -> bool {
        self.commit_kv_copy
    }

    /// Whether bookings compute the booked worker's copy source.
    pub fn compute_booked_copy_source(&self) -> bool {
        self.compute_booked_copy_source
    }

    /// The role's hint-policy names, in run order.
    pub fn hint_policy_names(&self) -> impl Iterator<Item = &str> {
        self.run_hint_policies.iter().map(|slot| slot.name.as_ref())
    }
}

impl std::fmt::Debug for RouterResponsibilities {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RouterResponsibilities")
            .field("prepare_inputs", &self.prepare_inputs)
            .field("commit_kv_copy", &self.commit_kv_copy)
            .field(
                "compute_booked_copy_source",
                &self.compute_booked_copy_source,
            )
            .field(
                "run_hint_policies",
                &self.hint_policy_names().collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// A role's plugins don't fit together. Each message names the fix.
#[derive(Debug, thiserror::Error)]
pub enum PluginConfigError {
    #[error(
        "hint policies {first:?} and {second:?} for {role} both send committed KV copies; keep one of them"
    )]
    ConflictingCopyConsumers {
        role: &'static str,
        first: String,
        second: String,
    },
    #[error(
        "hint policy {policy:?} for {role} reads KV copy sources, which need a local, event-driven KV index without a remote or predicted-TTL indexer; enable KV events or remove the policy"
    )]
    CopySourcesUnavailable { role: &'static str, policy: String },
    #[error(
        "hint policy {policy:?} for {role} declares {inputs}, which hint policies can't read; a hint policy may declare only WorkerInputs::KV_TRANSFER"
    )]
    UnreadableHintInputs {
        role: &'static str,
        policy: String,
        inputs: String,
    },
}

/// Work out the router's responsibilities for one role, or reject the config with a message that
/// names the fix.
///
/// Rejects the config when more than one hint policy declares
/// [`Commitments::KV_COPY`](crate::plugins::worker_selection::Commitments::KV_COPY), or when a
/// configured hint policy reads [`WorkerInputs::KV_TRANSFER`] but the KV index can't keep the
/// request's block chain. Default hint policies are passive: they add no inputs, never move the
/// role onto the KV path, and do without copy data where the index can't provide it.
///
/// Also rejects any hint policy, default or configured, that declares an input other than
/// [`WorkerInputs::KV_TRANSFER`]: [`HintContext`](super::hint_policy::HintContext) can't serve
/// it, so the router would start resources the policy never reads.
pub fn resolve_role(
    role: WorkerType,
    selection: &WorkerSelectionPolicy,
    hint_policies: RoleHintPolicies,
    config: &KvRouterConfig,
) -> Result<RouterResponsibilities, PluginConfigError> {
    let role_label = role.as_str();
    let (policies, defaults) = hint_policies.into_parts();
    let selection_inputs = selection.input_requirements();
    let mut prepare_inputs = selection_inputs;
    let mut consumers: Vec<HintPolicySlot> = Vec::new();
    let mut others = Vec::with_capacity(policies.len());
    for (name, policy) in policies {
        let slot = HintPolicySlot::new(name, policy);
        let inputs = slot.policy.required_worker_inputs();
        let unreadable = inputs.without(WorkerInputs::KV_TRANSFER);
        if unreadable != WorkerInputs::NONE {
            return Err(PluginConfigError::UnreadableHintInputs {
                role: role_label,
                policy: slot.name.to_string(),
                inputs: input_names(unreadable),
            });
        }
        if !defaults {
            prepare_inputs = prepare_inputs | inputs;
            if slot.reads_copy_source && !config.supports_kv_transfer_chain_retention() {
                return Err(PluginConfigError::CopySourcesUnavailable {
                    role: role_label,
                    policy: slot.name.to_string(),
                });
            }
        }
        if slot.consumes_kv_copy {
            if let Some(first) = consumers.first() {
                return Err(PluginConfigError::ConflictingCopyConsumers {
                    role: role_label,
                    first: first.name.to_string(),
                    second: slot.name.to_string(),
                });
            }
            consumers.push(slot);
        } else {
            others.push(slot);
        }
    }
    consumers.append(&mut others);
    Ok(RouterResponsibilities {
        prepare_inputs,
        commit_kv_copy: selection_inputs.contains(WorkerInputs::KV_TRANSFER),
        // The router sends its own fetch from the booked worker's copy source for every booking,
        // so every role computes it; hint policies that read KV_TRANSFER read the same source.
        compute_booked_copy_source: true,
        run_hint_policies: consumers,
    })
}

/// `inputs` as the constants that name them, for error messages.
fn input_names(inputs: WorkerInputs) -> String {
    [
        (WorkerInputs::CACHE, "CACHE"),
        (WorkerInputs::LOAD, "LOAD"),
        (WorkerInputs::PREFERRED_TAINT, "PREFERRED_TAINT"),
        (WorkerInputs::ACTIVE_REQUESTS, "ACTIVE_REQUESTS"),
    ]
    .into_iter()
    .filter(|(input, _)| inputs.contains(*input))
    .map(|(_, name)| format!("WorkerInputs::{name}"))
    .collect::<Vec<_>>()
    .join(" | ")
}
