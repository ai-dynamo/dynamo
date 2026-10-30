// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition of a booked selection's KV hint from its hint policies.

use std::collections::HashSet;

use super::*;
use crate::kv_hints::KV_FETCH_ACTION_TYPE;
use crate::plugins::hint_policy::runtime::ActionInputs;
use crate::plugins::hint_policy::{HintPolicy, HintRequirements};

/// Upper bound on the actions one composed hint carries.
const MAX_KV_HINT_ACTIONS: usize = 32;

/// Whether any worker in the published partition snapshot advertises a router
/// hint worker type (worker-level metadata, so one rank suffices).
pub(super) fn hint_capable_partition(configs: &HashMap<WorkerId, SelectionWorkerConfig>) -> bool {
    configs.values().any(|config| {
        config
            .router_hint_worker_type
            .as_deref()
            .is_some_and(|worker_type| !worker_type.is_empty())
    })
}

/// One configured hint policy of a partition that runs at booking, in configured order.
pub(super) struct HintPolicySlot {
    pub(super) name: Arc<str>,
    pub(super) requirements: HintRequirements,
    pub(super) policy: Arc<dyn HintPolicy>,
}

/// Compose one booking's KV hint from its partition's hint policies' actions, in configured
/// order.
///
/// A policy's output is accepted or dropped whole. Output is dropped when
/// [`HintPolicy::hint_kv_locality`] fails, an action ID repeats, a second `kv.fetch` appears, or
/// the hint would exceed [`MAX_KV_HINT_ACTIONS`]. Dropping one policy's output keeps every other
/// policy's.
///
/// When the selection committed the booked worker to a fetch, the policy that delivers committed
/// fetches is composed first, so no other output can crowd it out, and its output must contain
/// that fetch. Otherwise this fails, and the caller releases the booking.
pub(super) fn compose_kv_hint(
    policies: &[HintPolicySlot],
    inputs: &ActionInputs<'_>,
) -> Result<Option<KvHint>, SelectionError> {
    let committed = inputs.committed_kv_fetch.is_some();
    let delivers =
        |slot: &&HintPolicySlot| committed && slot.requirements.delivers_committed_kv_fetch();
    // Partition setup pairs a committing policy with exactly one delivering hint policy.
    if committed && !policies.iter().any(|slot| delivers(&slot)) {
        return Err(SelectionError::Internal(
            "the selection committed a KV fetch, but no hint policy delivers it".to_string(),
        ));
    }
    let mut composer = Composer::default();
    let mut actions = Vec::new();
    for slot in policies
        .iter()
        .filter(delivers)
        .chain(policies.iter().filter(|slot| !delivers(slot)))
    {
        let delivering = delivers(&slot);
        let context = inputs.context(slot.requirements);
        let output = match slot.policy.hint_kv_locality(&context) {
            Ok(output) => output,
            Err(error) => {
                reject(slot, delivering, inputs, &error.to_string())?;
                continue;
            }
        };
        if delivering {
            let expected = inputs
                .committed_kv_fetch
                .as_ref()
                .and_then(|fetch| context.kv_fetch_action(fetch).ok());
            if expected.is_none_or(|expected| !output.contains(&expected)) {
                reject(
                    slot,
                    true,
                    inputs,
                    "its output omits the committed kv.fetch",
                )?;
            }
        }
        match composer.admit(&output) {
            Ok(()) => actions.extend(output),
            Err(reason) => reject(slot, delivering, inputs, &reason)?,
        }
    }
    Ok((!actions.is_empty()).then(|| KvHint::new(inputs.request_id, actions)))
}

/// The action IDs, fetch, and size of the hint composed so far.
#[derive(Default)]
struct Composer {
    action_ids: HashSet<String>,
    has_fetch: bool,
    len: usize,
}

impl Composer {
    /// Add one policy's output if it composes, or explain why it does not.
    fn admit(&mut self, output: &[KvHintAction]) -> Result<(), String> {
        let mut new_ids = HashSet::with_capacity(output.len());
        let mut fetches = 0;
        for action in output {
            if self.action_ids.contains(&action.action_id)
                || !new_ids.insert(action.action_id.as_str())
            {
                return Err(format!("action ID {:?} is already used", action.action_id));
            }
            fetches += usize::from(action.action_type == KV_FETCH_ACTION_TYPE);
        }
        if fetches > usize::from(!self.has_fetch) {
            return Err("a hint carries at most one kv.fetch".to_string());
        }
        if self.len + output.len() > MAX_KV_HINT_ACTIONS {
            return Err(format!(
                "a hint carries at most {MAX_KV_HINT_ACTIONS} actions"
            ));
        }
        self.has_fetch |= fetches > 0;
        self.len += output.len();
        self.action_ids
            .extend(output.iter().map(|action| action.action_id.clone()));
        Ok(())
    }
}

/// Drop a policy's output with a warning, or fail the booking when the policy had to deliver the
/// committed fetch.
fn reject(
    policy: &HintPolicySlot,
    delivering: bool,
    inputs: &ActionInputs<'_>,
    reason: &str,
) -> Result<(), SelectionError> {
    if delivering {
        return Err(SelectionError::Internal(format!(
            "hint policy {:?} could not deliver the committed KV fetch: {reason}",
            policy.name
        )));
    }
    tracing::warn!(
        request_id = inputs.request_id,
        hint_policy = %policy.name,
        reason,
        "Dropping hint-policy actions"
    );
    Ok(())
}
