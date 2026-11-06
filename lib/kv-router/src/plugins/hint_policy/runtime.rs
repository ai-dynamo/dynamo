// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side inputs for running hint policies after booking, and composition of a booked
//! selection's KV hint from the router's fetch and its hint policies.

use std::collections::HashSet;
use std::sync::Arc;

use super::{HintActionContext, HintPolicy};
use crate::kv_hints::{KvHint, KvHintAction};
use crate::protocols::WorkerWithDpRank;

/// Action ID of the router's `kv.fetch` action. No other action may use it.
pub(crate) const KV_HINT_FETCH_ACTION_ID: &str = "a1";
/// Upper bound on the actions one composed hint carries.
const MAX_KV_HINT_ACTIONS: usize = 32;
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";

/// Request values the selection core shares with every hint policy of one booking.
pub(crate) struct ActionInputs<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) prompt_tokens: usize,
    pub(crate) block_size: u32,
}

impl<'a> ActionInputs<'a> {
    /// The context every policy sees.
    pub(crate) fn context(&self) -> HintActionContext<'a> {
        HintActionContext {
            request_id: self.request_id,
            worker: self.worker,
            prompt_tokens: self.prompt_tokens,
            block_size: self.block_size,
        }
    }
}

/// One configured hint policy of a partition that runs at booking, in configured order.
pub(crate) struct HintPolicySlot {
    pub(crate) name: Arc<str>,
    pub(crate) policy: Arc<dyn HintPolicy>,
}

/// Compose one booking's KV hint: the router's fetch, if any, then its partition's hint
/// policies' actions in configured order.
///
/// A policy's output is accepted or dropped whole. Output is dropped when
/// [`HintPolicy::hint_kv_locality`] fails, an action ID repeats, an action other than the
/// router's `kv.fetch` uses its ID, a second `kv.fetch` appears, or the hint would exceed
/// [`MAX_KV_HINT_ACTIONS`]. Dropping one policy's output keeps every other policy's.
pub(crate) fn compose_kv_hint(
    policies: &[HintPolicySlot],
    inputs: &ActionInputs<'_>,
    router_fetch: Option<KvHintAction>,
) -> Option<KvHint> {
    let context = inputs.context();
    let mut composer = Composer::new(router_fetch.as_ref());
    let mut actions: Vec<KvHintAction> = router_fetch.into_iter().collect();
    for slot in policies {
        let output = match slot.policy.hint_kv_locality(&context) {
            Ok(output) => output,
            Err(error) => {
                drop_output(slot, inputs, &error.to_string());
                continue;
            }
        };
        match composer.admit(&output) {
            Ok(()) => actions.extend(output),
            Err(reason) => drop_output(slot, inputs, &reason),
        }
    }
    (!actions.is_empty()).then(|| KvHint::new(inputs.request_id, actions))
}

/// The action IDs, fetch, and size of the hint composed so far.
struct Composer {
    action_ids: HashSet<String>,
    has_fetch: bool,
    len: usize,
}

impl Composer {
    fn new(router_fetch: Option<&KvHintAction>) -> Self {
        Self {
            action_ids: router_fetch
                .map(|fetch| fetch.action_id.clone())
                .into_iter()
                .collect(),
            has_fetch: router_fetch.is_some(),
            len: usize::from(router_fetch.is_some()),
        }
    }

    /// Add one policy's output if it composes, or explain why it does not.
    fn admit(&mut self, output: &[KvHintAction]) -> Result<(), String> {
        let mut new_ids = HashSet::with_capacity(output.len());
        let mut fetches = 0;
        for action in output {
            let is_fetch = action.action_type == KV_FETCH_ACTION_TYPE;
            if action.action_id == KV_HINT_FETCH_ACTION_ID && !is_fetch {
                return Err(format!(
                    "action ID {KV_HINT_FETCH_ACTION_ID:?} is reserved for the router's kv.fetch"
                ));
            }
            if self.action_ids.contains(&action.action_id)
                || !new_ids.insert(action.action_id.as_str())
            {
                return Err(format!("action ID {:?} is already used", action.action_id));
            }
            fetches += usize::from(is_fetch);
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

fn drop_output(policy: &HintPolicySlot, inputs: &ActionInputs<'_>, reason: &str) {
    tracing::warn!(
        request_id = inputs.request_id,
        hint_policy = %policy.name,
        reason,
        "Dropping hint-policy actions"
    );
}
