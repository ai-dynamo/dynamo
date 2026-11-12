// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hint stage: runs a booking's hint policies and composes their actions into one KV hint.
//!
//! The stage never reads an action's payload. It numbers the actions, caps the hint at
//! [`MAX_KV_HINT_ACTIONS`](crate::kv_hints::MAX_KV_HINT_ACTIONS), and treats a committed copy as
//! delivered when the policy that sends committed copies returns at least one action.

use std::sync::Arc;

use super::{Commitments, HintContext, HintPolicy};
use crate::kv_hints::{KvCopyLocations, KvCopySource, KvHint};
use crate::types::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_worker_selection::WorkerInputs;

/// One hint policy of a partition, with the declarations the stage reads. Built once per role,
/// in run order, by [`crate::resolve::resolve_role`].
pub(crate) struct HintPolicySlot {
    pub(crate) name: Arc<str>,
    pub(crate) policy: Arc<dyn HintPolicy>,
    /// Declares [`Commitments::KV_COPY`]: it sends the copy the selection committed to.
    pub(crate) consumes_kv_copy: bool,
    /// Reads [`WorkerInputs::KV_COPY`]: it receives the booked worker's copy source.
    pub(crate) reads_copy_source: bool,
}

impl HintPolicySlot {
    /// Read the policy's declarations once.
    pub(crate) fn new(name: Arc<str>, policy: Arc<dyn HintPolicy>) -> Self {
        Self {
            consumes_kv_copy: policy.commitments().contains(Commitments::KV_COPY),
            reads_copy_source: policy
                .required_worker_inputs()
                .contains(WorkerInputs::KV_COPY),
            name,
            policy,
        }
    }
}

/// A copy the booking can send, with its locations resolved from the current snapshots.
pub(crate) struct BookedCopy<'a> {
    copy: KvCopySource,
    /// `None` when the source no longer advertises a control endpoint.
    locations: Option<KvCopyLocations<'a>>,
}

impl<'a> BookedCopy<'a> {
    pub(crate) fn new(copy: KvCopySource, locations: Option<KvCopyLocations<'a>>) -> Self {
        Self { copy, locations }
    }

    pub(crate) fn copy(&self) -> &KvCopySource {
        &self.copy
    }

    pub(crate) fn locations(&self) -> Option<&KvCopyLocations<'a>> {
        self.locations.as_ref()
    }
}

/// One booking's facts, shared by every hint policy it runs.
pub(crate) struct HintInputs<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) isl_tokens: usize,
    pub(crate) block_size: u32,
    /// Whether the booked worker advertises `router_hint`. Other workers get no hint, and their
    /// policies don't run.
    pub(crate) worker_accepts_hints: bool,
    pub(crate) session_context: Option<&'a SessionContext>,
    /// The copy the selection's cost counted for the booked worker.
    pub(crate) committed_kv_copy: Option<BookedCopy<'a>>,
    /// The booked worker's best copy source, when the selection committed none.
    pub(crate) kv_copy_source: Option<BookedCopy<'a>>,
}

impl HintInputs<'_> {
    /// The context one policy sees: the copy accessors only for what it declared.
    fn context(&self, slot: &HintPolicySlot) -> HintContext<'_> {
        HintContext {
            request_id: self.request_id,
            worker: self.worker,
            isl_tokens: self.isl_tokens,
            block_size: self.block_size,
            session_context: self.session_context,
            committed_kv_copy: self
                .committed_kv_copy
                .as_ref()
                .filter(|_| slot.consumes_kv_copy),
            kv_copy_source: self
                .kv_copy_source
                .as_ref()
                .filter(|_| slot.reads_copy_source),
        }
    }
}

/// The committed copy went unsent, so the booking fails: the selection's cost assumed it.
#[derive(Debug, thiserror::Error)]
#[error("hint policy {policy:?} did not send the committed KV copy: {reason}")]
pub(crate) struct UndeliveredCommitment {
    policy: String,
    reason: String,
}

/// Compose one booking's KV hint: the policy that sends committed copies, then the other policies
/// in configured order.
///
/// Each policy's output is accepted or dropped whole, and numbered `a1`, `a2`, and so on in hint
/// order. A policy's output is dropped, with a warning, when [`HintPolicy::build_hints`] fails or
/// the output would take the hint past its action limit; the other policies' output is kept.
/// When the selection committed a copy, the policy that sends it returning an error or no action
/// fails the whole booking instead.
pub(crate) fn compose_kv_hint(
    policies: &[HintPolicySlot],
    inputs: &HintInputs<'_>,
) -> Result<Option<KvHint>, UndeliveredCommitment> {
    if !inputs.worker_accepts_hints {
        return Ok(None);
    }
    let committed = inputs.committed_kv_copy.is_some();
    if committed && !policies.iter().any(|slot| slot.consumes_kv_copy) {
        return Err(UndeliveredCommitment {
            policy: String::new(),
            reason: "no hint policy sends committed KV copies".to_string(),
        });
    }
    let mut hint = KvHint::for_request(inputs.request_id);
    // `resolve_role` already puts the consumer first; keep that order even for hand-built slots.
    let run_order = policies
        .iter()
        .filter(|slot| slot.consumes_kv_copy)
        .chain(policies.iter().filter(|slot| !slot.consumes_kv_copy));
    for slot in run_order {
        let delivering = committed && slot.consumes_kv_copy;
        let outcome = match slot.policy.build_hints(&inputs.context(slot)) {
            Ok(actions) if delivering && actions.is_empty() => {
                Err("it returned no action".to_string())
            }
            Ok(actions) => hint
                .append_numbered(actions)
                .map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let Err(reason) = outcome else {
            continue;
        };
        if delivering {
            return Err(UndeliveredCommitment {
                policy: slot.name.to_string(),
                reason,
            });
        }
        tracing::warn!(
            request_id = inputs.request_id,
            hint_policy = %slot.name,
            reason,
            "Dropping hint-policy actions"
        );
    }
    Ok((!hint.is_empty()).then_some(hint))
}
