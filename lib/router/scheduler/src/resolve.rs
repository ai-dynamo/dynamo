// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the router does for a role, from its worker-selection policy's declarations.
//!
//! Resolved once per partition when the policy is built. The policy's filters, scorers, and
//! picker say which worker inputs selection needs; its hint emitter, if any, says whether a
//! booking also needs the booked worker's copy source. Everything the router then does for the
//! role, from which indexes it keeps to whether it computes copy sources, follows from these.

use std::sync::Arc;

use dynamo_router_protocol::worker_type::WorkerType;
use dynamo_worker_selection::{HintEmitter, WorkerInputs, WorkerSelectionPolicy};

/// The router's responsibilities for one role, resolved from its policy.
#[derive(Clone)]
pub(crate) struct RouterResponsibilities {
    /// The optional worker inputs whose resources the router prepares: the policy's own, plus
    /// what its hint emitter reads at booking.
    pub(crate) prepare_inputs: WorkerInputs,
    /// The policy's scorers read copy offers, so every selection, preview or booking, computes
    /// copy sources and the booked worker's copy is the one its score credited.
    pub(crate) scores_kv_copy: bool,
    /// The hint emitter reads the booked worker's copy source, so bookings compute it even when
    /// scoring did not.
    pub(crate) books_kv_copy: bool,
    /// The emitter the router calls after each booking; `None` sends no hints.
    pub(crate) hint_emitter: Option<Arc<dyn HintEmitter>>,
}

impl RouterResponsibilities {
    pub(crate) fn prepare_inputs(&self) -> WorkerInputs {
        self.prepare_inputs
    }

    /// Whether bookings run a hint emitter.
    pub(crate) fn emits_hints(&self) -> bool {
        self.hint_emitter.is_some()
    }
}

impl std::fmt::Debug for RouterResponsibilities {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RouterResponsibilities")
            .field("prepare_inputs", &self.prepare_inputs)
            .field("scores_kv_copy", &self.scores_kv_copy)
            .field("books_kv_copy", &self.books_kv_copy)
            .field("emits_hints", &self.emits_hints())
            .finish()
    }
}

/// A worker-selection policy whose declarations the router cannot serve for its role.
#[derive(Debug, thiserror::Error)]
pub enum PluginConfigError {
    #[error(
        "the worker-selection policy for {role} reads KV copy offers, which need a local, event-driven KV index without a remote or predicted-TTL indexer; enable KV events or select a policy without copy credit"
    )]
    CopyOffersUnavailable { role: &'static str },
    #[error(
        "the worker-selection policy for {role} reads KV copy offers, so the worker it books is committed to a copy, but its hints do not read the copy to send it; attach a hint emitter that declares WorkerInputs::KV_COPY, such as kv_fetch"
    )]
    UnsentCopies { role: &'static str },
    #[error(
        "the hint emitter of the worker-selection policy for {role} declares {inputs}, which hints can't read; a hint emitter may declare only WorkerInputs::KV_COPY"
    )]
    UnreadableHintInputs { role: &'static str, inputs: String },
}

/// Resolve the router's responsibilities for `role` from `selection`.
///
/// `kv_chain_retention` is whether the partition's KV index keeps the matched block chain, which
/// copy sources need. A policy that scores copy offers without it is rejected; one whose emitter
/// merely reads the booked worker's copy source runs without copies, since that is best effort.
pub(crate) fn resolve_role(
    role: WorkerType,
    selection: &WorkerSelectionPolicy,
    kv_chain_retention: bool,
) -> Result<RouterResponsibilities, PluginConfigError> {
    let role_label = role.as_str();
    let selection_inputs = selection.required_worker_inputs();
    let hint_inputs = selection.hint_inputs();
    let unreadable = hint_inputs.without(WorkerInputs::KV_COPY);
    if unreadable != WorkerInputs::NONE {
        return Err(PluginConfigError::UnreadableHintInputs {
            role: role_label,
            inputs: input_names(unreadable),
        });
    }
    let scores_kv_copy = selection_inputs.contains(WorkerInputs::KV_COPY);
    let books_kv_copy = hint_inputs.contains(WorkerInputs::KV_COPY);
    if scores_kv_copy && !kv_chain_retention {
        return Err(PluginConfigError::CopyOffersUnavailable { role: role_label });
    }
    if scores_kv_copy && !books_kv_copy {
        return Err(PluginConfigError::UnsentCopies { role: role_label });
    }
    Ok(RouterResponsibilities {
        prepare_inputs: selection_inputs | hint_inputs,
        scores_kv_copy,
        books_kv_copy,
        hint_emitter: selection.hint_emitter().cloned(),
    })
}

/// Check that `selection` fits `role`, and return the worker inputs whose resources the router
/// must prepare for it: the policy's own, plus what its hint emitter reads at booking. Hosts
/// call this when they build a router, so a bad policy fails before it serves.
pub fn check_role(
    role: WorkerType,
    selection: &WorkerSelectionPolicy,
    kv_chain_retention: bool,
) -> Result<WorkerInputs, PluginConfigError> {
    resolve_role(role, selection, kv_chain_retention).map(|resolved| resolved.prepare_inputs())
}

fn input_names(inputs: WorkerInputs) -> String {
    [
        (WorkerInputs::CACHE, "CACHE"),
        (WorkerInputs::LOAD, "LOAD"),
        (WorkerInputs::PREFERRED_TAINT, "PREFERRED_TAINT"),
        (WorkerInputs::ACTIVE_REQUESTS, "ACTIVE_REQUESTS"),
        (WorkerInputs::RESIDENT_BLOCKS, "RESIDENT_BLOCKS"),
    ]
    .into_iter()
    .filter(|(input, _)| inputs.contains(*input))
    .map(|(_, name)| name)
    .collect::<Vec<_>>()
    .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_hints::KvHintAction;
    use dynamo_worker_selection::{
        HintContext, HintError, SelectionConfig, WorkerCandidates, WorkerInputView, WorkerPicker,
        WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
    };

    struct Declaring(WorkerInputs);

    impl WorkerScorer for Declaring {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.0
        }

        fn score(
            &self,
            _context: &WorkerSelectionContext<'_>,
            _candidates: WorkerCandidates<'_>,
            _costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            Ok(())
        }
    }

    struct FirstPicker;

    impl WorkerPicker for FirstPicker {
        fn pick(
            &self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
    }

    struct Emitter(WorkerInputs);

    impl HintEmitter for Emitter {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.0
        }

        fn emit(&self, _context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintError> {
            Ok(Vec::new())
        }
    }

    fn selection(inputs: WorkerInputs) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            SelectionConfig::default(),
            "test",
            vec![Box::new(Declaring(inputs))],
            Box::new(FirstPicker),
        )
    }

    #[test]
    fn a_policy_without_hints_prepares_only_its_own_inputs() {
        let resolved =
            resolve_role(WorkerType::Prefill, &selection(WorkerInputs::LOAD), true).unwrap();
        assert_eq!(resolved.prepare_inputs(), WorkerInputs::LOAD);
        assert!(!resolved.scores_kv_copy && !resolved.books_kv_copy);
        assert!(!resolved.emits_hints());
    }

    #[test]
    fn an_emitter_that_reads_copies_adds_them_to_the_prepared_inputs() {
        let policy =
            selection(WorkerInputs::LOAD).with_hints(Arc::new(Emitter(WorkerInputs::KV_COPY)));
        let resolved = resolve_role(WorkerType::Prefill, &policy, true).unwrap();
        assert_eq!(
            resolved.prepare_inputs(),
            WorkerInputs::LOAD | WorkerInputs::KV_COPY
        );
        assert!(!resolved.scores_kv_copy && resolved.books_kv_copy);
        assert!(resolved.emits_hints());
        // Best effort: without chain retention the emitter runs and reads no copy.
        resolve_role(WorkerType::Prefill, &policy, false).unwrap();
    }

    #[test]
    fn a_scorer_that_reads_copy_offers_needs_an_emitter_that_sends_them() {
        let committing = || selection(WorkerInputs::CACHE | WorkerInputs::KV_COPY);
        let paired = committing().with_hints(Arc::new(Emitter(WorkerInputs::KV_COPY)));
        let resolved = resolve_role(WorkerType::Prefill, &paired, true).unwrap();
        assert!(resolved.scores_kv_copy && resolved.books_kv_copy);

        for policy in [
            committing(),
            committing().with_hints(Arc::new(Emitter(WorkerInputs::NONE))),
        ] {
            let error = resolve_role(WorkerType::Prefill, &policy, true).unwrap_err();
            assert!(
                matches!(error, PluginConfigError::UnsentCopies { role: "prefill" }),
                "{error}"
            );
        }

        // Copy offers need an index that keeps the block chain.
        let paired = committing().with_hints(Arc::new(Emitter(WorkerInputs::KV_COPY)));
        let error = resolve_role(WorkerType::Prefill, &paired, false).unwrap_err();
        assert!(
            matches!(error, PluginConfigError::CopyOffersUnavailable { .. }),
            "{error}"
        );
    }

    #[test]
    fn an_emitter_may_declare_only_copies() {
        let policy = selection(WorkerInputs::LOAD).with_hints(Arc::new(Emitter(
            WorkerInputs::CACHE | WorkerInputs::KV_COPY | WorkerInputs::ACTIVE_REQUESTS,
        )));
        let error = resolve_role(WorkerType::Decode, &policy, true).unwrap_err();
        assert!(
            matches!(
                &error,
                PluginConfigError::UnreadableHintInputs { role: "decode", inputs }
                    if inputs == "CACHE | ACTIVE_REQUESTS"
            ),
            "{error}"
        );
        assert_eq!(
            check_role(WorkerType::Decode, &selection(WorkerInputs::LOAD), true).unwrap(),
            WorkerInputs::LOAD
        );
    }
}
