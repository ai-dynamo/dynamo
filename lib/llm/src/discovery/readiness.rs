// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The serving-readiness contract shared by the request plane and the Pool Relay.
//!
//! Equivalent [`ReadinessUnit`] inputs produce the same readiness answer. The frontend
//! builds units from committed WorkerSet membership; the relay topology projection
//! evaluates discovery independently. Under conflicting registrations, the relay can
//! remain conservative while a frontend serves its selected incumbent.

use std::collections::HashSet;

use crate::local_model::runtime_config::{
    ENCODER_HANDOFF_CONTRACT, ENCODER_HANDOFF_TRANSFER_MODE, ModelRuntimeConfig,
};
use crate::model_card::ModelDeploymentCard;
use crate::worker_type::WorkerType;

/// The backend owns both identities. Discovery compares the receipt contract on
/// every hop and the embedding transport only where embeddings actually move.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderHandoff<'a> {
    contract: Option<&'a serde_json::Value>,
    transfer_mode: Option<&'a str>,
    route_to_encoder: bool,
}

impl<'a> From<&'a ModelRuntimeConfig> for EncoderHandoff<'a> {
    fn from(config: &'a ModelRuntimeConfig) -> Self {
        Self {
            contract: config.runtime_data.get(ENCODER_HANDOFF_CONTRACT),
            transfer_mode: config
                .runtime_data
                .get(ENCODER_HANDOFF_TRANSFER_MODE)
                .and_then(serde_json::Value::as_str),
            route_to_encoder: config
                .runtime_data
                .get("encoder_result_handoff")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        }
    }
}

impl EncoderHandoff<'_> {
    pub(super) fn accepts_encoder(self, encoder: Self) -> bool {
        self.route_to_encoder
            && self.contract == encoder.contract
            && (self.contract.is_none() || self.transfer_mode.is_some())
            && self.transfer_mode == encoder.transfer_mode
    }

    pub(super) fn accepts_prefill(self, prefill: Self) -> bool {
        self.contract == prefill.contract
    }
}

/// The readiness-relevant projection of one WorkerSet: one distinct worker card shape
/// with the number of live workers behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessUnit<'a> {
    /// `None` is a legacy card without a declared worker type.
    pub worker_type: Option<WorkerType>,
    pub live_count: usize,
    pub needs: Vec<Vec<WorkerType>>,
    pub handoff: EncoderHandoff<'a>,
    pub standalone_encoder: bool,
}

impl ReadinessUnit<'_> {
    fn requires_encoder_target(&self) -> bool {
        (self.handoff.route_to_encoder && self.handoff.contract.is_some())
            || (!self.needs.is_empty()
                && self
                    .needs
                    .iter()
                    .all(|alternative| alternative.contains(&WorkerType::Encode)))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadinessEval {
    pub ready: bool,
    pub has_legacy: bool,
    pub legacy_live_workers: usize,
    pub present: HashSet<WorkerType>,
    pub missing: HashSet<WorkerType>,
    pub ambiguous: HashSet<WorkerType>,
    pub incompatible: HashSet<WorkerType>,
}

/// Project the topology implicit in a pre-`worker_type` prefill card into the
/// explicit contract used by current workers.
///
/// TODO(v1.5): Remove this projection together with the missing-role fallback
/// in `effective_worker_type` and the legacy readiness bypass after the v1.2
/// MDC compatibility window expires.
pub fn normalize_legacy_prefill_topology(card: &mut ModelDeploymentCard) {
    if card.worker_type.is_some() || !card.model_type.supports_prefill() {
        return;
    }

    card.worker_type = Some(WorkerType::Prefill);
    if card.needs.is_empty() {
        card.needs = vec![vec![WorkerType::Decode]];
    }
}

/// A required handoff needs a unique compatible encoder for each receiving
/// pipeline. Different receiver cohorts may use different encoders. Multiple
/// compatible endpoints for the same receiver remain ambiguous.
fn check_encoder_handoffs(units: &[ReadinessUnit<'_>], eval: &mut ReadinessEval) {
    let live = || units.iter().filter(|unit| unit.live_count > 0);
    let mut checked_receiver = false;
    let mut ambiguous_encoder = false;
    for receiver in live() {
        match receiver.worker_type {
            Some(WorkerType::Decode) => {
                // Routers bind to a committed group until discovery removes it,
                // even if its availability watch temporarily contains no workers.
                for role in [WorkerType::Prefill, WorkerType::Decode] {
                    if units
                        .iter()
                        .filter(|peer| peer.worker_type == Some(role))
                        .count()
                        > 1
                    {
                        eval.ambiguous.insert(role);
                    }
                }
                let prefills =
                    || live().filter(|peer| peer.worker_type == Some(WorkerType::Prefill));
                if prefills().next().is_some()
                    && !prefills().any(|p| receiver.handoff.accepts_prefill(p.handoff))
                {
                    eval.incompatible.insert(WorkerType::Prefill);
                }
            }
            // A native worker may accept receipts without routing to Dynamo E.
            // Only a receiver configured for that hop needs a compatible target.
            _ if !receiver.standalone_encoder && receiver.requires_encoder_target() => {
                checked_receiver = true;
                let mut eligible = units.iter().filter(|peer| {
                    peer.standalone_encoder && receiver.handoff.accepts_encoder(peer.handoff)
                });
                match (eligible.next(), eligible.next()) {
                    (_, Some(_)) => ambiguous_encoder = true,
                    (Some(encoder), None) if encoder.live_count > 0 => {}
                    (None, None) if eval.present.contains(&WorkerType::Encode) => {
                        eval.incompatible.insert(WorkerType::Encode);
                    }
                    _ => {
                        eval.missing.insert(WorkerType::Encode);
                    }
                }
            }
            _ => {}
        }
    }
    if checked_receiver {
        if ambiguous_encoder {
            eval.ambiguous.insert(WorkerType::Encode);
        } else {
            eval.ambiguous.remove(&WorkerType::Encode);
        }
    }
}

/// Whether a namespace's units are ready to serve traffic.
pub fn evaluate_readiness(units: &[ReadinessUnit<'_>]) -> ReadinessEval {
    let mut present: HashSet<WorkerType> = HashSet::new();
    let mut missing: HashSet<WorkerType> = HashSet::new();
    let mut has_legacy = false;
    let mut legacy_live_workers = 0usize;
    let mut has_live_worker = false;
    let mut has_handoff_contract = false;
    let mut live_units_by_type = std::collections::HashMap::new();

    // First pass: which worker types have a live worker (+ legacy detection).
    for unit in units {
        if unit.live_count > 0 {
            has_live_worker = true;
            has_handoff_contract |= unit.handoff.contract.is_some();
        }
        match unit.worker_type {
            Some(worker_type) => {
                if unit.live_count > 0 {
                    present.insert(worker_type);
                    *live_units_by_type.entry(worker_type).or_insert(0usize) += 1;
                }
            }
            // No declared worker_type → legacy card.
            None => {
                has_legacy = true;
                legacy_live_workers += unit.live_count;
            }
        }
    }

    // Legacy cards lack declared roles. Apply their liveness fallback unless
    // a live worker advertises an external handoff contract; that explicit
    // protocol requires identifiable peers even during a rolling upgrade.
    if has_legacy && !has_handoff_contract {
        return ReadinessEval {
            ready: has_live_worker,
            has_legacy,
            legacy_live_workers,
            present,
            ..Default::default()
        };
    }

    let ambiguous = live_units_by_type
        .into_iter()
        .filter_map(|(worker_type, count)| {
            (worker_type != WorkerType::Aggregated && count > 1).then_some(worker_type)
        })
        .collect::<HashSet<_>>();

    // Strict path: a registered worker type with no live worker anywhere is
    // missing; a *live* unit whose `needs` DNF is unsatisfied flags its
    // absent peers.
    for unit in units {
        let Some(worker_type) = unit.worker_type else {
            continue;
        };
        if !present.contains(&worker_type) {
            missing.insert(worker_type);
        }
        if unit.live_count == 0 || unit.needs.is_empty() {
            continue;
        }
        let satisfied = unit
            .needs
            .iter()
            .any(|alt| alt.iter().all(|t| present.contains(t)));
        if !satisfied {
            for alt in &unit.needs {
                for t in alt {
                    if !present.contains(t) {
                        missing.insert(*t);
                    }
                }
            }
        }
    }

    let mut eval = ReadinessEval {
        ready: false,
        has_legacy: false,
        legacy_live_workers,
        present,
        missing,
        ambiguous,
        incompatible: HashSet::new(),
    };
    if has_handoff_contract {
        check_encoder_handoffs(units, &mut eval);
    }
    eval.ready = has_live_worker
        && eval.missing.is_empty()
        && eval.ambiguous.is_empty()
        && eval.incompatible.is_empty();
    eval
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(
        worker_type: Option<WorkerType>,
        live_count: usize,
        needs: Vec<Vec<WorkerType>>,
    ) -> ReadinessUnit<'static> {
        ReadinessUnit {
            worker_type,
            live_count,
            needs,
            handoff: Default::default(),
            standalone_encoder: worker_type == Some(WorkerType::Encode),
        }
    }

    #[test]
    fn evaluator_matches_aggregated_pd_epd_and_dead_worker_semantics() {
        let aggregated = evaluate_readiness(&[unit(Some(WorkerType::Aggregated), 1, vec![])]);
        assert!(aggregated.ready);
        assert!(aggregated.present.contains(&WorkerType::Aggregated));

        let pd = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 1, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(pd.ready);

        let epd = evaluate_readiness(&[
            unit(
                Some(WorkerType::Encode),
                1,
                vec![
                    vec![WorkerType::Prefill, WorkerType::Decode],
                    vec![WorkerType::Aggregated],
                ],
            ),
            unit(Some(WorkerType::Prefill), 1, vec![]),
            unit(Some(WorkerType::Decode), 1, vec![]),
        ]);
        assert!(epd.ready);

        let missing_decode = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(!missing_decode.ready);
        assert!(missing_decode.missing.contains(&WorkerType::Decode));
        assert_eq!(missing_decode.missing.len(), 1);
    }

    #[test]
    fn evaluator_matches_legacy_fallback_for_legacy_only_and_mixed_inputs() {
        for units in [
            vec![unit(None, 1, vec![])],
            vec![
                unit(None, 1, vec![]),
                unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
            ],
        ] {
            let evaluation = evaluate_readiness(&units);
            assert!(evaluation.ready);
            assert!(evaluation.has_legacy);
            assert!(evaluation.missing.is_empty());
        }
        let unavailable = evaluate_readiness(&[unit(None, 0, vec![])]);
        assert!(!unavailable.ready);
        assert!(unavailable.has_legacy);
    }

    #[test]
    fn staged_receipt_advertisement_respects_routing_opt_in_and_legacy_peers() {
        use WorkerType::{Aggregated, Decode, Encode, Prefill};
        let contract = serde_json::json!({"receipt":"a"});
        let staged = |role, needs| ReadinessUnit {
            worker_type: Some(role),
            live_count: 1,
            needs,
            standalone_encoder: role == Encode,
            handoff: EncoderHandoff {
                contract: Some(&contract),
                ..Default::default()
            },
        };
        let mut pd = staged(Aggregated, vec![]);
        // Native external mode also accepts pre-staged traffic without a Dynamo E.
        assert!(evaluate_readiness(&[pd.clone()]).ready);
        pd.needs = vec![vec![Encode]];
        let mut encoder = staged(Encode, vec![vec![Aggregated]]);
        assert!(!evaluate_readiness(&[pd.clone(), encoder.clone()]).ready);
        pd.handoff.route_to_encoder = true;
        // The transport was absent on old experimental sidecar advertisements.
        assert!(!evaluate_readiness(&[pd.clone(), encoder.clone()]).ready);
        for peer in [&mut pd, &mut encoder] {
            peer.handoff.transfer_mode = Some("nixl");
        }
        assert!(evaluate_readiness(&[pd.clone(), encoder.clone()]).ready);
        let mut unavailable_duplicate = encoder.clone();
        unavailable_duplicate.live_count = 0;
        let duplicate = evaluate_readiness(&[pd, encoder, unavailable_duplicate]);
        assert!(!duplicate.ready);
        assert!(duplicate.ambiguous.contains(&Encode));

        let decode = staged(Decode, vec![vec![Prefill]]);
        let legacy = unit(None, 1, vec![]);
        let result = evaluate_readiness(&[decode.clone(), legacy]);
        assert!(!result.ready);
        assert!(result.missing.contains(&Prefill));
        let inline_prefill = unit(Some(Prefill), 1, vec![vec![Decode]]);
        let result = evaluate_readiness(&[decode, inline_prefill]);
        assert!(!result.ready);
        assert!(result.incompatible.contains(&Prefill));
    }

    #[test]
    fn evaluator_ignores_needs_of_dead_workers_but_reports_their_declared_role() {
        let evaluation = evaluate_readiness(&[
            unit(Some(WorkerType::Aggregated), 1, vec![]),
            unit(
                Some(WorkerType::Encode),
                0,
                vec![vec![WorkerType::Prefill, WorkerType::Decode]],
            ),
        ]);
        assert!(!evaluation.ready);
        assert_eq!(evaluation.missing, HashSet::from([WorkerType::Encode]),);
    }

    #[test]
    fn duplicate_live_units_of_one_role_are_ambiguous_and_not_ready() {
        let evaluation = evaluate_readiness(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 1, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(!evaluation.ready);
        assert_eq!(evaluation.ambiguous, HashSet::from([WorkerType::Prefill]));

        let scale_out = evaluate_readiness(&[
            unit(Some(WorkerType::Aggregated), 1, vec![]),
            unit(Some(WorkerType::Aggregated), 1, vec![]),
        ]);
        assert!(scale_out.ready);
        assert!(scale_out.ambiguous.is_empty());
    }

    #[test]
    fn normalization_projects_legacy_prefill_cards() {
        use crate::model_card::ModelDeploymentCard;
        use crate::model_type::ModelType;

        let mut legacy_prefill = ModelDeploymentCard::with_name_only("m");
        legacy_prefill.model_type = ModelType::Prefill;
        normalize_legacy_prefill_topology(&mut legacy_prefill);
        assert_eq!(legacy_prefill.worker_type, Some(WorkerType::Prefill));
        assert_eq!(legacy_prefill.needs, vec![vec![WorkerType::Decode]]);

        let mut plain_legacy = ModelDeploymentCard::with_name_only("m");
        normalize_legacy_prefill_topology(&mut plain_legacy);
        assert_eq!(plain_legacy.worker_type, None);
    }

    #[test]
    fn empty_topology_is_not_ready() {
        let evaluation = evaluate_readiness(&[]);
        assert!(!evaluation.ready);
        assert!(!evaluation.has_legacy);
        assert!(evaluation.present.is_empty());
        assert!(evaluation.missing.is_empty());
    }
}
