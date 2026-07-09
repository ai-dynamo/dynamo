// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::{
    AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
    PolicyClassAdmissionStrategy, WorkerEligibility,
};
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::policy_config::PolicyProfile;

pub type PolicyClassAdmissionStrategies = HashMap<String, Box<dyn PolicyClassAdmissionStrategy>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionTicket {
    pub class_index: usize,
    pub id: AdmissionId,
}

pub(crate) struct ClassAdmissionAction {
    pub class_index: usize,
    pub action: AdmissionAction,
}

pub(crate) struct PolicyClassAdmissionController {
    strategies: Vec<Option<Box<dyn PolicyClassAdmissionStrategy>>>,
    next_id: u64,
}

impl PolicyClassAdmissionController {
    pub fn new(profile: &PolicyProfile, mut strategies: PolicyClassAdmissionStrategies) -> Self {
        let resolved = profile
            .classes()
            .iter()
            .map(|class| strategies.remove(&class.name))
            .collect();
        for class_name in strategies.keys() {
            tracing::warn!(%class_name, "Ignoring admission strategy for unknown policy class");
        }
        Self {
            strategies: resolved,
            next_id: 0,
        }
    }

    pub fn has_strategy(&self, class_index: usize) -> bool {
        self.strategies[class_index].is_some()
    }

    pub fn admit(
        &mut self,
        class_index: usize,
        session_id: Option<&str>,
        input_tokens: usize,
        worker_eligibility: WorkerEligibility,
    ) -> Option<(AdmissionTicket, AdmissionDecision)> {
        let strategy = self.strategies[class_index].as_mut()?;
        let id = AdmissionId::new(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let ticket = AdmissionTicket { class_index, id };
        let decision = strategy.admit(AdmissionRequest::new(
            id,
            session_id,
            input_tokens,
            worker_eligibility,
        ));
        Some((ticket, decision))
    }

    pub fn dispatched(
        &mut self,
        ticket: AdmissionTicket,
        worker: WorkerWithDpRank,
    ) -> Vec<ClassAdmissionAction> {
        self.event(
            ticket,
            AdmissionEvent::Dispatched {
                id: ticket.id,
                worker,
            },
        )
    }

    pub fn finished(
        &mut self,
        ticket: AdmissionTicket,
        total_tokens: usize,
    ) -> Vec<ClassAdmissionAction> {
        self.event(
            ticket,
            AdmissionEvent::Finished {
                id: ticket.id,
                total_tokens,
            },
        )
    }

    pub fn progress(
        &mut self,
        ticket: AdmissionTicket,
        output_tokens: usize,
    ) -> Vec<ClassAdmissionAction> {
        self.event(
            ticket,
            AdmissionEvent::OutputTokens {
                id: ticket.id,
                cumulative: output_tokens,
            },
        )
    }

    pub fn reconcile(&mut self) -> Vec<ClassAdmissionAction> {
        let mut actions = Vec::new();
        for (class_index, strategy) in self.strategies.iter_mut().enumerate() {
            let Some(strategy) = strategy else {
                continue;
            };
            actions.extend(
                strategy
                    .on_event(AdmissionEvent::Reconcile)
                    .into_iter()
                    .map(|action| ClassAdmissionAction {
                        class_index,
                        action,
                    }),
            );
        }
        actions
    }

    fn event(
        &mut self,
        ticket: AdmissionTicket,
        event: AdmissionEvent,
    ) -> Vec<ClassAdmissionAction> {
        let Some(strategy) = self.strategies[ticket.class_index].as_mut() else {
            return Vec::new();
        };
        strategy
            .on_event(event)
            .into_iter()
            .map(|action| ClassAdmissionAction {
                class_index: ticket.class_index,
                action,
            })
            .collect()
    }
}
