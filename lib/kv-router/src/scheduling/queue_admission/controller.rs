// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::{
    AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
    PolicyClassAdmissionStrategy, RequestProgress, RequestProgressUpdater, WorkerEligibility,
};
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::policy_config::PolicyProfile;
use crate::scheduling::types::KvSchedulerError;

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
    pub fn new(
        profile: &PolicyProfile,
        mut strategies: PolicyClassAdmissionStrategies,
    ) -> Result<Self, KvSchedulerError> {
        if let Some(class_name) = strategies.iter().find_map(|(class_name, strategy)| {
            strategy
                .reconcile_interval()
                .is_some_and(|interval| interval.is_zero())
                .then_some(class_name)
        }) {
            return Err(KvSchedulerError::InitFailed(format!(
                "admission strategy for policy class {class_name:?} returned a zero reconcile interval"
            )));
        }

        let resolved = profile
            .classes()
            .iter()
            .map(|class| {
                let strategy = strategies.remove(&class.name);
                if let Some(config) = &class.queue_admission
                    && strategy.is_none()
                {
                    return Err(KvSchedulerError::InitFailed(format!(
                        "policy class {:?} configures admission strategy {:?}, but no implementation was registered",
                        class.name, config.strategy
                    )));
                }
                Ok(strategy)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for class_name in strategies.keys() {
            tracing::warn!(%class_name, "Ignoring admission strategy for unknown policy class");
        }
        Ok(Self {
            strategies: resolved,
            next_id: 0,
        })
    }

    pub fn has_strategy(&self, class_index: usize) -> bool {
        self.strategies[class_index].is_some()
    }

    pub fn admit(
        &mut self,
        class_index: usize,
        session_id: Option<&str>,
        context_tokens: usize,
        worker_eligibility: WorkerEligibility,
    ) -> Option<(AdmissionTicket, RequestProgressUpdater, AdmissionDecision)> {
        let strategy = self.strategies[class_index].as_mut()?;
        let id = AdmissionId::new(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let ticket = AdmissionTicket { class_index, id };
        let (progress, updater) = RequestProgress::new(context_tokens);
        let decision = strategy.admit(AdmissionRequest::with_progress(
            id,
            session_id,
            context_tokens,
            progress,
            worker_eligibility,
        ));
        Some((ticket, updater, decision))
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

    pub fn completed(
        &mut self,
        ticket: AdmissionTicket,
        context_tokens: usize,
    ) -> Vec<ClassAdmissionAction> {
        self.event(
            ticket,
            AdmissionEvent::Completed {
                id: ticket.id,
                context_tokens,
            },
        )
    }

    pub fn aborted(&mut self, ticket: AdmissionTicket) -> Vec<ClassAdmissionAction> {
        self.event(ticket, AdmissionEvent::Aborted { id: ticket.id })
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::RouterQueuePolicy;
    use crate::scheduling::policy_config::RouterPolicyConfig;

    struct ReadyStrategy;

    impl PolicyClassAdmissionStrategy for ReadyStrategy {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(super::super::WorkerPlacement::Any)
        }
    }

    struct ZeroIntervalStrategy;

    impl PolicyClassAdmissionStrategy for ZeroIntervalStrategy {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Bypass
        }

        fn reconcile_interval(&self) -> Option<Duration> {
            Some(Duration::ZERO)
        }
    }

    fn configured_profile() -> PolicyProfile {
        RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents
    queue_admission:
      type: test
    quantum: 1
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
    }

    #[test]
    fn rejects_configured_class_without_strategy() {
        let error = PolicyClassAdmissionController::new(
            &configured_profile(),
            PolicyClassAdmissionStrategies::new(),
        )
        .err()
        .unwrap();

        assert!(matches!(error, KvSchedulerError::InitFailed(message) if
            message.contains("agents") && message.contains("test")));
    }

    #[test]
    fn rejects_zero_reconcile_interval() {
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert(
            profile.default_class().name.clone(),
            Box::new(ZeroIntervalStrategy),
        );

        let error = PolicyClassAdmissionController::new(&profile, strategies)
            .err()
            .unwrap();

        assert!(matches!(error, KvSchedulerError::InitFailed(message) if
            message.contains("zero reconcile interval")));
    }

    #[test]
    fn accepts_programmatic_strategy_without_config() {
        let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert(
            profile.default_class().name.clone(),
            Box::new(ReadyStrategy),
        );

        let controller = PolicyClassAdmissionController::new(&profile, strategies).unwrap();

        assert!(controller.has_strategy(0));
    }
}
