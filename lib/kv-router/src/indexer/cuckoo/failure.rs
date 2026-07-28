// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Commit-boundary dispositions shared by the CKF producer and consumer.

/// State domain in which a failure was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkfFailureDomain {
    ProducerCore,
    ConsumerLane,
}

/// What is known about the relevant domain at the point of failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkfCommitState {
    /// This operation made no authoritative write in the relevant domain.
    KnownUnchanged,
    /// At least one authoritative write may have occurred and completion cannot be proven.
    Uncertain,
}

/// Concrete recovery or continuation selected from the commit boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkfFailureAction {
    ContinueCapacityOmission,
    ReportResourceFailure,
    RejectSource,
    FenceAndRebuildProducer,
    DeactivateAndSnapshot,
    RetrySnapshot,
}

/// A small exhaustive set of failure points where commit certainty changes the disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkfFailurePoint {
    BoundedRelocationFailure,
    PrecommitAllocationFailure,
    SourceProtocolFailure,
    PrewriteInvariantMismatch,
    ConsumerGapOrMalformedBeforeWrite,
    ConsumerWorkerFailureMidApply,
    InactiveSnapshotInstallFailure,
}

/// Decision-complete result used by fault wiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CkfFailureDisposition {
    pub domain: CkfFailureDomain,
    pub commit: CkfCommitState,
    /// Domain whose state must be recovered, if different from the observation domain.
    pub recovery_domain: Option<CkfFailureDomain>,
    pub action: CkfFailureAction,
}

impl CkfFailurePoint {
    /// Recovery is selected from the state whose commit became uncertain—not merely from the
    /// error's name.
    pub const fn disposition(self) -> CkfFailureDisposition {
        use CkfCommitState::{KnownUnchanged, Uncertain};
        use CkfFailureAction::{
            ContinueCapacityOmission, DeactivateAndSnapshot, FenceAndRebuildProducer, RejectSource,
            ReportResourceFailure, RetrySnapshot,
        };
        use CkfFailureDomain::{ConsumerLane, ProducerCore};

        match self {
            Self::BoundedRelocationFailure => CkfFailureDisposition {
                domain: ProducerCore,
                commit: KnownUnchanged,
                recovery_domain: None,
                action: ContinueCapacityOmission,
            },
            Self::PrecommitAllocationFailure => CkfFailureDisposition {
                domain: ProducerCore,
                commit: KnownUnchanged,
                recovery_domain: None,
                action: ReportResourceFailure,
            },
            Self::SourceProtocolFailure => CkfFailureDisposition {
                domain: ProducerCore,
                commit: KnownUnchanged,
                recovery_domain: None,
                action: RejectSource,
            },
            Self::PrewriteInvariantMismatch => CkfFailureDisposition {
                domain: ProducerCore,
                commit: KnownUnchanged,
                recovery_domain: Some(ProducerCore),
                action: FenceAndRebuildProducer,
            },
            Self::ConsumerGapOrMalformedBeforeWrite => CkfFailureDisposition {
                domain: ConsumerLane,
                commit: KnownUnchanged,
                recovery_domain: Some(ConsumerLane),
                action: DeactivateAndSnapshot,
            },
            Self::ConsumerWorkerFailureMidApply => CkfFailureDisposition {
                domain: ConsumerLane,
                commit: Uncertain,
                recovery_domain: Some(ConsumerLane),
                action: DeactivateAndSnapshot,
            },
            Self::InactiveSnapshotInstallFailure => CkfFailureDisposition {
                domain: ConsumerLane,
                commit: KnownUnchanged,
                recovery_domain: Some(ConsumerLane),
                action: RetrySnapshot,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispositions_target_only_the_domain_requiring_recovery() {
        let capacity = CkfFailurePoint::BoundedRelocationFailure.disposition();
        assert_eq!(capacity.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(capacity.recovery_domain, None);
        assert_eq!(capacity.action, CkfFailureAction::ContinueCapacityOmission);

        let consumer = CkfFailurePoint::ConsumerWorkerFailureMidApply.disposition();
        assert_eq!(consumer.commit, CkfCommitState::Uncertain);
        assert_eq!(
            consumer.recovery_domain,
            Some(CkfFailureDomain::ConsumerLane)
        );
    }

    #[test]
    fn resource_and_capacity_failures_are_distinct() {
        let capacity = CkfFailurePoint::BoundedRelocationFailure.disposition();
        let allocation = CkfFailurePoint::PrecommitAllocationFailure.disposition();
        assert_ne!(capacity.action, allocation.action);
        assert_eq!(allocation.commit, CkfCommitState::KnownUnchanged);
    }
}
