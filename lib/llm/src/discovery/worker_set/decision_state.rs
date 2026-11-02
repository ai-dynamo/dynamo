// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::{DecisionAdmission, DecisionAvailability};

pub(crate) enum DecisionState<'a> {
    Absent,
    Unqualified,
    Pending,
    Failed,
    Qualified {
        profile: DecisionExecutionProfile,
        capacity: usize,
        resources: &'a Arc<DecisionResources>,
        live_workers: usize,
    },
}

impl DecisionState<'_> {
    pub(crate) fn is_present(&self) -> bool {
        !matches!(self, Self::Absent)
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.live_workers() > 0
    }

    pub(crate) fn is_qualified(&self) -> bool {
        matches!(self, Self::Qualified { .. })
    }

    pub(crate) fn live_workers(&self) -> usize {
        match self {
            Self::Qualified { live_workers, .. } => *live_workers,
            _ => 0,
        }
    }
}

impl WorkerSet {
    pub(crate) fn decision_state(&self) -> DecisionState<'_> {
        let state = self
            .decision
            .as_ref()
            .and_then(|surface| surface.fleet.current());
        self.decision_state_for(state.as_deref())
    }

    pub(crate) fn decision_state_for(
        &self,
        availability: Option<&DecisionAvailability>,
    ) -> DecisionState<'_> {
        let Some(surface) = &self.decision else {
            return DecisionState::Absent;
        };
        let admission = match availability {
            Some(DecisionAvailability::Qualified(admission)) => admission,
            Some(DecisionAvailability::Unadvertised) => return DecisionState::Unqualified,
            Some(DecisionAvailability::Pending) | None => return DecisionState::Pending,
            Some(DecisionAvailability::Failed) => return DecisionState::Failed,
        };
        let (profile, capacity) = admission.qualification;
        let resources = &surface.resources;
        if !resources.supports(profile.kind()) {
            return DecisionState::Unqualified;
        }
        DecisionState::Qualified {
            profile,
            capacity,
            resources,
            live_workers: self.live_decision_workers(admission),
        }
    }

    fn live_decision_workers(&self, admission: &DecisionAdmission) -> usize {
        let Some(cancellation) = self.lifecycle_cancellation() else {
            return 0;
        };
        if cancellation.is_cancelled() || admission.ids.is_empty() {
            return 0;
        }
        let Some(available) = &self.instance_count_rx else {
            return 0;
        };
        if available.has_changed().is_err() {
            return 0;
        }
        let Some(admitted) = self.decision_admissions() else {
            return 0;
        };
        if admitted.has_changed().is_err() {
            return 0;
        }
        let admitted = admitted.borrow();
        available
            .borrow()
            .iter()
            .filter(|id| admitted.contains(id) && admission.ids.contains(id))
            .count()
    }
}
