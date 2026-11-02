// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    model_card::{ModelDeploymentCard, decision_profile::DecisionKinds},
    worker_type::WorkerType,
};
use dynamo_runtime::pipeline::RouterMode;

#[derive(Clone, Copy)]
pub(crate) struct DecisionCapabilities {
    pub(crate) kinds: DecisionKinds,
    pub(crate) token_possible: bool,
    pub(crate) mode: RouterMode,
}

impl Default for DecisionCapabilities {
    fn default() -> Self {
        Self {
            kinds: DecisionKinds::default(),
            token_possible: false,
            mode: RouterMode::RoundRobin,
        }
    }
}

impl DecisionCapabilities {
    pub(crate) fn for_card(card: &ModelDeploymentCard, mode: RouterMode, enabled: bool) -> Self {
        let eligible = if enabled && card.worker_type == Some(WorkerType::Aggregated) {
            DecisionKinds::eligible(card)
        } else {
            DecisionKinds::default()
        };
        let native = matches!(
            mode,
            RouterMode::RoundRobin
                | RouterMode::Random
                | RouterMode::PowerOfTwoChoices
                | RouterMode::LeastLoaded
                | RouterMode::DeviceAwareWeighted
        );
        Self {
            kinds: DecisionKinds {
                token: eligible.token,
                head: eligible.head && native,
                read: eligible.read && native,
            },
            token_possible: eligible.token && card.has_tokenizer(),
            mode,
        }
    }

    pub(crate) fn with_inputs(mut self, score: bool, load: bool) -> Self {
        self.kinds = self.kinds.intersection(DecisionKinds {
            token: score && load,
            head: load,
            read: load,
        });
        self
    }

    pub(crate) fn intent(self, advertised: DecisionKinds) -> DecisionKinds {
        self.kinds.intersection(advertised)
    }
}
