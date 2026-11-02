// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    model_card::{ModelDeploymentCard, decision_profile::DecisionKinds},
    worker_type::WorkerType,
};
use dynamo_runtime::pipeline::RouterMode;

pub(crate) fn supported_decision_kinds(
    card: &ModelDeploymentCard,
    mode: RouterMode,
    enabled: bool,
) -> DecisionKinds {
    if !enabled || card.worker_type != Some(WorkerType::Aggregated) {
        return DecisionKinds::default();
    }
    let eligible = DecisionKinds::eligible(card);
    let native = matches!(
        mode,
        RouterMode::RoundRobin
            | RouterMode::Random
            | RouterMode::PowerOfTwoChoices
            | RouterMode::LeastLoaded
            | RouterMode::DeviceAwareWeighted
    );
    DecisionKinds {
        token: eligible.token,
        head: eligible.head && native,
        read: eligible.read && native,
    }
}
