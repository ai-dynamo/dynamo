// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing decisions and actions

/// What the router will do with this request.
#[derive(Debug, Clone)]
pub enum Decision {
    PassThrough,
    /// Override the `model` in the outbound JSON body.
    Override { model_id: String },
    /// Duplicate to another route but do not block main response.
    Shadow { route_to: String },
    /// Reject (policy says we can't run this).
    Reject { reason: String },
}

/// How to act for a given class label.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RouteAction {
    PassThrough,
    Override { model: String },
    Shadow { route_to: String },
    Reject { reason: String },
}

/// Global override policy when user specifies a model.
#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverridePolicy {
    /// Never override if request already has a non-alias explicit model, unless header forces it.
    #[default]
    NeverWhenExplicit,
    /// Allow override when header says `auto` or `force`, or when model matches router alias.
    AllowWhenOptIn,
    /// Always override (dangerous; only for testing).
    Always,
}

/// Router operating mode (can be used as a global switch).
#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Auto,   // classify & maybe override depending on policy/thresholds
    Force,  // override if a class matches threshold regardless of explicit model
    Shadow, // always pass through but tee according to rules
    Off,    // disabled: passthrough
}

impl Decision {
    pub fn is_override(&self) -> bool {
        matches!(self, Decision::Override { .. })
    }
}

/// Helper for turning a configured action into a decision.
pub fn action_to_decision(action: &RouteAction) -> Decision {
    match action {
        RouteAction::PassThrough => Decision::PassThrough,
        RouteAction::Override { model } => Decision::Override { model_id: model.clone() },
        RouteAction::Shadow { route_to } => Decision::Shadow { route_to: route_to.clone() },
        RouteAction::Reject { reason } => Decision::Reject { reason: reason.clone() },
    }
}

/// Utility: choose first class above threshold, in descending score order.
pub fn choose_action_for_scored_labels<'a>(
    ordered: impl IntoIterator<Item = (&'a str, f32)>,
    lookup: impl Fn(&str) -> Option<(&'a RouteAction, f32)>,
) -> Decision {
    for (label, score) in ordered {
        if let Some((action, threshold)) = lookup(label) {
            if score >= threshold {
                return action_to_decision(action);
            }
        }
    }
    Decision::PassThrough
}

