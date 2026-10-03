// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `learned-choice`: a conditional-logit worker choice over router-observable features.
//!
//! Each eligible worker `i` in candidate set `S` gets a feature vector `x_i` (see `FEATURES.md`)
//! and a utility
//!
//! ```text
//! u_i = θ·x_i + Σ_k c_k (p_k·x_i),   c_k = q_k·x̄_S  (pooled form)   or   c_k = z_{S,k}  (sources form)
//! ```
//!
//! where `x̄_S` is the candidate mean of `x` and `z_S` a vector of named request-level sources.
//! At `temperature: 0` the policy takes the highest utility; otherwise it samples from
//! `softmax(u / temperature)`. Unlike `dynamo-default-cost-fn`, this temperature is not
//! range-normalized. Ties and samples use a per-instance RNG seeded by `seed` and visit workers
//! in canonical order, so replay is reproducible. Host pins and eligibility always win.
//!
//! Feature 0 is the default cost function's own logit at default weights, computed by the
//! default scorer composed into this policy, so `θ = −e₀` with no context term reproduces the
//! default's argmin. The policy evaluates in O(N·d) per request and reads nothing from a
//! performance model, so the same parameters run unchanged in the live router.

mod features;
mod parameters;

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    SessionContext, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

use crate::choice::{Chooser, row_of};
use crate::session_map::SessionMap;
use features::{FeatureTable, RequestInputs};
use parameters::{ContextTerm, Model, Parameters};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "learned-choice";

struct LearnedChoicePicker {
    model: Model,
    chooser: Chooser,
    sessions: SessionMap,
    table: FeatureTable,
    mean: Vec<f64>,
    weights: Vec<f64>,
    utilities: Vec<f64>,
}

impl LearnedChoicePicker {
    fn new(model: Model) -> Self {
        Self {
            chooser: Chooser::new(model.seed, model.tie_break),
            sessions: SessionMap::new(model.max_sessions),
            table: FeatureTable::new(model.feature_set),
            mean: Vec::new(),
            weights: Vec::new(),
            utilities: Vec::new(),
            model,
        }
    }

    fn decide(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
        session: Option<&str>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let request = RequestInputs {
            previous_worker: session.and_then(|session| self.sessions.get(session)),
        };
        let order = self.chooser.order();
        self.table.compute(context, input, order, &request)?;
        effective_weights(
            &self.model,
            &self.table,
            order,
            &mut self.mean,
            &mut self.weights,
        );
        self.utilities.clear();
        self.utilities.extend(
            (0..input.candidates().len()).map(|row| dot(&self.weights, self.table.row(row))),
        );
        if let Some(row) = self.utilities.iter().position(|value| !value.is_finite()) {
            return Err(WorkerSelectionPolicyError::failed(format!(
                "learned-choice utility is not finite for candidate row {row}"
            )));
        }
        let utilities = &self.utilities;
        Ok(if self.model.temperature == 0.0 {
            self.chooser.lowest(|row| -utilities[row])
        } else {
            self.chooser
                .sample_utilities(|row| utilities[row], self.model.temperature)
        })
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// The decision's effective coefficients w = θ + Σ_k c_k p_k, so that u_i = w·x_i.
fn effective_weights(
    model: &Model,
    table: &FeatureTable,
    order: &[usize],
    mean: &mut Vec<f64>,
    weights: &mut Vec<f64>,
) {
    weights.clear();
    weights.extend_from_slice(&model.theta);
    let mut add = |scale: f64, row: &[f64]| {
        for (weight, value) in weights.iter_mut().zip(row) {
            *weight += scale * value;
        }
    };
    match &model.context {
        ContextTerm::None => {}
        ContextTerm::Pooled { p, q } => {
            mean.clear();
            mean.extend((0..table.set.dim()).map(|feature| table.mean(feature, order)));
            for (p_k, q_k) in p.iter().zip(q) {
                add(dot(q_k, mean), p_k);
            }
        }
        ContextTerm::Sources { sources, p } => {
            for (source, p_k) in sources.iter().zip(p) {
                add(table.source(*source, order), p_k);
            }
        }
    }
}

impl WorkerPicker for LearnedChoicePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        self.chooser.begin(candidates);
        let session = context.session_context().map(SessionContext::session_id);
        let row = match context.pinned_worker() {
            Some(pinned) => row_of(candidates, pinned).unwrap_or(0),
            None => self.decide(context, input, session)?,
        };
        if let Some(session) = session {
            self.sessions.bind(session, candidates[row].worker());
        }
        Ok(row)
    }
}

fn policy(
    config: &KvRouterConfig,
    role: dynamo_kv_router::WorkerType,
    model: Model,
) -> WorkerSelectionPolicy {
    // Feature 0 is the default logit at default weights, whatever weights the host configures.
    let scorer = crate::default::cost_scorer(config, &KvRouterConfig::default(), role);
    WorkerSelectionPolicy::new(
        config.clone(),
        role.default_selector_label(),
        vec![scorer],
        Box::new(LearnedChoicePicker::new(model)),
    )
    .with_exclusive_affinity(true)
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let model = parameters.deserialize::<Parameters>()?.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, role, _partition| policy(config, role, model.clone()),
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests;
