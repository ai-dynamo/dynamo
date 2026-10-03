// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters of `learned-choice`. Unknown keys and wrong lengths fail startup.

use dynamo_kv_router::plugins::WorkerSelectionPolicyProviderError;

use super::features::{ContextSource, FeatureSet};
use crate::choice::TieBreak;
use crate::session_map::DEFAULT_MAX_SESSIONS;

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Parameters {
    feature_set: FeatureSet,
    theta: Vec<f64>,
    #[serde(default)]
    context: ContextParameters,
    #[serde(default)]
    temperature: f64,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    tie_break: TieBreak,
    #[serde(default = "default_max_sessions")]
    max_sessions: usize,
}

fn default_max_sessions() -> usize {
    DEFAULT_MAX_SESSIONS
}

/// `context.p` with either `q` (the contract's pooled form, rank `len(p)`) or `sources` (named
/// context sources, one per row of `p`). Empty or absent means no context term.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextParameters {
    #[serde(default)]
    p: Vec<Vec<f64>>,
    #[serde(default)]
    q: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    sources: Option<Vec<String>>,
}

/// The context term u_i += Σ_k c_k (p_k · x_i), where c_k is either q_k · x̄_S or a named source.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ContextTerm {
    None,
    Pooled {
        p: Vec<Vec<f64>>,
        q: Vec<Vec<f64>>,
    },
    Sources {
        sources: Vec<ContextSource>,
        p: Vec<Vec<f64>>,
    },
}

/// Validated model and decision parameters.
#[derive(Clone, Debug)]
pub(super) struct Model {
    pub(super) feature_set: FeatureSet,
    pub(super) theta: Vec<f64>,
    pub(super) context: ContextTerm,
    pub(super) temperature: f64,
    pub(super) seed: Option<u64>,
    pub(super) tie_break: TieBreak,
    pub(super) max_sessions: usize,
}

fn error(message: String) -> WorkerSelectionPolicyProviderError {
    WorkerSelectionPolicyProviderError::new(message)
}

fn check_vector(
    name: &str,
    values: &[f64],
    feature_set: FeatureSet,
) -> Result<(), WorkerSelectionPolicyProviderError> {
    let dim = feature_set.dim();
    if values.len() != dim {
        return Err(error(format!(
            "{name} has {} entries; feature_set {} needs {dim}",
            values.len(),
            feature_set.label(),
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(error(format!("{name}[{index}] must be finite")));
    }
    Ok(())
}

impl Parameters {
    pub(super) fn validate(self) -> Result<Model, WorkerSelectionPolicyProviderError> {
        let feature_set = self.feature_set;
        check_vector("theta", &self.theta, feature_set)?;
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(error(
                "temperature must be finite and non-negative".to_owned(),
            ));
        }
        if self.max_sessions == 0 {
            return Err(error("max_sessions must be positive".to_owned()));
        }
        let ContextParameters { p, q, sources } = self.context;
        for (k, row) in p.iter().enumerate() {
            check_vector(&format!("context.p[{k}]"), row, feature_set)?;
        }
        let context = match (q, sources) {
            (Some(_), Some(_)) => {
                return Err(error("context takes q or sources, not both".to_owned()));
            }
            (Some(q), None) => {
                if q.len() != p.len() {
                    return Err(error(format!(
                        "context.q has {} rows but context.p has {}; the rank must match",
                        q.len(),
                        p.len()
                    )));
                }
                for (k, row) in q.iter().enumerate() {
                    check_vector(&format!("context.q[{k}]"), row, feature_set)?;
                }
                if p.is_empty() {
                    ContextTerm::None
                } else {
                    ContextTerm::Pooled { p, q }
                }
            }
            (None, Some(names)) => {
                if names.len() != p.len() {
                    return Err(error(format!(
                        "context.sources has {} entries but context.p has {} rows",
                        names.len(),
                        p.len()
                    )));
                }
                let mut sources = Vec::with_capacity(names.len());
                for name in &names {
                    let source = ContextSource::parse(name).ok_or_else(|| {
                        error(format!(
                            "unknown context source `{name}`; expected one of {}",
                            ContextSource::known_names().collect::<Vec<_>>().join(", ")
                        ))
                    })?;
                    if sources.contains(&source) {
                        return Err(error(format!("context source `{name}` is repeated")));
                    }
                    sources.push(source);
                }
                if p.is_empty() {
                    ContextTerm::None
                } else {
                    ContextTerm::Sources { sources, p }
                }
            }
            (None, None) if p.is_empty() => ContextTerm::None,
            (None, None) => {
                return Err(error(
                    "context.p needs context.q or context.sources of the same length".to_owned(),
                ));
            }
        };
        Ok(Model {
            feature_set,
            theta: self.theta,
            context,
            temperature: self.temperature,
            seed: self.seed,
            tie_break: self.tie_break,
            max_sessions: self.max_sessions,
        })
    }
}
