// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local policy configuration, applied between scheduler decisions.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use super::{WorkerInputs, WorkerSelectionPolicy};

/// Resolved policy parameters and their revision within one router instance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimePolicyConfig {
    revision: u64,
    parameters: Map<String, Value>,
}

impl RuntimePolicyConfig {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn parameters(&self) -> &Map<String, Value> {
        &self.parameters
    }
}

/// A conditional update. Null parameter values restore their startup values.
pub struct RuntimePolicyUpdate {
    expected_revision: u64,
    parameters: Option<Map<String, Value>>,
}

impl RuntimePolicyUpdate {
    /// Replace the supplied parameter values, leaving omitted parameters unchanged.
    pub fn patch(expected_revision: u64, parameters: Map<String, Value>) -> Self {
        Self {
            expected_revision,
            parameters: Some(parameters),
        }
    }

    /// Restore all parameters to the values resolved at policy construction.
    pub fn reset(expected_revision: u64) -> Self {
        Self {
            expected_revision,
            parameters: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum RuntimePolicyError {
    #[error("the selected worker policy does not support runtime configuration")]
    Unsupported,
    #[error("policy revision conflict: expected {expected}, current {current}")]
    Conflict { expected: u64, current: u64 },
    #[error("invalid policy parameters: {0}")]
    InvalidParameters(String),
    #[error("router policy control is unavailable")]
    Unavailable,
}

/// Construct a replacement from complete resolved parameters. Called on the scheduler actor;
/// implementations must validate without I/O or modifying the active policy or shared state.
pub type RuntimePolicyBuilder = Arc<
    dyn Fn(Map<String, Value>) -> Result<WorkerSelectionPolicy, RuntimePolicyError> + Send + Sync,
>;

pub(crate) struct RuntimePolicyState {
    pub(crate) available_inputs: WorkerInputs,
    initial: Map<String, Value>,
    current: RuntimePolicyConfig,
    builder: RuntimePolicyBuilder,
}

impl RuntimePolicyState {
    pub(crate) fn new(
        parameters: Map<String, Value>,
        builder: RuntimePolicyBuilder,
        available_inputs: WorkerInputs,
    ) -> Self {
        Self {
            available_inputs,
            initial: parameters.clone(),
            current: RuntimePolicyConfig {
                revision: 0,
                parameters,
            },
            builder,
        }
    }

    pub(crate) fn config(&self) -> RuntimePolicyConfig {
        self.current.clone()
    }

    pub(crate) fn prepare(
        &self,
        update: RuntimePolicyUpdate,
    ) -> Result<(WorkerSelectionPolicy, Self), RuntimePolicyError> {
        if update.expected_revision != self.current.revision {
            return Err(RuntimePolicyError::Conflict {
                expected: update.expected_revision,
                current: self.current.revision,
            });
        }
        let parameters = match update.parameters {
            None => self.initial.clone(),
            Some(patch) => {
                if patch.is_empty() {
                    return Err(RuntimePolicyError::InvalidParameters("empty patch".into()));
                }
                let mut parameters = self.current.parameters.clone();
                for (name, value) in patch {
                    let initial = self.initial.get(&name).ok_or_else(|| {
                        RuntimePolicyError::InvalidParameters(format!("unknown parameter {name:?}"))
                    })?;
                    parameters.insert(
                        name,
                        if value.is_null() {
                            initial.clone()
                        } else {
                            value
                        },
                    );
                }
                parameters
            }
        };
        let replacement = (self.builder)(parameters.clone())?;
        let revision = self.current.revision + 1;
        Ok((
            replacement,
            Self {
                available_inputs: self.available_inputs,
                initial: self.initial.clone(),
                current: RuntimePolicyConfig {
                    revision,
                    parameters,
                },
                builder: self.builder.clone(),
            },
        ))
    }
}
