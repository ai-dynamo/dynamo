// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::{Capabilities, QuestionKind, QuestionOutcome, Usage};
use serde::{Deserialize, Serialize};

/// Whole-request canonical execution, independent of the serving framework.
pub(crate) const WORKER_CANONICAL_V1: &str = "worker_canonical_decision_v1";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerDecisionOutcome {
    pub ordinal: usize,
    pub outcome: QuestionOutcome,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerDecisionResult {
    pub outcomes: Vec<WorkerDecisionOutcome>,
    pub usage: Usage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerDecisionCapabilities {
    pub max_questions: usize,
    pub max_candidates: usize,
    pub question_types: Vec<QuestionKind>,
    pub measured_cache_reads: bool,
    pub prompt_format_version: u32,
    pub routing: WorkerDecisionRouting,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkerDecisionRouting {
    RoundRobin,
}

impl WorkerDecisionCapabilities {
    pub(crate) fn parse(value: &serde_json::Value) -> anyhow::Result<Self> {
        let mut capabilities: Self = match value {
            serde_json::Value::String(value) => serde_json::from_str(value)?,
            value => serde_json::from_value(value.clone())?,
        };
        anyhow::ensure!(
            capabilities.max_questions > 0
                && capabilities.max_candidates > 0
                && capabilities.prompt_format_version == 1
                && !capabilities.question_types.is_empty()
                && !capabilities
                    .question_types
                    .iter()
                    .enumerate()
                    .any(|(i, kind)| capabilities.question_types[..i].contains(kind))
                && (!capabilities
                    .question_types
                    .contains(&QuestionKind::Predicate)
                    || capabilities.max_candidates >= 2),
            "invalid canonical decision capabilities"
        );
        capabilities.question_types.sort_by_key(|kind| match kind {
            QuestionKind::Predicate => 0,
            QuestionKind::Choice => 1,
            QuestionKind::Score => 2,
        });
        Ok(capabilities)
    }

    pub(crate) fn decision_capabilities(&self) -> Capabilities {
        Capabilities {
            max_questions: self.max_questions,
            max_candidates: self.max_candidates,
            supports_predicate: self.question_types.contains(&QuestionKind::Predicate),
            supports_choice: self.question_types.contains(&QuestionKind::Choice),
            supports_score: self.question_types.contains(&QuestionKind::Score),
            measured_cache_reads: self.measured_cache_reads,
            prompt_format_version: self.prompt_format_version,
        }
    }
}
