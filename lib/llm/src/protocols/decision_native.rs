// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::{
    CanonicalQuestion, CanonicalRequest, ChoiceValue, DecisionError, Distribution, QuestionKind,
    QuestionOutcome, Usage,
};
use serde::{Deserialize, Serialize};

use crate::{
    local_model::runtime_config::DecisionExecutionProfile,
    protocols::systemone::MAX_DECISION_ENCODING_BYTES,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeDecisionRequestV1 {
    pub protocol_version: u32,
    pub profile: DecisionExecutionProfile,
    pub model: String,
    pub input: serde_json::Value,
    pub questions: Vec<NativeQuestionV1>,
    pub temperature: f64,
    pub cache_salt: String,
    pub max_input_tokens: usize,
    pub max_tokenization_work: usize,
    pub max_encoding_bytes: usize,
}

impl NativeDecisionRequestV1 {
    pub(crate) fn new(
        request: &CanonicalRequest,
        model: &str,
        profile: DecisionExecutionProfile,
        cache_salt: String,
        max_input_tokens: usize,
        max_tokenization_work: usize,
    ) -> Self {
        Self {
            protocol_version: 1,
            profile,
            model: model.to_owned(),
            input: request.input.clone(),
            questions: request
                .questions
                .iter()
                .map(NativeQuestionV1::from)
                .collect(),
            temperature: request.temperature,
            cache_salt,
            max_input_tokens,
            max_tokenization_work,
            max_encoding_bytes: MAX_DECISION_ENCODING_BYTES,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeQuestionV1 {
    pub ordinal: usize,
    pub id: Option<String>,
    pub kind: NativeQuestionKindV1,
    pub instructions: Option<serde_json::Value>,
    pub candidates: Vec<NativeCandidateV1>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum NativeQuestionKindV1 {
    #[serde(rename = "Predicate")]
    Predicate,
    #[serde(rename = "Choice")]
    Choice,
    #[serde(rename = "Score")]
    Score,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeCandidateV1 {
    pub value: NativeChoiceValueV1,
    pub description: Option<serde_json::Value>,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum NativeChoiceValueV1 {
    String(String),
    Bool(bool),
}

impl From<&CanonicalQuestion> for NativeQuestionV1 {
    fn from(question: &CanonicalQuestion) -> Self {
        Self {
            ordinal: question.ordinal,
            id: question.id.clone(),
            kind: match question.kind {
                QuestionKind::Predicate => NativeQuestionKindV1::Predicate,
                QuestionKind::Choice => NativeQuestionKindV1::Choice,
                QuestionKind::Score => NativeQuestionKindV1::Score,
            },
            instructions: question.instructions.clone(),
            candidates: question
                .candidates
                .iter()
                .map(|candidate| NativeCandidateV1 {
                    value: match &candidate.value {
                        ChoiceValue::String(value) => NativeChoiceValueV1::String(value.clone()),
                        ChoiceValue::Bool(value) => NativeChoiceValueV1::Bool(*value),
                    },
                    description: candidate.description.clone(),
                    label: candidate.label.clone(),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: Option<u64>,
}

pub(crate) trait NativeOutcome {
    const DESCRIPTION: &'static str;
    fn into_distribution(self) -> (usize, Distribution);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeResponse<O> {
    pub protocol_version: u32,
    pub profile: DecisionExecutionProfile,
    pub outcomes: Vec<O>,
    pub usage: NativeUsage,
}

impl<O: NativeOutcome> NativeResponse<O> {
    pub(crate) fn validate(
        self,
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        max_input_tokens: usize,
    ) -> Result<(Vec<QuestionOutcome>, Usage), DecisionError> {
        let invalid = || DecisionError::execution(request.dialect, O::DESCRIPTION);
        if self.protocol_version != 1
            || self.profile != profile
            || self.outcomes.len() != request.questions.len()
            || self.usage.input_tokens == 0
            || self.usage.input_tokens > max_input_tokens as u64
            || self.usage.output_tokens > profile.max_native_output_tokens(request.questions.len())
            || self
                .usage
                .cached_tokens
                .is_some_and(|cached| cached > self.usage.input_tokens)
        {
            return Err(invalid());
        }
        let outcomes = self
            .outcomes
            .into_iter()
            .zip(&request.questions)
            .map(|(outcome, question)| {
                let (ordinal, distribution) = outcome.into_distribution();
                if ordinal != question.ordinal {
                    return Err(invalid());
                }
                distribution
                    .validate(question.candidates.len())
                    .map_err(|_| invalid())?;
                Ok(QuestionOutcome::Answer(distribution))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            outcomes,
            Usage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
                cached_tokens: self.usage.cached_tokens,
                reasoning_tokens: 0,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_decisions::{Route, parse_request};
    use serde_json::json;

    #[test]
    fn native_question_v1_matches_the_shared_golden_wire() {
        let request = parse_request(
            &serde_json::to_vec(&json!({
                "model":"native", "input":"Unicode λ state", "questions":[
                    {"type":"predicate","instructions":"True?"},
                    {"type":"choice","name":"duplicate","instructions":"Choose", "choices":[
                        {"value":true,"description":"boolean"},{"value":"true"}
                    ]},
                    {"type":"score","name":"duplicate","instructions":"Rate","levels":[
                        {"label":"low","description":"Bad"},{"label":"high"}
                    ]},
                    {"type":"predicate","name":"","instructions":"Again?"}
                ]
            }))
            .unwrap(),
            Route::Decisions,
        )
        .unwrap();
        let wire = NativeDecisionRequestV1::new(
            &request,
            "native",
            DecisionExecutionProfile::VllmD3V1,
            "salt".into(),
            64,
            16448,
        );
        let golden = include_str!("../../../../tests/fixtures/decisions/native-question-v1.json");
        assert_eq!(
            format!("{}\n", serde_json::to_string(&wire).unwrap()),
            golden
        );
        let decoded: NativeDecisionRequestV1 = serde_json::from_str(golden).unwrap();
        assert_eq!(
            serde_json::to_value(&decoded).unwrap(),
            serde_json::to_value(&wire).unwrap()
        );
        let mut malformed = serde_json::to_value(&wire).unwrap();
        malformed["questions"][0]["unknown"] = json!(true);
        assert!(serde_json::from_value::<NativeDecisionRequestV1>(malformed).is_err());
    }
}
