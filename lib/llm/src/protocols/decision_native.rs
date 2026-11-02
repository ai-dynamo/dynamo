// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::{
    CanonicalQuestion, CanonicalRequest, ChoiceValue, DecisionError, Distribution, QuestionKind,
    QuestionOutcome, Usage,
};
use serde::{Deserialize, Serialize};

use crate::{
    decision_profile::DecisionExecutionProfile, protocols::systemone::MAX_DECISION_ENCODING_BYTES,
};

pub(crate) const MAX_DECISION_TOKENIZATION_WORK: usize = 1_048_576;

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
    ) -> anyhow::Result<Self> {
        Ok(Self {
            protocol_version: 1,
            profile,
            model: model.to_owned(),
            input: native_input(request)?,
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
        })
    }
}

fn native_input(request: &CanonicalRequest) -> anyhow::Result<serde_json::Value> {
    use dynamo_decisions::{
        Dialect,
        protocols::openai::{Content, Input, InputPart},
    };
    if request.dialect == Dialect::Jev {
        return Ok(request.input.clone());
    }
    let input = serde_json::from_value::<Input>(request.input.clone())?;
    let text = match input {
        Input::Text(text) => text,
        Input::Messages(messages) => messages
            .into_iter()
            .map(|message| match message.content {
                Content::Text(text) => Ok(text),
                Content::Parts(parts) => parts
                    .into_iter()
                    .map(|part| match part {
                        InputPart::InputText { text } => Ok(text),
                        InputPart::InputImage { .. } => {
                            anyhow::bail!("native decision input requires text parts")
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
                    .map(|parts| parts.join("\n")),
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n"),
    };
    Ok(serde_json::Value::String(text))
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
    pub cached_tokens: u64,
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
    pub finish_reason: NativeFinishReason,
    pub outcomes: Vec<O>,
    pub usage: NativeUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NativeFinishReason {
    Stop,
}

impl<O: NativeOutcome> NativeResponse<O> {
    pub(crate) fn validate(
        self,
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        max_input_tokens: usize,
    ) -> Result<(Vec<QuestionOutcome>, Usage), DecisionError> {
        let invalid = || DecisionError::execution(request.dialect, O::DESCRIPTION);
        let max_output_tokens = match profile {
            DecisionExecutionProfile::VllmD3V1
            | DecisionExecutionProfile::VllmClefFlashV1
            | DecisionExecutionProfile::VllmClefOmniV1 => 0,
            DecisionExecutionProfile::VllmLiquidV1 => request.questions.len() as u64 + 1,
            DecisionExecutionProfile::VllmDiffusionV1 => request.questions.len() as u64 * 2,
            DecisionExecutionProfile::SglangSerialV1 | DecisionExecutionProfile::VllmCausalV1 => {
                return Err(invalid());
            }
        };
        if self.protocol_version != 1
            || self.profile != profile
            || self.outcomes.len() != request.questions.len()
            || self.usage.input_tokens == 0
            || self.usage.input_tokens > max_input_tokens as u64
            || self.usage.output_tokens > max_output_tokens
            || self.usage.cached_tokens > self.usage.input_tokens
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
                cached_tokens: Some(self.usage.cached_tokens),
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
    fn native_openai_input_uses_the_pinned_text_adapter_without_changing_jev_state() {
        let input = json!([
            {"role":"user", "content":[{"type":"input_text","text":"first λ"},{"type":"input_text","text":"second"}]},
            {"role":"user", "content":"third"}
        ]);
        let mut request = parse_request(
            &serde_json::to_vec(&json!({
                "model":"native", "input":input,
                "questions":[{"type":"predicate","instructions":"True?"}]
            }))
            .unwrap(),
            Route::Decisions,
        )
        .unwrap();
        for profile in [
            DecisionExecutionProfile::VllmD3V1,
            DecisionExecutionProfile::VllmClefFlashV1,
            DecisionExecutionProfile::VllmClefOmniV1,
            DecisionExecutionProfile::VllmLiquidV1,
            DecisionExecutionProfile::VllmDiffusionV1,
        ] {
            let wire =
                NativeDecisionRequestV1::new(&request, "native", profile, "salt".into(), 64, 400)
                    .unwrap();
            assert_eq!(wire.input, json!("first λ\nsecond\n\nthird"));
            assert_eq!(
                request.input, input,
                "native projection must not modify the causal canonical input"
            );
            request.dialect = dynamo_decisions::Dialect::Jev;
            let wire =
                NativeDecisionRequestV1::new(&request, "native", profile, "salt".into(), 64, 400)
                    .unwrap();
            assert_eq!(wire.input, input);
            request.dialect = dynamo_decisions::Dialect::OpenAi;
        }
        request.input = json!("unchanged λ");
        let wire = NativeDecisionRequestV1::new(
            &request,
            "native",
            DecisionExecutionProfile::VllmD3V1,
            "salt".into(),
            64,
            400,
        )
        .unwrap();
        assert_eq!(wire.input, request.input);
    }

    #[test]
    fn native_openai_input_rejects_invalid_canonical_values() {
        let mut request = parse_request(
            &serde_json::to_vec(&json!({
                "model":"native", "input":"state",
                "questions":[{"type":"predicate","instructions":"True?"}]
            }))
            .unwrap(),
            Route::Decisions,
        )
        .unwrap();
        for input in [
            json!(true),
            json!({"content":"state"}),
            json!([{"role":"assistant","content":"state"}]),
            json!([{"role":"user","content":[{"type":"input_image","image_url":"https://example.invalid/image"}]}]),
        ] {
            request.input = input;
            assert!(
                NativeDecisionRequestV1::new(
                    &request,
                    "native",
                    DecisionExecutionProfile::VllmD3V1,
                    "salt".into(),
                    64,
                    400
                )
                .is_err()
            );
        }
    }

    #[test]
    fn native_wire_bounds_match_the_cross_language_contract() {
        use crate::protocols::systemone::{MAX_DECISION_ENCODING_BYTES, MAX_DECISION_INPUT_TOKENS};
        let bounds: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/decisions/native-bounds-v1.json"
        ))
        .unwrap();
        let (questions, candidates) = DecisionExecutionProfile::VllmD3V1.limits();
        assert_eq!(bounds["protocol_version"], 1);
        assert_eq!(bounds["max_questions"], questions);
        assert_eq!(bounds["max_candidates"], candidates);
        assert_eq!(bounds["max_input_tokens"], MAX_DECISION_INPUT_TOKENS);
        assert_eq!(
            bounds["max_tokenization_work"],
            MAX_DECISION_TOKENIZATION_WORK
        );
        assert_eq!(bounds["max_encoding_bytes"], MAX_DECISION_ENCODING_BYTES);
    }

    #[test]
    fn native_usage_requires_actual_cache_measurement_on_success() {
        for missing in [
            json!({"input_tokens":3,"output_tokens":0}),
            json!({"input_tokens":3,"output_tokens":0,"cached_tokens":null}),
        ] {
            assert!(serde_json::from_value::<NativeUsage>(missing).is_err());
        }
        for cached in [0, 2] {
            let value = json!({"input_tokens":3,"output_tokens":0,"cached_tokens":cached});
            let usage: NativeUsage = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(usage).unwrap(), value);
        }
    }

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
        )
        .unwrap();
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
