// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::{
    CanonicalQuestion, CanonicalRequest, Dialect, Distribution, QuestionOutcome, Usage,
};
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, SingleIn},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::Annotated;

pub(crate) const CLEF_FLASH_PROFILE: &str = "vllm_clef_flash_joint_decision_v1";
pub(crate) type HeadEngine =
    Arc<dyn AsyncEngine<SingleIn<HeadRequest>, ManyOut<Annotated<HeadResponse>>, Error>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadRequest {
    pub decision_head: HeadBody,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadBody {
    pub protocol_version: u32,
    pub profile: String,
    pub model: String,
    pub input: Value,
    pub questions: Vec<CanonicalQuestion>,
    pub temperature: f64,
    pub cache_salt: String,
    pub max_input_tokens: usize,
    pub max_tokenization_work: usize,
    pub max_encoding_bytes: usize,
}

impl HeadRequest {
    pub(crate) fn new(
        request: &CanonicalRequest,
        canonical_model: &str,
        cache_salt: String,
        max_input_tokens: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            request.temperature == 1.0,
            "Clef decisions require temperature 1"
        );
        let wire = Self {
            decision_head: HeadBody {
                protocol_version: 1,
                profile: CLEF_FLASH_PROFILE.into(),
                model: canonical_model.into(),
                input: native_input(request)?,
                questions: request.questions.clone(),
                temperature: 1.0,
                cache_salt,
                max_input_tokens,
                max_tokenization_work: 1_048_576,
                max_encoding_bytes: 16 * 1024 * 1024,
            },
        };
        anyhow::ensure!(
            serde_json::to_vec(&wire)?.len() <= wire.decision_head.max_encoding_bytes,
            "native decision wire exceeds its byte budget"
        );
        Ok(wire)
    }
}

fn native_input(request: &CanonicalRequest) -> anyhow::Result<Value> {
    use dynamo_decisions::protocols::openai::{Content, Input, InputPart};
    if request.dialect == Dialect::Jev {
        return Ok(request.input.clone());
    }
    let input: Input = serde_json::from_value(request.input.clone())?;
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
                            anyhow::bail!("native decisions require text input")
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
                    .map(|parts| parts.join("\n")),
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n"),
    };
    Ok(Value::String(text))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadResponse {
    pub protocol_version: u32,
    pub profile: String,
    pub finish_reason: String,
    pub outcomes: Vec<HeadOutcome>,
    pub usage: HeadUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadOutcome {
    pub ordinal: usize,
    pub probabilities: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

impl HeadResponse {
    pub(crate) fn validate(
        self,
        request: &CanonicalRequest,
        max_input_tokens: usize,
    ) -> anyhow::Result<(Vec<QuestionOutcome>, Usage)> {
        anyhow::ensure!(
            self.protocol_version == 1
                && self.profile == CLEF_FLASH_PROFILE
                && self.finish_reason == "stop",
            "invalid native decision terminal response"
        );
        anyhow::ensure!(
            self.outcomes.len() == request.questions.len(),
            "native decision answer count differs from request"
        );
        anyhow::ensure!(
            self.usage.input_tokens > 0
                && self.usage.input_tokens <= max_input_tokens as u64
                && self.usage.output_tokens == 0
                && self.usage.cached_tokens <= self.usage.input_tokens,
            "invalid native decision usage"
        );
        let mut answers = Vec::with_capacity(self.outcomes.len());
        for (index, (outcome, question)) in self
            .outcomes
            .into_iter()
            .zip(&request.questions)
            .enumerate()
        {
            anyhow::ensure!(
                outcome.ordinal == index
                    && question.ordinal == index
                    && outcome.probabilities.len() == question.candidates.len(),
                "invalid native decision answer order or cardinality"
            );
            anyhow::ensure!(
                outcome
                    .probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                    && (outcome.probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-9,
                "invalid native decision probability distribution"
            );
            answers.push(QuestionOutcome::Answer(Distribution {
                probabilities: outcome.probabilities,
                label_mass: None,
            }));
        }
        Ok((
            answers,
            Usage {
                input_tokens: self.usage.input_tokens,
                output_tokens: 0,
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

    fn request() -> CanonicalRequest {
        parse_request(&serde_json::to_vec(&json!({"model":"alias", "input":"state", "questions":[{"type":"choice", "instructions":"choose", "choices":[{"value":true},{"value":"true"}]}]})).unwrap(), Route::Decisions).unwrap()
    }

    fn response() -> Value {
        json!({"protocol_version":1, "profile":CLEF_FLASH_PROFILE, "finish_reason":"stop", "outcomes":[{"ordinal":0,"probabilities":[0.2,0.8]}], "usage":{"input_tokens":20,"output_tokens":0,"cached_tokens":2}})
    }

    #[test]
    fn head_wire_keeps_typed_choices_and_normalizes_user_messages() {
        let mut request = request();
        request.input = json!([{"role":"user","content":[{"type":"input_text","text":"one"},{"type":"input_text","text":"two"}]},{"role":"user","content":"three"}]);
        let wire = HeadRequest::new(&request, "canonical", "salt".into(), 100).unwrap();
        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(value["decision_head"]["input"], "one\ntwo\n\nthree");
        assert_eq!(
            value["decision_head"]["questions"][0]["candidates"][0]["value"],
            true
        );
        assert_eq!(
            value["decision_head"]["questions"][0]["candidates"][1]["value"],
            "true"
        );
        request.dialect = Dialect::Jev;
        request.input = json!({"context":[1,2]});
        assert_eq!(
            HeadRequest::new(&request, "canonical", "salt".into(), 100)
                .unwrap()
                .decision_head
                .input,
            request.input
        );
        request.temperature = 0.5;
        assert!(HeadRequest::new(&request, "canonical", "salt".into(), 100).is_err());
    }

    #[test]
    fn head_response_requires_complete_ordered_measured_distributions() {
        let request = request();
        let valid: HeadResponse = serde_json::from_value(response()).unwrap();
        let (answers, usage) = valid.validate(&request, 100).unwrap();
        assert_eq!(usage.cached_tokens, Some(2));
        assert!(matches!(&answers[0], QuestionOutcome::Answer(d) if d.label_mass.is_none()));
        for (field, bad) in [
            ("protocol_version", json!(2)),
            ("profile", json!("other")),
            ("finish_reason", json!("length")),
            ("outcomes", json!([])),
            ("outcomes", json!([{"ordinal":1,"probabilities":[0.2,0.8]}])),
            ("outcomes", json!([{"ordinal":0,"probabilities":[0.2,0.2]}])),
            (
                "usage",
                json!({"input_tokens":20,"output_tokens":1,"cached_tokens":2}),
            ),
            (
                "usage",
                json!({"input_tokens":20,"output_tokens":0,"cached_tokens":21}),
            ),
        ] {
            let mut value = response();
            value[field] = bad;
            assert!(
                serde_json::from_value::<HeadResponse>(value)
                    .unwrap()
                    .validate(&request, 100)
                    .is_err()
            );
        }
        let mut missing = response();
        missing["usage"]
            .as_object_mut()
            .unwrap()
            .remove("cached_tokens");
        assert!(serde_json::from_value::<HeadResponse>(missing).is_err());
    }
}
