// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::{
    CanonicalQuestion, CanonicalRequest, DecisionError, Distribution, QuestionOutcome, Usage,
};
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, SingleIn},
};
use serde::{Deserialize, Serialize};

use crate::{local_model::runtime_config::DecisionExecutionProfile, protocols::Annotated};

pub(crate) type DecisionHeadStreamingEngine = Arc<
    dyn AsyncEngine<SingleIn<DecisionHeadRequest>, ManyOut<Annotated<DecisionHeadResponse>>, Error>,
>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionHeadRequest {
    pub decision_head: HeadRequestV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadRequestV1 {
    pub protocol_version: u32,
    pub profile: String,
    pub model: String,
    pub input: serde_json::Value,
    pub questions: Vec<CanonicalQuestion>,
    pub temperature: f64,
    pub cache_salt: String,
    pub max_input_tokens: usize,
    pub max_tokenization_work: usize,
    pub max_encoding_bytes: usize,
}

impl DecisionHeadRequest {
    pub(crate) fn new(
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        cache_salt: String,
        max_input_tokens: usize,
    ) -> Self {
        Self {
            decision_head: HeadRequestV1 {
                protocol_version: 1,
                profile: profile.generate_capability().into(),
                model: request.model.clone(),
                input: request.input.clone(),
                questions: request.questions.clone(),
                temperature: request.temperature,
                cache_salt,
                max_input_tokens: max_input_tokens.min(1_048_576),
                max_tokenization_work: 1_048_576,
                max_encoding_bytes: 16 * 1024 * 1024,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionHeadResponse {
    pub protocol_version: u32,
    pub profile: String,
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
    pub cached_tokens: Option<u64>,
}

impl DecisionHeadResponse {
    pub(crate) fn validate(
        self,
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        max_input_tokens: usize,
    ) -> Result<(Vec<QuestionOutcome>, Usage), DecisionError> {
        let invalid =
            || DecisionError::execution(request.dialect, "malformed native head response");
        if self.protocol_version != 1
            || self.profile != profile.generate_capability()
            || self.outcomes.len() != request.questions.len()
            || self.usage.input_tokens == 0
            || self.usage.input_tokens > max_input_tokens as u64
            || self.usage.output_tokens != 0
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
                if outcome.ordinal != question.ordinal {
                    return Err(invalid());
                }
                let distribution = Distribution {
                    probabilities: outcome.probabilities,
                    label_mass: None,
                };
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
                output_tokens: 0,
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

    fn request() -> CanonicalRequest {
        parse_request(&serde_json::to_vec(&json!({
            "model":"head", "input":"state", "questions":[
                {"type":"predicate","instructions":"True?"},
                {"type":"choice","instructions":"Choose", "choices":[{"value":true},{"value":"true"}]}
            ]
        })).unwrap(), Route::Decisions).unwrap()
    }

    #[test]
    fn decision_head_wire_preserves_canonical_order_and_types() {
        let request = request();
        let wire = DecisionHeadRequest::new(
            &request,
            DecisionExecutionProfile::VllmD3V1,
            "salt".into(),
            100,
        );
        let value = serde_json::to_value(wire).unwrap();
        assert_eq!(value["decision_head"]["protocol_version"], 1);
        assert_eq!(value["decision_head"]["profile"], "vllm_d3_decision_v1");
        assert_eq!(
            value["decision_head"]["questions"][1]["candidates"][0]["value"],
            true
        );
        assert_eq!(
            value["decision_head"]["questions"][1]["candidates"][1]["value"],
            "true"
        );
    }

    #[test]
    fn decision_head_result_rejects_reordered_or_malformed_results() {
        let request = request();
        let value = json!({"protocol_version":1,"profile":"vllm_d3_decision_v1", "outcomes":[
            {"ordinal":0,"probabilities":[0.2,0.8]}, {"ordinal":1,"probabilities":[0.6,0.4]}
        ], "usage":{"input_tokens":25,"output_tokens":0,"cached_tokens":3}});
        let response: DecisionHeadResponse = serde_json::from_value(value.clone()).unwrap();
        let (answers, usage) = response
            .validate(&request, DecisionExecutionProfile::VllmD3V1, 100)
            .unwrap();
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.cached_tokens, Some(3));
        assert!(matches!(&answers[0], QuestionOutcome::Answer(d) if d.label_mass.is_none()));
        for (field, malformed) in [
            (
                "outcomes",
                json!([{"ordinal":1,"probabilities":[0.2,0.8]}, {"ordinal":0,"probabilities":[0.6,0.4]}]),
            ),
            ("outcomes", json!([{"ordinal":0,"probabilities":[0.2,0.2]}])),
            (
                "usage",
                json!({"input_tokens":25,"output_tokens":1,"cached_tokens":0}),
            ),
            (
                "usage",
                json!({"input_tokens":25,"output_tokens":0,"cached_tokens":26}),
            ),
            ("profile", json!("vllm_clef_flash_joint_decision_v1")),
        ] {
            let mut bad = value.clone();
            bad[field] = malformed;
            let result: DecisionHeadResponse = serde_json::from_value(bad).unwrap();
            assert!(
                result
                    .validate(&request, DecisionExecutionProfile::VllmD3V1, 100)
                    .is_err()
            );
        }
    }
}
