// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::{CanonicalRequest, DecisionError, Distribution, QuestionOutcome, Usage};
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, SingleIn},
};
use serde::{Deserialize, Serialize};

use super::decision_head::HeadRequestV1;
use crate::{local_model::runtime_config::DecisionExecutionProfile, protocols::Annotated};

pub(crate) type DecisionReadStreamingEngine = Arc<
    dyn AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>,
>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionReadRequest {
    pub decision_read: HeadRequestV1,
}

impl DecisionReadRequest {
    pub(crate) fn new(
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        cache_salt: String,
        max_input_tokens: usize,
    ) -> Self {
        Self {
            decision_read: super::decision_head::DecisionHeadRequest::new(
                request,
                profile,
                cache_salt,
                max_input_tokens,
            )
            .decision_head,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionReadResponse {
    pub protocol_version: u32,
    pub profile: String,
    pub outcomes: Vec<ReadOutcome>,
    pub usage: super::decision_head::HeadUsage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadOutcome {
    pub ordinal: usize,
    pub probabilities: Vec<f64>,
    pub label_mass: f64,
}

impl DecisionReadResponse {
    pub(crate) fn validate(
        self,
        request: &CanonicalRequest,
        profile: DecisionExecutionProfile,
        max_input_tokens: usize,
    ) -> Result<(Vec<QuestionOutcome>, Usage), DecisionError> {
        let invalid =
            || DecisionError::execution(request.dialect, "malformed native read response");
        // Native reads may include one shared-prefix warmup and up to two canvas output tokens.
        let per_question = if profile == DecisionExecutionProfile::VllmDiffusionV1 {
            2
        } else {
            1
        };
        let max_output = request.questions.len() as u64 * per_question + 1;
        if !profile.is_native_read()
            || self.protocol_version != 1
            || self.profile != profile.generate_capability()
            || self.outcomes.len() != request.questions.len()
            || self.usage.input_tokens == 0
            || self.usage.input_tokens > max_input_tokens as u64
            || self.usage.output_tokens > max_output
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
                    label_mass: Some(outcome.label_mass),
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

    fn request() -> CanonicalRequest {
        parse_request(&serde_json::to_vec(&json!({
            "model":"native", "input":"state", "questions":[
                {"type":"predicate","instructions":"True?"},
                {"type":"choice","instructions":"Choose", "choices":[{"value":true},{"value":"true"}]}
            ]
        })).unwrap(), Route::Decisions).unwrap()
    }

    fn response() -> serde_json::Value {
        json!({"protocol_version":1,"profile":"vllm_diffusion_canvas_decision_v1", "outcomes":[
            {"ordinal":0,"probabilities":[0.2,0.8],"label_mass":0.4},
            {"ordinal":1,"probabilities":[0.6,0.4],"label_mass":0.9}
        ], "usage":{"input_tokens":25,"output_tokens":4,"cached_tokens":null}})
    }

    #[test]
    fn decision_read_wire_preserves_canonical_order_and_types() {
        let wire = DecisionReadRequest::new(
            &request(),
            DecisionExecutionProfile::VllmLiquidV1,
            "salt".into(),
            100,
        );
        let value = serde_json::to_value(wire).unwrap();
        assert_eq!(value["decision_read"]["profile"], "vllm_liquid_decision_v1");
        assert_eq!(
            value["decision_read"]["questions"][1]["candidates"][0]["value"],
            true
        );
        assert_eq!(
            value["decision_read"]["questions"][1]["candidates"][1]["value"],
            "true"
        );
        assert!(value.get("decision_head").is_none());
    }

    #[test]
    fn decision_read_result_preserves_native_mass_and_actual_usage() {
        let result: DecisionReadResponse = serde_json::from_value(response()).unwrap();
        let (answers, usage) = result
            .validate(&request(), DecisionExecutionProfile::VllmDiffusionV1, 100)
            .unwrap();
        assert!(matches!(&answers[0], QuestionOutcome::Answer(d) if d.label_mass == Some(0.4)));
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.cached_tokens, None);
    }

    #[test]
    fn decision_read_result_rejects_invalid_identity_mass_order_and_usage() {
        for (field, invalid) in [
            ("profile", json!("vllm_d3_decision_v1")),
            ("protocol_version", json!(2)),
            (
                "outcomes",
                json!([{ "ordinal":1,"probabilities":[0.2,0.8],"label_mass":0.4 }]),
            ),
            (
                "outcomes",
                json!([
                    {"ordinal":0,"probabilities":[0.2,0.8],"label_mass":1.01},
                    {"ordinal":1,"probabilities":[0.6,0.4],"label_mass":0.9}
                ]),
            ),
            (
                "usage",
                json!({"input_tokens":25,"output_tokens":6,"cached_tokens":null}),
            ),
            (
                "usage",
                json!({"input_tokens":25,"output_tokens":4,"cached_tokens":26}),
            ),
        ] {
            let mut bad = response();
            bad[field] = invalid;
            let parsed: DecisionReadResponse = serde_json::from_value(bad).unwrap();
            assert!(
                parsed
                    .validate(&request(), DecisionExecutionProfile::VllmDiffusionV1, 100)
                    .is_err()
            );
        }
        let mut missing_mass = response();
        missing_mass["outcomes"][0]
            .as_object_mut()
            .unwrap()
            .remove("label_mass");
        assert!(serde_json::from_value::<DecisionReadResponse>(missing_mass).is_err());
    }
}
