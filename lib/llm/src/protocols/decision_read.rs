// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_decisions::Distribution;
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, SingleIn},
};
use serde::{Deserialize, Serialize};

use super::decision_native::{NativeDecisionRequestV1, NativeOutcome, NativeResponse};
use crate::protocols::Annotated;

pub(crate) type DecisionReadStreamingEngine = Arc<
    dyn AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>,
>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionReadRequest {
    pub decision_read: NativeDecisionRequestV1,
}

pub(crate) type DecisionReadResponse = NativeResponse<ReadOutcome>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadOutcome {
    pub ordinal: usize,
    pub probabilities: Vec<f64>,
    pub label_mass: f64,
}

impl NativeOutcome for ReadOutcome {
    const DESCRIPTION: &'static str = "malformed native read response";
    fn into_distribution(self) -> (usize, Distribution) {
        (
            self.ordinal,
            Distribution {
                probabilities: self.probabilities,
                label_mass: Some(self.label_mass),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::DecisionExecutionProfile;
    use dynamo_decisions::{CanonicalRequest, QuestionOutcome, Route, parse_request};
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
        let wire = DecisionReadRequest {
            decision_read: NativeDecisionRequestV1::new(
                &request(),
                "native",
                DecisionExecutionProfile::VllmLiquidV1,
                "salt".into(),
                100,
                400,
            ),
        };
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
