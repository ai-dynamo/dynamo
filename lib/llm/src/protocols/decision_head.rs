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

pub(crate) type DecisionHeadStreamingEngine = Arc<
    dyn AsyncEngine<SingleIn<DecisionHeadRequest>, ManyOut<Annotated<DecisionHeadResponse>>, Error>,
>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionHeadRequest {
    pub decision_head: NativeDecisionRequestV1,
}

pub(crate) type DecisionHeadResponse = NativeResponse<HeadOutcome>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeadOutcome {
    pub ordinal: usize,
    pub probabilities: Vec<f64>,
}

impl NativeOutcome for HeadOutcome {
    const DESCRIPTION: &'static str = "malformed native head response";
    fn into_distribution(self) -> (usize, Distribution) {
        (
            self.ordinal,
            Distribution {
                probabilities: self.probabilities,
                label_mass: None,
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
            "model":"head", "input":"state", "questions":[
                {"type":"predicate","instructions":"True?"},
                {"type":"choice","instructions":"Choose", "choices":[{"value":true},{"value":"true"}]}
            ]
        })).unwrap(), Route::Decisions).unwrap()
    }

    #[test]
    fn decision_head_wire_preserves_canonical_order_and_types() {
        let request = request();
        let wire = DecisionHeadRequest {
            decision_head: NativeDecisionRequestV1::new(
                &request,
                &request.model,
                DecisionExecutionProfile::VllmD3V1,
                "salt".into(),
                100,
                400,
            ),
        };
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
        let mut unknown = value.clone();
        unknown["profile"] = json!("unknown_native_profile");
        assert!(serde_json::from_value::<DecisionHeadResponse>(unknown).is_err());
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
