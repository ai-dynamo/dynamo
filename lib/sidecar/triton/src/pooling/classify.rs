// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI `/v1/classify` on top of a Triton classifier, mirroring the Python worker's `classify_handler.py`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use dynamo_backend_common::DynamoError;
use dynamo_llm::protocols::openai::classify::{
    ClassificationData, ClassificationInput, ClassificationUsage, NvCreateClassifyRequest,
    NvCreateClassifyResponse,
};
use dynamo_sidecar_common::invalid_argument;
use serde_json::Value;
use tokio::sync::Mutex;

use super::{
    PromptInput, Rows, TextModel, TextTask, reject_controls, response_model, text_prompts,
};

/// Class labels by class index, learned from Triton once per model.
type Labels = Arc<Vec<Option<String>>>;

#[derive(Default)]
pub(crate) struct Classify {
    labels: Mutex<Option<Labels>>,
}

impl Classify {
    /// Labels for at least `class_count` classes.
    ///
    /// Triton keeps a model's `label_filename` server-side and exposes labels only through the KServe classification extension, whose scores are rounded to six decimals.
    /// Probabilities therefore come from the raw FP32 output, and the labels are fetched once with a classification request for every class, then cached.
    async fn labels(&self, model: &TextModel, prompt: &str, class_count: usize) -> Labels {
        let mut cached = self.labels.lock().await;
        if let Some(labels) = cached.as_ref().filter(|labels| labels.len() >= class_count) {
            return labels.clone();
        }
        let labels = match model.classify(prompt, class_count).await {
            Ok(classes) => parse_labels(&classes, class_count),
            Err(error) => {
                // Labels are optional in the response; keep serving without them rather than retrying on every request.
                tracing::warn!(model = model.name(), %error, "Could not read Triton class labels");
                vec![None; class_count]
            }
        };
        let labels = Arc::new(labels);
        *cached = Some(labels.clone());
        labels
    }
}

#[async_trait]
impl TextTask for Classify {
    const LABEL: &'static str = "classify";
    const ENDPOINT: &'static str = "/v1/classify";
    const INPUT_FLAG: &'static str = "--classify-input-name";
    const OUTPUT_FLAG: &'static str = "--classify-output-name";

    type Request = NvCreateClassifyRequest;

    fn parse(
        &self,
        model: &TextModel,
        request: Value,
    ) -> Result<(Self::Request, Vec<String>), DynamoError> {
        // Pooling requests always carry `encoding_format` and classify requests never do.
        if request.get("encoding_format").is_some() {
            return Err(invalid_argument(
                "the Triton worker does not yet serve /v1/pooling; register with a pooling-capable backend",
            ));
        }
        let mut request: NvCreateClassifyRequest = serde_json::from_value(request)
            .map_err(|error| invalid_argument(format!("invalid classify request: {error}")))?;
        let input =
            match std::mem::replace(&mut request.input, ClassificationInput::Batch(Vec::new())) {
                ClassificationInput::Single(prompt) => PromptInput::Single(prompt),
                ClassificationInput::Batch(prompts) => PromptInput::Batch(prompts),
                ClassificationInput::Tokens(_) | ClassificationInput::TokenBatch(_) => {
                    PromptInput::Tokens
                }
            };
        let prompts = text_prompts::<Self>(input)?;
        model.check_batch::<Self>(&prompts)?;
        // Tokenization, the classifier activation, processor kwargs, and prefix caching are owned by the Triton model.
        reject_controls::<Self>(
            &[
                ("use_activation", request.use_activation.is_some()),
                ("add_special_tokens", request.add_special_tokens.is_some()),
                (
                    "truncate_prompt_tokens",
                    request.truncate_prompt_tokens.is_some(),
                ),
                ("truncation_side", request.truncation_side.is_some()),
                ("mm_processor_kwargs", request.mm_processor_kwargs.is_some()),
                ("cache_salt", request.cache_salt.is_some()),
            ],
            request.priority,
        )?;
        Ok((request, prompts))
    }

    async fn respond(
        &self,
        model: &TextModel,
        request: Self::Request,
        prompts: &[String],
        rows: Rows,
        context_id: &str,
    ) -> Result<Value, DynamoError> {
        let mut labels: Option<Labels> = None;
        let mut data = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let label = match argmax(row) {
                Some(class) if model.labelled() => {
                    if labels.is_none() {
                        labels = Some(self.labels(model, &prompts[0], row.len()).await);
                    }
                    labels
                        .as_ref()
                        .and_then(|labels| labels.get(class).cloned().flatten())
                }
                _ => None,
            };
            data.push(ClassificationData {
                index: index as u32,
                label,
                probs: row.to_vec(),
                num_classes: row.len() as u32,
            });
        }

        // Fall back to the Dynamo request ID so concurrent responses stay correlatable.
        let request_id = request
            .request_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| context_id.to_string());
        let response = NvCreateClassifyResponse {
            id: if request_id.is_empty() {
                "classify".to_string()
            } else {
                format!("classify-{request_id}")
            },
            object: "list".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or_default(),
            model: response_model(request.model, model),
            data,
            // Triton does not report prompt token counts; the fields stay for wire parity with vLLM.
            usage: ClassificationUsage::default(),
        };
        serde_json::to_value(response).map_err(|error| {
            invalid_argument(format!("could not serialize classify response: {error}"))
        })
    }

    fn probe_response(&self, model_name: &str) -> Value {
        let mut response = NvCreateClassifyResponse::empty();
        response.model = model_name.to_string();
        serde_json::to_value(response).expect("classify response serializes")
    }
}

/// Index of the largest value, or of the first NaN, matching `numpy.argmax`.
fn argmax(row: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in row.iter().enumerate() {
        if value.is_nan() {
            return Some(index);
        }
        if best.is_none_or(|(_, max)| value > max) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

/// Map `"<score>:<index>[:<label>]"` classification strings to labels by class index.
fn parse_labels(classes: &[String], class_count: usize) -> Vec<Option<String>> {
    let mut labels = vec![None; class_count];
    for class in classes {
        let mut parts = class.splitn(3, ':');
        let (Some(_score), Some(index), Some(label)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if let Some(slot) = index
            .parse::<usize>()
            .ok()
            .and_then(|index| labels.get_mut(index))
            && !label.is_empty()
        {
            *slot = Some(label.to_string());
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use dynamo_llm::grpc::service::kserve::inference::DataType::{TypeFp32, TypeString};
    use serde_json::json;

    use super::super::tests::{config, resolve, tensor};
    use super::super::{Rows, TextTask};
    use super::{Classify, argmax, parse_labels};

    fn parse(request: serde_json::Value) -> Result<Vec<String>, String> {
        let model = resolve(
            &config(
                4,
                &[tensor("TEXT", TypeString, &[1])],
                &[tensor("probs", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        Classify::default()
            .parse(&model, request)
            .map(|(_, prompts)| prompts)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn parses_text_requests() {
        assert_eq!(parse(json!({"model": "m", "input": "hi"})).unwrap(), ["hi"]);
        assert_eq!(
            parse(json!({"model": "m", "input": ["a", "b"], "priority": 0})).unwrap(),
            ["a", "b"]
        );
    }

    #[test]
    fn rejects_unsupported_requests() {
        for (request, expected) in [
            (
                json!({"model": "m", "input": "a", "encoding_format": "float"}),
                "/v1/pooling",
            ),
            (json!({"model": "m", "input": [1, 2]}), "token-ID input"),
            (json!({"model": "m", "input": [[1], [2]]}), "token-ID input"),
            (
                json!({"model": "m", "input": ["a", "a", "a", "a", "a"]}),
                "at most 4 prompts",
            ),
            (
                json!({"model": "m", "input": "a", "use_activation": false}),
                "'use_activation'",
            ),
            (
                json!({"model": "m", "input": "a", "truncation_side": "left"}),
                "'truncation_side'",
            ),
            (
                json!({"model": "m", "input": "a", "cache_salt": "s"}),
                "'cache_salt'",
            ),
            (
                json!({"model": "m", "input": "a", "priority": 1}),
                "'priority'",
            ),
            (json!({"model": "m"}), "invalid classify request"),
        ] {
            let error = parse(request.clone()).unwrap_err();
            assert!(error.contains(expected), "{request}: {error}");
        }
    }

    #[tokio::test]
    async fn responds_with_probabilities_and_request_id() {
        let model = resolve(
            &config(
                4,
                &[tensor("TEXT", TypeString, &[1])],
                &[tensor("probs", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        let task = Classify::default();
        let (request, prompts) = task
            .parse(
                &model,
                json!({"model": "", "input": ["a", "b"], "request_id": "r1"}),
            )
            .unwrap();
        let rows = Rows::new(vec![0.1, 0.7, 0.2, 0.5, 0.4, 0.1], &[2, 3], true).unwrap();
        let response = task
            .respond(&model, request, &prompts, rows, "ctx")
            .await
            .unwrap();
        assert_eq!(response["id"], "classify-r1");
        assert_eq!(response["object"], "list");
        assert_eq!(response["model"], "model");
        assert_eq!(response["data"][1]["index"], 1);
        assert_eq!(response["data"][1]["num_classes"], 3);
        assert_eq!(response["data"][0]["label"], serde_json::Value::Null);
        assert_eq!(
            response["usage"],
            json!({"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0})
        );

        let (request, prompts) = task
            .parse(&model, json!({"model": "m", "input": "a"}))
            .unwrap();
        let rows = Rows::new(vec![1.0, 0.0, 0.0], &[1, 3], true).unwrap();
        let response = task
            .respond(&model, request, &prompts, rows, "ctx")
            .await
            .unwrap();
        assert_eq!(response["id"], "classify-ctx");
        assert_eq!(response["model"], "m");
    }

    #[test]
    fn argmax_matches_numpy() {
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[0.1, 0.7, 0.7, 0.2]), Some(1));
        assert_eq!(argmax(&[0.1, f32::NAN, 0.9]), Some(1));
        assert_eq!(argmax(&[f32::NEG_INFINITY]), Some(0));
    }

    #[test]
    fn parses_classification_labels() {
        let classes = [
            "0.700000:1:positive".to_string(),
            "0.200000:2".to_string(),
            "0.100000:0:neutral:extra".to_string(),
            "bad".to_string(),
            "0.0:9:out-of-range".to_string(),
        ];
        assert_eq!(
            parse_labels(&classes, 3),
            [
                Some("neutral:extra".to_string()),
                Some("positive".to_string()),
                None
            ]
        );
    }
}
