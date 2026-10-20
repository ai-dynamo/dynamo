// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI `/v1/embeddings` on top of a Triton embedding model, mirroring the Python worker's `embedding_handler.py`.

use async_trait::async_trait;
use base64::Engine as _;
use dynamo_backend_common::DynamoError;
use dynamo_llm::protocols::openai::embeddings::{
    NvCreateEmbeddingRequest, NvCreateEmbeddingResponse,
};
use dynamo_protocols::types::{
    CreateEmbeddingResponse, Embedding, EmbeddingInput, EmbeddingUsage, EmbeddingVector,
};
use dynamo_sidecar_common::invalid_argument;
use serde_json::Value;

use super::{
    PromptInput, Rows, TextModel, TextTask, reject_controls, response_model, text_prompts,
};

pub(crate) struct Embeddings;

#[async_trait]
impl TextTask for Embeddings {
    const LABEL: &'static str = "embedding";
    const ENDPOINT: &'static str = "/v1/embeddings";
    const INPUT_FLAG: &'static str = "--embeddings-input-name";
    const OUTPUT_FLAG: &'static str = "--embeddings-output-name";

    type Request = NvCreateEmbeddingRequest;

    fn parse(
        &self,
        model: &TextModel,
        request: Value,
    ) -> Result<(Self::Request, Vec<String>), DynamoError> {
        match request.get("encoding_format") {
            None | Some(Value::Null) => {}
            Some(Value::String(format)) if matches!(format.as_str(), "float" | "base64") => {}
            Some(other) => {
                return Err(invalid_argument(format!(
                    "the Triton embedding worker only supports encoding_format values ['float', 'base64']; got {other}."
                )));
            }
        }
        let mut request: NvCreateEmbeddingRequest = serde_json::from_value(request)
            .map_err(|error| invalid_argument(format!("invalid embedding request: {error}")))?;
        let input = match std::mem::replace(
            &mut request.inner.input,
            EmbeddingInput::StringArray(Vec::new()),
        ) {
            EmbeddingInput::String(prompt) => PromptInput::Single(prompt),
            EmbeddingInput::StringArray(prompts) => PromptInput::Batch(prompts),
            EmbeddingInput::IntegerArray(_) | EmbeddingInput::ArrayOfIntegerArray(_) => {
                PromptInput::Tokens
            }
        };
        let prompts = text_prompts::<Self>(input)?;
        model.check_batch::<Self>(&prompts)?;
        // The output dimension and tokenization are fixed by the Triton model.
        reject_controls::<Self>(
            &[
                ("dimensions", request.inner.dimensions.is_some()),
                ("add_special_tokens", request.add_special_tokens.is_some()),
                (
                    "truncate_prompt_tokens",
                    request.truncate_prompt_tokens.is_some(),
                ),
            ],
            0,
        )?;
        Ok((request, prompts))
    }

    async fn respond(
        &self,
        model: &TextModel,
        request: Self::Request,
        _prompts: &[String],
        rows: Rows,
        _context_id: &str,
    ) -> Result<Value, DynamoError> {
        // Always base64 on the worker wire; the frontend decodes to floats unless the client asked for base64.
        let data = rows
            .iter()
            .enumerate()
            .map(|(index, row)| Embedding {
                index: index as u32,
                object: "embedding".to_string(),
                embedding: EmbeddingVector::Base64(encode_base64(row)),
            })
            .collect();
        let response = NvCreateEmbeddingResponse {
            inner: CreateEmbeddingResponse {
                object: "list".to_string(),
                model: response_model(request.inner.model, model),
                data,
                // Triton does not report prompt token counts; the fields stay for wire parity with vLLM.
                usage: EmbeddingUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            },
        };
        serde_json::to_value(response).map_err(|error| {
            invalid_argument(format!("could not serialize embedding response: {error}"))
        })
    }

    fn probe_response(&self, model_name: &str) -> Value {
        let mut response = NvCreateEmbeddingResponse::empty();
        response.inner.model = model_name.to_string();
        serde_json::to_value(response).expect("embedding response serializes")
    }
}

/// OpenAI's base64 embedding format: little-endian FP32 bytes, standard alphabet.
fn encode_base64(row: &[f32]) -> String {
    let bytes: Vec<u8> = row.iter().flat_map(|value| value.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use dynamo_llm::grpc::service::kserve::inference::DataType::{TypeFp32, TypeString};
    use serde_json::json;

    use super::super::tests::{config, resolve, tensor};
    use super::super::{Rows, TextTask};
    use super::Embeddings;

    fn model(max_batch_size: i32) -> super::super::TextModel {
        resolve(
            &config(
                max_batch_size,
                &[tensor("TEXT", TypeString, &[1])],
                &[tensor("embedding", TypeFp32, &[2])],
            ),
            None,
            None,
        )
        .unwrap()
    }

    fn parse(request: serde_json::Value) -> Result<Vec<String>, String> {
        Embeddings
            .parse(&model(4), request)
            .map(|(_, prompts)| prompts)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn parses_text_requests() {
        assert_eq!(parse(json!({"model": "m", "input": "hi"})).unwrap(), ["hi"]);
        for format in ["float", "base64"] {
            assert_eq!(
                parse(json!({"model": "m", "input": ["a", "b"], "encoding_format": format}))
                    .unwrap(),
                ["a", "b"]
            );
        }
    }

    #[test]
    fn rejects_unsupported_requests() {
        for (request, expected) in [
            (
                json!({"model": "m", "input": "a", "encoding_format": "int8"}),
                "encoding_format",
            ),
            (json!({"model": "m", "input": [1, 2]}), "token-ID input"),
            (json!({"model": "m", "input": ""}), "empty string"),
            (
                json!({"model": "m", "input": "a", "dimensions": 8}),
                "'dimensions'",
            ),
            (
                json!({"model": "m", "input": "a", "add_special_tokens": true}),
                "'add_special_tokens'",
            ),
            (
                json!({"model": "m", "input": "a", "truncate_prompt_tokens": 5}),
                "'truncate_prompt_tokens'",
            ),
        ] {
            let error = parse(request.clone()).unwrap_err();
            assert!(error.contains(expected), "{request}: {error}");
        }
        let error = Embeddings
            .parse(&model(0), json!({"model": "m", "input": ["a", "b"]}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unbatched"), "{error}");
    }

    #[tokio::test]
    async fn responds_with_base64_vectors() {
        let model = model(4);
        let (request, prompts) = Embeddings
            .parse(&model, json!({"model": "m", "input": ["a", "b"]}))
            .unwrap();
        let rows = Rows::new(vec![1.0, -2.0, 0.5, 0.25], &[2, 2], true).unwrap();
        let response = Embeddings
            .respond(&model, request, &prompts, rows, "ctx")
            .await
            .unwrap();
        assert_eq!(response["object"], "list");
        assert_eq!(response["model"], "m");
        assert_eq!(
            response["usage"],
            json!({"prompt_tokens": 0, "total_tokens": 0})
        );
        assert_eq!(response["data"][1]["index"], 1);
        assert_eq!(response["data"][1]["object"], "embedding");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(response["data"][1]["embedding"].as_str().unwrap())
            .unwrap();
        let expected: Vec<u8> = [0.5_f32, 0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn probe_response_names_the_model() {
        let response = Embeddings.probe_response("text_embedder");
        assert_eq!(response["model"], "text_embedder");
        assert_eq!(response["data"], json!([]));
    }
}
