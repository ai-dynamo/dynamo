// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI `/v1/classify` and `/v1/embeddings` serving on top of Triton models with one `TYPE_STRING` input and one `TYPE_FP32` output.
//!
//! This is the Rust counterpart of the Python Triton worker's `pooling_common.py`, `classify_handler.py`, and `embedding_handler.py`.
//! The frontend sends the OpenAI request as JSON; the sidecar forwards its text prompts to Triton as one BYTES tensor and wraps the `FP32` rows Triton returns in the OpenAI response.
//! Tokenization stays inside the Triton model (typically an ensemble), so token-ID input is rejected.

mod classify;
mod embeddings;

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    BackendError, DynamoError, EngineConfig, ErrorType, GenerateContext, HEALTH_CHECK_KEY,
    RawEngine,
};
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_llm::grpc::service::kserve::inference::infer_parameter::ParameterChoice;
use dynamo_sidecar_common::{cannot_connect, invalid_argument};
use futures::stream::BoxStream;
use serde_json::{Value, json};

use crate::client::TritonClient;

pub(crate) use classify::Classify;
pub(crate) use embeddings::Embeddings;

/// A task served on top of a STRING-in, FP32-out Triton model.
#[async_trait]
pub(crate) trait TextTask: Send + Sync + 'static {
    /// Task name used in error messages (`classify` or `embedding`).
    const LABEL: &'static str;
    /// OpenAI endpoint this task serves.
    const ENDPOINT: &'static str;
    /// CLI flag that selects the input tensor.
    const INPUT_FLAG: &'static str;
    /// CLI flag that selects the output tensor.
    const OUTPUT_FLAG: &'static str;

    type Request: Send + 'static;

    /// Validate `request` for `model` and return it with its text prompts.
    fn parse(
        &self,
        model: &TextModel,
        request: Value,
    ) -> Result<(Self::Request, Vec<String>), DynamoError>;

    /// Build the OpenAI response from one FP32 row per prompt.
    async fn respond(
        &self,
        model: &TextModel,
        request: Self::Request,
        prompts: &[String],
        rows: Rows,
        context_id: &str,
    ) -> Result<Value, DynamoError>;

    /// Empty response returned to a health-check probe.
    fn probe_response(&self, model_name: &str) -> Value;
}

/// A Triton model resolved to the tensors a [`TextTask`] reads and writes.
pub(crate) struct TextModel {
    name: String,
    input: String,
    output: String,
    max_batch_size: i32,
    /// Whether the output declares a `label_filename`, so Triton can name its classes.
    labelled: bool,
    client: TritonClient,
}

impl TextModel {
    /// Resolve the STRING input and FP32 output from `config`, failing at startup for a model the task cannot serve.
    pub(crate) fn new<T: TextTask>(
        name: &str,
        config: &inference::ModelConfig,
        input_override: Option<&str>,
        output_override: Option<&str>,
        client: TritonClient,
    ) -> Result<Self, DynamoError> {
        let label = T::LABEL;
        let string_inputs: Vec<&str> = config
            .input
            .iter()
            .filter(|input| input.data_type == inference::DataType::TypeString as i32)
            .map(|input| input.name.as_str())
            .collect();
        let input = resolve_tensor(
            &string_inputs,
            input_override,
            || format!("Triton {label} model '{name}'"),
            "TYPE_STRING input",
            T::INPUT_FLAG,
        )?;
        // Requests carry one string per item, so the input must be shaped [1] or [-1].
        let dims = &config
            .input
            .iter()
            .find(|candidate| candidate.name == input)
            .expect("resolved input is one of the model's inputs")
            .dims;
        if !matches!(dims.as_slice(), [1] | [-1]) {
            return Err(invalid_argument(format!(
                "Triton {label} model '{name}' STRING input '{input}' has dims={dims:?}; the {} path supports only dims=[1] or dims=[-1] (one string per request item). \
                 Change the model's config.pbtxt to one of the supported layouts, or serve the model with --task tensor and address it over KServe gRPC.",
                T::ENDPOINT
            )));
        }

        let fp32_outputs: Vec<&str> = config
            .output
            .iter()
            .filter(|output| output.data_type == inference::DataType::TypeFp32 as i32)
            .map(|output| output.name.as_str())
            .collect();
        let output = resolve_tensor(
            &fp32_outputs,
            output_override,
            || format!("Triton {label} model '{name}'"),
            "TYPE_FP32 output",
            T::OUTPUT_FLAG,
        )?;
        let labelled = config
            .output
            .iter()
            .any(|candidate| candidate.name == output && !candidate.label_filename.is_empty());

        tracing::info!(
            task = label,
            model = name,
            input,
            output,
            batched = config.max_batch_size > 0,
            labelled,
            "Triton text model resolved"
        );
        Ok(Self {
            name: name.to_string(),
            input,
            output,
            max_batch_size: config.max_batch_size,
            labelled,
            client,
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn labelled(&self) -> bool {
        self.labelled
    }

    fn batched(&self) -> bool {
        self.max_batch_size > 0
    }

    /// Reject a prompt count the model's batching configuration cannot accept.
    pub(crate) fn check_batch<T: TextTask>(&self, prompts: &[String]) -> Result<(), DynamoError> {
        let (label, name, max) = (T::LABEL, &self.name, self.max_batch_size);
        if !self.batched() && prompts.len() > 1 {
            return Err(invalid_argument(format!(
                "Triton {label} model '{name}' is unbatched (max_batch_size={max}) and received {} prompts. \
                 Send one prompt per request, or raise max_batch_size in the model's config.pbtxt.",
                prompts.len()
            )));
        }
        if self.batched() && prompts.len() > max as usize {
            return Err(invalid_argument(format!(
                "Triton {label} model '{name}' accepts at most {max} prompts per request (max_batch_size); received {}. \
                 Split the batch or raise max_batch_size in the model's config.pbtxt.",
                prompts.len()
            )));
        }
        Ok(())
    }

    /// Infer request carrying `prompts` as the model's BYTES input and asking for `output`.
    fn request(
        &self,
        prompts: &[String],
        output: inference::model_infer_request::InferRequestedOutputTensor,
    ) -> inference::ModelInferRequest {
        // Shape [N, 1] for a batching model, else [N].
        let count = prompts.len() as i64;
        let shape = if self.batched() {
            vec![count, 1]
        } else {
            vec![count]
        };
        inference::ModelInferRequest {
            model_name: self.name.clone(),
            inputs: vec![inference::model_infer_request::InferInputTensor {
                name: self.input.clone(),
                datatype: "BYTES".to_string(),
                shape,
                ..Default::default()
            }],
            outputs: vec![output],
            raw_input_contents: vec![bytes_tensor(prompts.iter().map(String::as_str))],
            ..Default::default()
        }
    }

    /// Run `prompts` through Triton and return one FP32 row per prompt.
    pub(crate) async fn infer_rows(&self, prompts: &[String]) -> Result<Rows, DynamoError> {
        let request = self.request(
            prompts,
            inference::model_infer_request::InferRequestedOutputTensor {
                name: self.output.clone(),
                ..Default::default()
            },
        );
        let response = self.client.infer(request).await?;
        let (shape, values) = self.fp32_output(response)?;
        let rows = Rows::new(values, &shape, self.batched()).map_err(|detail| {
            unknown(format!(
                "Triton model '{}' {detail} for output '{}' (max_batch_size={})",
                self.name, self.output, self.max_batch_size
            ))
        })?;
        // A batch of N prompts must produce exactly N rows.
        if rows.len() != prompts.len() {
            return Err(unknown(format!(
                "Triton model '{}' returned {} row(s) for {} input(s); expected one row per input. \
                 Check the model's batching config or ensemble output shape.",
                self.name,
                rows.len(),
                prompts.len()
            )));
        }
        Ok(rows)
    }

    /// Ask Triton for the top `class_count` classes of `prompt` as `"<score>:<index>[:<label>]"` strings.
    pub(crate) async fn classify(
        &self,
        prompt: &str,
        class_count: usize,
    ) -> Result<Vec<String>, DynamoError> {
        let request = self.request(
            std::slice::from_ref(&prompt.to_string()),
            inference::model_infer_request::InferRequestedOutputTensor {
                name: self.output.clone(),
                parameters: [(
                    "classification".to_string(),
                    inference::InferParameter {
                        parameter_choice: Some(ParameterChoice::Int64Param(class_count as i64)),
                    },
                )]
                .into(),
            },
        );
        let response = self.client.infer(request).await?;
        let index = self.output_index(&response)?;
        let raw = response
            .raw_output_contents
            .get(index)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if raw.is_empty() {
            let contents = response.outputs[index].contents.as_ref();
            return Ok(contents
                .map(|contents| {
                    contents
                        .bytes_contents
                        .iter()
                        .map(|class| String::from_utf8_lossy(class).into_owned())
                        .collect()
                })
                .unwrap_or_default());
        }
        parse_bytes_tensor(raw).ok_or_else(|| {
            unknown(format!(
                "Triton model '{}' returned a malformed classification for output '{}'",
                self.name, self.output
            ))
        })
    }

    fn output_index(&self, response: &inference::ModelInferResponse) -> Result<usize, DynamoError> {
        response
            .outputs
            .iter()
            .position(|output| output.name == self.output)
            .ok_or_else(|| {
                unknown(format!(
                    "Triton model '{}' returned no output '{}'",
                    self.name, self.output
                ))
            })
    }

    fn fp32_output(
        &self,
        mut response: inference::ModelInferResponse,
    ) -> Result<(Vec<i64>, Vec<f32>), DynamoError> {
        let index = self.output_index(&response)?;
        let output = &mut response.outputs[index];
        if output.datatype != "FP32" {
            return Err(unknown(format!(
                "Triton model '{}' output '{}' has datatype {}; expected FP32",
                self.name, self.output, output.datatype
            )));
        }
        let values = match response.raw_output_contents.get(index) {
            Some(raw) => {
                if !raw.len().is_multiple_of(4) {
                    return Err(unknown(format!(
                        "Triton model '{}' output '{}' has {} raw bytes, which is not a whole number of FP32 values",
                        self.name,
                        self.output,
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("chunk of four")))
                    .collect()
            }
            None => output
                .contents
                .take()
                .map(|contents| contents.fp32_contents)
                .unwrap_or_default(),
        };
        Ok((std::mem::take(&mut output.shape), values))
    }

    async fn ready(&self) -> Result<(), DynamoError> {
        if self.client.model_ready(&self.name).await? {
            Ok(())
        } else {
            Err(cannot_connect(format!(
                "Triton model '{}' is not ready",
                self.name
            )))
        }
    }
}

fn resolve_tensor(
    candidates: &[&str],
    override_name: Option<&str>,
    model: impl Fn() -> String,
    kind: &str,
    flag: &str,
) -> Result<String, DynamoError> {
    if let Some(name) = override_name {
        if !candidates.contains(&name) {
            return Err(invalid_argument(format!(
                "{} has no {kind} named '{name}'; {kind}s: {candidates:?}.",
                model()
            )));
        }
        return Ok(name.to_string());
    }
    match candidates {
        [name] => Ok((*name).to_string()),
        _ => Err(invalid_argument(format!(
            "{} has {} {kind} tensor(s); expected exactly 1. Pass {flag} to disambiguate.",
            model(),
            candidates.len()
        ))),
    }
}

/// One FP32 row per prompt, stored contiguously.
pub(crate) struct Rows {
    values: Vec<f32>,
    count: usize,
}

impl Rows {
    /// Split `values` into rows by the model's batching contract rather than the tensor rank:
    /// an unbatched model with dims=[a, b] returns one row, not `a` rows.
    fn new(values: Vec<f32>, shape: &[i64], batched: bool) -> Result<Self, String> {
        if !batched {
            return Ok(Self { values, count: 1 });
        }
        let count = match shape {
            [count, _, ..] => usize::try_from(*count).ok(),
            _ => None,
        }
        .ok_or_else(|| {
            format!("declares batching but returned shape {shape:?}; expected a leading batch axis")
        })?;
        // `is_multiple_of(0)` holds only for zero values.
        if !values.len().is_multiple_of(count) {
            return Err(format!(
                "returned {} values, which do not split into {count} rows",
                values.len()
            ));
        }
        Ok(Self { values, count })
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &[f32]> {
        let width = self.values.len().checked_div(self.count).unwrap_or(0);
        (0..self.count).map(move |row| &self.values[row * width..(row + 1) * width])
    }
}

/// Text form of a request's `input`; token-ID variants carry no text.
pub(crate) enum PromptInput {
    Single(String),
    Batch(Vec<String>),
    Tokens,
}

/// Validate a request's `input` and return its prompts.
pub(crate) fn text_prompts<T: TextTask>(input: PromptInput) -> Result<Vec<String>, DynamoError> {
    let label = T::LABEL;
    match input {
        PromptInput::Single(prompt) if prompt.is_empty() => Err(invalid_argument(format!(
            "{label} 'input' cannot be an empty string"
        ))),
        PromptInput::Single(prompt) => Ok(vec![prompt]),
        PromptInput::Batch(prompts) if prompts.is_empty() => Err(invalid_argument(format!(
            "{label} 'input' cannot be an empty list"
        ))),
        PromptInput::Batch(prompts) if prompts.iter().any(String::is_empty) => {
            Err(invalid_argument(format!(
                "{label} 'input' list must not contain empty strings"
            )))
        }
        PromptInput::Batch(prompts) => Ok(prompts),
        PromptInput::Tokens => Err(invalid_argument(format!(
            "the Triton {label} worker does not yet accept token-ID input (only text 'input' strings); \
             ask the client to send text or run the model behind a backend that owns tokenization"
        ))),
    }
}

/// Reject request controls the Triton model plan owns, so a client relying on them gets an error instead of a silently different result.
pub(crate) fn reject_controls<T: TextTask>(
    controls: &[(&str, bool)],
    priority: i64,
) -> Result<(), DynamoError> {
    let label = T::LABEL;
    if let Some((field, _)) = controls.iter().find(|(_, set)| *set) {
        return Err(invalid_argument(format!(
            "the Triton {label} worker does not honor '{field}'. \
             Send the field unset, or run the model behind a backend that honors it (vLLM)."
        )));
    }
    // `priority` is always on the wire with a default of 0, so only a non-default value is rejected.
    if priority != 0 {
        return Err(invalid_argument(format!(
            "the Triton {label} worker does not honor 'priority'; Triton's scheduling queue is not exposed to the sidecar. \
             Send priority unset (0), or run the model behind a backend that honors it (vLLM)."
        )));
    }
    Ok(())
}

/// Response model name: the request's, or the Triton model's when the request leaves it empty.
pub(crate) fn response_model(requested: String, model: &TextModel) -> String {
    if requested.is_empty() {
        model.name().to_string()
    } else {
        requested
    }
}

fn unknown(message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}

/// KServe raw BYTES layout: each element is a little-endian u32 length followed by its bytes.
fn bytes_tensor<'a>(elements: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut raw = Vec::new();
    for element in elements {
        raw.extend_from_slice(&(element.len() as u32).to_le_bytes());
        raw.extend_from_slice(element.as_bytes());
    }
    raw
}

fn parse_bytes_tensor(mut raw: &[u8]) -> Option<Vec<String>> {
    let mut elements = Vec::new();
    while !raw.is_empty() {
        let (length, rest) = raw.split_first_chunk::<4>()?;
        let length = u32::from_le_bytes(*length) as usize;
        let element = rest.get(..length)?;
        elements.push(String::from_utf8_lossy(element).into_owned());
        raw = &rest[length..];
    }
    Some(elements)
}

/// Serves one [`TextTask`] for one Triton model.
pub(crate) struct TextEngine<T> {
    task: Arc<T>,
    model: Arc<TextModel>,
}

impl<T: TextTask> TextEngine<T> {
    pub(crate) fn new(task: T, model: TextModel) -> Self {
        Self {
            task: Arc::new(task),
            model: Arc::new(model),
        }
    }
}

#[async_trait]
impl<T: TextTask> RawEngine for TextEngine<T> {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        self.model.ready().await?;
        Ok(EngineConfig {
            model: self.model.name.clone(),
            served_model_name: Some(self.model.name.clone()),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        if request.get(HEALTH_CHECK_KEY).and_then(Value::as_bool) == Some(true) {
            self.model.ready().await?;
            let response = self.task.probe_response(&self.model.name);
            return Ok(Box::pin(futures::stream::once(async move { Ok(response) })));
        }

        let (request, prompts) = self.task.parse(&self.model, request)?;
        let task = self.task.clone();
        let model = self.model.clone();
        let context_id = ctx.id().to_string();
        let context = ctx.inner_arc();
        Ok(Box::pin(async_stream::stream! {
            let response = async {
                let rows = model.infer_rows(&prompts).await?;
                task.respond(&model, request, &prompts, rows, &context_id).await
            };
            tokio::select! {
                _ = context.stopped() => return,
                _ = context.killed() => return,
                response = response => yield response,
            }
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }

    async fn health_check_payload(&self) -> Result<Option<Value>, DynamoError> {
        // Probes are answered from Triton's readiness without running inference.
        Ok(Some(
            json!({"model": self.model.name, HEALTH_CHECK_KEY: true}),
        ))
    }
}

#[cfg(test)]
mod tests {
    use dynamo_llm::grpc::service::kserve::inference;

    use super::{
        Classify, Embeddings, PromptInput, Rows, TextModel, bytes_tensor, parse_bytes_tensor,
        reject_controls, text_prompts,
    };

    pub(super) fn tensor(
        name: &str,
        data_type: inference::DataType,
        dims: &[i64],
    ) -> (String, i32, Vec<i64>) {
        (name.to_string(), data_type as i32, dims.to_vec())
    }

    pub(super) fn config(
        max_batch_size: i32,
        inputs: &[(String, i32, Vec<i64>)],
        outputs: &[(String, i32, Vec<i64>)],
    ) -> inference::ModelConfig {
        inference::ModelConfig {
            name: "model".to_string(),
            max_batch_size,
            input: inputs
                .iter()
                .map(|(name, data_type, dims)| inference::ModelInput {
                    name: name.clone(),
                    data_type: *data_type,
                    dims: dims.clone(),
                    ..Default::default()
                })
                .collect(),
            output: outputs
                .iter()
                .map(|(name, data_type, dims)| inference::ModelOutput {
                    name: name.clone(),
                    data_type: *data_type,
                    dims: dims.clone(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    pub(super) fn resolve(
        config: &inference::ModelConfig,
        input: Option<&str>,
        output: Option<&str>,
    ) -> Result<TextModel, String> {
        TextModel::new::<Classify>(
            "model",
            config,
            input,
            output,
            crate::client::TritonClient::for_tests(),
        )
        .map_err(|error| error.to_string())
    }

    #[test]
    fn resolves_the_only_string_input_and_fp32_output() {
        use inference::DataType::{TypeFp32, TypeInt32, TypeString};
        let config = config(
            8,
            &[
                tensor("TEXT", TypeString, &[1]),
                tensor("IDS", TypeInt32, &[-1]),
            ],
            &[
                tensor("probs", TypeFp32, &[3]),
                tensor("ids", TypeInt32, &[-1]),
            ],
        );
        let model = resolve(&config, None, None).unwrap();
        assert_eq!(
            (model.input.as_str(), model.output.as_str()),
            ("TEXT", "probs")
        );
        assert!(model.batched());
        assert!(!model.labelled());

        let mut labelled = config.clone();
        labelled.output[0].label_filename = "labels.txt".to_string();
        assert!(resolve(&labelled, None, None).unwrap().labelled());
    }

    #[test]
    fn rejects_ambiguous_or_unsupported_tensors() {
        use inference::DataType::{TypeFp32, TypeString};
        let two_inputs = config(
            0,
            &[
                tensor("A", TypeString, &[1]),
                tensor("B", TypeString, &[-1]),
            ],
            &[tensor("out", TypeFp32, &[3])],
        );
        let error = resolve(&two_inputs, None, None).err().unwrap();
        assert!(error.contains("2 TYPE_STRING input tensor(s)"), "{error}");
        assert!(error.contains("--classify-input-name"), "{error}");
        assert_eq!(resolve(&two_inputs, Some("B"), None).unwrap().input, "B");
        let error = resolve(&two_inputs, Some("C"), None).err().unwrap();
        assert!(error.contains("no TYPE_STRING input named 'C'"), "{error}");

        let no_output = config(0, &[tensor("A", TypeString, &[1])], &[]);
        assert!(
            resolve(&no_output, None, None)
                .err()
                .unwrap()
                .contains("0 TYPE_FP32 output")
        );

        let wide = config(
            0,
            &[tensor("A", TypeString, &[2])],
            &[tensor("out", TypeFp32, &[3])],
        );
        let error = resolve(&wide, None, None).err().unwrap();
        assert!(error.contains("dims=[2]"), "{error}");
        assert!(error.contains("/v1/classify"), "{error}");
    }

    #[test]
    fn checks_batch_against_max_batch_size() {
        use inference::DataType::{TypeFp32, TypeString};
        let prompts = |count: usize| vec!["text".to_string(); count];
        let unbatched = resolve(
            &config(
                0,
                &[tensor("A", TypeString, &[1])],
                &[tensor("o", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        assert!(unbatched.check_batch::<Classify>(&prompts(1)).is_ok());
        assert!(unbatched.check_batch::<Classify>(&prompts(2)).is_err());

        let batched = resolve(
            &config(
                2,
                &[tensor("A", TypeString, &[1])],
                &[tensor("o", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        assert!(batched.check_batch::<Embeddings>(&prompts(2)).is_ok());
        let error = batched
            .check_batch::<Embeddings>(&prompts(3))
            .unwrap_err()
            .to_string();
        assert!(error.contains("at most 2 prompts"), "{error}");
        assert!(error.contains("embedding"), "{error}");
    }

    #[test]
    fn builds_bytes_requests_shaped_by_batching() {
        use inference::DataType::{TypeFp32, TypeString};
        let batched = resolve(
            &config(
                4,
                &[tensor("TEXT", TypeString, &[1])],
                &[tensor("o", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        let prompts = ["ab".to_string(), "c".to_string()];
        let request = batched.request(&prompts, Default::default());
        assert_eq!(request.inputs[0].datatype, "BYTES");
        assert_eq!(request.inputs[0].shape, [2, 1]);
        assert_eq!(
            request.raw_input_contents,
            [vec![2, 0, 0, 0, b'a', b'b', 1, 0, 0, 0, b'c']]
        );

        let unbatched = resolve(
            &config(
                0,
                &[tensor("TEXT", TypeString, &[-1])],
                &[tensor("o", TypeFp32, &[3])],
            ),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            unbatched.request(&prompts[..1], Default::default()).inputs[0].shape,
            [1]
        );
    }

    #[test]
    fn splits_rows_by_batching_contract() {
        let values: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let rows = Rows::new(values.clone(), &[2, 3], true).unwrap();
        let rows: Vec<_> = rows.iter().collect();
        assert_eq!(rows, [&[0.0, 1.0, 2.0][..], &[3.0, 4.0, 5.0][..]]);

        // An unbatched [2, 3] output is a single row.
        let rows = Rows::new(values.clone(), &[2, 3], false).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows.iter().next().unwrap().len(), 6);

        assert!(Rows::new(values.clone(), &[6], true).is_err());
        assert!(Rows::new(values, &[4, 1], true).is_err());
        assert_eq!(Rows::new(Vec::new(), &[0, 3], true).unwrap().len(), 0);
    }

    #[test]
    fn validates_text_prompts() {
        let prompts = |input| text_prompts::<Classify>(input).map_err(|error| error.to_string());
        assert_eq!(prompts(PromptInput::Single("a".into())).unwrap(), ["a"]);
        assert_eq!(
            prompts(PromptInput::Batch(vec!["a".into(), "b".into()])).unwrap(),
            ["a", "b"]
        );
        assert!(
            prompts(PromptInput::Single(String::new()))
                .unwrap_err()
                .contains("empty string")
        );
        assert!(
            prompts(PromptInput::Batch(Vec::new()))
                .unwrap_err()
                .contains("empty list")
        );
        assert!(
            prompts(PromptInput::Batch(vec!["a".into(), String::new()]))
                .unwrap_err()
                .contains("must not contain empty strings")
        );
        assert!(
            prompts(PromptInput::Tokens)
                .unwrap_err()
                .contains("token-ID input")
        );
    }

    #[test]
    fn rejects_controls_and_priority() {
        assert!(reject_controls::<Classify>(&[("use_activation", false)], 0).is_ok());
        let error = reject_controls::<Classify>(&[("a", false), ("cache_salt", true)], 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not honor 'cache_salt'"), "{error}");
        let error = reject_controls::<Classify>(&[], 3).unwrap_err().to_string();
        assert!(error.contains("'priority'"), "{error}");
    }

    #[test]
    fn round_trips_bytes_tensors() {
        let raw = bytes_tensor(["0.9:1:dog", "", "0.1:0"].into_iter());
        assert_eq!(
            parse_bytes_tensor(&raw).unwrap(),
            ["0.9:1:dog", "", "0.1:0"]
        );
        assert!(parse_bytes_tensor(&raw[..raw.len() - 1]).is_none());
        assert!(parse_bytes_tensor(&[1, 0, 0]).is_none());
    }

    /// Binary KServe HTTP infer response with one output.
    fn infer_reply(
        name: &str,
        datatype: &str,
        shape: &[i64],
        raw: Vec<u8>,
    ) -> crate::http::tests::Reply {
        let mut body = serde_json::to_vec(&serde_json::json!({
            "model_name": "model",
            "outputs": [{
                "name": name,
                "datatype": datatype,
                "shape": shape,
                "parameters": {"binary_data_size": raw.len()},
            }],
        }))
        .unwrap();
        let header_length = body.len();
        body.extend_from_slice(&raw);
        crate::http::tests::Reply {
            status: "200 OK",
            headers: vec![("Inference-Header-Content-Length", header_length.to_string())],
            body,
        }
    }

    fn fp32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    async fn generate<T: super::TextTask>(
        engine: &super::TextEngine<T>,
        request: serde_json::Value,
    ) -> serde_json::Value {
        use dynamo_backend_common::RawEngine;
        use futures::StreamExt;
        let context = dynamo_backend_common::GenerateContext::new(
            dynamo_backend_common::testing::mock_context(),
            None,
        );
        let mut stream = engine.generate(request, context).await.unwrap();
        let response = stream.next().await.unwrap().unwrap();
        assert!(stream.next().await.is_none());
        response
    }

    #[tokio::test]
    async fn serves_labelled_classification_through_triton() {
        use crate::http::tests::{Reply, serve, transport};
        use inference::DataType::{TypeFp32, TypeString};

        let (endpoint, server) = serve(vec![
            Reply::json("200 OK", serde_json::json!({})),
            infer_reply(
                "probs",
                "FP32",
                &[2, 3],
                fp32(&[0.1, 0.7, 0.2, 0.6, 0.3, 0.1]),
            ),
            infer_reply(
                "probs",
                "BYTES",
                &[1, 3],
                bytes_tensor(["0.7:1:positive", "0.2:2:neutral", "0.1:0:negative"].into_iter()),
            ),
            infer_reply("probs", "FP32", &[1, 3], fp32(&[0.2, 0.2, 0.6])),
            Reply::json("200 OK", serde_json::json!({})),
        ])
        .await;
        let endpoint = dynamo_sidecar_common::HttpEndpoint::parse(&endpoint, "test").unwrap();
        let client = crate::client::TritonClient::from_http(
            crate::http::HttpClient::connect(&endpoint, transport())
                .await
                .unwrap(),
        );
        let mut model_config = config(
            8,
            &[tensor("TEXT", TypeString, &[1])],
            &[tensor("probs", TypeFp32, &[3])],
        );
        model_config.output[0].label_filename = "labels.txt".to_string();
        let model = TextModel::new::<Classify>("model", &model_config, None, None, client).unwrap();
        let engine = super::TextEngine::new(Classify::default(), model);

        let response = generate(
            &engine,
            serde_json::json!({"model": "model", "input": ["good", "fine"]}),
        )
        .await;
        assert_eq!(response["data"][0]["label"], "positive");
        assert_eq!(response["data"][1]["label"], "negative");
        assert_eq!(response["data"][0]["probs"].as_array().unwrap().len(), 3);

        // Labels are cached, so the next request runs one inference.
        let response = generate(
            &engine,
            serde_json::json!({"model": "model", "input": "ok"}),
        )
        .await;
        assert_eq!(response["data"][0]["label"], "neutral");

        let probe = generate(
            &engine,
            serde_json::json!({"model": "model", dynamo_backend_common::HEALTH_CHECK_KEY: true}),
        )
        .await;
        assert_eq!(probe["data"], serde_json::json!([]));

        let requests = server.await.unwrap();
        let lines: Vec<_> = requests.iter().map(|(line, _)| line.as_str()).collect();
        assert_eq!(
            lines,
            [
                "GET /triton/v2/health/ready HTTP/1.1",
                "POST /triton/v2/models/model/infer HTTP/1.1",
                "POST /triton/v2/models/model/infer HTTP/1.1",
                "POST /triton/v2/models/model/infer HTTP/1.1",
                "GET /triton/v2/models/model/ready HTTP/1.1",
            ]
        );
        // The label request asks for every class of the first prompt.
        let header = String::from_utf8_lossy(&requests[2].1);
        assert!(header.contains(r#""classification":3"#), "{header}");
        assert!(header.contains(r#""shape":[1,1]"#), "{header}");
    }

    #[tokio::test]
    async fn serves_embeddings_through_triton() {
        use crate::http::tests::{Reply, serve, transport};
        use inference::DataType::{TypeFp32, TypeString};

        let (endpoint, server) = serve(vec![
            Reply::json("200 OK", serde_json::json!({})),
            infer_reply("embedding", "FP32", &[2], fp32(&[0.5, -1.0])),
            infer_reply("embedding", "FP32", &[4], fp32(&[0.5, -1.0, 0.0, 1.0])),
        ])
        .await;
        let endpoint = dynamo_sidecar_common::HttpEndpoint::parse(&endpoint, "test").unwrap();
        let client = crate::client::TritonClient::from_http(
            crate::http::HttpClient::connect(&endpoint, transport())
                .await
                .unwrap(),
        );
        // Unbatched: input shape [1], and the whole output is one row.
        let model = TextModel::new::<Embeddings>(
            "model",
            &config(
                0,
                &[tensor("TEXT", TypeString, &[-1])],
                &[tensor("embedding", TypeFp32, &[-1])],
            ),
            None,
            None,
            client,
        )
        .unwrap();
        let engine = super::TextEngine::new(Embeddings, model);
        let response = generate(
            &engine,
            serde_json::json!({"model": "model", "input": "hello"}),
        )
        .await;
        assert_eq!(response["data"][0]["embedding"], "AAAAPwAAgL8=");

        let response = generate(
            &engine,
            serde_json::json!({"model": "model", "input": "world"}),
        )
        .await;
        assert_eq!(response["data"].as_array().unwrap().len(), 1);

        let requests = server.await.unwrap();
        let header = String::from_utf8_lossy(&requests[1].1);
        assert!(header.contains(r#""shape":[1]"#), "{header}");
        assert!(header.contains(r#""name":"embedding""#), "{header}");
    }
}
