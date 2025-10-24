// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    engine::AsyncEngineContext,
    pipeline::{AsyncEngineContextProvider, Context},
    protocols::annotated::AnnotationsProvider,
};
use futures::{Stream, StreamExt, stream};
use std::str::FromStr;
use std::sync::Arc;

use crate::types::Annotated;

use super::kserve;

// [gluo NOTE] These are common utilities that should be shared between frontends
use crate::http::service::{
    disconnect::{ConnectionHandle, create_connection_monitor},
    metrics::{Endpoint, ResponseMetricCollector},
};
use crate::{http::service::metrics::InflightGuard, preprocessor::LLMMetricAnnotation};

use crate::protocols::tensor;
use crate::protocols::tensor::{
    NvCreateTensorRequest, NvCreateTensorResponse, Tensor, TensorMetadata,
};

use crate::grpc::service::kserve::inference;
use crate::grpc::service::kserve::inference::DataType;

use tonic::Status;

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

use inference::infer_parameter::ParameterChoice;

// Extend the NvCreateTensorResponse to include options to control
// the conversion to ModelInferResponse / ModelStreamInferResponse
pub struct ExtendedNvCreateTensorResponse {
    pub response: NvCreateTensorResponse,
    pub set_raw_output_contents: bool,
}

/// Tensor Request Handler
///
/// This method will handle the incoming request for model type tensor. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
pub async fn tensor_response_stream(
    state: Arc<kserve::State>,
    request: NvCreateTensorRequest,
    streaming: bool,
) -> Result<impl Stream<Item = Annotated<NvCreateTensorResponse>>, Status> {
    // create the context for the request
    let request_id = get_or_create_request_id(request.id.as_deref());
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    // [gluo TODO] revisit metrics to properly expose it
    // create the connection handles
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.model;

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_tensor_engine(model)
        .map_err(|_| Status::not_found("model not found"))?;

    let inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Tensor, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        Status::internal(format!("Failed to generate tensor response stream: {}", e))
    })?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // prepare any requested annotations
    let annotations = annotations.map_or(Vec::new(), |annotations| {
        annotations
            .iter()
            .filter_map(|annotation| {
                if annotation == ANNOTATION_REQUEST_ID {
                    Annotated::<NvCreateTensorResponse>::from_annotation(
                        ANNOTATION_REQUEST_ID,
                        &request_id,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    // apply any annotations to the front of the stream
    let stream = stream::iter(annotations).chain(stream);

    // Tap on the stream to collect response metrics
    let stream = stream.inspect(move |response| {
        process_metrics_only(response, &mut response_collector);
    });

    let stream = grpc_monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    Ok(stream)
}

/// This method will consume an AsyncEngineStream and monitor for disconnects or context cancellation.
/// This is gRPC variant of `monitor_for_disconnects` as that implementation has SSE specific handling.
/// Should decouple and reuse `monitor_for_disconnects`
///
/// Uses `tokio::select!` to choose between receiving responses from the source stream or detecting when
/// the context is stopped. If the context is stopped, we break the stream. If the source stream ends
/// naturally, we mark the request as successful and send the final `[DONE]` event.
pub fn grpc_monitor_for_disconnects<T>(
    stream: impl Stream<Item = Annotated<T>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
    mut stream_handle: ConnectionHandle,
) -> impl Stream<Item = Annotated<T>> {
    stream_handle.arm();
    async_stream::stream! {
        tokio::pin!(stream);
        loop {
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(response) => {
                            yield response;
                        }
                        None => {
                            // Stream ended normally
                            inflight_guard.mark_ok();
                            stream_handle.disarm();
                            break;
                        }
                    }
                }
                // todo - test request cancellation with kserve frontend and tensor-based models
                _ = context.stopped() => {
                    tracing::trace!("Context stopped; breaking stream");
                    break;
                }
            }
        }
    }
}

fn process_metrics_only<T>(
    annotated: &Annotated<T>,
    response_collector: &mut ResponseMetricCollector,
) {
    // update metrics
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(annotated) {
        response_collector.observe_current_osl(metrics.output_tokens);
        response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);
    }
}

/// Get the request ID from a primary source, or lastly create a new one if not present
fn get_or_create_request_id(primary: Option<&str>) -> String {
    // Try to get the request ID from the primary source
    if let Some(primary) = primary
        && let Ok(uuid) = uuid::Uuid::parse_str(primary)
    {
        return uuid.to_string();
    }

    // Try to parse the request ID as a UUID, or generate a new one if missing/invalid
    let uuid = uuid::Uuid::new_v4();
    uuid.to_string()
}

/// Convert KServe InferParameter to Dynamo ParameterValue
#[allow(clippy::result_large_err)]
pub fn kserve_param_to_dynamo(
    key: &str,
    param: &inference::InferParameter,
) -> Result<tensor::ParameterValue, Status> {
    param
        .parameter_choice
        .as_ref()
        .ok_or_else(|| Status::invalid_argument(format!("Parameter '{}' has no value", key)))
        .map(|choice| match choice {
            ParameterChoice::BoolParam(v) => tensor::ParameterValue::Bool(*v),
            ParameterChoice::Int64Param(v) => tensor::ParameterValue::Int64(*v),
            ParameterChoice::StringParam(v) => tensor::ParameterValue::String(v.clone()),
            ParameterChoice::DoubleParam(v) => tensor::ParameterValue::Double(*v),
            ParameterChoice::Uint64Param(v) => tensor::ParameterValue::Uint64(*v),
        })
}

/// Convert Dynamo ParameterValue to KServe InferParameter
pub fn dynamo_param_to_kserve(param: &tensor::ParameterValue) -> inference::InferParameter {
    let parameter_choice = match param {
        tensor::ParameterValue::Bool(v) => ParameterChoice::BoolParam(*v),
        tensor::ParameterValue::Int64(v) => ParameterChoice::Int64Param(*v),
        tensor::ParameterValue::String(v) => ParameterChoice::StringParam(v.clone()),
        tensor::ParameterValue::Double(v) => ParameterChoice::DoubleParam(*v),
        tensor::ParameterValue::Uint64(v) => ParameterChoice::Uint64Param(*v),
    };

    inference::InferParameter {
        parameter_choice: Some(parameter_choice),
    }
}

/// Convert KServe parameter map to Dynamo Parameters
#[allow(clippy::result_large_err)]
fn convert_kserve_to_dynamo_params(
    kserve_params: &std::collections::HashMap<String, inference::InferParameter>,
) -> Result<tensor::Parameters, Status> {
    kserve_params
        .iter()
        .map(|(k, v)| kserve_param_to_dynamo(k, v).map(|param_value| (k.clone(), param_value)))
        .collect()
}

/// Convert Dynamo Parameters to KServe parameter map
fn convert_dynamo_to_kserve_params(
    dynamo_params: &tensor::Parameters,
) -> std::collections::HashMap<String, inference::InferParameter> {
    dynamo_params
        .iter()
        .map(|(k, v)| (k.clone(), dynamo_param_to_kserve(v)))
        .collect()
}

impl TryFrom<inference::ModelInferRequest> for NvCreateTensorRequest {
    type Error = Status;

    fn try_from(request: inference::ModelInferRequest) -> Result<Self, Self::Error> {
        // Protocol requires if `raw_input_contents` is used to hold input data,
        // it must be used for all inputs.
        if !request.raw_input_contents.is_empty()
            && request.inputs.len() != request.raw_input_contents.len()
        {
            return Err(Status::invalid_argument(
                "`raw_input_contents` must be used for all inputs",
            ));
        }

        // Extract request-level parameters
        let parameters = convert_kserve_to_dynamo_params(&request.parameters)?;

        let mut tensor_request = NvCreateTensorRequest {
            id: if !request.id.is_empty() {
                Some(request.id.clone())
            } else {
                None
            },
            model: request.model_name.clone(),
            tensors: Vec::new(),
            parameters,
            nvext: None,
        };

        // iterate through inputs
        for (idx, input) in request.inputs.into_iter().enumerate() {
            // Extract per-tensor parameters
            let tensor_parameters = convert_kserve_to_dynamo_params(&input.parameters)?;

            let mut tensor = Tensor {
                metadata: TensorMetadata {
                    name: input.name.clone(),
                    data_type: tensor::DataType::from_str(&input.datatype)
                        .map_err(|err| Status::invalid_argument(err.to_string()))?,
                    shape: input.shape.clone(),
                    parameters: tensor_parameters,
                },
                // Placeholder, will be filled below
                data: tensor::FlattenTensor::Bool(Vec::new()),
            };
            match &input.contents {
                // If contents is provided in InferInputTensor
                Some(contents) => {
                    tensor.set_data_from_tensor_contents(contents);
                }
                // If not in InferInputTensor, contents is provided in raw_input_contents
                None => {
                    tensor.set_data_from_raw_contents(&request.raw_input_contents[idx])?;
                }
            }
            tensor_request.tensors.push(tensor);
        }
        Ok(tensor_request)
    }
}

impl tensor::Tensor {
    fn set_data_from_tensor_contents(&mut self, contents: &inference::InferTensorContents) {
        self.data = match self.metadata.data_type {
            tensor::DataType::Bool => tensor::FlattenTensor::Bool(contents.bool_contents.clone()),
            tensor::DataType::Uint8 => tensor::FlattenTensor::Uint8(
                contents.uint_contents.iter().map(|&x| x as u8).collect(),
            ),
            tensor::DataType::Uint16 => tensor::FlattenTensor::Uint16(
                contents.uint_contents.iter().map(|&x| x as u16).collect(),
            ),
            tensor::DataType::Uint32 => {
                tensor::FlattenTensor::Uint32(contents.uint_contents.clone())
            }
            tensor::DataType::Uint64 => {
                tensor::FlattenTensor::Uint64(contents.uint64_contents.clone())
            }
            tensor::DataType::Int8 => tensor::FlattenTensor::Int8(
                contents.int_contents.iter().map(|&x| x as i8).collect(),
            ),
            tensor::DataType::Int16 => tensor::FlattenTensor::Int16(
                contents.int_contents.iter().map(|&x| x as i16).collect(),
            ),
            tensor::DataType::Int32 => tensor::FlattenTensor::Int32(contents.int_contents.clone()),
            tensor::DataType::Int64 => {
                tensor::FlattenTensor::Int64(contents.int64_contents.clone())
            }

            tensor::DataType::Float32 => {
                tensor::FlattenTensor::Float32(contents.fp32_contents.clone())
            }

            tensor::DataType::Float64 => {
                tensor::FlattenTensor::Float64(contents.fp64_contents.clone())
            }

            tensor::DataType::Bytes => {
                tensor::FlattenTensor::Bytes(contents.bytes_contents.clone())
            }
        }
    }

    #[allow(clippy::result_large_err)]
    fn set_data_from_raw_contents(&mut self, raw_input: &[u8]) -> Result<(), Status> {
        let element_count = self.metadata.shape.iter().try_fold(1usize, |acc, &d| {
            if d < 0 {
                Err(Status::invalid_argument(format!(
                    "Shape contains negative dimension: {}",
                    d
                )))
            } else {
                acc.checked_mul(d as usize).ok_or_else(|| {
                    Status::invalid_argument("Overflow occurred while calculating element count")
                })
            }
        })?;

        let data_size = self.metadata.data_type.size();

        // For BYTES type, we need to parse length-prefixed strings and properly slice them
        // into bytes of array, and early return
        if data_size == 0 {
            self.data = self.raw_input_to_bytes_tensor(element_count, raw_input)?;
            return Ok(());
        }

        // Control reaches here on non-bytes types
        // validate raw input length before conversion
        if !raw_input.len().is_multiple_of(data_size) {
            return Err(Status::invalid_argument(format!(
                "Raw input length must be a multiple of {}",
                data_size
            )));
        } else if raw_input.len() / data_size != element_count {
            return Err(Status::invalid_argument(format!(
                "Raw input element count for '{}' does not match expected size, expected {} elements, got {} elements",
                self.metadata.name,
                element_count,
                raw_input.len() / data_size
            )));
        }
        self.data = self.raw_input_to_typed_tensor(raw_input)?;

        Ok(())
    }

    #[allow(clippy::result_large_err)]
    fn raw_input_to_bytes_tensor(
        &self,
        element_count: usize,
        raw_input: &[u8],
    ) -> Result<tensor::FlattenTensor, Status> {
        // element is not fixed size for bytes type, so the raw input has
        // length-prefixed bytes for each element.
        let mut bytes_contents = vec![];
        let mut offset = 0;
        while offset + 4 <= raw_input.len() {
            let len =
                u32::from_le_bytes(raw_input[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + len > raw_input.len() {
                return Err(Status::invalid_argument(format!(
                    "Invalid length-prefixed BYTES input for '{}', length exceeds raw input size",
                    self.metadata.name
                )));
            }
            bytes_contents.push(raw_input[offset..offset + len].to_vec());
            offset += len;
        }
        if offset != raw_input.len() {
            return Err(Status::invalid_argument(format!(
                "Invalid length-prefixed BYTES input for '{}', extra bytes at the end",
                self.metadata.name
            )));
        }
        if element_count != bytes_contents.len() {
            return Err(Status::invalid_argument(format!(
                "Raw input element count for '{}' does not match expected size, expected {} elements, got {} elements",
                self.metadata.name,
                element_count,
                bytes_contents.len()
            )));
        }
        Ok(tensor::FlattenTensor::Bytes(bytes_contents))
    }

    #[allow(clippy::result_large_err)]
    fn raw_input_to_typed_tensor(&self, raw_input: &[u8]) -> Result<tensor::FlattenTensor, Status> {
        // In Rust, we can not "reinterpret cast" a Vec<u8> to Vec<T> directly
        // as Vec require the pointer to be aligned with the type T, which can not
        // be guaranteed from Vec<u8>. We will have to reconstruct the Vec<T> element
        // by element which results in data copy.
        // Here we assume little endianess for all types as the KServe protocol doesn't
        // specify the endianness while it should have.
        match self.metadata.data_type {
            tensor::DataType::Bool => Ok(tensor::FlattenTensor::Bool(
                raw_input.iter().map(|&b| b != 0).collect(),
            )),
            tensor::DataType::Uint8 => Ok(tensor::FlattenTensor::Uint8(
                raw_input.chunks_exact(1).map(|chunk| chunk[0]).collect(),
            )),
            tensor::DataType::Uint16 => Ok(tensor::FlattenTensor::Uint16(
                raw_input
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect(),
            )),
            tensor::DataType::Uint32 => Ok(tensor::FlattenTensor::Uint32(
                raw_input
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            )),
            tensor::DataType::Uint64 => Ok(tensor::FlattenTensor::Uint64(
                raw_input
                    .chunks_exact(8)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            )),
            tensor::DataType::Int8 => Ok(tensor::FlattenTensor::Int8(
                raw_input
                    .chunks_exact(1)
                    .map(|chunk| chunk[0] as i8)
                    .collect(),
            )),
            tensor::DataType::Int16 => Ok(tensor::FlattenTensor::Int16(
                raw_input
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect(),
            )),
            tensor::DataType::Int32 => Ok(tensor::FlattenTensor::Int32(
                raw_input
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            )),
            tensor::DataType::Int64 => Ok(tensor::FlattenTensor::Int64(
                raw_input
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            )),
            tensor::DataType::Float32 => Ok(tensor::FlattenTensor::Float32(
                raw_input
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            )),
            tensor::DataType::Float64 => Ok(tensor::FlattenTensor::Float64(
                raw_input
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            )),
            tensor::DataType::Bytes => Err(Status::internal(format!(
                "Unexpected BYTES type in non-bytes branch for input '{}'",
                self.metadata.name
            ))),
        }
    }
}

impl TryFrom<ExtendedNvCreateTensorResponse> for inference::ModelInferResponse {
    type Error = anyhow::Error;

    fn try_from(extended_response: ExtendedNvCreateTensorResponse) -> Result<Self, Self::Error> {
        let response = extended_response.response;

        // Convert response-level parameters
        let parameters = convert_dynamo_to_kserve_params(&response.parameters);

        let mut infer_response = inference::ModelInferResponse {
            model_name: response.model,
            model_version: "1".to_string(),
            id: response.id.unwrap_or_default(),
            outputs: vec![],
            parameters,
            raw_output_contents: vec![],
        };
        for tensor in &response.tensors {
            // Convert per-tensor parameters
            let tensor_parameters = convert_dynamo_to_kserve_params(&tensor.metadata.parameters);

            infer_response
                .outputs
                .push(inference::model_infer_response::InferOutputTensor {
                    name: tensor.metadata.name.clone(),
                    datatype: tensor.metadata.data_type.to_string(),
                    shape: tensor.metadata.shape.clone(),
                    parameters: tensor_parameters,
                    ..Default::default()
                });
            if extended_response.set_raw_output_contents {
                infer_response.add_raw_output_contents(tensor)?;
            } else {
                infer_response.fill_last_tensor_contents(tensor);
            }
        }

        Ok(infer_response)
    }
}

impl inference::ModelInferResponse {
    /// Serializes the tensor data into a standardized little-endian byte format
    /// and appends it to the raw_output_contents field.
    ///
    /// This ensures consistent cross-platform representation of numerical values
    /// regardless of the host machine's native endianness. Each tensor element is
    /// flattened and converted to its corresponding little-endian byte sequence,
    /// matching the protocol format expected by Triton Inference Server and
    /// similar inference runtimes.
    pub fn add_raw_output_contents(
        &mut self,
        tensor: &tensor::Tensor,
    ) -> Result<(), anyhow::Error> {
        let raw_content = match &tensor.data {
            tensor::FlattenTensor::Bool(data) => {
                data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect()
            }
            tensor::FlattenTensor::Uint8(data) => data.clone(),
            tensor::FlattenTensor::Uint16(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Uint32(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Uint64(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Int8(data) => data.iter().map(|&x| x as u8).collect(),
            tensor::FlattenTensor::Int16(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Int32(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Int64(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Float32(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Float64(data) => {
                data.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            tensor::FlattenTensor::Bytes(data) => {
                let mut bytes = Vec::new();
                for item in data {
                    let len = item.len() as u32;
                    bytes.extend_from_slice(&len.to_le_bytes());
                    bytes.extend_from_slice(item);
                }
                bytes
            }
        };
        self.raw_output_contents.push(raw_content);
        Ok(())
    }

    pub fn fill_last_tensor_contents(&mut self, tensor: &tensor::Tensor) {
        if self.outputs.is_empty() {
            return;
        }
        self.outputs.last_mut().unwrap().contents = match &tensor.data {
            tensor::FlattenTensor::Bool(data) => Some(inference::InferTensorContents {
                bool_contents: data.clone(),
                ..Default::default()
            }),
            tensor::FlattenTensor::Uint8(data) => Some(inference::InferTensorContents {
                uint_contents: data.iter().map(|&x| x as u32).collect(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Uint16(data) => Some(inference::InferTensorContents {
                uint_contents: data.iter().map(|&x| x as u32).collect(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Uint32(data) => Some(inference::InferTensorContents {
                uint_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Uint64(data) => Some(inference::InferTensorContents {
                uint64_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Int8(data) => Some(inference::InferTensorContents {
                int_contents: data.iter().map(|&x| x as i32).collect(),
                ..Default::default()
            }),
            tensor::FlattenTensor::Int16(data) => Some(inference::InferTensorContents {
                int_contents: data.iter().map(|&x| x as i32).collect(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Int32(data) => Some(inference::InferTensorContents {
                int_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Int64(data) => Some(inference::InferTensorContents {
                int64_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Float32(data) => Some(inference::InferTensorContents {
                fp32_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Float64(data) => Some(inference::InferTensorContents {
                fp64_contents: data.clone(),
                ..Default::default()
            }),

            tensor::FlattenTensor::Bytes(data) => Some(inference::InferTensorContents {
                bytes_contents: data.clone(),
                ..Default::default()
            }),
        };
    }
}

impl TryFrom<ExtendedNvCreateTensorResponse> for inference::ModelStreamInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: ExtendedNvCreateTensorResponse) -> Result<Self, Self::Error> {
        match inference::ModelInferResponse::try_from(response) {
            Ok(response) => Ok(inference::ModelStreamInferResponse {
                infer_response: Some(response),
                ..Default::default()
            }),
            Err(e) => Ok(inference::ModelStreamInferResponse {
                infer_response: None,
                error_message: format!("Failed to convert response: {}", e),
            }),
        }
    }
}

impl tensor::DataType {
    pub fn to_kserve(&self) -> i32 {
        match *self {
            tensor::DataType::Bool => DataType::TypeBool as i32,
            tensor::DataType::Uint8 => DataType::TypeUint8 as i32,
            tensor::DataType::Uint16 => DataType::TypeUint16 as i32,
            tensor::DataType::Uint32 => DataType::TypeUint32 as i32,
            tensor::DataType::Uint64 => DataType::TypeUint64 as i32,
            tensor::DataType::Int8 => DataType::TypeInt8 as i32,
            tensor::DataType::Int16 => DataType::TypeInt16 as i32,
            tensor::DataType::Int32 => DataType::TypeInt32 as i32,
            tensor::DataType::Int64 => DataType::TypeInt64 as i32,
            tensor::DataType::Float32 => DataType::TypeFp32 as i32,
            tensor::DataType::Float64 => DataType::TypeFp64 as i32,
            tensor::DataType::Bytes => DataType::TypeString as i32,
        }
    }
}

impl std::fmt::Display for tensor::DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            tensor::DataType::Bool => write!(f, "BOOL"),
            tensor::DataType::Uint8 => write!(f, "UINT8"),
            tensor::DataType::Uint16 => write!(f, "UINT16"),
            tensor::DataType::Uint32 => write!(f, "UINT32"),
            tensor::DataType::Uint64 => write!(f, "UINT64"),
            tensor::DataType::Int8 => write!(f, "INT8"),
            tensor::DataType::Int16 => write!(f, "INT16"),
            tensor::DataType::Int32 => write!(f, "INT32"),
            tensor::DataType::Int64 => write!(f, "INT64"),
            tensor::DataType::Float32 => write!(f, "FP32"),
            tensor::DataType::Float64 => write!(f, "FP64"),
            tensor::DataType::Bytes => write!(f, "BYTES"),
        }
    }
}

impl FromStr for tensor::DataType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "BOOL" => Ok(tensor::DataType::Bool),
            "UINT8" => Ok(tensor::DataType::Uint8),
            "UINT16" => Ok(tensor::DataType::Uint16),
            "UINT32" => Ok(tensor::DataType::Uint32),
            "UINT64" => Ok(tensor::DataType::Uint64),
            "INT8" => Ok(tensor::DataType::Int8),
            "INT16" => Ok(tensor::DataType::Int16),
            "INT32" => Ok(tensor::DataType::Int32),
            "INT64" => Ok(tensor::DataType::Int64),
            "FP32" => Ok(tensor::DataType::Float32),
            "FP64" => Ok(tensor::DataType::Float64),
            "BYTES" => Ok(tensor::DataType::Bytes),
            _ => Err(anyhow::anyhow!("Invalid data type")),
        }
    }
}
