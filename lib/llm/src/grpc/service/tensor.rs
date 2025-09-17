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

impl TryFrom<inference::ModelInferRequest> for NvCreateTensorRequest {
    type Error = Status;

    fn try_from(mut request: inference::ModelInferRequest) -> Result<Self, Self::Error> {
        // Protocol requires if `raw_input_contents` is used to hold input data,
        // it must be used for all inputs.
        if !request.raw_input_contents.is_empty()
            && request.inputs.len() != request.raw_input_contents.len()
        {
            return Err(Status::invalid_argument(
                "`raw_input_contents` must be used for all inputs",
            ));
        }

        let mut tensor_request = NvCreateTensorRequest {
            id: if !request.id.is_empty() {
                Some(request.id.clone())
            } else {
                None
            },
            model: request.model_name.clone(),
            tensors: Vec::new(),
            nvext: None,
        };

        // iterate through inputs
        for input in request.inputs {
            let mut tensor = Tensor {
                metadata: TensorMetadata {
                    name: input.name.clone(),
                    data_type: tensor::DataType::from_str(&input.datatype)
                        .map_err(|err| Status::invalid_argument(err.to_string()))?,
                    shape: input.shape.clone(),
                },
                // Placeholder, will be filled below
                data: tensor::FlattenTensor::Bool(Vec::new()),
            };
            tensor.data = match &input.contents {
                Some(content) => match tensor.metadata.data_type {
                    tensor::DataType::Bool => {
                        tensor::FlattenTensor::Bool(content.bool_contents.clone())
                    }
                    tensor::DataType::Uint8 => tensor::FlattenTensor::Uint8(
                        content.uint_contents.iter().map(|&x| x as u8).collect(),
                    ),
                    tensor::DataType::Uint16 => tensor::FlattenTensor::Uint16(
                        content.uint_contents.iter().map(|&x| x as u16).collect(),
                    ),
                    tensor::DataType::Uint32 => {
                        tensor::FlattenTensor::Uint32(content.uint_contents.clone())
                    }
                    tensor::DataType::Uint64 => {
                        tensor::FlattenTensor::Uint64(content.uint64_contents.clone())
                    }
                    tensor::DataType::Int8 => tensor::FlattenTensor::Int8(
                        content.int_contents.iter().map(|&x| x as i8).collect(),
                    ),
                    tensor::DataType::Int16 => tensor::FlattenTensor::Int16(
                        content.int_contents.iter().map(|&x| x as i16).collect(),
                    ),
                    tensor::DataType::Int32 => {
                        tensor::FlattenTensor::Int32(content.int_contents.clone())
                    }
                    tensor::DataType::Int64 => {
                        tensor::FlattenTensor::Int64(content.int64_contents.clone())
                    }

                    tensor::DataType::Float32 => {
                        tensor::FlattenTensor::Float32(content.fp32_contents.clone())
                    }

                    tensor::DataType::Float64 => {
                        tensor::FlattenTensor::Float64(content.fp64_contents.clone())
                    }

                    tensor::DataType::Bytes => {
                        tensor::FlattenTensor::Bytes(content.bytes_contents.clone())
                    }
                },
                // data is provided in raw_input_contents
                None => {
                    // Take ownership of the raw input for later pointer operation
                    let raw_input = request.raw_input_contents.remove(0);
                    let data_size = match tensor.metadata.data_type {
                        tensor::DataType::Bool => 1,
                        tensor::DataType::Uint8 => 1,
                        tensor::DataType::Uint16 => 2,
                        tensor::DataType::Uint32 => 4,
                        tensor::DataType::Uint64 => 8,
                        tensor::DataType::Int8 => 1,
                        tensor::DataType::Int16 => 2,
                        tensor::DataType::Int32 => 4,
                        tensor::DataType::Int64 => 8,
                        tensor::DataType::Float32 => 4,
                        tensor::DataType::Float64 => 8,
                        tensor::DataType::Bytes => 0,
                    };
                    // Non-bytes type, simply reinterpret cast the raw input bytes
                    if data_size > 0 {
                        let element_count =
                            tensor.metadata.shape.iter().try_fold(1usize, |acc, &d| {
                                if d < 0 {
                                    Err(Status::invalid_argument(format!(
                                        "Shape contains negative dimension: {}",
                                        d
                                    )))
                                } else {
                                    acc.checked_mul(d as usize).ok_or_else(|| {
                                        Status::invalid_argument(
                                            "Overflow occurred while calculating element count",
                                        )
                                    })
                                }
                            })?;
                        if raw_input.len() % data_size != 0 {
                            return Err(Status::invalid_argument(format!(
                                "Raw input length must be a multiple of {}",
                                data_size
                            )));
                        } else if raw_input.len() / data_size != element_count {
                            return Err(Status::invalid_argument(format!(
                                "Raw input element count for '{}' does not match expected size, expected {} elements, got {} elements",
                                input.name,
                                element_count,
                                raw_input.len() / data_size
                            )));
                        }

                        // Here we "reinterpret cast" vec<u8> to be vec<T> of the corresponding data type
                        // To do so we extract the raw pointer to construct new vector<T>,
                        // and forget the original vector to avoid double-free.
                        let ptr = raw_input.as_ptr();
                        std::mem::forget(raw_input); // Prevent deallocation of the vector
                        unsafe {
                            match tensor.metadata.data_type {
                                tensor::DataType::Bool => {
                                    tensor::FlattenTensor::Bool(Vec::from_raw_parts(
                                        ptr as *mut bool,
                                        element_count,
                                        element_count,
                                    ))
                                }
                                tensor::DataType::Uint8 => {
                                    tensor::FlattenTensor::Uint8(Vec::from_raw_parts(
                                        ptr as *mut u8,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Uint16 => {
                                    tensor::FlattenTensor::Uint16(Vec::from_raw_parts(
                                        ptr as *mut u16,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Uint32 => {
                                    tensor::FlattenTensor::Uint32(Vec::from_raw_parts(
                                        ptr as *mut u32,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Uint64 => {
                                    tensor::FlattenTensor::Uint64(Vec::from_raw_parts(
                                        ptr as *mut u64,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Int8 => {
                                    tensor::FlattenTensor::Int8(Vec::from_raw_parts(
                                        ptr as *mut i8,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Int16 => {
                                    tensor::FlattenTensor::Int16(Vec::from_raw_parts(
                                        ptr as *mut i16,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Int32 => {
                                    tensor::FlattenTensor::Int32(Vec::from_raw_parts(
                                        ptr as *mut i32,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Int64 => {
                                    tensor::FlattenTensor::Int64(Vec::from_raw_parts(
                                        ptr as *mut i64,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Float32 => {
                                    tensor::FlattenTensor::Float32(Vec::from_raw_parts(
                                        ptr as *mut f32,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Float64 => {
                                    tensor::FlattenTensor::Float64(Vec::from_raw_parts(
                                        ptr as *mut f64,
                                        element_count,
                                        element_count,
                                    ))
                                }

                                tensor::DataType::Bytes => {
                                    return Err(Status::internal(format!(
                                        "Unexpected BYTES type in non-bytes branch for input '{}'",
                                        input.name
                                    )));
                                }
                            }
                        }
                    } else {
                        // For BYTES type, we need to parse length-prefixed strings and properly slice them
                        // into bytes of array.
                        let mut bytes_contents = vec![];
                        let mut offset = 0;
                        while offset + 4 <= raw_input.len() {
                            let len = u32::from_le_bytes(
                                raw_input[offset..offset + 4].try_into().unwrap(),
                            ) as usize;
                            offset += 4;
                            if offset + len > raw_input.len() {
                                return Err(Status::invalid_argument(format!(
                                    "Invalid length-prefixed BYTES input for '{}', length exceeds raw input size",
                                    input.name
                                )));
                            }
                            bytes_contents.push(raw_input[offset..offset + len].to_vec());
                            offset += len;
                        }
                        if offset != raw_input.len() {
                            return Err(Status::invalid_argument(format!(
                                "Invalid length-prefixed BYTES input for '{}', extra bytes at the end",
                                input.name
                            )));
                        }
                        tensor::FlattenTensor::Bytes(bytes_contents)
                    }
                }
            };
            tensor_request.tensors.push(tensor);
        }
        Ok(tensor_request)
    }
}

impl TryFrom<NvCreateTensorResponse> for inference::ModelInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateTensorResponse) -> Result<Self, Self::Error> {
        let mut infer_response = inference::ModelInferResponse {
            model_name: response.model,
            model_version: "1".to_string(),
            id: response.id.unwrap_or_default(),
            outputs: vec![],
            parameters: ::std::collections::HashMap::<String, inference::InferParameter>::new(),
            raw_output_contents: vec![],
        };
        for tensor in &response.tensors {
            infer_response
                .outputs
                .push(inference::model_infer_response::InferOutputTensor {
                    name: tensor.metadata.name.clone(),
                    datatype: tensor.metadata.data_type.to_string(),
                    shape: tensor.metadata.shape.clone(),
                    contents: match &tensor.data {
                        tensor::FlattenTensor::Bool(data) => Some(inference::InferTensorContents {
                            bool_contents: data.clone(),
                            ..Default::default()
                        }),
                        tensor::FlattenTensor::Uint8(data) => {
                            Some(inference::InferTensorContents {
                                uint_contents: data.iter().map(|&x| x as u32).collect(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Uint16(data) => {
                            Some(inference::InferTensorContents {
                                uint_contents: data.iter().map(|&x| x as u32).collect(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Uint32(data) => {
                            Some(inference::InferTensorContents {
                                uint_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Uint64(data) => {
                            Some(inference::InferTensorContents {
                                uint64_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Int8(data) => Some(inference::InferTensorContents {
                            int_contents: data.iter().map(|&x| x as i32).collect(),
                            ..Default::default()
                        }),
                        tensor::FlattenTensor::Int16(data) => {
                            Some(inference::InferTensorContents {
                                int_contents: data.iter().map(|&x| x as i32).collect(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Int32(data) => {
                            Some(inference::InferTensorContents {
                                int_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Int64(data) => {
                            Some(inference::InferTensorContents {
                                int64_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Float32(data) => {
                            Some(inference::InferTensorContents {
                                fp32_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Float64(data) => {
                            Some(inference::InferTensorContents {
                                fp64_contents: data.clone(),
                                ..Default::default()
                            })
                        }

                        tensor::FlattenTensor::Bytes(data) => {
                            Some(inference::InferTensorContents {
                                bytes_contents: data.clone(),
                                ..Default::default()
                            })
                        }
                    },
                    ..Default::default()
                });
        }

        Ok(infer_response)
    }
}

impl TryFrom<NvCreateTensorResponse> for inference::ModelStreamInferResponse {
    type Error = anyhow::Error;

    fn try_from(response: NvCreateTensorResponse) -> Result<Self, Self::Error> {
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
