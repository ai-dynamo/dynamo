// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Translation between the KServe v2 protobuf messages and Triton's HTTP/REST
//! JSON encoding with the binary tensor data extension.
//!
//! The HTTP transport accepts and returns the same protobuf messages as the
//! gRPC transport so request and response conversion stays shared. Tensor
//! contents travel as binary data in both directions, matching gRPC's raw
//! contents, so neither path converts tensors through JSON arrays.

use std::collections::HashMap;
use std::sync::LazyLock;

use dynamo_backend_common::DynamoError;
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_llm::grpc::service::kserve::inference::infer_parameter::ParameterChoice;
use dynamo_sidecar_common::invalid_argument;
use prost::Message;
use prost_reflect::{DescriptorPool, DeserializeOptions, DynamicMessage};
use serde::Deserialize;
use serde_json::{Map, Number, Value, json};

use super::protocol_error;

/// Length of the JSON header that precedes binary tensor data in a body.
pub(super) const HEADER_LENGTH: &str = "inference-header-content-length";
const BINARY_DATA_SIZE: &str = "binary_data_size";
const BINARY_DATA: &str = "binary_data";
const BINARY_DATA_OUTPUT: &str = "binary_data_output";

static DESCRIPTORS: LazyLock<DescriptorPool> = LazyLock::new(|| {
    DescriptorPool::decode(
        include_bytes!(concat!(env!("OUT_DIR"), "/model_config_descriptor.bin")).as_slice(),
    )
    .expect("build script emits a valid model_config.proto descriptor set")
});

pub(super) struct EncodedRequest {
    pub(super) body: Vec<u8>,
    pub(super) header_length: usize,
}

impl EncodedRequest {
    pub(super) fn has_binary_data(&self) -> bool {
        self.body.len() > self.header_length
    }
}

/// Encode a gRPC infer request as a JSON header followed by its raw inputs,
/// asking Triton to return every output as binary data.
pub(super) fn encode_infer_request(
    request: &inference::ModelInferRequest,
) -> Result<EncodedRequest, DynamoError> {
    if request.raw_input_contents.len() != request.inputs.len() {
        return Err(invalid_argument(format!(
            "Triton HTTP requests need raw contents for every input; got {} inputs and {} raw buffers",
            request.inputs.len(),
            request.raw_input_contents.len()
        )));
    }

    let mut parameters = parameters_to_json(&request.parameters)?;
    parameters.insert(BINARY_DATA_OUTPUT.into(), Value::Bool(true));
    let inputs = request
        .inputs
        .iter()
        .zip(&request.raw_input_contents)
        .map(|(input, raw)| {
            if input.contents.is_some() {
                return Err(invalid_argument(format!(
                    "input tensor '{}' has both typed and raw contents",
                    input.name
                )));
            }
            let mut parameters = parameters_to_json(&input.parameters)?;
            parameters.insert(BINARY_DATA_SIZE.into(), Value::from(raw.len()));
            Ok(json!({
                "name": input.name,
                "datatype": input.datatype,
                "shape": input.shape,
                "parameters": parameters,
            }))
        })
        .collect::<Result<Vec<_>, DynamoError>>()?;
    let outputs = request
        .outputs
        .iter()
        .map(|output| {
            let mut parameters = parameters_to_json(&output.parameters)?;
            parameters.insert(BINARY_DATA.into(), Value::Bool(true));
            Ok(json!({"name": output.name, "parameters": parameters}))
        })
        .collect::<Result<Vec<_>, DynamoError>>()?;

    let mut header = Map::new();
    if !request.id.is_empty() {
        header.insert("id".into(), Value::String(request.id.clone()));
    }
    header.insert("parameters".into(), Value::Object(parameters));
    header.insert("inputs".into(), Value::Array(inputs));
    if !outputs.is_empty() {
        header.insert("outputs".into(), Value::Array(outputs));
    }

    let mut body = serde_json::to_vec(&header).map_err(|error| {
        invalid_argument(format!("could not serialize Triton HTTP request: {error}"))
    })?;
    let header_length = body.len();
    body.reserve(request.raw_input_contents.iter().map(Vec::len).sum());
    for raw in &request.raw_input_contents {
        body.extend_from_slice(raw);
    }
    Ok(EncodedRequest {
        body,
        header_length,
    })
}

#[derive(Deserialize)]
struct InferResponse {
    model_name: String,
    #[serde(default)]
    model_version: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    parameters: Map<String, Value>,
    #[serde(default)]
    outputs: Vec<OutputTensor>,
}

#[derive(Deserialize)]
struct OutputTensor {
    name: String,
    datatype: String,
    #[serde(default)]
    shape: Vec<i64>,
    #[serde(default)]
    parameters: Map<String, Value>,
    #[serde(default)]
    data: Option<Value>,
}

/// Decode an HTTP infer response into the gRPC message, with every output in
/// `raw_output_contents` as Triton's gRPC endpoint returns it.
pub(super) fn decode_infer_response(
    body: &[u8],
    header_length: Option<usize>,
) -> Result<inference::ModelInferResponse, DynamoError> {
    let header_length = header_length.unwrap_or(body.len());
    if header_length > body.len() {
        return Err(protocol_error(format!(
            "JSON header length {header_length} exceeds the {}-byte body",
            body.len()
        )));
    }
    let (header, mut binary) = body.split_at(header_length);
    let response: InferResponse = serde_json::from_slice(header)
        .map_err(|error| protocol_error(format!("invalid inference response: {error}")))?;

    let mut outputs = Vec::with_capacity(response.outputs.len());
    let mut raw_output_contents = Vec::with_capacity(response.outputs.len());
    for mut output in response.outputs {
        let raw = match output.parameters.remove(BINARY_DATA_SIZE) {
            Some(size) => {
                let size = size
                    .as_u64()
                    .and_then(|size| usize::try_from(size).ok())
                    .filter(|size| *size <= binary.len())
                    .ok_or_else(|| {
                        protocol_error(format!(
                            "output '{}' has binary_data_size {size} but {} bytes remain",
                            output.name,
                            binary.len()
                        ))
                    })?;
                let (raw, rest) = binary.split_at(size);
                binary = rest;
                raw.to_vec()
            }
            None => {
                let data = output.data.take().ok_or_else(|| {
                    protocol_error(format!("output '{}' has no data", output.name))
                })?;
                json_to_raw(&output.datatype, &data)
                    .map_err(|error| protocol_error(format!("output '{}': {error}", output.name)))?
            }
        };
        outputs.push(inference::model_infer_response::InferOutputTensor {
            name: output.name,
            datatype: output.datatype,
            shape: output.shape,
            parameters: parameters_from_json(output.parameters)?,
            contents: None,
        });
        raw_output_contents.push(raw);
    }
    if !binary.is_empty() {
        return Err(protocol_error(format!(
            "{} bytes of binary data follow the last output",
            binary.len()
        )));
    }

    Ok(inference::ModelInferResponse {
        model_name: response.model_name,
        model_version: response.model_version,
        id: response.id,
        parameters: parameters_from_json(response.parameters)?,
        outputs,
        raw_output_contents,
    })
}

/// Convert Triton's JSON model configuration into the protobuf message the
/// gRPC `ModelConfig` RPC returns.
pub(super) fn model_config_from_json(json: &[u8]) -> Result<inference::ModelConfig, DynamoError> {
    let descriptor = DESCRIPTORS
        .get_message_by_name("inference.ModelConfig")
        .expect("model_config.proto defines inference.ModelConfig");
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    // Newer Triton releases may report fields this proto does not define yet.
    let options = DeserializeOptions::new().deny_unknown_fields(false);
    let config = DynamicMessage::deserialize_with_options(descriptor, &mut deserializer, &options)
        .and_then(|config| deserializer.end().map(|()| config))
        .map_err(|error| protocol_error(format!("invalid model configuration: {error}")))?;
    let encoded = prost_reflect::prost::Message::encode_to_vec(&config);
    inference::ModelConfig::decode(encoded.as_slice())
        .map_err(|error| protocol_error(format!("invalid model configuration: {error}")))
}

fn parameters_to_json(
    parameters: &HashMap<String, inference::InferParameter>,
) -> Result<Map<String, Value>, DynamoError> {
    parameters
        .iter()
        .map(|(key, parameter)| {
            let value = match parameter.parameter_choice.as_ref() {
                Some(ParameterChoice::BoolParam(value)) => Value::Bool(*value),
                Some(ParameterChoice::Int64Param(value)) => Value::from(*value),
                Some(ParameterChoice::Uint64Param(value)) => Value::from(*value),
                Some(ParameterChoice::StringParam(value)) => Value::String(value.clone()),
                Some(ParameterChoice::DoubleParam(value)) => {
                    Number::from_f64(*value).map(Value::Number).ok_or_else(|| {
                        invalid_argument(format!("parameter '{key}' is not a finite number"))
                    })?
                }
                None => {
                    return Err(invalid_argument(format!("parameter '{key}' has no value")));
                }
            };
            Ok((key.clone(), value))
        })
        .collect()
}

fn parameters_from_json(
    parameters: Map<String, Value>,
) -> Result<HashMap<String, inference::InferParameter>, DynamoError> {
    parameters
        .into_iter()
        .map(|(key, value)| {
            let choice = match value {
                Value::Bool(value) => ParameterChoice::BoolParam(value),
                Value::String(value) => ParameterChoice::StringParam(value),
                Value::Number(value) => {
                    if let Some(value) = value.as_i64() {
                        ParameterChoice::Int64Param(value)
                    } else if let Some(value) = value.as_u64() {
                        ParameterChoice::Uint64Param(value)
                    } else if let Some(value) = value.as_f64() {
                        ParameterChoice::DoubleParam(value)
                    } else {
                        return Err(protocol_error(format!(
                            "parameter '{key}' has unsupported number {value}"
                        )));
                    }
                }
                other => {
                    return Err(protocol_error(format!(
                        "parameter '{key}' has unsupported value {other}"
                    )));
                }
            };
            Ok((
                key,
                inference::InferParameter {
                    parameter_choice: Some(choice),
                },
            ))
        })
        .collect()
}

/// Encode JSON tensor data in the KServe raw little-endian layout.
fn json_to_raw(datatype: &str, data: &Value) -> Result<Vec<u8>, String> {
    fn flatten<'a>(value: &'a Value, elements: &mut Vec<&'a Value>) {
        match value {
            Value::Array(values) => values.iter().for_each(|value| flatten(value, elements)),
            value => elements.push(value),
        }
    }
    let mut elements = Vec::new();
    flatten(data, &mut elements);

    let invalid = |element: &Value| format!("invalid {datatype} element {element}");
    let mut raw = Vec::new();
    macro_rules! encode {
        ($ty:ty, $get:ident) => {
            for element in &elements {
                let value = element
                    .$get()
                    .and_then(|value| <$ty>::try_from(value).ok())
                    .ok_or_else(|| invalid(element))?;
                raw.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    match datatype {
        "BOOL" => {
            for element in &elements {
                raw.push(u8::from(element.as_bool().ok_or_else(|| invalid(element))?));
            }
        }
        "UINT8" => encode!(u8, as_u64),
        "UINT16" => encode!(u16, as_u64),
        "UINT32" => encode!(u32, as_u64),
        "UINT64" => encode!(u64, as_u64),
        "INT8" => encode!(i8, as_i64),
        "INT16" => encode!(i16, as_i64),
        "INT32" => encode!(i32, as_i64),
        "INT64" => encode!(i64, as_i64),
        "FP32" => {
            for element in &elements {
                let value = element.as_f64().ok_or_else(|| invalid(element))? as f32;
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        "FP64" => encode!(f64, as_f64),
        "BYTES" => {
            for element in &elements {
                let value = element.as_str().ok_or_else(|| invalid(element))?;
                let length = u32::try_from(value.len()).map_err(|_| invalid(element))?;
                raw.extend_from_slice(&length.to_le_bytes());
                raw.extend_from_slice(value.as_bytes());
            }
        }
        other => return Err(format!("{other} data must use the binary data extension")),
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_llm::grpc::service::kserve::inference;
    use dynamo_llm::grpc::service::kserve::inference::infer_parameter::ParameterChoice;
    use serde_json::{Value, json};

    use super::{decode_infer_response, encode_infer_request, model_config_from_json};

    fn parameter(choice: ParameterChoice) -> inference::InferParameter {
        inference::InferParameter {
            parameter_choice: Some(choice),
        }
    }

    fn body(header: Value, binary: &[u8]) -> (Vec<u8>, usize) {
        let mut body = serde_json::to_vec(&header).unwrap();
        let header_length = body.len();
        body.extend_from_slice(binary);
        (body, header_length)
    }

    #[test]
    fn request_sends_raw_inputs_as_binary_data() {
        let request = inference::ModelInferRequest {
            model_name: "identity".to_string(),
            id: "request-1".to_string(),
            parameters: HashMap::from([(
                "priority".to_string(),
                parameter(ParameterChoice::Int64Param(2)),
            )]),
            inputs: vec![
                inference::model_infer_request::InferInputTensor {
                    name: "INPUT0".to_string(),
                    datatype: "FP32".to_string(),
                    shape: vec![2],
                    ..Default::default()
                },
                inference::model_infer_request::InferInputTensor {
                    name: "INPUT1".to_string(),
                    datatype: "UINT8".to_string(),
                    shape: vec![1],
                    ..Default::default()
                },
            ],
            outputs: vec![inference::model_infer_request::InferRequestedOutputTensor {
                name: "OUTPUT0".to_string(),
                parameters: HashMap::from([(
                    "classification".to_string(),
                    parameter(ParameterChoice::Int64Param(1)),
                )]),
            }],
            raw_input_contents: vec![vec![1; 8], vec![7]],
            ..Default::default()
        };
        let encoded = encode_infer_request(&request).unwrap();
        assert!(encoded.has_binary_data());
        let (header, binary) = encoded.body.split_at(encoded.header_length);
        let header: Value = serde_json::from_slice(header).unwrap();
        assert_eq!(
            header,
            json!({
                "id": "request-1",
                "parameters": {"priority": 2, "binary_data_output": true},
                "inputs": [
                    {
                        "name": "INPUT0",
                        "datatype": "FP32",
                        "shape": [2],
                        "parameters": {"binary_data_size": 8},
                    },
                    {
                        "name": "INPUT1",
                        "datatype": "UINT8",
                        "shape": [1],
                        "parameters": {"binary_data_size": 1},
                    },
                ],
                "outputs": [
                    {"name": "OUTPUT0", "parameters": {"classification": 1, "binary_data": true}},
                ],
            })
        );
        assert_eq!(binary, [1, 1, 1, 1, 1, 1, 1, 1, 7]);
    }

    #[test]
    fn request_rejects_typed_contents() {
        let request = inference::ModelInferRequest {
            inputs: vec![inference::model_infer_request::InferInputTensor {
                name: "INPUT0".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1],
                contents: Some(inference::InferTensorContents {
                    int_contents: vec![1],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(encode_infer_request(&request).is_err());
    }

    #[test]
    fn response_reads_binary_outputs_in_order() {
        let (body, header_length) = body(
            json!({
                "model_name": "identity",
                "model_version": "1",
                "id": "request-1",
                "parameters": {"sequence_end": true, "big": u64::MAX, "ratio": 0.5},
                "outputs": [
                    {
                        "name": "OUTPUT0",
                        "datatype": "INT32",
                        "shape": [2],
                        "parameters": {"binary_data_size": 8},
                    },
                    {
                        "name": "OUTPUT1",
                        "datatype": "BOOL",
                        "shape": [1],
                        "parameters": {"binary_data_size": 1, "custom": "kept"},
                    },
                ],
            }),
            &[3, 0, 0, 0, 4, 0, 0, 0, 1],
        );
        let response = decode_infer_response(&body, Some(header_length)).unwrap();
        assert_eq!(response.model_name, "identity");
        assert_eq!(response.model_version, "1");
        assert_eq!(response.id, "request-1");
        assert_eq!(
            response.parameters["sequence_end"],
            parameter(ParameterChoice::BoolParam(true))
        );
        assert_eq!(
            response.parameters["big"],
            parameter(ParameterChoice::Uint64Param(u64::MAX))
        );
        assert_eq!(
            response.parameters["ratio"],
            parameter(ParameterChoice::DoubleParam(0.5))
        );
        assert_eq!(
            response.raw_output_contents,
            [vec![3, 0, 0, 0, 4, 0, 0, 0], vec![1]]
        );
        assert!(response.outputs[0].parameters.is_empty());
        assert_eq!(
            response.outputs[1].parameters["custom"],
            parameter(ParameterChoice::StringParam("kept".to_string()))
        );

        let dynamo = crate::convert::from_triton_response(response).unwrap();
        assert_eq!(
            dynamo.tensors[0].data,
            dynamo_llm::protocols::tensor::FlattenTensor::Int32(vec![3, 4])
        );
    }

    #[test]
    fn response_encodes_json_outputs_as_raw() {
        let (body, _) = body(
            json!({
                "model_name": "m",
                "outputs": [
                    {"name": "I", "datatype": "INT16", "shape": [2, 1], "data": [[-1], [2]]},
                    {"name": "F", "datatype": "FP32", "shape": [1], "data": [1.5]},
                    {"name": "S", "datatype": "BYTES", "shape": [1], "data": ["ab"]},
                    {"name": "B", "datatype": "BOOL", "shape": [2], "data": [true, false]},
                ],
            }),
            &[],
        );
        let response = decode_infer_response(&body, None).unwrap();
        assert_eq!(
            response.raw_output_contents,
            [
                vec![0xff, 0xff, 2, 0],
                1.5_f32.to_le_bytes().to_vec(),
                vec![2, 0, 0, 0, b'a', b'b'],
                vec![1, 0],
            ]
        );
    }

    #[test]
    fn response_rejects_malformed_binary_layout() {
        let header = json!({
            "model_name": "m",
            "outputs": [{
                "name": "O",
                "datatype": "UINT8",
                "shape": [2],
                "parameters": {"binary_data_size": 2},
            }],
        });
        // Too few bytes, too many bytes, and a header longer than the body.
        let (short, length) = body(header.clone(), &[1]);
        assert!(decode_infer_response(&short, Some(length)).is_err());
        let (long, length) = body(header.clone(), &[1, 2, 3]);
        assert!(decode_infer_response(&long, Some(length)).is_err());
        let (exact, length) = body(header, &[1, 2]);
        assert!(decode_infer_response(&exact, Some(length)).is_ok());
        assert!(decode_infer_response(&exact, Some(exact.len() + 1)).is_err());

        let (out_of_range, _) = body(
            json!({
                "model_name": "m",
                "outputs": [{"name": "O", "datatype": "UINT8", "shape": [1], "data": [256]}],
            }),
            &[],
        );
        assert!(decode_infer_response(&out_of_range, None).is_err());
    }

    #[test]
    fn model_config_matches_grpc_message() {
        // Triton's HTTP config uses proto field names, enum names, and may
        // render 64-bit integers either as numbers or strings.
        let config = model_config_from_json(
            json!({
                "name": "identity",
                "platform": "",
                "backend": "identity",
                "max_batch_size": 8,
                "input": [{
                    "name": "INPUT0",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, "16"],
                }],
                "output": [{"name": "OUTPUT0", "data_type": "TYPE_STRING", "dims": [1]}],
                "instance_group": [{"kind": "KIND_GPU", "count": 1, "gpus": [0]}],
                "dynamic_batching": {"max_queue_delay_microseconds": "100"},
                "field_from_a_newer_triton": {"ignored": true},
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        assert_eq!(config.name, "identity");
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.input[0].dims, [-1, 16]);
        assert_eq!(
            config.input[0].data_type,
            inference::DataType::TypeFp32 as i32
        );
        assert_eq!(
            config.output[0].data_type,
            inference::DataType::TypeString as i32
        );
        assert_eq!(
            config.instance_group[0].kind,
            inference::model_instance_group::Kind::Gpu as i32
        );
        assert!(matches!(
            config.scheduling_choice,
            Some(inference::model_config::SchedulingChoice::DynamicBatching(
                _
            ))
        ));

        assert!(model_config_from_json(b"{\"max_batch_size\": \"eight\"}").is_err());
        assert!(model_config_from_json(b"{} trailing").is_err());
    }
}
