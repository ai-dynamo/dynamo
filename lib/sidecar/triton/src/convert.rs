// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::DynamoError;
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_llm::grpc::service::tensor::dynamo_param_to_kserve;
use dynamo_llm::protocols::tensor::{NvCreateTensorRequest, NvCreateTensorResponse};
use validator::Validate;

use dynamo_sidecar_common::{invalid_argument, protocol_error};

pub(crate) fn to_triton_request(
    request: NvCreateTensorRequest,
    model: &str,
) -> Result<inference::ModelInferRequest, DynamoError> {
    request
        .validate()
        .map_err(|error| invalid_argument(format!("invalid tensor request: {error}")))?;
    if request.model != model {
        return Err(invalid_argument(format!(
            "request model '{}' does not match sidecar model '{model}'",
            request.model
        )));
    }

    let mut raw = inference::ModelInferResponse::default();
    let mut inputs = Vec::with_capacity(request.tensors.len());
    for tensor in &request.tensors {
        raw.add_raw_output_contents(tensor).map_err(|error| {
            invalid_argument(format!(
                "could not serialize input tensor '{}': {error}",
                tensor.metadata.name
            ))
        })?;
        inputs.push(inference::model_infer_request::InferInputTensor {
            name: tensor.metadata.name.clone(),
            datatype: tensor.metadata.data_type.to_string(),
            shape: tensor.metadata.shape.clone(),
            parameters: tensor
                .metadata
                .parameters
                .iter()
                .map(|(key, value)| (key.clone(), dynamo_param_to_kserve(value)))
                .collect(),
            contents: None,
        });
    }

    Ok(inference::ModelInferRequest {
        model_name: model.to_string(),
        model_version: String::new(),
        id: request.id.unwrap_or_default(),
        parameters: request
            .parameters
            .iter()
            .map(|(key, value)| (key.clone(), dynamo_param_to_kserve(value)))
            .collect(),
        inputs,
        outputs: request
            .outputs
            .into_iter()
            .map(
                |output| inference::model_infer_request::InferRequestedOutputTensor {
                    name: output.name,
                    parameters: output
                        .parameters
                        .iter()
                        .map(|(key, value)| (key.clone(), dynamo_param_to_kserve(value)))
                        .collect(),
                },
            )
            .collect(),
        raw_input_contents: raw.raw_output_contents,
    })
}

pub(crate) fn from_triton_response(
    response: inference::ModelInferResponse,
) -> Result<NvCreateTensorResponse, DynamoError> {
    if !response.raw_output_contents.is_empty()
        && response.outputs.len() != response.raw_output_contents.len()
    {
        return Err(protocol_error(
            "Triton",
            format!(
                "returned {} output descriptors but {} raw output buffers",
                response.outputs.len(),
                response.raw_output_contents.len()
            ),
        ));
    }

    let request = inference::ModelInferRequest {
        model_name: response.model_name,
        model_version: response.model_version,
        id: response.id,
        parameters: response.parameters,
        inputs: response
            .outputs
            .into_iter()
            .map(|output| inference::model_infer_request::InferInputTensor {
                name: output.name,
                datatype: output.datatype,
                shape: output.shape,
                parameters: output.parameters,
                contents: output.contents,
            })
            .collect(),
        outputs: Vec::new(),
        raw_input_contents: response.raw_output_contents,
    };
    let request = NvCreateTensorRequest::try_from(request).map_err(|status| {
        protocol_error("Triton", format!("invalid inference response: {status}"))
    })?;
    Ok(NvCreateTensorResponse {
        id: request.id,
        model: request.model,
        tensors: request.tensors,
        parameters: request.parameters,
    })
}

#[cfg(test)]
mod tests {
    use dynamo_llm::grpc::service::kserve::inference;
    use dynamo_llm::protocols::tensor::{
        DataType, FlattenTensor, NvCreateTensorRequest, Tensor, TensorMetadata,
    };

    use super::{from_triton_response, to_triton_request};

    #[test]
    fn request_uses_raw_kserve_contents() {
        let request = NvCreateTensorRequest {
            id: Some("request-1".to_string()),
            model: "identity".to_string(),
            tensors: vec![Tensor {
                metadata: TensorMetadata {
                    name: "INPUT0".to_string(),
                    data_type: DataType::Float32,
                    shape: vec![2],
                    parameters: Default::default(),
                },
                data: FlattenTensor::Float32(vec![1.5, 2.5]),
            }],
            outputs: Vec::new(),
            parameters: Default::default(),
            nvext: None,
        };
        let triton = to_triton_request(request, "identity").unwrap();
        assert_eq!(triton.inputs.len(), 1);
        assert!(triton.inputs[0].contents.is_none());
        assert_eq!(triton.raw_input_contents[0].len(), 8);
    }

    #[test]
    fn response_reuses_kserve_tensor_validation() {
        let response = inference::ModelInferResponse {
            model_name: "identity".to_string(),
            id: "request-1".to_string(),
            outputs: vec![inference::model_infer_response::InferOutputTensor {
                name: "OUTPUT0".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![2],
                contents: Some(inference::InferTensorContents {
                    int_contents: vec![3, 4],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let dynamo = from_triton_response(response).unwrap();
        assert_eq!(dynamo.model, "identity");
        assert_eq!(dynamo.tensors[0].data, FlattenTensor::Int32(vec![3, 4]));
    }
}
