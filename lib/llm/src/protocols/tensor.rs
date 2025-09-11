// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::Annotated;
use anyhow::Result;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use validator::Validate;

// [gluo TODO] whether it makes sense to have aggregator for tensor..
// we could if considering aggregation to be stacking the tensors by adding
// one more dimension. i.e. stream of [2, 2] tensors to be aggregated to
// [-1, 2, 2]. Will decide it later and currently do not allow aggregation.
// mod aggregator;

// pub use aggregator::DeltaAggregator;

// [gluo TODO] nvext is LLM specific, we really only use the annotation field
pub use super::openai::nvext::{NvExt, NvExtProvider};

#[derive(Debug, Serialize, Clone, Eq, PartialEq, Deserialize)]
pub enum DataType {
    Bool,
    Uint32,
    Int32,
    Float32,
    Bytes,
}

#[derive(Debug, Serialize, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum FlattenTensor {
    Bool(Vec<bool>),
    // [gluo WIP] fill all numerical types in different precisions
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    // Typically use to store string data, but really it can store
    // arbitrary data such as serialized handles for custom worker behavior.
    Bytes(Vec<Vec<u8>>),
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, Eq, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: DataType,
    pub shape: Vec<i64>,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, Eq, PartialEq)]
pub struct TensorModelConfig {
    pub name: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct Tensor {
    pub metadata: TensorMetadata,
    // [gluo WIP] data type is determined by the enum variant.
    // verify if the request can be properly pass to Python side as
    // here is enum with data and there is no proper mapping in Python.
    pub data: FlattenTensor,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateTensorRequest {
    /// ID of the request
    pub id: Option<String>,

    /// ID of the model to use.
    pub model: String,

    /// Input tensors.
    pub tensors: Vec<Tensor>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateTensorResponse {
    /// ID of the corresponding request.
    pub id: Option<String>,

    /// ID of the model.
    pub model: String,

    /// Output tensors.
    pub tensors: Vec<Tensor>,
}

/// Implements `NvExtProvider` for `NvCreateTensorRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateTensorRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        // Not really apply here.
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateTensorRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateTensorRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

pub struct DeltaAggregator {
    response: Option<NvCreateTensorResponse>,
    error: Option<String>,
}

impl NvCreateTensorResponse {
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateTensorResponse>>,
    ) -> Result<NvCreateTensorResponse> {
        let aggregator = stream
            .fold(
                DeltaAggregator {
                    response: None,
                    error: None,
                },
                |mut aggregator, delta| async move {
                    let delta = match delta.ok() {
                        Ok(delta) => delta,
                        Err(error) => {
                            aggregator.error = Some(error);
                            return aggregator;
                        }
                    };

                    if aggregator.response.is_none() {
                        if delta.data.is_some() {
                            aggregator.response = Some(delta.data.unwrap());
                        } else {
                            aggregator.error = Some("No data in response".to_string());
                        }
                    } else {
                        aggregator.error =
                            Some("Multiple responses in non-streaming mode".to_string());
                    }
                    aggregator
                },
            )
            .await;
        if let Some(error) = aggregator.error {
            Err(anyhow::anyhow!(error))
        } else if let Some(response) = aggregator.response {
            Ok(response)
        } else {
            Err(anyhow::anyhow!("No response received"))
        }
    }
}
