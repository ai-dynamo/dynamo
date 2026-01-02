// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::Annotated;
use anyhow::Result;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bytes,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::Bool => size_of::<bool>(),
            DataType::Uint8 => size_of::<u8>(),
            DataType::Uint16 => size_of::<u16>(),
            DataType::Uint32 => size_of::<u32>(),
            DataType::Uint64 => size_of::<u64>(),
            DataType::Int8 => size_of::<i8>(),
            DataType::Int16 => size_of::<i16>(),
            DataType::Int32 => size_of::<i32>(),
            DataType::Int64 => size_of::<i64>(),
            DataType::Float32 => size_of::<f32>(),
            DataType::Float64 => size_of::<f64>(),
            DataType::Bytes => 0, // variable length, return 0 as indicator
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Deserialize)]
// Self-describing encoding removes ambiguity between signed/unsigned and width variants.
#[serde(tag = "data_type", content = "values")]
pub enum FlattenTensor {
    Bool(Vec<bool>),
    // [gluo NOTE] f16, and bf16 is not stably supported
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    // Typically use to store string data, but really it can store
    // arbitrary data such as serialized handles for custom worker behavior.
    Bytes(Vec<Vec<u8>>),
}

#[allow(clippy::len_without_is_empty)]
impl FlattenTensor {
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Uint8(v) => v.len(),
            Self::Uint16(v) => v.len(),
            Self::Uint32(v) => v.len(),
            Self::Uint64(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Bytes(v) => v.len(),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::Uint8(_) => DataType::Uint8,
            Self::Uint16(_) => DataType::Uint16,
            Self::Uint32(_) => DataType::Uint32,
            Self::Uint64(_) => DataType::Uint64,
            Self::Int8(_) => DataType::Int8,
            Self::Int16(_) => DataType::Int16,
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::Float32(_) => DataType::Float32,
            Self::Float64(_) => DataType::Float64,
            Self::Bytes(_) => DataType::Bytes,
        }
    }
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: DataType,
    pub shape: Vec<i64>,

    /// Optional parameters for this tensor
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub parameters: Parameters,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq, Default)]
pub struct TensorModelConfig {
    pub name: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
    // Optional Triton model config in serialized protobuf string,
    // if provided, it supersedes the basic model config defined above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triton_model_config: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tensor {
    pub metadata: TensorMetadata,
    pub data: FlattenTensor,
}

impl validator::Validate for Tensor {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        use validator::{ValidationError, ValidationErrors};
        let mut errs = ValidationErrors::new();

        // dtype must match
        if self.metadata.data_type != self.data.data_type() {
            let mut e = ValidationError::new("dtype_mismatch");
            e.message = Some("metadata.data_type does not match data variant".into());
            errs.add("data_type", e);
        }

        let mut product: usize = 1;
        for &d in &self.metadata.shape {
            if d < 0 {
                let mut e = ValidationError::new("negative_dim");
                e.message = Some("only -1 is allowed as a wildcard dimension".into());
                errs.add("shape", e);
                break;
            }
            product = product.saturating_mul(d as usize);
        }
        // bytes payloads may be variable-length per item; enforce outer count only
        let expect_count = self.data.len();
        if product != expect_count {
            let mut e = ValidationError::new("element_count_mismatch");
            e.message = Some(
                format!(
                    "shape implies {} elements but data has {}",
                    product, expect_count
                )
                .into(),
            );
            errs.add("shape", e);
        }

        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateTensorRequest {
    /// ID of the request
    pub id: Option<String>,

    /// ID of the model to use.
    pub model: String,

    /// Input tensors.
    pub tensors: Vec<Tensor>,

    /// Optional request-level parameters
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub parameters: Parameters,

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

    /// Optional response-level parameters
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub parameters: Parameters,
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
                            if aggregator.error.is_none() {
                                aggregator.error = Some(error);
                            }
                            return aggregator;
                        }
                    };
                    match delta.data {
                        Some(resp) => {
                            if aggregator.response.is_none() {
                                aggregator.response = Some(resp);
                            } else if aggregator.error.is_none() {
                                aggregator.error =
                                    Some("Multiple responses in non-streaming mode".to_string());
                            }
                        }
                        None => {
                            // Ignore metadata-only deltas in non-streaming mode.
                        }
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

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ParameterValue {
    Bool(bool),
    Int64(i64),
    String(String),
    Double(f64),
    Uint64(u64),
}

pub type Parameters = HashMap<String, ParameterValue>;
