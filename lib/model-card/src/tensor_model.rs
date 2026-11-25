// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tensor model metadata a model deployment card carries: the model's input and output
//! tensors, their data types, and their parameters. The tensor request and response
//! protocol builds on these types.

use std::collections::HashMap;
use std::str::FromStr;

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use validator::Validate;

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
    Float16,
    BFloat16,
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
            DataType::Float16 => size_of::<f16>(),
            DataType::BFloat16 => size_of::<bf16>(),
            DataType::Float32 => size_of::<f32>(),
            DataType::Float64 => size_of::<f64>(),
            DataType::Bytes => 0, // variable length, return 0 as indicator
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            DataType::Bool => write!(f, "BOOL"),
            DataType::Uint8 => write!(f, "UINT8"),
            DataType::Uint16 => write!(f, "UINT16"),
            DataType::Uint32 => write!(f, "UINT32"),
            DataType::Uint64 => write!(f, "UINT64"),
            DataType::Int8 => write!(f, "INT8"),
            DataType::Int16 => write!(f, "INT16"),
            DataType::Int32 => write!(f, "INT32"),
            DataType::Int64 => write!(f, "INT64"),
            DataType::Float16 => write!(f, "FP16"),
            DataType::BFloat16 => write!(f, "BF16"),
            DataType::Float32 => write!(f, "FP32"),
            DataType::Float64 => write!(f, "FP64"),
            DataType::Bytes => write!(f, "BYTES"),
        }
    }
}

impl FromStr for DataType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "BOOL" => Ok(DataType::Bool),
            "UINT8" => Ok(DataType::Uint8),
            "UINT16" => Ok(DataType::Uint16),
            "UINT32" => Ok(DataType::Uint32),
            "UINT64" => Ok(DataType::Uint64),
            "INT8" => Ok(DataType::Int8),
            "INT16" => Ok(DataType::Int16),
            "INT32" => Ok(DataType::Int32),
            "INT64" => Ok(DataType::Int64),
            "FP16" => Ok(DataType::Float16),
            "BF16" => Ok(DataType::BFloat16),
            "FP32" => Ok(DataType::Float32),
            "FP64" => Ok(DataType::Float64),
            "BYTES" => Ok(DataType::Bytes),
            _ => Err(anyhow::anyhow!("Invalid data type")),
        }
    }
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: DataType,
    pub shape: Vec<i64>,

    /// Optional parameters for this tensor
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub parameters: Parameters,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TensorModelConfig {
    pub name: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
    // Optional Triton model config in serialized protobuf string,
    // if provided, it supersedes the basic model config defined above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triton_model_config: Option<Vec<u8>>,
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
