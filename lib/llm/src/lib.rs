// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

use std::{fs::File, io::BufReader, path::Path};

use anyhow::Context as _;

pub mod backend;
pub mod common;
pub mod disagg_router;
pub mod discovery;
pub mod engines;
pub mod gguf;
pub mod http;
pub mod hub;
pub mod key_value_store;
pub mod kv_router;
pub mod local_model;
pub mod mocker;
pub mod model_card;
pub mod model_type;
pub mod preprocessor;
pub mod protocols;
pub mod recorder;
pub mod request_template;
pub mod tokenizers;
pub mod tokens;
pub mod types;

#[cfg(feature = "block-manager")]
pub mod block_manager;

/// Reads a JSON file, extracts a specific field, and deserializes it into type T.
///
/// # Arguments
///
/// * `json_file_path`: Path to the JSON file.
/// * `field_name`: The name of the field to extract from the JSON map.
///
/// # Returns
///
/// A `Result` containing the deserialized value of type `T` if successful,
/// or an `anyhow::Error` if any step fails (file I/O, JSON parsing, field not found,
/// or deserialization to `T` fails).
///
/// # Type Parameters
///
/// * `T`: The expected type of the field's value. `T` must implement `serde::de::DeserializeOwned`.
pub fn file_json_field<T: serde::de::DeserializeOwned>(
    json_file_path: &Path,
    field_name: &str,
) -> anyhow::Result<T> {
    // 1. Open the file
    let file = File::open(json_file_path)
        .with_context(|| format!("Failed to open file: {:?}", json_file_path))?;
    let reader = BufReader::new(file);

    // 2. Parse the JSON file into a generic serde_json::Value
    // We parse into `serde_json::Value` first because we need to look up a specific field.
    // If we tried to deserialize directly into `T`, `T` would need to represent the whole JSON structure.
    let json_data: serde_json::Value = serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse JSON from file: {:?}", json_file_path))?;

    // 3. Ensure the root of the JSON is an object (map)
    let map = json_data.as_object().ok_or_else(|| {
        anyhow::anyhow!("JSON root is not an object in file: {:?}", json_file_path)
    })?;

    // 4. Get the specific field's value
    let field_value = map.get(field_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Field '{}' not found in JSON file: {:?}",
            field_name,
            json_file_path
        )
    })?;

    // 5. Deserialize the field's value into the target type T
    // We need to clone `field_value` because `from_value` consumes its input.
    serde_json::from_value(field_value.clone()).with_context(|| {
        format!(
            "Failed to deserialize field '{}' (value: {:?}) to the expected type from file: {:?}",
            field_name, field_value, json_file_path
        )
    })
}
