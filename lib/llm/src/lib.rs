// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

use std::{fs::File, io::BufReader, path::Path};

use anyhow::Context as _;

pub mod backend;
pub mod common;
pub mod discovery;
pub mod endpoint_type;
pub mod engines;
pub mod entrypoint;
pub mod grpc;
pub mod http;
pub mod hub;
// pub mod key_value_store;
pub mod audit;
pub mod kv_router;
pub mod local_model;
pub mod lora;
pub mod migration;
pub mod mocker;
pub mod model_card;
pub mod model_type;
pub mod namespace;
pub mod perf;
pub mod preprocessor;
pub mod protocols;
pub mod recorder;
pub mod request_template;
pub mod tokenizers;
pub mod tokens;
pub mod types;
pub mod utils;

#[cfg(feature = "block-manager")]
pub mod block_manager;

#[cfg(feature = "cuda")]
pub mod cuda;

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

/// Pretty-print the part of JSON that has an error.
pub fn log_json_err(filename: &str, json: &str, err: &serde_json::Error) {
    const ERROR_PREFIX: &str = ">>     ";

    // Only log errors that relate to the content of the JSON file
    if !(err.is_syntax() || err.is_data()) {
        return;
    }
    // These are 1 based for humans so subtract
    let line = err.line().saturating_sub(1);
    let column = err.column().saturating_sub(1);

    let json_lines: Vec<&str> = json.lines().collect();
    if json_lines.is_empty() {
        tracing::error!("JSON parsing error in {filename}: File is empty.");
        return;
    }

    // Two lines before
    let start_index = (line - 2).max(0);
    // The problem line and two lines after
    let end_index = (line + 3).min(json_lines.len());

    // Collect the context
    let mut context_lines: Vec<String> = (start_index..end_index)
        .map(|i| {
            if i == line {
                format!("{ERROR_PREFIX}{}", json_lines[i])
            } else {
                // Six places because tokenizer.json is very long
                format!("{:06} {}", i + 1, json_lines[i])
            }
        })
        .collect();

    // Insert the column indicator
    let col_indicator = "_".to_string().repeat(column + ERROR_PREFIX.len()) + "^";
    let error_in_context_idx = line - start_index;
    if error_in_context_idx < context_lines.len() {
        context_lines.insert(error_in_context_idx + 1, col_indicator);
    }

    tracing::error!(
        "JSON parsing error in {filename}: Line {}, column {}:\n{}",
        err.line(),
        err.column(),
        context_lines.join("\n")
    );
}

#[cfg(test)]
mod file_json_field_tests {
    use super::file_json_field;
    use serde::Deserialize;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    // Helper function to create a temporary JSON file
    fn create_temp_json_file(dir: &Path, file_name: &str, content: &str) -> PathBuf {
        let file_path = dir.join(file_name);
        let mut file = File::create(&file_path)
            .unwrap_or_else(|_| panic!("Failed to create test file: {:?}", file_path));
        file.write_all(content.as_bytes())
            .unwrap_or_else(|_| panic!("Failed to write to test file: {:?}", file_path));
        file_path
    }

    // Define a custom struct for testing deserialization
    #[derive(Debug, PartialEq, Deserialize)]
    struct MyConfig {
        version: String,
        enabled: bool,
        count: u32,
    }

    #[test]
    fn test_success_basic() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "test_basic.json",
            r#"{ "name": "Rust", "age": 30, "is_active": true }"#,
        );

        let name: String = file_json_field(&file_path, "name").unwrap();
        assert_eq!(name, "Rust");

        let age: i32 = file_json_field(&file_path, "age").unwrap();
        assert_eq!(age, 30);

        let is_active: bool = file_json_field(&file_path, "is_active").unwrap();
        assert!(is_active);
    }

    #[test]
    fn test_success_custom_struct_field() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "test_struct.json",
            r#"{
                "config": {
                    "version": "1.0.0",
                    "enabled": true,
                    "count": 123
                },
                "other_field": "value"
            }"#,
        );

        let config: MyConfig = file_json_field(&file_path, "config").unwrap();
        assert_eq!(
            config,
            MyConfig {
                version: "1.0.0".to_string(),
                enabled: true,
                count: 123,
            }
        );
    }

    #[test]
    fn test_file_not_found() {
        let tmp_dir = tempdir().unwrap();
        let non_existent_path = tmp_dir.path().join("non_existent.json");

        let result: anyhow::Result<String> = file_json_field(&non_existent_path, "field");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to open file"));
    }

    #[test]
    fn test_invalid_json_syntax() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "invalid.json",
            r#"{ "key": "value", "bad_syntax": }"#, // Malformed JSON
        );

        let result: anyhow::Result<String> = file_json_field(&file_path, "key");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to parse JSON from file"));
    }

    #[test]
    fn test_json_root_not_object_array() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "root_array.json",
            r#"[ { "item": 1 }, { "item": 2 } ]"#, // Root is an array
        );

        let result: anyhow::Result<String> = file_json_field(&file_path, "item");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("JSON root is not an object"));
    }

    #[test]
    fn test_json_root_not_object_primitive() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "root_primitive.json",
            r#""just_a_string""#, // Root is a string
        );

        let result: anyhow::Result<String> = file_json_field(&file_path, "field");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("JSON root is not an object"));
    }

    #[test]
    fn test_field_not_found() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "missing_field.json",
            r#"{ "existing_field": "hello" }"#,
        );

        let result: anyhow::Result<String> = file_json_field(&file_path, "non_existent_field");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Field 'non_existent_field' not found")
        );
    }

    #[test]
    fn test_field_type_mismatch() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(
            tmp_dir.path(),
            "type_mismatch.json",
            r#"{ "count": "not_an_integer" }"#,
        );

        let result: anyhow::Result<u32> = file_json_field(&file_path, "count");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize field 'count'")
        );
    }

    #[test]
    fn test_empty_file() {
        let tmp_dir = tempdir().unwrap();
        let file_path = create_temp_json_file(tmp_dir.path(), "empty.json", "");

        let result: anyhow::Result<String> = file_json_field(&file_path, "field");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to parse JSON from file"));
    }
}
