// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

pub mod agents;
pub mod backend;
pub mod common;
pub mod discovery;
pub mod endpoint_type;
pub mod engines;
pub mod entrypoint;
pub mod fpm_publisher;
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
pub mod telemetry;
pub use dynamo_tokenizers as tokenizers;
pub use dynamo_tokenizers::{file_json_field, log_json_err};
pub mod tokens;
pub mod types;
pub mod utils;

#[cfg(feature = "block-manager")]
pub mod block_manager;

#[cfg(feature = "cuda")]
pub mod cuda;

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
