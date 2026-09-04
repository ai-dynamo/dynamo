// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::protocols::TokenIdType;

pub(super) fn read_model_config(
    model_id: &str,
    expected_model_type: &str,
    expected_architecture: &str,
    family: &str,
    model_dir: &Path,
) -> Result<Value> {
    let config = read_json(model_dir, "config.json")?;
    let actual_model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .with_context(|| format!("mm-routing: {family} model_type is missing from config.json"))?;
    anyhow::ensure!(
        actual_model_type == expected_model_type,
        "mm-routing: model_type mismatch for {model_id}: deployment reports {expected_model_type:?}, config.json reports {actual_model_type:?}"
    );

    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .with_context(|| {
            format!("mm-routing: {family} architectures are missing from config.json")
        })?;
    anyhow::ensure!(
        architectures
            .iter()
            .any(|architecture| architecture.as_str() == Some(expected_architecture)),
        "mm-routing: {family} model_type {actual_model_type:?} requires architecture {expected_architecture:?}"
    );
    Ok(config)
}

pub(super) fn read_json(model_dir: &Path, filename: &str) -> Result<Value> {
    let path = model_dir.join(filename);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("mm-routing: failed to read {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("mm-routing: failed to parse {}", path.display()))
}

pub(super) fn required_usize(value: &Value, field: &str, family: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .with_context(|| format!("mm-routing: missing or invalid {family} field {field:?}"))
}

pub(super) fn required_token_id(value: &Value, field: &str, family: &str) -> Result<TokenIdType> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| TokenIdType::try_from(value).ok())
        .with_context(|| format!("mm-routing: missing or invalid {family} token id {field:?}"))
}
