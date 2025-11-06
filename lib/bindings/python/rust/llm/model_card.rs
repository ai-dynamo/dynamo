// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use llm_rs::model_card::ModelDeploymentCard as RsModelDeploymentCard;

#[pyclass]
#[derive(Clone)]
pub struct ModelDeploymentCard {
    pub(crate) inner: RsModelDeploymentCard,
}

impl ModelDeploymentCard {}

#[pymethods]
impl ModelDeploymentCard {
    /// Build an in-memory ModelDeploymentCard from a folder containing config.json,
    /// tokenizer.json and tokenizer_config.json (i.e. a huggingface repo checkout).
    ///
    /// # Arguments
    /// * `path` - Path to the local model directory
    /// * `model_name` - Name of the model
    ///
    /// # Returns
    /// A new ModelDeploymentCard instance
    ///
    /// # Errors
    /// Returns an error if the model directory does not exist or the model name is invalid.
    #[staticmethod]
    fn load(path: String, model_name: String) -> PyResult<ModelDeploymentCard> {
        let mut card = RsModelDeploymentCard::load_from_disk(&path, None).map_err(to_pyerr)?;
        card.set_name(&model_name);
        Ok(ModelDeploymentCard { inner: card })
    }

    #[staticmethod]
    fn from_json_str(json: String) -> PyResult<ModelDeploymentCard> {
        let card = RsModelDeploymentCard::load_from_json_str(&json).map_err(to_pyerr)?;
        Ok(ModelDeploymentCard { inner: card })
    }

    fn to_json_str(&self) -> PyResult<String> {
        let json = self.inner.to_json().map_err(to_pyerr)?;
        Ok(json)
    }
}
