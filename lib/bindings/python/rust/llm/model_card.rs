// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use llm_rs::model_card::ModelDeploymentCard as RsModelDeploymentCard;

#[pyclass]
#[derive(Clone)]
pub(crate) struct ModelDeploymentCard {
    pub(crate) inner: RsModelDeploymentCard,
}

impl ModelDeploymentCard {}

#[pymethods]
impl ModelDeploymentCard {
    // Previously called "from_local_path"
    #[staticmethod]
    fn load(path: String, model_name: String, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut card = RsModelDeploymentCard::load(&path, None)
                .await
                .map_err(to_pyerr)?;
            card.set_name(&model_name);
            Ok(ModelDeploymentCard { inner: card })
        })
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
