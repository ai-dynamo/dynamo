// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_llm::{self as llm_rs};
use llm_rs::model_card::ModelDeploymentCard as RsModelDeploymentCard;
use llm_rs::model_type::{ModelInput, ModelType};
use pyo3::prelude::*;

use crate::{CancellationToken, engine::*, llm::local_model::ModelRuntimeConfig, to_pyerr};

pub use dynamo_llm::grpc::service::kserve;

#[pyclass]
pub struct KserveGrpcService {
    inner: kserve::KserveService,
}

#[pymethods]
impl KserveGrpcService {
    #[new]
    #[pyo3(signature = (port=None, host=None))]
    pub fn new(port: Option<u16>, host: Option<String>) -> PyResult<Self> {
        let mut builder = kserve::KserveService::builder();
        if let Some(port) = port {
            builder = builder.port(port);
        }
        if let Some(host) = host {
            builder = builder.host(host);
        }
        let inner = builder.build().map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    pub fn add_completions_model(
        &self,
        model: String,
        checksum: String,
        engine: PythonAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_completions_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    pub fn add_chat_completions_model(
        &self,
        model: String,
        checksum: String,
        engine: PythonAsyncEngine,
    ) -> PyResult<()> {
        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_chat_completions_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (model, checksum, engine, runtime_config=None))]
    pub fn add_tensor_model(
        &self,
        model: String,
        checksum: String,
        engine: PythonAsyncEngine,
        runtime_config: Option<ModelRuntimeConfig>,
    ) -> PyResult<()> {
        // If runtime_config is provided, create and save a ModelDeploymentCard
        // so the ModelConfig endpoint can return model configuration
        if let Some(runtime_config) = runtime_config {
            let mut card = RsModelDeploymentCard::with_name_only(&model);
            card.model_type = ModelType::TensorBased;
            card.model_input = ModelInput::Tensor;
            card.runtime_config = runtime_config.inner;

            self.inner
                .model_manager()
                .save_model_card(&model, card)
                .map_err(to_pyerr)?;
        }

        let engine = Arc::new(engine);
        self.inner
            .model_manager()
            .add_tensor_model(&model, &checksum, engine)
            .map_err(to_pyerr)
    }

    pub fn remove_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn remove_chat_completions_model(&self, model: String) -> PyResult<()> {
        self.inner
            .model_manager()
            .remove_chat_completions_model(&model)
            .map_err(to_pyerr)
    }

    pub fn remove_tensor_model(&self, model: String) -> PyResult<()> {
        // Remove the engine
        self.inner
            .model_manager()
            .remove_tensor_model(&model)
            .map_err(to_pyerr)?;

        // Also remove the model card if it exists
        // (It's ok if it doesn't exist since runtime_config is optional, we just ignore the None return)
        let _ = self.inner.model_manager().remove_model_card(&model);

        Ok(())
    }

    pub fn list_chat_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_chat_completions_models())
    }

    pub fn list_completions_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_completions_models())
    }

    pub fn list_tensor_models(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.model_manager().list_tensor_models())
    }

    fn run<'p>(&self, py: Python<'p>, token: CancellationToken) -> PyResult<Bound<'p, PyAny>> {
        let service = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token.inner).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}
