// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use crate::{CancellationToken, engine::*, llm::model_card::ModelDeploymentCard, to_pyerr};

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

    pub fn add_tensor_model(
        &self,
        model: String,
        checksum: String,
        engine: PythonAsyncEngine,
    ) -> PyResult<()> {
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
        self.inner
            .model_manager()
            .remove_tensor_model(&model)
            .map_err(to_pyerr)
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

    pub fn save_model_card(&self, key: String, card: ModelDeploymentCard) -> PyResult<()> {
        self.inner
            .model_manager()
            .save_model_card(&key, card.inner)
            .map_err(to_pyerr)
    }

    pub fn remove_model_card(&self, key: String) -> PyResult<Option<ModelDeploymentCard>> {
        let card = self.inner.model_manager().remove_model_card(&key);
        Ok(card.map(|inner| ModelDeploymentCard { inner }))
    }

    pub fn get_model_cards(&self) -> PyResult<Vec<ModelDeploymentCard>> {
        let cards = self
            .inner
            .model_manager()
            .get_model_cards()
            .into_iter()
            .map(|inner| ModelDeploymentCard { inner })
            .collect();
        Ok(cards)
    }

    fn run<'p>(&self, py: Python<'p>, token: CancellationToken) -> PyResult<Bound<'p, PyAny>> {
        let service = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            service.run(token.inner).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}
