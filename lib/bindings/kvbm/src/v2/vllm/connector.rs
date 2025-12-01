// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::to_pyerr;
use crate::v2::vllm::config::PyKvbmVllmConfig;

use std::collections::HashMap;
use std::sync::Mutex;

use dynamo_kvbm::v2::G1;
use dynamo_kvbm::v2::integrations::connector::ConnectorMetadataBuilder;
use dynamo_kvbm::v2::integrations::connector::leader::{
    Blocks, ConnectorLeader, LeaderRuntime, Request, SchedulerOutput, data::BlocksView,
};
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Python wrapper for the connector metadata builder.
#[pyclass(name = "ConnectorMetadataBuilder")]
pub struct PyConnectorMetadataBuilder {
    inner: Mutex<ConnectorMetadataBuilder>,
}

#[pyclass(name = "KvbmRequest")]
pub struct PyKvbmRequest {
    inner: Request,
}

#[pymethods]
impl PyKvbmRequest {
    #[new]
    #[pyo3(signature = (request_id, lora_name=None, salt_hash=None, max_tokens=None))]
    pub fn new(
        request_id: String,
        lora_name: Option<String>,
        salt_hash: Option<String>,
        max_tokens: Option<usize>,
    ) -> PyResult<Self> {
        if max_tokens.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_tokens is required",
            ));
        }
        Ok(Self {
            inner: Request::new(request_id, lora_name, salt_hash, max_tokens.unwrap()),
        })
    }

    #[getter]
    pub fn request_id(&self) -> &str {
        &self.inner.request_id
    }
}

impl From<&PyKvbmRequest> for Request {
    fn from(value: &PyKvbmRequest) -> Self {
        value.inner.clone()
    }
}

#[pyclass(name = "SchedulerOutput")]
pub struct PyRustSchedulerOutput {
    inner: Mutex<SchedulerOutput>,
}

#[pymethods]
impl PyRustSchedulerOutput {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(SchedulerOutput::new()),
        }
    }

    #[pyo3(signature = (request_id, prompt_token_ids, block_ids, num_computed_tokens))]
    pub fn add_new_request(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) {
        let block_ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        let blocks = Blocks::from_external(block_ids);
        self.inner.lock().unwrap().add_new_request(
            request_id.to_owned(),
            prompt_token_ids,
            blocks,
            num_computed_tokens,
        );
    }

    #[pyo3(signature = (request_id, resumed_from_preemption, new_token_ids, new_block_ids, num_computed_tokens))]
    pub fn add_cached_request(
        &self,
        request_id: &str,
        resumed_from_preemption: bool,
        new_token_ids: Vec<u32>,
        new_block_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) {
        let new_block_ids: Vec<usize> = new_block_ids.iter().map(|&id| id as usize).collect();
        let blocks = Blocks::from_external(new_block_ids);
        self.inner.lock().unwrap().add_cached_request(
            request_id.to_owned(),
            resumed_from_preemption,
            new_token_ids,
            blocks,
            num_computed_tokens,
        );
    }

    pub fn set_num_scheduled_tokens(&self, mapping: Bound<'_, PyDict>) -> PyResult<()> {
        let mut counts = HashMap::new();
        for (key, value) in mapping.iter() {
            counts.insert(key.extract::<String>()?, value.extract::<usize>()?);
        }
        self.inner.lock().unwrap().set_num_scheduled_tokens(counts);
        Ok(())
    }
}

#[pyclass(name = "KvConnectorLeader")]
pub struct PyKvConnectorLeader {
    inner: Mutex<ConnectorLeader>,
}

#[pymethods]
impl PyKvConnectorLeader {
    #[new]
    pub fn new(engine_id: String, config: &PyKvbmVllmConfig) -> Self {
        Self {
            inner: Mutex::new(ConnectorLeader::new(engine_id, config.inner().as_generic())),
        }
    }

    pub fn engine_id(&self) -> PyResult<String> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.engine_id().to_string())
    }

    pub fn has_slot(&self, request_id: &str) -> PyResult<bool> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.has_slot(request_id))
    }

    pub fn create_slot(
        &self,
        request: &PyKvbmRequest,
        all_token_ids: Vec<Vec<i64>>,
    ) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .create_slot((&*request).into(), all_token_ids)
            .map_err(crate::to_pyerr)
    }

    #[pyo3(signature = (request_id, num_computed_tokens))]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<(Option<usize>, bool)> {
        use dynamo_kvbm::v2::integrations::connector::leader::MatchResult;
        let guard = self.inner.lock().unwrap();
        let result = guard
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
            .map_err(crate::to_pyerr)?;
        match result {
            MatchResult::Evaluating => Ok((None, false)),
            MatchResult::NoMatches => Ok((Some(0), false)),
            MatchResult::Matched(count) => Ok((Some(count.get()), true)),
        }
    }

    pub fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<u32>,
        num_external_tokens: usize,
    ) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        let block_ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        let blocks_view = BlocksView::<G1>::from(block_ids);
        guard
            .update_state_after_alloc(request_id, blocks_view, num_external_tokens)
            .map_err(crate::to_pyerr)
    }

    pub fn request_finished(&self, request_id: &str, block_ids: Vec<u32>) -> PyResult<bool> {
        use dynamo_kvbm::v2::integrations::connector::leader::FinishedStatus;
        let mut guard = self.inner.lock().unwrap();
        let block_ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        let blocks_view = BlocksView::<G1>::from(block_ids);
        let blocks = Blocks::View(blocks_view);
        let status = guard
            .request_finished(request_id, blocks)
            .map_err(crate::to_pyerr)?;
        match status {
            FinishedStatus::Finished => Ok(false), // false = not pending
            FinishedStatus::Pending => Ok(true),   // true = is pending
        }
    }

    pub fn build_connector_metadata(
        &self,
        py: Python<'_>,
        output: &PyRustSchedulerOutput,
    ) -> PyResult<Py<PyBytes>> {
        let mut guard = self.inner.lock().unwrap();
        let bytes = {
            let output_ref = output.inner.lock().unwrap();
            guard
                .build_connector_metadata(&output_ref)
                .map_err(crate::to_pyerr)?
        };
        Ok(PyBytes::new(py, &bytes).into())
    }

    /// Handle finished offloading for a request.
    ///
    /// Called when workers report they've finished offloading (device->host/disk).
    /// This is invoked from `update_connector_output` when `finished_sending` is reported.
    ///
    /// Args:
    ///     request_id: The request ID that finished offloading.
    #[pyo3(name = "handle_finished_offload")]
    pub fn py_handle_finished_offload(&self, request_id: &str) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .handle_finished_offload(request_id)
            .map_err(crate::to_pyerr)
    }

    /// Handle finished onboarding for a request.
    ///
    /// Called when workers report they've finished onboarding (host/disk->device).
    /// This is invoked from `update_connector_output` when `finished_recving` is reported.
    ///
    /// Args:
    ///     request_id: The request ID that finished onboarding.
    #[pyo3(name = "handle_finished_onboard")]
    pub fn py_handle_finished_onboard(&self, request_id: &str) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .handle_finished_onboard(request_id)
            .map_err(crate::to_pyerr)
    }
}

#[pymethods]
impl PyConnectorMetadataBuilder {
    #[new]
    #[pyo3(signature = (protocol_version = 1))]
    pub fn new(protocol_version: u32) -> Self {
        Self {
            inner: Mutex::new(ConnectorMetadataBuilder::new(protocol_version)),
        }
    }

    pub fn queue_slot_create(&self, request_id: &str, create_event: &str) {
        self.inner
            .lock()
            .unwrap()
            .queue_slot_create(request_id.to_owned(), create_event.to_owned());
    }

    pub fn queue_forward_event(&self, request_id: &str, rank: u32, event: &str) {
        self.inner.lock().unwrap().queue_forward_event(
            request_id.to_owned(),
            rank,
            event.to_owned(),
        );
    }

    pub fn queue_slot_delete(&self, request_id: &str) {
        self.inner
            .lock()
            .unwrap()
            .queue_slot_delete(request_id.to_owned());
    }

    /// Queue a transaction payload encoded as a JSON string.
    pub fn queue_transaction(&self, request_id: &str, payload_json: &str) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(payload_json)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        self.inner
            .lock()
            .unwrap()
            .queue_transaction(request_id.to_owned(), value);
        Ok(())
    }

    pub fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }

    pub fn build_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = self.inner.lock().unwrap().build_bytes().map_err(to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).into())
    }
}
