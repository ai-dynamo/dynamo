// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for ConnectorLeader.
//!
//! Provides a PyO3 wrapper around the Rust ConnectorLeader, enabling Python code
//! to send Nova RPCs to workers for leader-driven initialization.

use pyo3::prelude::*;
use std::collections::HashSet;

use dynamo_kvbm::InstanceId;
use dynamo_kvbm::integrations::connector::leader::{ConnectorLeader, FinishedStatus, Request};

use dynamo_nova_backend::{PeerInfo, WorkerAddress};
use uuid::Uuid;

use crate::to_pyerr;

mod request;
pub use request::PyRequest;

/// Python wrapper for WorkerClient.
///
/// This client provides leader-specific coordination operations via Nova RPC.
/// It is separate from KvbmRuntime because the runtime is shared infrastructure
/// while this client handles leader-to-worker RPC communication during initialization.
///
/// Example:
///     service = ConnectorLeader(runtime)
///     metadata = service.configure_layouts(instance_id, host_block_count=1000)
#[pyclass(name = "ConnectorLeader")]
pub struct PyConnectorLeader {
    inner: Arc<ConnectorLeader>,
}

#[pymethods]
impl PyConnectorLeader {
    /// Create a new ConnectorLeader from a KvbmRuntime.
    ///
    /// Args:
    ///     runtime: The KvbmRuntime instance to use for Nova RPC communication
    ///
    /// Raises:
    ///     RuntimeError: If the runtime doesn't have a Nova instance
    #[new]
    pub fn new(runtime: &PyKvbmRuntime, block_size: usize) -> PyResult<Self> {
        let runtime = runtime.inner();
        let leader = Arc::new(ConnectorLeader::new(runtime, block_size));
        Ok(Self { inner: leader })
    }

    pub fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    pub fn create_slot(&self, request: PyRequest) -> PyResult<()> {
        self.inner
            .create_slot(request.inner.clone())
            .map_err(to_pyerr)
    }

    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<(Option<usize>, bool)> {
        self.inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
            .map_err(to_pyerr)
    }

    /// See [`ConnectorLeader::request_finished`] for more details.
    pub fn request_finished(&self, request_id: &str) -> bool {
        match self.inner.request_finished(request_id) {
            FinishedStatus::Finished => false,
            FinishedStatus::Pending => true,
            FinishedStatus::UntrackedRequest => false,
        }
    }

    /// See [`ConnectorLeader::update_connector_output`] for more details.
    pub fn update_connector_output(
        &self,
        finished_sending: &PyAny,
        finished_recving: &PyAny,
    ) -> PyResult<()> {
        let finished_sending: HashSet<String> = finished_sending.extract()?;
        let finished_recving: HashSet<String> = finished_recving.extract()?;

        self.inner
            .update_connector_output(finished_sending, finished_recving)
            .map_err(to_pyerr)?;
        Ok(())
    }

    /// Register a worker peer with Nova.
    ///
    /// This registers the worker as a Nova peer so the leader can communicate
    /// with it via RPC. Workers should be registered in rank order (0, 1, 2, ...).
    ///
    /// Args:
    ///     rank: The worker's rank (0-indexed)
    ///     instance_id_bytes: 16-byte UUID of the worker instance
    ///     worker_address_bytes: JSON-serialized WorkerAddress of the worker peer
    ///
    /// Raises:
    ///     ValueError: If the bytes cannot be deserialized
    ///     RuntimeError: If peer registration fails
    pub fn register_worker(
        &self,
        rank: usize,
        instance_id_bytes: &[u8],
        worker_address_bytes: &[u8],
    ) -> PyResult<()> {
        // Parse instance ID from bytes (16-byte UUID)
        if instance_id_bytes.len() != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "instance_id must be 16 bytes, got {}",
                instance_id_bytes.len()
            )));
        }
        let uuid_bytes: [u8; 16] = instance_id_bytes.try_into().map_err(to_pyerr)?;
        let uuid = Uuid::from_bytes(uuid_bytes);
        let instance_id = InstanceId::from(uuid);

        // Deserialize worker address from JSON
        let worker_address: WorkerAddress =
            serde_json::from_slice(worker_address_bytes).map_err(to_pyerr)?;

        self.inner
            .register_worker(rank, instance_id, worker_address)
            .map_err(to_pyerr)?;

        Ok(())
    }

    /// After all workers have been registered, initialize them all.
    pub fn initialize_workers(&self) -> PyResult<()> {
        self.inner.initialize_workers().map_err(to_pyerr)?;
        Ok(())
    }
}
