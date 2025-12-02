// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for ConnectorLeader.
//!
//! Provides a PyO3 wrapper around the Rust ConnectorLeader, enabling Python code
//! to send Nova RPCs to workers for leader-driven initialization.

use pyo3::prelude::*;

use dynamo_kvbm::InstanceId;
use dynamo_kvbm::integrations::connector::leader::ConnectorLeader;

use dynamo_nova_backend::{PeerInfo, WorkerAddress};
use uuid::Uuid;

use crate::to_pyerr;
use crate::v2::runtime::PyKvbmRuntime;

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
    inner: ConnectorLeader,
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
    pub fn new(runtime: &PyKvbmRuntime) -> PyResult<Self> {
        let runtime = runtime.inner();
        let leader = ConnectorLeader::new(runtime);
        Ok(Self { inner: leader })
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

    pub fn initialize_workers(&self) -> PyResult<()> {
        self.inner.initialize_workers().map_err(to_pyerr)?;
        Ok(())
    }

    // /// Send a configure_layouts RPC to a worker to trigger deferred initialization.
    // ///
    // /// This is called by the leader after collecting handshake metadata from all workers.
    // /// It triggers the worker to complete NIXL registration and create G1/G2/G3 layouts.
    // ///
    // /// Args:
    // ///     instance_id_bytes: 16-byte UUID of the target worker instance
    // ///     host_block_count: Number of host/pinned blocks for G2 tier
    // ///     disk_block_count: Number of disk blocks for G3 tier (None = no disk tier)
    // ///     enable_posix: Enable POSIX backend for host memory transfers
    // ///     enable_gds: Enable GDS backend for GPU Direct Storage
    // ///
    // /// Returns:
    // ///     bytes: Serialized metadata from the worker after configuration
    // ///
    // /// Raises:
    // ///     ValueError: If instance_id is not 16 bytes
    // ///     RuntimeError: If the RPC fails or worker returns an error
    // #[pyo3(signature = (instance_id_bytes, host_block_count, disk_block_count=None, enable_posix=false, enable_gds=false))]
    // pub fn configure_layouts<'py>(
    //     &self,
    //     py: Python<'py>,
    //     instance_id_bytes: &[u8],
    //     host_block_count: usize,
    //     disk_block_count: Option<usize>,
    //     enable_posix: bool,
    //     enable_gds: bool,
    // ) -> PyResult<Bound<'py, PyBytes>> {
    //     // Parse instance ID from bytes (16-byte UUID)
    //     if instance_id_bytes.len() != 16 {
    //         return Err(pyo3::exceptions::PyValueError::new_err(format!(
    //             "instance_id must be 16 bytes, got {}",
    //             instance_id_bytes.len()
    //         )));
    //     }
    //     let uuid_bytes: [u8; 16] = instance_id_bytes.try_into().map_err(to_pyerr)?;
    //     let uuid = Uuid::from_bytes(uuid_bytes);
    //     let instance_id = InstanceId::from(uuid);

    //     // Build the config
    //     let config = LeaderLayoutConfig {
    //         host_block_count,
    //         disk_block_count,
    //         backend_config: NixlBackendConfigMessage {
    //             enable_posix,
    //             enable_gds,
    //             gds_params: None,
    //         },
    //     };

    //     // Send the RPC and block on the result
    //     let handle = self.inner.nova().runtime().clone();
    //     let result = handle
    //         .block_on(async { self.inner.configure_layouts(instance_id, config).await })
    //         .map_err(to_pyerr)?;

    //     // Serialize the metadata response
    //     let metadata_bytes = serde_json::to_vec(&result.metadata).map_err(to_pyerr)?;
    //     Ok(PyBytes::new(py, &metadata_bytes))
    // }
}
