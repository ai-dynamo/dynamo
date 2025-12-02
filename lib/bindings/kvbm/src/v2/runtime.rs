// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for KvbmRuntime.
//!
//! Provides a PyO3 wrapper around the Rust KvbmRuntime, enabling Python code
//! to build and interact with the KVBM runtime infrastructure including Nova
//! for distributed communication.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use dynamo_kvbm::{KvbmRuntime, KvbmRuntimeBuilder};
use dynamo_kvbm_config::KvbmConfig;
use dynamo_nova_backend::PeerInfo;

use crate::to_pyerr;

/// Python wrapper for KvbmRuntime.
///
/// The runtime contains the shared infrastructure for KVBM operations:
/// - Tokio runtime for async execution
/// - NixlAgent for RDMA/UCX transfers
/// - Nova for distributed RPC
#[pyclass(name = "KvbmRuntime")]
pub struct PyKvbmRuntime {
    inner: Arc<KvbmRuntime>,
}

impl PyKvbmRuntime {
    /// Get the Nova instance for internal use by other bindings.
    pub(crate) fn nova(&self) -> anyhow::Result<Arc<dynamo_nova::Nova>> {
        Ok(self.inner.nova().clone())
    }

    pub fn inner(&self) -> KvbmRuntime {
        self.inner.clone()
    }
}

#[pymethods]
impl PyKvbmRuntime {
    /// Build a KvbmRuntime configured for a worker role.
    ///
    /// This creates a runtime with Nova configured for worker-side operations.
    /// Configuration is loaded from environment variables (KVBM_* prefix) with
    /// the "worker" profile selected (merges `profile.worker.*` values).
    ///
    /// Args:
    ///     config_json: Optional JSON string with config overrides from Python.
    ///         JSON can include both default values and profile-specific values:
    ///         - Top-level keys apply to all profiles
    ///         - `profile.worker.*` keys are overlaid for worker profile
    ///
    /// Returns:
    ///     KvbmRuntime: The initialized runtime instance.
    ///
    /// Raises:
    ///     RuntimeError: If configuration is invalid or runtime construction fails.
    #[staticmethod]
    #[pyo3(signature = (config_json=None))]
    fn build_worker(config_json: Option<&str>) -> PyResult<Self> {
        // Build config with worker profile selected
        let config = match config_json {
            Some(json) => KvbmConfig::from_figment_with_json_for_worker(json).map_err(to_pyerr)?,
            None => KvbmConfig::from_env_for_worker().map_err(to_pyerr)?,
        };

        // Create tokio runtime from config
        let tokio_rt = config.tokio.build_runtime().map_err(to_pyerr)?;
        let handle = tokio_rt.handle().clone();

        // Build KvbmRuntime using block_on
        let runtime = handle
            .block_on(async {
                KvbmRuntimeBuilder::new(config)
                    .with_runtime(Arc::new(tokio_rt))
                    .build_worker()
                    .await
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(runtime),
        })
    }

    /// Build a KvbmRuntime configured for a leader/scheduler role.
    ///
    /// This creates a runtime with Nova configured for leader-side operations.
    /// Configuration is loaded from environment variables (KVBM_* prefix) with
    /// the "leader" profile selected (merges `profile.leader.*` values).
    ///
    /// Args:
    ///     config_json: Optional JSON string with config overrides from Python.
    ///         JSON can include both default values and profile-specific values:
    ///         - Top-level keys apply to all profiles
    ///         - `profile.leader.*` keys are overlaid for leader profile
    ///         Use this to pass vLLM's `kv_connector_extra_config` dict as JSON.
    ///
    /// Returns:
    ///     KvbmRuntime: The initialized runtime instance.
    ///
    /// Raises:
    ///     RuntimeError: If configuration is invalid or runtime construction fails.
    #[staticmethod]
    #[pyo3(signature = (config_json=None))]
    fn build_leader(config_json: Option<&str>) -> PyResult<Self> {
        // Build config with leader profile selected
        let config = match config_json {
            Some(json) => KvbmConfig::from_figment_with_json_for_leader(json).map_err(to_pyerr)?,
            None => KvbmConfig::from_env_for_leader().map_err(to_pyerr)?,
        };

        // Create tokio runtime from config
        let tokio_rt = config.tokio.build_runtime().map_err(to_pyerr)?;
        let handle = tokio_rt.handle().clone();

        // Build KvbmRuntime using block_on
        let runtime = handle
            .block_on(async {
                KvbmRuntimeBuilder::new(config)
                    .with_runtime(Arc::new(tokio_rt))
                    .build_leader()
                    .await
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(runtime),
        })
    }

    /// Get Nova peer information for this runtime instance.
    ///
    /// Returns the instance ID and worker address as bytes, which can be
    /// serialized and sent to remote instances for peer discovery.
    ///
    /// Returns:
    ///     tuple[bytes, bytes]: (instance_id_bytes, worker_address_bytes)
    ///         - instance_id_bytes: 16-byte UUID identifying this instance
    ///         - worker_address_bytes: JSON-serialized WorkerAddress for TCP connection
    fn peer_info<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let info = self.inner.nova().peer_info();

        // Instance ID as raw bytes (16 bytes for UUID)
        let instance_id_bytes = PyBytes::new(py, info.instance_id.as_bytes());

        // Worker address serialized as JSON (more portable than bincode)
        let worker_address_json = serde_json::to_vec(&info.worker_address).map_err(to_pyerr)?;
        let worker_address_bytes = PyBytes::new(py, &worker_address_json);

        Ok((instance_id_bytes, worker_address_bytes))
    }

    /// Register a remote peer with this runtime's Nova instance.
    ///
    /// This allows the runtime to establish connections to the remote peer
    /// for Nova active message communication.
    ///
    /// Args:
    ///     instance_id_bytes: 16-byte UUID of the remote instance
    ///     worker_address_bytes: JSON-serialized WorkerAddress of the remote peer
    ///
    /// Raises:
    ///     ValueError: If the bytes cannot be deserialized
    ///     RuntimeError: If peer registration fails
    fn register_peer(&self, instance_id_bytes: &[u8], worker_address_bytes: &[u8]) -> PyResult<()> {
        use dynamo_kvbm::InstanceId;
        use uuid::Uuid;

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
        let worker_address = serde_json::from_slice(worker_address_bytes).map_err(to_pyerr)?;

        // Create PeerInfo and register
        let peer_info = PeerInfo::new(instance_id, worker_address);
        self.inner.nova().register_peer(peer_info).map_err(to_pyerr)
    }

    /// Get the instance ID of this runtime as bytes.
    ///
    /// Returns:
    ///     bytes: 16-byte UUID identifying this instance
    fn instance_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let info = self.inner.nova().peer_info();
        PyBytes::new(py, info.instance_id.as_bytes())
    }

    /// Wait for a handler to become available on a remote instance.
    ///
    /// This is CRITICAL to call after `register_peer` and before sending any
    /// RPCs to that peer. Nova performs an asynchronous handshake after peer
    /// registration to discover available handlers. Without waiting, RPCs
    /// may fail or hang because the handler isn't yet discoverable.
    ///
    /// Args:
    ///     instance_id_bytes: 16-byte UUID of the remote instance
    ///     handler_name: Name of the handler to wait for (e.g., "kvbm.connector.configure_layouts")
    ///
    /// Raises:
    ///     ValueError: If instance_id_bytes is not 16 bytes
    ///     RuntimeError: If the handler never becomes available (timeout)
    fn wait_for_handler(&self, instance_id_bytes: &[u8], handler_name: &str) -> PyResult<()> {
        use dynamo_kvbm::InstanceId;
        use uuid::Uuid;

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

        // Use the runtime's tokio handle to block on the async wait
        let nova = self.inner.nova().clone();
        let handler_name = handler_name.to_string();

        self.inner
            .handle()
            .block_on(async move { nova.wait_for_handler(instance_id, &handler_name).await })
            .map_err(to_pyerr)
    }
}
