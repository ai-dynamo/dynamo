// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use dynamo_model_protection::{
    CancellationToken, ModelProtectionKind, ProtectionError, detect_model_protection,
};
#[cfg(target_os = "linux")]
use dynamo_model_protection::{enforce_process_persistence_policy, runtime::PreparedRuntime};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

pyo3::create_exception!(dynamo._core, ModelProtectionError, PyException);

#[pyclass(name = "ProtectedModelSession", module = "dynamo._core")]
pub struct PyProtectedModelSession {
    #[cfg(target_os = "linux")]
    inner: Option<PreparedRuntime>,
    #[cfg(target_os = "linux")]
    cancellation: CancellationToken,
}

#[pyclass(name = "ModelProtectionCancellation", module = "dynamo._core", frozen)]
pub struct PyModelProtectionCancellation(CancellationToken);

#[pymethods]
impl PyModelProtectionCancellation {
    fn cancel(&self) {
        self.0.cancel();
    }
}

#[pymethods]
impl PyProtectedModelSession {
    #[getter]
    fn model_path(&self) -> PyResult<String> {
        #[cfg(target_os = "linux")]
        {
            self.inner
                .as_ref()
                .and_then(|session| session.model_path().to_str())
                .map(str::to_owned)
                .ok_or_else(config_error)
        }
        #[cfg(not(target_os = "linux"))]
        Err(config_error())
    }

    fn cleanup(&mut self) -> PyResult<()> {
        #[cfg(target_os = "linux")]
        {
            self.cancellation.cancel();
            if let Some(session) = self.inner.take() {
                session.cleanup().map_err(protection_error)?;
            }
        }
        Ok(())
    }

    fn cancellation(&self) -> PyModelProtectionCancellation {
        #[cfg(target_os = "linux")]
        {
            PyModelProtectionCancellation(self.cancellation.clone())
        }
        #[cfg(not(target_os = "linux"))]
        PyModelProtectionCancellation(CancellationToken::default())
    }

    fn materialize(&mut self, py: Python<'_>) -> PyResult<()> {
        #[cfg(target_os = "linux")]
        {
            let session = self.inner.as_mut().ok_or_else(config_error)?;
            py.allow_threads(|| session.materialize(&self.cancellation))
                .map_err(protection_error)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = py;
            Err(ModelProtectionError::new_err("RUNTIME_UNSUPPORTED"))
        }
    }
}

#[pyfunction]
fn is_protected_model(model_path: &str) -> PyResult<bool> {
    detect_model_protection(Path::new(model_path))
        .map(|kind| kind == ModelProtectionKind::ProtectedCandidate)
        .map_err(protection_error)
}

#[pyfunction]
#[pyo3(signature = (model_path, namespace, config_path=None))]
fn prepare_protected_model(
    model_path: &str,
    namespace: &str,
    config_path: Option<&str>,
) -> PyResult<Option<PyProtectedModelSession>> {
    let package_root = Path::new(model_path);
    if detect_model_protection(package_root).map_err(protection_error)?
        == ModelProtectionKind::Plain
    {
        return Ok(None);
    }
    #[cfg(target_os = "linux")]
    {
        let runtime =
            PreparedRuntime::prepare(package_root, namespace, config_path.unwrap_or("{}"))
                .map_err(protection_error)?;
        Ok(Some(PyProtectedModelSession {
            inner: Some(runtime),
            cancellation: CancellationToken::default(),
        }))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (namespace, config_path);
        Err(ModelProtectionError::new_err("RUNTIME_UNSUPPORTED"))
    }
}

#[pyfunction]
fn enforce_model_protection_process_policy() -> PyResult<()> {
    #[cfg(target_os = "linux")]
    {
        enforce_process_persistence_policy().map_err(protection_error)
    }
    #[cfg(not(target_os = "linux"))]
    Err(ModelProtectionError::new_err("RUNTIME_UNSUPPORTED"))
}

fn config_error() -> PyErr {
    ModelProtectionError::new_err("MODEL_PROTECTION_CONFIG_INVALID")
}

fn protection_error(error: ProtectionError) -> PyErr {
    match error {
        ProtectionError::ProtectionLayerDisabled(layer) => {
            ModelProtectionError::new_err(format!("MODEL_PROTECTION_LAYER_DISABLED: {layer}"))
        }
        _ => ModelProtectionError::new_err(error.code().as_str()),
    }
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "model_protection_runtime_enabled",
        cfg!(target_os = "linux"),
    )?;
    m.add(
        "model_protection_tpm_enabled",
        cfg!(all(target_os = "linux", feature = "model-protection-tpm2")),
    )?;
    m.add(
        "ModelProtectionError",
        m.py().get_type::<ModelProtectionError>(),
    )?;
    m.add_function(wrap_pyfunction!(is_protected_model, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_protected_model, m)?)?;
    m.add_function(wrap_pyfunction!(
        enforce_model_protection_process_policy,
        m
    )?)?;
    m.add_class::<PyProtectedModelSession>()?;
    m.add_class::<PyModelProtectionCancellation>()?;
    Ok(())
}
