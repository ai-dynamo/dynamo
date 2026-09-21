// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use std::fs::File;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use std::io::Read;
use std::path::Path;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use std::path::PathBuf;

use dynamo_model_protection::{
    CancellationToken, ModelProtectionKind, ProtectionError, detect_model_protection,
};
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use dynamo_model_protection::{
    AuthorizedModel, SecureModelSession, enforce_process_persistence_policy,
    load_authorized_model, materialize_tpm_model,
};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use rustix::fs::{CWD, FileType, Mode, OFlags, ResolveFlags, fstat, openat2};
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use rustix::process::geteuid;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
use serde::Deserialize;

pyo3::create_exception!(dynamo._core, ModelProtectionError, PyException);

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
const MAX_CONFIG_BYTES: u64 = 64 * 1024;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
const MAX_MEMORY_MARGIN_BYTES: u64 = 1 << 40;
#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
const TPM_TCTI: &str = "device:/dev/tpmrm0";

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeConfig {
    license_root: PathBuf,
    package_trust: TrustKey,
    license_trust: TrustKey,
    tpm: TpmConfig,
    process_memory_margin_bytes: u64,
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustKey {
    key_id: String,
    public_key_file: PathBuf,
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TpmConfig {
    device_key_handle: String,
    policy_authority_key_handle: String,
    policy_authority_key_id: String,
}

#[pyclass(name = "ProtectedModelSession", module = "dynamo._core")]
pub struct PyProtectedModelSession {
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    inner: Option<SecureModelSession>,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    authorized: AuthorizedModel,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    package_root: PathBuf,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    policy_authority_key_id: String,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    device_key_handle: u32,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    policy_authority_key_handle: u32,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    weights_ready: bool,
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
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
        #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
        {
            return self
                .inner
                .as_ref()
                .and_then(|session| session.model_path().to_str())
                .map(str::to_owned)
                .ok_or_else(config_error);
        }
        #[cfg(not(all(target_os = "linux", feature = "model-protection-tpm2")))]
        Err(config_error())
    }

    fn cleanup(&mut self) -> PyResult<()> {
        #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
        {
            self.cancellation.cancel();
            if let Some(session) = self.inner.take() {
                session.cleanup().map_err(protection_error)?;
            }
        }
        Ok(())
    }

    fn cancellation(&self) -> PyModelProtectionCancellation {
        #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
        {
            return PyModelProtectionCancellation(self.cancellation.clone());
        }
        #[cfg(not(all(target_os = "linux", feature = "model-protection-tpm2")))]
        PyModelProtectionCancellation(CancellationToken::default())
    }

    fn materialize(&mut self, _py: Python<'_>) -> PyResult<()> {
        #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
        {
            if self.weights_ready {
                return Err(ModelProtectionError::new_err("SESSION_CONFLICT"));
            }
            let session = self.inner.as_mut().ok_or_else(config_error)?;
            _py.allow_threads(|| {
                materialize_tpm_model(
                    session,
                    &self.package_root,
                    &self.authorized,
                    &self.policy_authority_key_id,
                    TPM_TCTI,
                    self.device_key_handle,
                    self.policy_authority_key_handle,
                    &self.cancellation,
                )
            })
            .map_err(protection_error)?;
            self.weights_ready = true;
            return Ok(());
        }
        #[cfg(not(all(target_os = "linux", feature = "model-protection-tpm2")))]
        Err(ModelProtectionError::new_err("TPM_UNAVAILABLE"))
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

    #[cfg(not(all(target_os = "linux", feature = "model-protection-tpm2")))]
    {
        let _ = (namespace, config_path);
        Err(ModelProtectionError::new_err("TPM_UNAVAILABLE"))
    }

    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    {
        enforce_process_persistence_policy().map_err(protection_error)?;
        let config = read_config(config_path.ok_or_else(config_error)?)?;
        validate_config(&config)?;
        let package_key = read_public_key(&config.package_trust.public_key_file)?;
        let license_key = read_public_key(&config.license_trust.public_key_file)?;
        let device_key_handle = parse_handle(&config.tpm.device_key_handle)?;
        let policy_authority_key_handle =
            parse_handle(&config.tpm.policy_authority_key_handle)?;
        let authorized = load_authorized_model(
            package_root,
            &config.license_root,
            &config.package_trust.key_id,
            &package_key,
            &config.license_trust.key_id,
            &license_key,
        )
        .map_err(protection_error)?;
        let mut session = SecureModelSession::prepare(
            namespace,
            &authorized,
            config.process_memory_margin_bytes,
        )
        .map_err(protection_error)?;
        session
            .stage_public_metadata(package_root, &authorized)
            .map_err(protection_error)?;
        Ok(Some(PyProtectedModelSession {
            inner: Some(session),
            authorized,
            package_root: package_root.to_path_buf(),
            policy_authority_key_id: config.tpm.policy_authority_key_id,
            device_key_handle,
            policy_authority_key_handle,
            weights_ready: false,
            cancellation: CancellationToken::default(),
        }))
    }
}

#[pyfunction]
fn enforce_model_protection_process_policy() -> PyResult<()> {
    #[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
    {
        return enforce_process_persistence_policy().map_err(protection_error);
    }
    #[cfg(not(all(target_os = "linux", feature = "model-protection-tpm2")))]
    Err(ModelProtectionError::new_err("TPM_UNAVAILABLE"))
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
fn read_config(path: &str) -> PyResult<RuntimeConfig> {
    let path = Path::new(path);
    let bytes = read_private_file(path, MAX_CONFIG_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|_| config_error())
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
fn validate_config(config: &RuntimeConfig) -> PyResult<()> {
    if !config.license_root.is_absolute()
        || config.process_memory_margin_bytes > MAX_MEMORY_MARGIN_BYTES
        || !valid_identifier(&config.package_trust.key_id)
        || !valid_identifier(&config.license_trust.key_id)
        || !valid_identifier(&config.tpm.policy_authority_key_id)
    {
        return Err(config_error());
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
fn read_public_key(path: &Path) -> PyResult<[u8; 32]> {
    read_private_file(path, 32)?
        .try_into()
        .map_err(|_| config_error())
}

#[cfg(all(target_os = "linux", feature = "model-protection-tpm2"))]
fn read_private_file(path: &Path, maximum: u64) -> PyResult<Vec<u8>> {
    if !path.is_absolute() {
        return Err(config_error());
    }
    let fd = openat2(
        CWD,
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|_| config_error())?;
    let stat = fstat(&fd).map_err(|_| config_error())?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_uid != geteuid().as_raw()
        || stat.st_size <= 0
        || stat.st_size as u64 > maximum
        || Mode::from_raw_mode(stat.st_mode).intersects(Mode::RWXG | Mode::RWXO)
    {
        return Err(config_error());
    }
    let mut bytes = Vec::with_capacity(stat.st_size as usize);
    File::from(fd)
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| config_error())?;
    if bytes.len() != stat.st_size as usize || bytes.len() as u64 > maximum {
        return Err(config_error());
    }
    Ok(bytes)
}

#[cfg(any(test, all(target_os = "linux", feature = "model-protection-tpm2")))]
fn parse_handle(value: &str) -> PyResult<u32> {
    value
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 8)
        .and_then(|digits| u32::from_str_radix(digits, 16).ok())
        .ok_or_else(config_error)
}

fn config_error() -> PyErr {
    ModelProtectionError::new_err("MODEL_PROTECTION_CONFIG_INVALID")
}

fn protection_error(error: ProtectionError) -> PyErr {
    ModelProtectionError::new_err(error.code().as_str())
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(enforce_model_protection_process_policy, m)?)?;
    m.add_class::<PyProtectedModelSession>()?;
    m.add_class::<PyModelProtectionCancellation>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_explicit_persistent_handle_syntax() {
        assert_eq!(parse_handle("0x81000001").unwrap(), 0x81000001);
        assert!(parse_handle("81000001").is_err());
        assert!(parse_handle("0x1").is_err());
    }
}
