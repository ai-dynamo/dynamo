// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service configuration (figment + clap). See `bin/kvbm_service.rs` for how
//! the CLI args layer on top of the env/TOML/default stack.

pub mod flags;
pub use flags::ServiceFlags;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

/// Minimum graceful shutdown window when one is explicitly set. The default
/// (`None`) waits indefinitely; non-`None` values below this floor are
/// rejected at config-load time because in-flight host → CUDA-IPC writes
/// can corrupt client state if interrupted prematurely.
pub const MIN_SHUTDOWN_GRACE: Duration = Duration::from_secs(60);

/// Configuration consumed by `KvbmService::start`.
///
/// Default `http_addr` is `127.0.0.1:0` so the OS picks a free port; the UDS
/// path is built from `<dir>/kvbm-service.<pid>.<http_port>.sock` if not
/// explicitly set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// HTTP sidecar listen address.
    #[serde(default = "ServiceConfig::default_http_addr")]
    pub http_addr: SocketAddr,
    /// Explicit UDS path. If `None`, derived from pid + resolved HTTP port.
    #[serde(default)]
    pub uds_path: Option<PathBuf>,
    /// Parent directory used when `uds_path` is None.
    #[serde(default = "ServiceConfig::default_uds_dir")]
    pub uds_dir: PathBuf,
    /// Graceful shutdown deadline in **milliseconds**. `None` (default)
    /// means wait indefinitely. If `Some`, must be ≥ 60 000 ms.
    #[serde(default)]
    pub shutdown_grace_ms: Option<u64>,
    /// Host-memory pool sizing + hugepage policy.
    #[serde(default)]
    pub pool: crate::pool::PoolConfig,
}

impl ServiceConfig {
    pub fn default_http_addr() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    pub fn default_uds_dir() -> PathBuf {
        PathBuf::from("/tmp")
    }

    /// Typed accessor for the configured grace period. `None` means
    /// "wait indefinitely".
    pub fn shutdown_grace(&self) -> Option<Duration> {
        self.shutdown_grace_ms.map(Duration::from_millis)
    }

    /// Compute the UDS path for the given resolved HTTP port. Honors an
    /// explicit `uds_path` if one was set.
    pub fn resolved_uds_path(&self, http_port: u16) -> PathBuf {
        if let Some(p) = &self.uds_path {
            return p.clone();
        }
        let pid = std::process::id();
        self.uds_dir
            .join(format!("kvbm-service.{pid}.{http_port}.sock"))
    }

    /// Load with the precedence: defaults < TOML file (if Some) < env
    /// (`KVBM_SERVICE_*`) < CLI flags.  figment merges in that order so the
    /// right-most wins.
    pub fn load(
        config_file: Option<&std::path::Path>,
        cli: ServiceConfigCli,
    ) -> Result<Self, ConfigError> {
        let mut figment = Figment::from(Serialized::defaults(ServiceConfig::default()));

        if let Some(path) = config_file {
            figment = figment.merge(Toml::file(path));
        }

        figment = figment.merge(Env::prefixed("KVBM_SERVICE_"));

        // CLI overrides last — only insert keys where the caller supplied Some.
        let mut cli_map = serde_json::Map::new();
        if let Some(v) = &cli.http_addr {
            cli_map.insert("http_addr".into(), serde_json::Value::String(v.to_string()));
        }
        if let Some(v) = &cli.uds_path {
            cli_map.insert("uds_path".into(), serde_json::to_value(v).unwrap());
        }
        if let Some(v) = &cli.uds_dir {
            cli_map.insert("uds_dir".into(), serde_json::to_value(v).unwrap());
        }
        if let Some(v) = &cli.shutdown_grace_ms {
            cli_map.insert("shutdown_grace_ms".into(), serde_json::Value::from(*v));
        }

        if !cli_map.is_empty() {
            figment = figment.merge(Serialized::defaults(serde_json::Value::Object(cli_map)));
        }

        let cfg: ServiceConfig = figment.extract()?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(ms) = self.shutdown_grace_ms {
            let d = Duration::from_millis(ms);
            if d < MIN_SHUTDOWN_GRACE {
                return Err(ConfigError::InvalidGrace(format!(
                    "shutdown_grace_ms must be >= {} ms (got {} ms)",
                    MIN_SHUTDOWN_GRACE.as_millis(),
                    ms
                )));
            }
        }
        Ok(())
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            http_addr: Self::default_http_addr(),
            uds_path: None,
            uds_dir: Self::default_uds_dir(),
            shutdown_grace_ms: None,
            pool: crate::pool::PoolConfig::default(),
        }
    }
}

/// CLI overrides for `ServiceConfig`. All optional; `None` means "fall through
/// to env/TOML/default".
#[derive(Debug, Clone, Default)]
pub struct ServiceConfigCli {
    pub http_addr: Option<SocketAddr>,
    pub uds_path: Option<PathBuf>,
    pub uds_dir: Option<PathBuf>,
    pub shutdown_grace_ms: Option<u64>,
}

/// Errors returned by [`ServiceConfig::load`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config load: {0}")]
    Figment(Box<figment::Error>),
    #[error("invalid grace period: {0}")]
    InvalidGrace(String),
}

impl From<figment::Error> for ConfigError {
    fn from(e: figment::Error) -> Self {
        ConfigError::Figment(Box::new(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_localhost_port_zero() {
        let cfg = ServiceConfig::default();
        assert_eq!(cfg.http_addr, "127.0.0.1:0".parse::<SocketAddr>().unwrap());
        assert!(cfg.uds_path.is_none());
        assert_eq!(cfg.uds_dir, PathBuf::from("/tmp"));
        assert!(cfg.shutdown_grace_ms.is_none());
    }

    #[test]
    fn cli_overrides_env_for_http_addr() {
        let cli = ServiceConfigCli {
            http_addr: Some("127.0.0.1:9999".parse().unwrap()),
            ..Default::default()
        };
        let cfg = ServiceConfig::load(None, cli).unwrap();
        assert_eq!(cfg.http_addr.port(), 9999);
    }

    #[test]
    fn env_layered_over_toml() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), r#"http_addr = "127.0.0.1:1000""#).unwrap();
        let key = "KVBM_SERVICE_HTTP_ADDR";
        // SAFETY: single-threaded test; no concurrent env reads.
        unsafe { std::env::set_var(key, "127.0.0.1:2000") };
        let cfg = ServiceConfig::load(Some(tmp.path()), ServiceConfigCli::default()).unwrap();
        unsafe { std::env::remove_var(key) };
        assert_eq!(cfg.http_addr.port(), 2000);
    }

    #[test]
    fn resolved_uds_path_uses_pid_and_port() {
        let cfg = ServiceConfig::default();
        let pid = std::process::id();
        let path = cfg.resolved_uds_path(8080);
        let expected = format!("kvbm-service.{pid}.8080.sock");
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), expected);
        assert_eq!(path.parent().unwrap(), PathBuf::from("/tmp"));
    }

    #[test]
    fn rejects_grace_below_60s() {
        let cli = ServiceConfigCli {
            shutdown_grace_ms: Some(30_000),
            ..Default::default()
        };
        let err = ServiceConfig::load(None, cli).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidGrace(_)), "got {err:?}");
    }

    #[test]
    fn accepts_exactly_60s_grace() {
        let cli = ServiceConfigCli {
            shutdown_grace_ms: Some(60_000),
            ..Default::default()
        };
        let cfg = ServiceConfig::load(None, cli).unwrap();
        assert_eq!(cfg.shutdown_grace(), Some(Duration::from_secs(60)));
    }

    #[test]
    fn accepts_none_grace() {
        let cfg = ServiceConfig::load(None, ServiceConfigCli::default()).unwrap();
        assert!(cfg.shutdown_grace().is_none());
    }
}
