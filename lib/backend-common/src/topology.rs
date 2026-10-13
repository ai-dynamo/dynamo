// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! File-backed worker topology, matching `dynamo.common.utils.topology`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use dynamo_llm::kv_router::protocols::KvTransferEnforcement;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use tokio::time::Instant;

use dynamo_runtime::config::parse_bool;

use crate::error::{BackendError, DynamoError, ErrorType};

const DEFAULT_MOUNT_PATH: &str = "/etc/dynamo/topology";
const POLL_INTERVAL: Duration = Duration::from_secs(1);
const POLL_TIMEOUT: Duration = Duration::from_secs(30);

/// Load deployment-provided topology before publishing the worker's model card.
/// Disabled topology leaves the runtime config untouched. Enabled topology must
/// resolve the selected transfer domain within the startup polling deadline.
pub(crate) async fn apply_topology_config(
    runtime_config: &mut ModelRuntimeConfig,
) -> Result<(), DynamoError> {
    apply_from_env(
        runtime_config,
        |name| std::env::var(name).ok(),
        POLL_INTERVAL,
        POLL_TIMEOUT,
    )
    .await
}

async fn apply_from_env(
    runtime_config: &mut ModelRuntimeConfig,
    env: impl Fn(&str) -> Option<String>,
    poll_interval: Duration,
    poll_timeout: Duration,
) -> Result<(), DynamoError> {
    let raw = env("DYN_TOPOLOGY_ENABLED").unwrap_or_default();
    let trimmed = raw.trim();
    match parse_bool(trimmed) {
        Ok(true) => {}
        Ok(false) => return Ok(()),
        Err(_) => {
            if !trimmed.is_empty() {
                tracing::warn!(
                    value = %trimmed,
                    "Unrecognized DYN_TOPOLOGY_ENABLED value, treating as disabled; use \"true\"/\"false\" (or 1/0, on/off, yes/no)"
                );
            }
            return Ok(());
        }
    }

    let domain = env("DYN_KV_TRANSFER_DOMAIN")
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            invalid_config("DYN_TOPOLOGY_ENABLED is set but DYN_KV_TRANSFER_DOMAIN is not")
        })?;
    let enforcement = match env("DYN_KV_TRANSFER_ENFORCEMENT").as_deref() {
        None | Some("required") => KvTransferEnforcement::Required,
        Some("preferred") => KvTransferEnforcement::Preferred,
        Some(value) => {
            return Err(invalid_config(format!(
                "DYN_KV_TRANSFER_ENFORCEMENT must be required or preferred, got {value:?}"
            )));
        }
    };
    let preferred_weight = env("DYN_KV_TRANSFER_PREFERRED_WEIGHT")
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.trim().parse::<f32>().map_err(|error| {
                invalid_config(format!("invalid DYN_KV_TRANSFER_PREFERRED_WEIGHT: {error}"))
            })
        })
        .transpose()?;
    let mount_path = PathBuf::from(
        env("DYN_TOPOLOGY_MOUNT_PATH").unwrap_or_else(|| DEFAULT_MOUNT_PATH.to_owned()),
    );

    // Pod labels may be projected after the worker starts. Do not register a
    // policy-less worker while waiting for the selected domain to appear.
    let deadline = Instant::now() + poll_timeout;
    let topology_domains = loop {
        let domains = read_topology_domains(&mount_path).await;
        if domains.contains_key(&domain) {
            break domains;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(invalid_config(format!(
                "topology domain {domain:?} in {} was not populated within {poll_timeout:?}",
                mount_path.display()
            )));
        }
        tracing::info!(%domain, path = %mount_path.display(), ?remaining, "Waiting for topology domain");
        tokio::time::sleep(poll_interval.min(remaining)).await;
    };

    runtime_config.topology_domains = topology_domains;
    runtime_config.kv_transfer_domain = Some(domain);
    runtime_config.kv_transfer_enforcement = Some(enforcement);
    runtime_config.kv_transfer_preferred_weight = preferred_weight;
    tracing::info!(
        domains = ?runtime_config.topology_domains,
        domain = ?runtime_config.kv_transfer_domain,
        ?enforcement,
        ?preferred_weight,
        "Loaded worker topology"
    );
    Ok(())
}

async fn read_topology_domains(mount_path: &Path) -> HashMap<String, String> {
    let mut domains = HashMap::new();
    let mut entries = match tokio::fs::read_dir(mount_path).await {
        Ok(entries) => entries,
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %mount_path.display(), %error, "Unable to list topology directory");
            }
            return domains;
        }
    };
    loop {
        let entry = match entries.next_entry().await {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            Err(error) => {
                tracing::warn!(path = %mount_path.display(), %error, "Unable to list topology entry");
                // Discard the partial scan so the bounded outer poll retries
                // the directory, even if the transfer domain was already read.
                return HashMap::new();
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str().filter(|name| !name.starts_with('.')) else {
            continue;
        };
        // Follow symlinks: Downward API files point through the hidden ..data
        // directory, which itself must not become a published topology domain.
        let path = entry.path();
        match tokio::fs::metadata(&path).await {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => continue,
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(path = %path.display(), %error, "Unable to inspect topology file");
                }
                continue;
            }
        }
        match tokio::fs::read_to_string(&path).await {
            Ok(value) if !value.trim().is_empty() => {
                domains.insert(name.to_owned(), value.trim().to_owned());
            }
            Ok(_) => {}
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(path = %path.display(), %error, "Unable to read topology file");
                }
            }
        }
    }
    domains
}

fn invalid_config(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message.into())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reads_projected_domains_and_required_or_preferred_policy() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("..data");
        std::fs::create_dir(&data).unwrap();
        std::fs::write(data.join("zone"), " zone-a\n").unwrap();
        std::os::unix::fs::symlink("..data/zone", dir.path().join("zone")).unwrap();
        std::fs::write(dir.path().join("Rack"), "rack-1").unwrap();
        std::fs::write(dir.path().join("empty"), " \n").unwrap();
        std::fs::write(dir.path().join(".hidden"), "ignored").unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        for (enforcement, weight, expected) in [
            (None, None, KvTransferEnforcement::Required),
            (Some("required"), None, KvTransferEnforcement::Required),
            (
                Some("preferred"),
                Some("0.85"),
                KvTransferEnforcement::Preferred,
            ),
        ] {
            let mut config = ModelRuntimeConfig::default();
            apply_from_env(
                &mut config,
                |name| match name {
                    "DYN_TOPOLOGY_ENABLED" => Some(" TRUE ".into()),
                    "DYN_TOPOLOGY_MOUNT_PATH" => Some(dir.path().display().to_string()),
                    "DYN_KV_TRANSFER_DOMAIN" => Some(" zone ".into()),
                    "DYN_KV_TRANSFER_ENFORCEMENT" => enforcement.map(str::to_owned),
                    "DYN_KV_TRANSFER_PREFERRED_WEIGHT" => weight.map(str::to_owned),
                    _ => None,
                },
                POLL_INTERVAL,
                POLL_TIMEOUT,
            )
            .await
            .unwrap();
            assert_eq!(
                config.topology_domains,
                HashMap::from([
                    ("zone".into(), "zone-a".into()),
                    ("Rack".into(), "rack-1".into())
                ])
            );
            assert_eq!(config.kv_transfer_domain.as_deref(), Some("zone"));
            assert_eq!(config.kv_transfer_enforcement, Some(expected));
            assert_eq!(
                config.kv_transfer_preferred_weight,
                weight.map(|v| v.parse::<f32>().unwrap())
            );
        }
    }

    #[tokio::test]
    async fn waits_for_delayed_projection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("topology");
        let mut config = ModelRuntimeConfig::default();
        let load = apply_from_env(
            &mut config,
            |name| match name {
                "DYN_TOPOLOGY_ENABLED" => Some("true".into()),
                "DYN_TOPOLOGY_MOUNT_PATH" => Some(path.display().to_string()),
                "DYN_KV_TRANSFER_DOMAIN" => Some("zone".into()),
                _ => None,
            },
            Duration::from_millis(5),
            Duration::from_secs(2),
        );
        let publish = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tokio::fs::create_dir(&path).await.unwrap();
            tokio::fs::write(path.join("zone"), "zone-a").await.unwrap();
        };
        let (result, ()) = tokio::join!(load, publish);
        result.unwrap();
        assert_eq!(config.topology_domains["zone"], "zone-a");
    }

    #[tokio::test]
    async fn missing_or_empty_transfer_domain_times_out() {
        for contents in [None, Some(""), Some(" \n")] {
            let dir = tempfile::tempdir().unwrap();
            if let Some(contents) = contents {
                std::fs::write(dir.path().join("zone"), contents).unwrap();
            }
            std::fs::write(dir.path().join("rack"), "rack-1").unwrap();
            let mut config = ModelRuntimeConfig::default();
            let error = apply_from_env(
                &mut config,
                |name| match name {
                    "DYN_TOPOLOGY_ENABLED" => Some("true".into()),
                    "DYN_TOPOLOGY_MOUNT_PATH" => Some(dir.path().display().to_string()),
                    "DYN_KV_TRANSFER_DOMAIN" => Some("zone".into()),
                    _ => None,
                },
                Duration::from_millis(5),
                Duration::from_millis(20),
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("was not populated"));
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(config.topology_domains.is_empty());
        }
    }

    #[tokio::test]
    async fn alt_truthy_values_enable_topology() {
        // parse_bool accepts "1", "on", "yes" as truthy in addition to "true".
        // Verify that the wider set enables topology so coverage tracks the helper.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("zone"), "zone-a").unwrap();
        for value in ["1", "on", "yes"] {
            let mut config = ModelRuntimeConfig::default();
            apply_from_env(
                &mut config,
                |name| match name {
                    "DYN_TOPOLOGY_ENABLED" => Some(value.into()),
                    "DYN_TOPOLOGY_MOUNT_PATH" => Some(dir.path().display().to_string()),
                    "DYN_KV_TRANSFER_DOMAIN" => Some("zone".into()),
                    _ => None,
                },
                POLL_INTERVAL,
                POLL_TIMEOUT,
            )
            .await
            .unwrap_or_else(|e| panic!("alt truthy {value:?} must enable topology: {e}"));
            assert!(
                !config.topology_domains.is_empty(),
                "expected topology enabled for DYN_TOPOLOGY_ENABLED={value:?}"
            );
        }
    }

    #[tokio::test]
    async fn invalid_enabled_values_disable_and_warn() {
        // Unrecognized values ("maybe", "2") must not crash the worker;
        // they disable topology (Ok(()) with empty domains).
        // Warning log coverage lives in the Python caplog test.
        for value in ["maybe", "2", "enabled", "truthy"] {
            let mut config = ModelRuntimeConfig::default();
            apply_from_env(
                &mut config,
                |name| match name {
                    "DYN_TOPOLOGY_ENABLED" => Some(value.into()),
                    _ => None,
                },
                POLL_INTERVAL,
                POLL_TIMEOUT,
            )
            .await
            .expect("invalid enabled value must not error");
            assert!(
                config.topology_domains.is_empty(),
                "expected topology disabled for DYN_TOPOLOGY_ENABLED={value:?}"
            );
        }
    }

    #[tokio::test]
    async fn rejects_invalid_topology_env() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("zone"), "zone-a").unwrap();
        for (domain, enforcement, weight) in [
            (None, None, None),
            (Some(" "), None, None),
            (Some("zone"), Some("fallback"), None),
            (Some("zone"), Some("preferred"), Some("heavy")),
        ] {
            let error = apply_from_env(
                &mut ModelRuntimeConfig::default(),
                |name| match name {
                    "DYN_TOPOLOGY_ENABLED" => Some("true".into()),
                    "DYN_TOPOLOGY_MOUNT_PATH" => Some(dir.path().display().to_string()),
                    "DYN_KV_TRANSFER_DOMAIN" => domain.map(str::to_owned),
                    "DYN_KV_TRANSFER_ENFORCEMENT" => enforcement.map(str::to_owned),
                    "DYN_KV_TRANSFER_PREFERRED_WEIGHT" => weight.map(str::to_owned),
                    _ => None,
                },
                POLL_INTERVAL,
                Duration::ZERO,
            )
            .await
            .expect_err("invalid environment must fail parsing");
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
        }
    }
}
