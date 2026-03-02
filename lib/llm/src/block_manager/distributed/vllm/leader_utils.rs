// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::block_manager::block::transfer::remote::RemoteKey;
use crate::block_manager::distributed::registry::{
    BinaryCodec, NoMetadata, PositionalKey, RegistryClient, RegistryClientConfig, ZmqTransport,
};
use dynamo_runtime::config::environment_names::kvbm as env_kvbm;

/// Type alias for the registry client used by the vLLM connector.
pub type DistributedRegistryClient = RegistryClient<
    PositionalKey,
    RemoteKey,
    NoMetadata,
    ZmqTransport,
    BinaryCodec<PositionalKey, RemoteKey, NoMetadata>,
>;

/// Create a DistributedRegistryClient if enabled via environment.
pub fn create_distributed_registry_client() -> Option<Arc<DistributedRegistryClient>> {
    if !RegistryClientConfig::is_enabled() {
        tracing::debug!("Distributed registry disabled (DYN_REGISTRY_ENABLE not set)");
        return None;
    }

    let config = RegistryClientConfig::from_env();
    tracing::info!(
        "Creating distributed registry client: query={}, register={}",
        config.hub_query_addr,
        config.hub_register_addr
    );

    let transport = config.create_transport()?;

    let client = RegistryClient::new(transport, BinaryCodec::new())
        .with_batch_size(config.batch_size)
        .with_batch_timeout(config.batch_timeout);

    tracing::info!("Distributed registry client created successfully");
    Some(Arc::new(client))
}

/// Check whether KVBM_DEV_MODE is enabled via environment variable.
pub fn is_dev_mode() -> bool {
    std::env::var(env_kvbm::KVBM_DEV_MODE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn kvbm_metrics_endpoint_enabled() -> bool {
    std::env::var(env_kvbm::DYN_KVBM_METRICS)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn parse_kvbm_metrics_port() -> u16 {
    match std::env::var(env_kvbm::DYN_KVBM_METRICS_PORT) {
        Ok(val) => match val.trim().parse::<u16>() {
            Ok(port) => port,
            Err(_) => {
                tracing::warn!(
                    "[kvbm] Invalid DYN_KVBM_METRICS_PORT='{}', falling back to 6880",
                    val
                );
                6880
            }
        },
        Err(_) => {
            tracing::warn!(
                "DYN_KVBM_METRICS_PORT not present or couldn’t be interpreted, falling back to 6880"
            );
            6880
        }
    }
}
