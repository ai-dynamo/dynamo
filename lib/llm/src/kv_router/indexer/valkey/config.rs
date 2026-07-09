// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated index identity and connection configuration.

use super::*;

pub(super) struct PreparedValkeyIndexer {
    pub(super) primary_endpoint: String,
    pub(super) pool_size: usize,
    pub(super) index_key: Vec<u8>,
    pub(super) replication_quorum: usize,
}

impl PreparedValkeyIndexer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        urls: &str,
        connection_pool_size: u32,
        required_replica_acks: Option<u32>,
        namespace: &str,
        component: &str,
        index_scope: Option<&str>,
        _model_name: Option<&str>,
        block_size: u32,
    ) -> Result<Self> {
        let endpoints: Vec<String> = urls
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(parse_endpoint)
            .collect::<Result<_>>()?;
        if endpoints.is_empty() {
            bail!("router_valkey_urls must contain at least one endpoint");
        }
        if endpoints.len() > 64 {
            bail!("router_valkey_urls must contain at most 64 endpoints");
        }
        let unique_endpoints = endpoints.iter().collect::<BTreeSet<_>>();
        if unique_endpoints.len() != endpoints.len() {
            bail!("router_valkey_urls endpoints must be distinct");
        }
        let pool_size = usize::try_from(connection_pool_size)
            .context("router_valkey_connection_pool_size does not fit usize")?;
        if !(1..=64).contains(&pool_size) {
            bail!("router_valkey_connection_pool_size must be in 1..=64");
        }

        let primary_endpoint = endpoints
            .first()
            .expect("endpoints checked non-empty")
            .clone();
        let replication_quorum =
            resolve_required_replica_acks(required_replica_acks, endpoints.len())?;
        if required_replica_acks.is_none() && endpoints.len() > 2 {
            tracing::warn!(
                configured_endpoints = endpoints.len(),
                "Valkey replica acknowledgements were inferred as one for compatibility; configure router_valkey_required_replica_acks explicitly"
            );
        }
        let index_scope = index_scope.unwrap_or(component);
        let namespace = encode_index_key_segment(namespace, "Valkey index namespace")?;
        // Worker-side direct publishers do not currently receive a model
        // name, while both workers and frontends address the same Dynamo
        // component. Use that shared discovery identity even when an explicit
        // scope is configured, so reusing a scope cannot merge two component
        // indexes. Keep `_model_name` in the API for a future identity which
        // can be supplied consistently by both sides.
        let component = encode_index_key_segment(component, "Valkey index component")?;
        let index_scope = encode_index_key_segment(index_scope, "router_valkey_index_scope")?;
        let index_key = format!(
            "dynamo:kv-router:{namespace}:component-{component}:scope-{index_scope}:block-size-{block_size}"
        )
        .into_bytes();

        Ok(Self {
            primary_endpoint,
            pool_size,
            index_key,
            replication_quorum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_indexer_bounds_connection_pool_allocation() {
        for pool_size in [0, 65, u32::MAX] {
            assert!(
                PreparedValkeyIndexer::new(
                    "valkey://127.0.0.1:6379",
                    pool_size,
                    None,
                    "namespace",
                    "component",
                    None,
                    None,
                    16,
                )
                .is_err()
            );
        }
    }
}
