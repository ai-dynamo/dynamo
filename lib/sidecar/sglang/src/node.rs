// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SGLang Rust listeners of one node, served as one Dynamo worker.
//!
//! SGLang's Rust server embeds one listener in every DP-group leader
//! scheduler, on `--port + offset` and `--grpc-port + offset`, and each
//! listener reports the node's whole listener table in
//! `DeploymentInfo.node_listeners`. One sidecar connects to all of them and
//! registers one worker that owns their contiguous DP ranks `[s, s + L)`.

use std::collections::HashSet;

use dynamo_backend_common::{DisaggregationMode, DynamoError};
use dynamo_sidecar_common::GrpcEndpoint;

use crate::client::{self, LocalKvEventSource, NodeMetadata};
use crate::discovery::EngineInfo;
use crate::engine::discovery_bootstrap_port;
use crate::proto as pb;

/// One Rust listener of the node, as `DeploymentInfo.node_listeners` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeListener {
    pub port_offset: u32,
    pub dp_rank: u32,
    pub http_port: u32,
    pub grpc_port: Option<u32>,
}

/// The node's listener table as one listener reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeTable {
    /// Every listener of the node, by DP rank: contiguous, unique, and below
    /// the deployment's DP size.
    pub listeners: Vec<NodeListener>,
    /// The entry of the listener that reported the table.
    pub own: usize,
}

impl NodeTable {
    /// The table `deployment` reports, checked against the listener that
    /// reported it. An SGLang build without the table can still serve DP=1:
    /// its one listener becomes the table.
    pub(crate) fn from_discovery(
        server_info: &pb::GetServerInfoResponse,
        deployment: &pb::DeploymentInfo,
    ) -> Result<Self, DynamoError> {
        let dp_size = deployment.dp_size.max(1);
        let own_rank = deployment.dp_rank.unwrap_or(0);
        let mut listeners: Vec<NodeListener> = if deployment.node_listeners.is_empty() {
            if dp_size > 1 {
                return Err(client::invalid_arg(format!(
                    "SGLang reports dp_size {dp_size} but no DeploymentInfo.node_listeners; \
                     DP>1 needs an SGLang build that reports its node's listeners"
                )));
            }
            vec![NodeListener {
                port_offset: 0,
                dp_rank: own_rank,
                http_port: server_info.port,
                grpc_port: None,
            }]
        } else {
            deployment
                .node_listeners
                .iter()
                .map(|listener| NodeListener {
                    port_offset: listener.port_offset,
                    dp_rank: listener.dp_rank,
                    http_port: listener.http_port,
                    grpc_port: listener.grpc_port,
                })
                .collect()
        };
        listeners.sort_by_key(|listener| listener.dp_rank);
        let mut offsets = HashSet::new();
        for (index, listener) in listeners.iter().enumerate() {
            if listener.dp_rank >= dp_size {
                return Err(client::invalid_arg(format!(
                    "SGLang lists a listener for DP rank {} of a {dp_size}-rank deployment",
                    listener.dp_rank
                )));
            }
            if index > 0 && listener.dp_rank != listeners[index - 1].dp_rank + 1 {
                return Err(client::invalid_arg(format!(
                    "SGLang's listeners on this node serve DP ranks {:?}, which are not \
                     contiguous",
                    listeners.iter().map(|l| l.dp_rank).collect::<Vec<_>>()
                )));
            }
            if !offsets.insert(listener.port_offset) {
                return Err(client::invalid_arg(format!(
                    "SGLang lists port offset {} on two listeners of this node",
                    listener.port_offset
                )));
            }
        }
        let own = listeners
            .iter()
            .position(|listener| listener.dp_rank == own_rank)
            .ok_or_else(|| {
                client::invalid_arg(format!(
                    "the SGLang listener reports DP rank {own_rank}, which its node's \
                     listener table does not list"
                ))
            })?;
        if listeners[own].http_port != server_info.port {
            return Err(client::invalid_arg(format!(
                "the SGLang listener for DP rank {own_rank} reports HTTP port {}, but its \
                 node's listener table lists {}",
                server_info.port, listeners[own].http_port
            )));
        }
        Ok(Self { listeners, own })
    }

    pub(crate) fn start_rank(&self) -> u32 {
        self.listeners[0].dp_rank
    }

    pub(crate) fn len(&self) -> usize {
        self.listeners.len()
    }
}

/// The node's listeners as one worker.
#[derive(Clone, Debug)]
pub(crate) struct NodeInfo {
    /// The start listener's discovery, with the node-wide capacity: the least
    /// KV capacity and host-cache capacity of any listener.
    pub engine: EngineInfo,
    pub start_rank: u32,
    pub size: u32,
    /// `DeploymentInfo.dp_size`: every DP rank of the deployment.
    pub global_dp_size: u32,
    /// Each listener's prefill bootstrap port, by DP rank (prefill only).
    pub bootstrap_ports: Vec<Option<u16>>,
    /// The union of every listener's KV-event publishers.
    pub kv_events: NodeMetadata,
}

impl NodeInfo {
    /// Aggregate the node's listeners into one registration; their KV sources
    /// must cover all or none of them. `infos` is in table order.
    pub(crate) fn from_listeners(
        table: &NodeTable,
        infos: &[&EngineInfo],
        mode: DisaggregationMode,
    ) -> Result<Self, DynamoError> {
        assert_eq!(infos.len(), table.len(), "one discovery per listener");
        let start = infos[0];
        let mut engine = start.clone();
        engine.max_total_num_tokens = min_present(infos, |info| info.max_total_num_tokens);
        engine.native_offloading_capacity =
            min_present(infos, |info| info.native_offloading_capacity);
        let bootstrap_ports = if mode.is_prefill() {
            infos
                .iter()
                .copied()
                .map(discovery_bootstrap_port)
                .collect::<Result<_, _>>()?
        } else {
            vec![None; infos.len()]
        };
        let kv_events = kv_union(infos)?;
        Ok(Self {
            engine,
            start_rank: table.start_rank(),
            size: u32::try_from(table.len()).expect("a node has few listeners"),
            global_dp_size: start.dp_size,
            bootstrap_ports,
            kv_events,
        })
    }
}

/// The least value any listener reports; unset if one reports none.
fn min_present(infos: &[&EngineInfo], value: impl Fn(&EngineInfo) -> Option<u64>) -> Option<u64> {
    infos
        .iter()
        .copied()
        .map(value)
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min()
}

/// Every listener publishes KV events for its own rank, or none does: a
/// router that sees only some ranks' events would misjudge the others.
fn kv_union(infos: &[&EngineInfo]) -> Result<NodeMetadata, DynamoError> {
    let mut sources: Vec<LocalKvEventSource> = Vec::new();
    let mut publishing = Vec::new();
    for info in infos {
        let metadata = info.kv_events.as_ref().map_err(Clone::clone)?;
        for source in &metadata.kv_event_sources {
            if source.dp_rank != info.dp_rank {
                return Err(client::protocol_error(format!(
                    "the SGLang listener for DP rank {} reports a KV-event source for DP rank {}",
                    info.dp_rank, source.dp_rank
                )));
            }
        }
        if !metadata.kv_event_sources.is_empty() {
            publishing.push(info.dp_rank);
        }
        sources.extend(metadata.kv_event_sources.iter().cloned());
    }
    if !publishing.is_empty() && publishing.len() != infos.len() {
        return Err(client::protocol_error(format!(
            "only the SGLang listeners for DP ranks {publishing:?} of this node publish KV \
             events; enable KV events on every DP rank or on none"
        )));
    }
    let start = infos[0].kv_events.as_ref().map_err(Clone::clone)?;
    NodeMetadata::new(
        start.node_rank,
        start.nnodes,
        start.dp_size,
        start.dist_init_addr.clone(),
        sources,
    )
    .map_err(|error| client::protocol_error(error.to_string()))
}

/// The gRPC endpoint of table entry `index`: the host of `--grpc-endpoint`
/// and the entry's own port. The entry the endpoint names keeps the endpoint
/// as configured.
pub(crate) fn listener_endpoint(
    endpoint: &GrpcEndpoint,
    table: &NodeTable,
    index: usize,
) -> Result<GrpcEndpoint, DynamoError> {
    if index == table.own {
        return Ok(endpoint.clone());
    }
    let listener = &table.listeners[index];
    let port = listener.grpc_port.ok_or_else(|| {
        client::invalid_arg(format!(
            "SGLang's listener for DP rank {} serves no native gRPC; launch SGLang with \
             --grpc-port so every listener serves sglang.api.v1",
            listener.dp_rank
        ))
    })?;
    GrpcEndpoint::parse(
        &format!("http://{}:{port}", endpoint.authority_host()),
        "--grpc-endpoint",
    )
}

#[cfg(test)]
pub(crate) fn node_of(
    infos: &[EngineInfo],
    mode: DisaggregationMode,
) -> Result<NodeInfo, DynamoError> {
    let infos: Vec<_> = infos.iter().collect();
    NodeInfo::from_listeners(&infos[0].table, &infos, mode)
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, ErrorType};

    use super::*;
    use crate::discovery::{engine_info, parse_discovery};

    const HTTP: u32 = 30_000;
    const GRPC: u32 = 30_010;

    fn listener(offset: u32, dp_rank: u32) -> pb::NodeRustListener {
        pb::NodeRustListener {
            port_offset: offset,
            dp_rank,
            http_port: HTTP + offset,
            grpc_port: Some(GRPC + offset),
        }
    }

    /// Discovery of the listener for `dp_rank` of a node whose listeners
    /// serve `ranks`, rewritten by `update`.
    fn report(
        dp_rank: u32,
        ranks: std::ops::Range<u32>,
        dp_size: u32,
        update: impl FnOnce(&mut pb::GetServerInfoResponse),
    ) -> Result<EngineInfo, DynamoError> {
        let table = ranks
            .clone()
            .map(|rank| listener(rank - ranks.start, rank))
            .collect();
        let offset = dp_rank - ranks.start;
        let mut server_info = pb::GetServerInfoResponse {
            incremental_streaming_output: true,
            port: HTTP + offset,
            disaggregation_mode: "prefill".into(),
            max_total_num_tokens: Some(4096 - u64::from(offset)),
            deployment_info: Some(pb::DeploymentInfo {
                nnodes: 1,
                dp_size,
                dp_rank: Some(dp_rank),
                dcp_size: 1,
                page_size: Some(16),
                enable_dp_attention: true,
                max_running_requests: Some(64),
                disaggregation_bootstrap_port: Some(HTTP + offset),
                node_listeners: table,
                ..Default::default()
            }),
            ..Default::default()
        };
        update(&mut server_info);
        engine_info(parse_discovery(
            pb::GetModelInfoResponse {
                model_path: "model".into(),
                ..Default::default()
            },
            server_info,
            Vec::new(),
        )?)
    }

    fn assert_invalid(error: &DynamoError, message: &str) {
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn node_listener_table_is_parsed() {
        // Node 1 of two lists its ranks 2 and 3 in any order.
        let info = report(3, 2..4, 4, |info| {
            let deployment = info.deployment_info.as_mut().unwrap();
            deployment.node_listeners.reverse();
            (deployment.nnodes, deployment.node_rank) = (2, 1);
            deployment.dist_init_addr = Some("10.0.0.1:5000".into());
        })
        .unwrap();
        let ranks: Vec<_> = info.table.listeners.iter().map(|l| l.dp_rank).collect();
        assert_eq!(
            (ranks, info.table.own, info.table.start_rank()),
            (vec![2, 3], 1, 2)
        );
        // Only node 0's rendezvous host is a bootstrap candidate; the worker
        // group still names the rendezvous.
        assert_eq!(info.dist_init_addr, None);
        assert_eq!(
            info.kv_events.as_ref().unwrap().dist_init_addr.as_deref(),
            Some("10.0.0.1:5000")
        );
        // A build without the table serves DP=1 as its one listener.
        let dp1 = report(0, 0..1, 1, |info| {
            info.deployment_info
                .as_mut()
                .unwrap()
                .node_listeners
                .clear()
        })
        .unwrap();
        assert_eq!(
            (dp1.table.len(), dp1.table.listeners[0].http_port),
            (1, HTTP)
        );
    }

    #[test]
    fn rejects_unusable_tables() {
        let error = report(0, 0..1, 4, |info| {
            info.deployment_info.as_mut().unwrap().node_listeners =
                vec![listener(0, 0), listener(1, 2)];
        })
        .unwrap_err();
        assert_invalid(&error, "not contiguous");
        assert_invalid(
            &report(0, 0..3, 2, |_| {}).unwrap_err(),
            "DP rank 2 of a 2-rank deployment",
        );
        let error = report(0, 0..2, 2, |info| {
            info.deployment_info
                .as_mut()
                .unwrap()
                .node_listeners
                .clear()
        })
        .unwrap_err();
        assert_invalid(&error, "no DeploymentInfo.node_listeners");
    }

    fn source(dp_rank: u32, port: u32) -> pb::KvEventSource {
        pb::KvEventSource {
            dp_rank,
            endpoint: format!("tcp://127.0.0.1:{port}"),
            topic: String::new(),
            block_size: 16,
        }
    }

    #[test]
    fn kv_union_requires_every_rank_or_none() {
        let with_sources = |rank: u32, sources: Vec<pb::KvEventSource>| {
            report(rank, 0..2, 2, move |info| {
                info.deployment_info.as_mut().unwrap().kv_event_sources = sources
            })
            .unwrap()
        };
        let infos = [
            with_sources(0, vec![source(0, 5557)]),
            with_sources(1, vec![source(1, 5558)]),
        ];
        let node = node_of(&infos, DisaggregationMode::Aggregated).unwrap();
        assert_eq!(
            node.kv_events
                .kv_event_sources
                .iter()
                .map(|source| source.dp_rank)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        for (infos, message) in [
            (
                [
                    with_sources(0, vec![source(0, 5557)]),
                    with_sources(1, Vec::new()),
                ],
                "only the SGLang listeners for DP ranks [0]",
            ),
            (
                [
                    with_sources(0, vec![source(0, 5557)]),
                    with_sources(1, vec![source(0, 5558)]),
                ],
                "reports a KV-event source for DP rank 0",
            ),
        ] {
            let error = node_of(&infos, DisaggregationMode::Aggregated).unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown),
                "{error}"
            );
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}
