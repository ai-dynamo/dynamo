// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::CommandFactory;
use clap::error::ErrorKind;
use dynamo_backend_common::CommonArgs;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs, SidecarArgs};

use crate::topology::check_socket_path;

const DEFAULT_TOPOLOGY_AGENT_SOCKET: &str = "/run/lpx-topology/agent.sock";
const DEFAULT_TOPOLOGY_SOCKET: &str = "/run/lpx-topology/topology.sock";

fn parse_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())
}

fn parse_socket_path(raw: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(raw);
    check_socket_path(&path)?;
    Ok(path)
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-lpx-sidecar",
    about = "Run a Dynamo worker against an LPX engine's OpenEngine gRPC server"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Engine gRPC endpoint as `host:port`, a plaintext gRPC/HTTP URL, or
    /// `unix:///ABSOLUTE/PATH` for a Unix socket. Required to serve; a
    /// topology publisher or a standby rank does not use it.
    #[arg(
        long,
        env = "DYN_SIDECAR_GRPC_ENDPOINT",
        value_parser = parse_grpc_endpoint
    )]
    pub grpc_endpoint: Option<GrpcEndpoint>,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,

    /// Hugging Face model ID or local path used for tokenization and the chat
    /// template. The engine's own model paths are inside its container, so the
    /// sidecar cannot use them. Required to serve.
    #[arg(long)]
    pub model_path: Option<String>,

    /// The model name the engine must serve. The sidecar registers the name
    /// the engine reports and fails at startup when it differs from this one.
    /// Optional; it lets a runtime container written for the TensorRT-LLM
    /// sidecar, which takes this flag, keep its arguments.
    #[arg(long, env = "DYN_SERVED_MODEL_NAME")]
    pub served_model_name: Option<String>,

    /// Rank of this Pod in a multi-Pod engine or Agent allocation; only rank 0
    /// serves or publishes topology. On any other rank the sidecar stands by:
    /// it answers only the system health and live probes and never connects to
    /// the engine or to Dynamo, unless it is a topology consumer.
    // Kubernetes resolves a rank bound to a missing Pod label to an empty
    // value. clap's u32 parser rejects it rather than applying the default,
    // which would give the engine a second serving rank 0.
    #[arg(long = "node-rank", env = "DYN_SIDECAR_NODE_RANK", default_value_t = 0)]
    pub node_rank: u32,

    #[command(flatten)]
    pub topology: TopologyArgs,
}

/// The sidecar's LPX partition-topology role, independent of serving.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct TopologyArgs {
    /// LPX partition-topology role. `publisher`, beside an LPX Agent
    /// allocation's leader (rank 0), registers the Agent's topology service in
    /// Dynamo discovery and serves no engine. `consumer`, in a conductor pod,
    /// serves the conductor a local topology API backed by that authority, on
    /// every rank and alongside the engine on rank 0. Unset: no topology work.
    #[arg(long = "topology-role", env = "DYN_LPX_TOPOLOGY_ROLE", value_enum)]
    pub role: Option<TopologyRole>,

    /// Publisher: the Unix socket on which the Agent serves its topology.
    #[arg(
        long = "topology-agent-socket",
        env = "DYN_LPX_TOPOLOGY_AGENT_SOCKET",
        default_value = DEFAULT_TOPOLOGY_AGENT_SOCKET,
        value_parser = parse_socket_path
    )]
    pub agent_socket: PathBuf,

    /// Consumer: the Unix socket on which the sidecar serves the topology API.
    #[arg(
        long = "topology-socket",
        env = "DYN_LPX_TOPOLOGY_SOCKET",
        default_value = DEFAULT_TOPOLOGY_SOCKET,
        value_parser = parse_socket_path
    )]
    pub socket: PathBuf,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TopologyRole {
    Publisher,
    Consumer,
}

/// The arguments of a sidecar that serves the engine, with the ones serving
/// requires.
#[derive(Clone, Debug)]
pub(crate) struct ServeArgs {
    pub sidecar: SidecarArgs,
    pub model_path: String,
    pub served_model_name: Option<String>,
}

impl Args {
    /// The serving arguments, or the CLI error a missing required one gives.
    pub(crate) fn serve(self) -> Result<ServeArgs, clap::Error> {
        let missing: Vec<&str> = [
            (
                self.grpc_endpoint.is_none(),
                "--grpc-endpoint <GRPC_ENDPOINT>",
            ),
            (self.model_path.is_none(), "--model-path <MODEL_PATH>"),
        ]
        .into_iter()
        .filter_map(|(is_missing, flag)| is_missing.then_some(flag))
        .collect();
        let (Some(grpc_endpoint), Some(model_path)) = (self.grpc_endpoint, self.model_path) else {
            return Err(Self::command().error(
                ErrorKind::MissingRequiredArgument,
                format!(
                    "the following required arguments were not provided:\n  {}",
                    missing.join("\n  ")
                ),
            ));
        };
        Ok(ServeArgs {
            sidecar: SidecarArgs {
                common: self.common,
                grpc_endpoint,
                grpc: self.grpc,
            },
            model_path,
            served_model_name: self.served_model_name,
        })
    }
}
