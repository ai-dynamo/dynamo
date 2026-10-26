// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! ```bash
//! cargo run -p dynamo-global-discovery-service -- \
//!     --store memory --router aws-ohio/global-router-0=http://10.42.0.5:50051
//! ```

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_discovery_service::backends::kubernetes::{
    EndpointSliceReplicaDirectory, McsPoolDirectory, RouterService, TrustedNetworkAuthenticator,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::pool_discovery::{
    LeasePolicy, McsImplementation, PoolDirectory, PoolRegistrar, ReplicaEndpoint, ReplicaId,
};
use dynamo_global_discovery_service::service::{self, DiscoveryStore, RouterSource};
use kube::Client;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Parser)]
#[command(about = "Serve Global Router pool discovery over HTTP")]
struct Args {
    /// Address to serve HTTP on. Use `[::]:8080` on clusters with IPv6 pods.
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// Where pools and router replicas come from.
    #[arg(long, value_enum, default_value_t = Store::Memory)]
    store: Store,
    /// Memory store: how often a PoolRelay registers again.
    #[arg(long, default_value_t = 5)]
    heartbeat_interval_secs: u64,
    /// Memory store: how long a registration lasts.
    #[arg(long, default_value_t = 15)]
    lease_duration_secs: u64,
    /// Memory store: a Global Router replica as `<id>=<url>`. Repeat it for
    /// each replica.
    #[arg(long = "router", value_parser = parse_router)]
    routers: Vec<ReplicaEndpoint>,
    /// Kubernetes stores: each PoolRelay Service is named
    /// `<DGD name><suffix>`.
    #[arg(long, default_value = "-pool-relay")]
    pool_service_suffix: String,
    /// Kubernetes stores: namespace of the Global Router Service.
    #[arg(long)]
    router_namespace: Option<String>,
    /// Kubernetes stores: the Global Router Service, exported through MCS.
    #[arg(long)]
    router_service: Option<String>,
    /// Kubernetes stores: the router Service's named port for state streams.
    #[arg(long)]
    router_port_name: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Store {
    /// Registrations in this process; router replicas from `--router`.
    Memory,
    /// Imported EndpointSlices, named the Karmada way. No registrations.
    Karmada,
    /// Imported EndpointSlices with the standard `multicluster.kubernetes.io`
    /// labels. No registrations.
    Mcs,
}

fn parse_router(value: &str) -> Result<ReplicaEndpoint, String> {
    let (id, url) = value
        .split_once('=')
        .ok_or("expected <id>=<url>".to_owned())?;
    if id.is_empty() {
        return Err("the replica id is empty".to_owned());
    }
    let url = Url::parse(url).map_err(|error| error.to_string())?;
    Ok(ReplicaEndpoint {
        replica: ReplicaId(id.to_owned()),
        url,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let store = match args.store {
        Store::Memory => memory_store(&args)?,
        Store::Karmada => kubernetes_store(&args, McsImplementation::Karmada).await?,
        Store::Mcs => kubernetes_store(&args, McsImplementation::Standard).await?,
    };
    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("bind the Global Discovery listener at {}", args.listen))?;
    tracing::info!(address = %args.listen, "Global Discovery Service listening");

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });
    service::serve(listener, store, shutdown)
        .await
        .context("serve Global Discovery HTTP")
}

fn memory_store(args: &Args) -> Result<DiscoveryStore> {
    let lease = LeasePolicy {
        heartbeat_interval: Duration::from_secs(args.heartbeat_interval_secs),
        lease_duration: Duration::from_secs(args.lease_duration_secs),
    };
    let store = Arc::new(
        MemoryPoolStore::new(lease)
            .context("check --heartbeat-interval-secs and --lease-duration-secs")?,
    );
    let mut ids = HashSet::new();
    if let Some(repeated) = args
        .routers
        .iter()
        .find(|router| !ids.insert(&router.replica))
    {
        bail!("--router {} is given twice", repeated.replica.0);
    }
    let pools: Arc<PoolDirectory> = store.clone();
    let registrar: Arc<dyn PoolRegistrar> = store;
    Ok(DiscoveryStore {
        pools,
        routers: Arc::new(StaticReplicaDirectory::new(args.routers.clone())),
        router_source: RouterSource::Configured,
        registrar: Some(registrar),
        authenticator: Arc::new(TrustedNetworkAuthenticator),
    })
}

async fn kubernetes_store(args: &Args, mcs: McsImplementation) -> Result<DiscoveryStore> {
    let (Some(namespace), Some(name), Some(port_name)) = (
        &args.router_namespace,
        &args.router_service,
        &args.router_port_name,
    ) else {
        bail!(
            "the Kubernetes stores need --router-namespace, --router-service and --router-port-name"
        );
    };
    let client = Client::try_default()
        .await
        .context("create the Kubernetes client")?;
    let router = RouterService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    Ok(DiscoveryStore {
        // Each pool record names its Ready PoolRelay's address, which routers
        // on a trusted network use to admit the Relay.
        pools: Arc::new(McsPoolDirectory::new(
            client.clone(),
            mcs,
            args.pool_service_suffix.clone(),
            true,
            HashMap::new(),
        )),
        routers: Arc::new(EndpointSliceReplicaDirectory::imported(
            client, &router, mcs,
        )),
        router_source: RouterSource::Mcs,
        registrar: None,
        authenticator: Arc::new(TrustedNetworkAuthenticator),
    })
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}
