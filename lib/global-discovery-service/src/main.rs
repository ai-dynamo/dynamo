// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! ```bash
//! cargo run -p dynamo-global-discovery-service -- \
//!     --store memory --router aws-ohio/global-router-0=http://10.42.0.5:50051
//! ```

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::Lease;
use dynamo_global_discovery_service::backends::kubernetes::{
    EndpointSliceReplicaDirectory, ExportedService, McsImplementation, McsPoolDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::pool_discovery::{
    PoolDirectory, PoolRegistrar, ReplicaEndpoint, ReplicaId,
};
use dynamo_global_discovery_service::service::{self, DiscoveryStore, Tls};
use kube::Client;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Parser)]
#[command(about = "Serve Global Router pool discovery over HTTP")]
struct Args {
    /// Address to serve on. Use `[::]:8080` on clusters with IPv6 pods.
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// Serve HTTPS with this certificate. Needs `--tls-key-path`.
    #[arg(long, requires = "tls_key_path")]
    tls_cert_path: Option<PathBuf>,
    /// The private key for `--tls-cert-path`.
    #[arg(long, requires = "tls_cert_path")]
    tls_key_path: Option<PathBuf>,
    /// Also require callers to present a certificate that this CA signed
    /// (mTLS). Probes need one too, so probe readiness with TCP or a command.
    #[arg(long, requires = "tls_cert_path")]
    tls_client_ca_cert_path: Option<PathBuf>,
    /// Where pools and router replicas come from.
    #[arg(long, value_enum, default_value_t = Store::Memory)]
    store: Store,
    /// Memory store: how often a PoolRelay registers again.
    #[arg(long, default_value_t = 5)]
    heartbeat_interval_secs: u64,
    /// Memory store: how long a registration lasts. Longer than the
    /// heartbeat interval, and at most a day.
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
    /// Registrations in this process; router replicas from `--router`. Run
    /// one replica: each keeps only the registrations it receives.
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
    cancel_on_signal(shutdown.clone())?;
    let tls = args
        .tls_cert_path
        .clone()
        .zip(args.tls_key_path.clone())
        .map(|(cert, key)| Tls {
            cert,
            key,
            client_ca: args.tls_client_ca_cert_path.clone(),
        });
    service::serve(listener, store, tls, shutdown)
        .await
        .context("serve the Global Discovery Service")
}

fn memory_store(args: &Args) -> Result<DiscoveryStore> {
    let lease = Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")?;
    let store = Arc::new(MemoryPoolStore::new(lease));
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
        registrar: Some(registrar),
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
    let router = ExportedService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    Ok(DiscoveryStore {
        pools: Arc::new(McsPoolDirectory::new(
            client.clone(),
            mcs,
            args.pool_service_suffix.clone(),
        )),
        routers: Arc::new(EndpointSliceReplicaDirectory::imported(
            client, &router, mcs,
        )),
        registrar: None,
    })
}

/// Cancels `shutdown` on SIGTERM or Ctrl-C. Fails at startup if it cannot
/// listen for SIGTERM, so the service never runs without a clean way to stop.
fn cancel_on_signal(shutdown: CancellationToken) -> Result<()> {
    #[cfg(unix)]
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("listen for SIGTERM")?;
    tokio::spawn(async move {
        #[cfg(unix)]
        let terminated = terminate.recv();
        #[cfg(not(unix))]
        let terminated = std::future::pending::<Option<()>>();
        tokio::select! {
            // If Ctrl-C cannot be watched, SIGTERM still stops the service.
            Ok(()) = tokio::signal::ctrl_c() => {}
            _ = terminated => {}
        }
        shutdown.cancel();
    });
    Ok(())
}
