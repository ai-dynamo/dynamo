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
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_discovery_service::backends::kubernetes::{
    EndpointSliceReplicaDirectory, FrontendService, McsPoolDirectory, RouterService,
    TrustedNetworkAuthenticator,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::backends::registered::RegisteredPoolDirectory;
use dynamo_global_discovery_service::peers::PeerSync;
use dynamo_global_discovery_service::pool_discovery::{
    LeasePolicy, McsImplementation, PoolDirectory, PoolRegistrar, ReplicaEndpoint, ReplicaId,
};
use dynamo_global_discovery_service::service::{self, DiscoveryStore, Sharing};
use kube::Client;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

#[derive(Parser)]
#[command(about = "Serve Global Router pool discovery over HTTP")]
struct Args {
    /// Address to serve HTTP on. Use `[::]:8080` on clusters with IPv6 pods.
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// Where pools and router replicas come from.
    #[arg(long, value_enum, default_value_t = Store::Memory)]
    store: Store,
    /// How often a PoolRelay registers again.
    #[arg(long, default_value_t = 5)]
    heartbeat_interval_secs: u64,
    /// How long a registration lasts.
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
    /// Kubernetes stores: each pool's frontend Service is named
    /// `<DGD name><suffix>`.
    #[arg(long, default_value = "-frontend")]
    frontend_service_suffix: String,
    /// Kubernetes stores: the frontend Service's named port for HTTP.
    #[arg(long, default_value = "http")]
    frontend_port_name: String,
    /// Kubernetes stores: namespace of the Global Router Service.
    #[arg(long)]
    router_namespace: Option<String>,
    /// Kubernetes stores: the Global Router Service, exported through MCS.
    #[arg(long)]
    router_service: Option<String>,
    /// Kubernetes stores: the router Service's named port for state streams.
    #[arg(long)]
    router_port_name: Option<String>,
    /// Kubernetes stores: this service's own Service, exported through MCS.
    /// With it, the instances in every cluster share their registrations.
    #[arg(long)]
    peer_service: Option<String>,
    /// Kubernetes stores: namespace of `--peer-service`. Defaults to
    /// `--router-namespace`.
    #[arg(long)]
    peer_namespace: Option<String>,
    /// Kubernetes stores: the named HTTP port of `--peer-service`.
    #[arg(long, default_value = "http")]
    peer_port_name: String,
}

#[derive(Clone, Copy, ValueEnum)]
enum Store {
    /// Registrations in this process; router replicas from `--router`.
    Memory,
    /// Registrations, joined with the imported EndpointSlices, named the
    /// Karmada way.
    Karmada,
    /// Registrations, joined with the imported EndpointSlices that have the
    /// standard `multicluster.kubernetes.io` labels.
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
    let shutdown = CancellationToken::new();
    let store = match args.store {
        Store::Memory => memory_store(&args)?,
        Store::Karmada => kubernetes_store(&args, McsImplementation::Karmada, &shutdown).await?,
        Store::Mcs => kubernetes_store(&args, McsImplementation::Standard, &shutdown).await?,
    };
    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("bind the Global Discovery listener at {}", args.listen))?;
    tracing::info!(address = %args.listen, "Global Discovery Service listening");

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

fn lease(args: &Args) -> LeasePolicy {
    LeasePolicy {
        heartbeat_interval: Duration::from_secs(args.heartbeat_interval_secs),
        lease_duration: Duration::from_secs(args.lease_duration_secs),
    }
}

fn registrations(args: &Args) -> Result<Arc<MemoryPoolStore>> {
    let store = MemoryPoolStore::new(lease(args))
        .context("check --heartbeat-interval-secs and --lease-duration-secs")?;
    Ok(Arc::new(store))
}

fn memory_store(args: &Args) -> Result<DiscoveryStore> {
    let store = registrations(args)?;
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
        authenticator: Arc::new(TrustedNetworkAuthenticator),
        sharing: None,
    })
}

/// A pool is listed while MCS imports a Ready PoolRelay endpoint for it and
/// its PoolRelay has registered within the lease. MCS gives the addresses;
/// the registration gives the model, capabilities, capacity and location.
async fn kubernetes_store(
    args: &Args,
    mcs: McsImplementation,
    shutdown: &CancellationToken,
) -> Result<DiscoveryStore> {
    let (Some(namespace), Some(name), Some(port_name)) = (
        &args.router_namespace,
        &args.router_service,
        &args.router_port_name,
    ) else {
        bail!(
            "the Kubernetes stores need --router-namespace, --router-service and --router-port-name"
        );
    };
    let registrations = registrations(args)?;
    let client = Client::try_default()
        .await
        .context("create the Kubernetes client")?;
    // Each pool record names its Ready PoolRelay's address, which routers on
    // a trusted network use to admit the Relay.
    let imported = McsPoolDirectory::with_frontends(
        client.clone(),
        mcs,
        args.pool_service_suffix.clone(),
        true,
        HashMap::new(),
        FrontendService {
            suffix: args.frontend_service_suffix.clone(),
            port_name: args.frontend_port_name.clone(),
        },
    )
    .context("check --frontend-service-suffix and --pool-service-suffix")?;
    let lease = lease(args);
    let pools = RegisteredPoolDirectory::new(Arc::new(imported), registrations.clone(), lease);
    if args.peer_service.is_some() && lease.lease_duration <= lease.heartbeat_interval * 2 {
        bail!(
            "with --peer-service, --lease-duration-secs must be more than twice \
             --heartbeat-interval-secs, so that copies stay alive between refreshes"
        );
    }
    let sharing = args.peer_service.as_ref().map(|peer_service| {
        let peers = RouterService {
            namespace: args
                .peer_namespace
                .clone()
                .unwrap_or_else(|| namespace.clone()),
            name: peer_service.clone(),
            port_name: args.peer_port_name.clone(),
        };
        let instance = Uuid::new_v4().to_string();
        let synced = Arc::new(AtomicBool::new(false));
        tokio::spawn(
            PeerSync {
                peers: Arc::new(EndpointSliceReplicaDirectory::imported(
                    client.clone(),
                    &peers,
                    mcs,
                )),
                store: registrations.clone(),
                instance: instance.clone(),
                every: lease.heartbeat_interval / 2,
                synced: synced.clone(),
            }
            .run(shutdown.clone()),
        );
        Sharing {
            instance,
            registrations: registrations.clone(),
            synced,
        }
    });
    if sharing.is_none() {
        tracing::info!(
            "no --peer-service: registrations stay in this instance, so only pools that \
             register here are listed"
        );
    }
    let router = RouterService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    let registrar: Arc<dyn PoolRegistrar> = registrations;
    Ok(DiscoveryStore {
        pools: Arc::new(pools),
        routers: Arc::new(EndpointSliceReplicaDirectory::imported(
            client, &router, mcs,
        )),
        registrar: Some(registrar),
        authenticator: Arc::new(TrustedNetworkAuthenticator),
        sharing,
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
