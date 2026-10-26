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
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::Lease;
use dynamo_global_discovery_service::backends::kubernetes::{
    ExportedService, FrontendService, McsImplementation, McsPoolDirectory, McsReplicaDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::backends::registered::RegisteredPoolDirectory;
use dynamo_global_discovery_service::peers::PeerSync;
use dynamo_global_discovery_service::pool_discovery::{
    PoolDirectory, PoolRegistrar, ReplicaEndpoint, ReplicaId,
};
use dynamo_global_discovery_service::service::{self, DiscoveryStore, Tls};
use kube::Client;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

/// The longest lease the service grants. `Lease` itself has no upper bound,
/// and the stores add the lease to the current time, which a lease near
/// `u64::MAX` seconds would overflow.
const MAX_LEASE_DURATION_SECS: u64 = 86_400;

/// How long the Kubernetes client waits to connect to the API server, TLS
/// included. kube's default is 30 s.
const KUBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the Kubernetes client waits for the API server to answer. The
/// Kubernetes stores need this and [`KUBE_CONNECT_TIMEOUT`] to add up to
/// under 90 s, so a server that stops answering shows up as failures; their
/// watches end after 50 s, so a working watch never reaches it. kube's
/// default is 295 s.
const KUBE_READ_TIMEOUT: Duration = Duration::from_secs(75);
const _: () = assert!(KUBE_CONNECT_TIMEOUT.as_secs() + KUBE_READ_TIMEOUT.as_secs() < 90);

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
    /// How often a PoolRelay registers again.
    #[arg(long, default_value_t = 5)]
    heartbeat_interval_secs: u64,
    /// How long a registration lasts. Longer than the heartbeat interval, and
    /// at most a day.
    #[arg(
        long,
        default_value_t = 15,
        value_parser = clap::value_parser!(u64).range(..=MAX_LEASE_DURATION_SECS)
    )]
    lease_duration_secs: u64,
    /// Memory store: a Global Router replica as `<id>=<url>`. Repeat it for
    /// each replica.
    #[arg(long = "router", value_parser = parse_router)]
    routers: Vec<ReplicaEndpoint>,
    /// Kubernetes stores: each PoolRelay Service is named
    /// `<DGD name><suffix>`.
    #[arg(
        long,
        default_value = "-pool-relay",
        allow_hyphen_values = true,
        value_parser = clap::builder::NonEmptyStringValueParser::new()
    )]
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
    /// With it, the instances in every cluster share their registrations,
    /// over plain HTTP, so not with `--tls-cert-path` yet. The lease must be
    /// more than twice the heartbeat interval. Other regions then drop a pool
    /// whose PoolRelay is more than the lease minus two heartbeat intervals
    /// late (5 s with the defaults); its own region allows the lease minus
    /// one.
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
    /// Registrations in this process; router replicas from `--router`. Run
    /// one replica: each keeps only the registrations it receives.
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

fn lease(args: &Args) -> Result<Lease> {
    Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")
}

fn memory_store(args: &Args) -> Result<DiscoveryStore> {
    if args.peer_service.is_some() {
        bail!("--peer-service requires --store karmada or --store mcs");
    }
    let store = Arc::new(MemoryPoolStore::new(lease(args)?));
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
    let lease = lease(args)?;
    if args.peer_service.is_some() {
        if args.tls_cert_path.is_some() {
            bail!(
                "--peer-service does not work with --tls-cert-path yet: instances ask each other \
                 over plain HTTP"
            );
        }
        if lease.lease_duration() <= lease.heartbeat_interval() * 2 {
            bail!(
                "with --peer-service, --lease-duration-secs must be more than twice \
                 --heartbeat-interval-secs, so that copies stay alive between refreshes"
            );
        }
    }
    let registrations = Arc::new(MemoryPoolStore::new(lease));
    let client = kubernetes_client()
        .await
        .context("create the Kubernetes client")?;
    let imported = McsPoolDirectory::new(
        client.clone(),
        mcs,
        args.pool_service_suffix.clone(),
        FrontendService {
            suffix: args.frontend_service_suffix.clone(),
            port_name: args.frontend_port_name.clone(),
        },
    )
    .context("check --frontend-service-suffix and --pool-service-suffix")?;
    let pools = RegisteredPoolDirectory::new(Arc::new(imported), registrations.clone(), lease);
    let sharing = match &args.peer_service {
        Some(peer_service) => {
            let peers = ExportedService {
                namespace: args
                    .peer_namespace
                    .clone()
                    .unwrap_or_else(|| namespace.clone()),
                name: peer_service.clone(),
                port_name: args.peer_port_name.clone(),
            };
            let (sync, sharing) = PeerSync::new(
                Arc::new(McsReplicaDirectory::new(client.clone(), &peers, mcs)),
                registrations.clone(),
                lease.heartbeat_interval(),
            )
            .context("create the HTTP client for --peer-service")?;
            tokio::spawn(sync.run(shutdown.clone()));
            Some(sharing)
        }
        None => {
            tracing::info!(
                "no --peer-service: registrations stay in this instance, so only pools that \
                 register here are listed"
            );
            None
        }
    };
    let router = ExportedService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    let registrar: Arc<dyn PoolRegistrar> = registrations;
    Ok(DiscoveryStore {
        pools: Arc::new(pools),
        routers: Arc::new(McsReplicaDirectory::new(client, &router, mcs)),
        registrar: Some(registrar),
        sharing,
    })
}

/// A client configured as `Client::try_default` does, from the pod's service
/// account or the local kubeconfig, but with [`KUBE_CONNECT_TIMEOUT`] and
/// [`KUBE_READ_TIMEOUT`].
async fn kubernetes_client() -> Result<Client> {
    let mut config = kube::Config::infer().await?;
    config.connect_timeout = Some(KUBE_CONNECT_TIMEOUT);
    config.read_timeout = Some(KUBE_READ_TIMEOUT);
    Ok(Client::try_from(config)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_service_suffix_can_start_with_a_hyphen_but_not_be_empty() {
        let args = Args::try_parse_from(["gds", "--pool-service-suffix", "-pool-gateway"]).unwrap();
        assert_eq!(args.pool_service_suffix, "-pool-gateway");
        // An empty suffix would make every imported Service a pool.
        assert!(Args::try_parse_from(["gds", "--pool-service-suffix="]).is_err());
    }

    #[test]
    fn a_lease_is_at_most_a_day() {
        let day = MAX_LEASE_DURATION_SECS.to_string();
        assert!(Args::try_parse_from(["gds", "--lease-duration-secs", &day]).is_ok());
        let longer = (MAX_LEASE_DURATION_SECS + 1).to_string();
        assert!(Args::try_parse_from(["gds", "--lease-duration-secs", &longer]).is_err());
    }
}
