// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! Two flags choose the backend behind the API:
//! - `--pool-discovery`: where the pool and router lists come from. Either
//!   the registrations PoolRelays send (`registrations`), or the
//!   EndpointSlices that MCS imports, named the Karmada way (`karmada`) or
//!   the standard way (`mcs`).
//! - `--registration-storage`: where registrations are kept. Only `memory`
//!   for now.
//!
//! ```bash
//! cargo run -p dynamo-global-discovery-service -- \
//!     --pool-discovery registrations --router aws-ohio/global-router-0=http://10.42.0.5:50051
//! ```

use std::collections::HashSet;
use std::ffi::OsString;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::error::ErrorKind;
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum};
use dynamo_global_common::Lease;
use dynamo_global_discovery_service::backends::kubernetes::{
    ExportedService, McsImplementation, McsPoolDirectory, McsReplicaDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::pool_discovery::{
    PoolDirectory, PoolRegistrar, RouterReplica, RouterReplicaId,
};
use dynamo_global_discovery_service::service::{self, DiscoveryBackend, Tls};
use kube::Client;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

/// The longest lease the service grants. `Lease` itself has no upper bound,
/// and the registration storage adds the lease to the current time, which a
/// lease near `u64::MAX` seconds would overflow.
const MAX_LEASE_DURATION_SECS: u64 = 86_400;

/// How long the Kubernetes client waits to connect to the API server, TLS
/// included. kube's default is 30 s.
const KUBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the Kubernetes client waits for the API server to answer. Pool
/// discovery from MCS needs this and [`KUBE_CONNECT_TIMEOUT`] to add up to
/// under 90 s, so a server that stops answering shows up as failures; its
/// watches end after 50 s, so a working watch never reaches it. kube's
/// default is 295 s.
const KUBE_READ_TIMEOUT: Duration = Duration::from_secs(75);
const _: () = assert!(KUBE_CONNECT_TIMEOUT.as_secs() + KUBE_READ_TIMEOUT.as_secs() < 90);

/// `--help` headings for the flags that work with only some `--pool-discovery`
/// values. A flag under the heading of another value fails at startup, rather
/// than being ignored.
const WITH_REGISTRATIONS: &str = "With --pool-discovery registrations";
const WITH_MCS: &str = "With --pool-discovery karmada or mcs";

#[derive(Debug, Parser)]
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
    /// Where the pool and router lists come from.
    #[arg(long, value_enum, default_value_t = PoolDiscovery::Registrations)]
    pool_discovery: PoolDiscovery,
    /// Where registrations are kept.
    #[arg(long, value_enum, default_value_t = RegistrationStorage::Memory)]
    registration_storage: RegistrationStorage,
    /// How often a PoolRelay registers again.
    #[arg(long, default_value_t = 5, help_heading = WITH_REGISTRATIONS)]
    heartbeat_interval_secs: u64,
    /// How long a registration lasts. Longer than the heartbeat interval, and
    /// at most a day.
    #[arg(
        long,
        default_value_t = 15,
        value_parser = clap::value_parser!(u64).range(..=MAX_LEASE_DURATION_SECS),
        help_heading = WITH_REGISTRATIONS
    )]
    lease_duration_secs: u64,
    /// A Global Router replica as `<id>=<url>`. Repeat it for each replica.
    #[arg(long = "router", value_parser = parse_router, help_heading = WITH_REGISTRATIONS)]
    routers: Vec<RouterReplica>,
    /// Each PoolRelay Service is named `<DGD name><suffix>`.
    #[arg(
        long,
        default_value = "-pool-relay",
        allow_hyphen_values = true,
        value_parser = clap::builder::NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS
    )]
    pool_service_suffix: String,
    /// Namespace of the Global Router Service.
    #[arg(long, help_heading = WITH_MCS)]
    router_namespace: Option<String>,
    /// The Global Router Service, exported through MCS.
    #[arg(long, help_heading = WITH_MCS)]
    router_service: Option<String>,
    /// The router Service's named port for state streams.
    #[arg(long, help_heading = WITH_MCS)]
    router_port_name: Option<String>,
}

/// Where the pool and router lists come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PoolDiscovery {
    /// The pools PoolRelays register (`POST /pools`), while their leases
    /// last. Router replicas come from `--router`.
    Registrations,
    /// The pools whose PoolRelay Services Karmada imports, and the replicas
    /// of the router Service, from EndpointSlices named the Karmada way.
    /// Takes no registrations.
    Karmada,
    /// As `karmada`, from EndpointSlices with the standard
    /// `multicluster.kubernetes.io` labels.
    Mcs,
}

/// Where registrations are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RegistrationStorage {
    /// In this process. Run one replica: each keeps only the registrations it
    /// receives.
    Memory,
}

fn parse_router(value: &str) -> Result<RouterReplica, String> {
    let (id, url) = value
        .split_once('=')
        .ok_or("expected <id>=<url>".to_owned())?;
    if id.is_empty() {
        return Err("the replica id is empty".to_owned());
    }
    let url = Url::parse(url).map_err(|error| error.to_string())?;
    Ok(RouterReplica {
        replica: RouterReplicaId(id.to_owned()),
        url,
    })
}

/// Parses the flags. A flag under the `--help` heading of another
/// `--pool-discovery` value fails, as clap's own errors do.
fn parse_args<I, T>(args: I) -> Result<Args, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let mut command = Args::command();
    let matches = command.try_get_matches_from_mut(args)?;
    let parsed = Args::from_arg_matches(&matches)?;
    let refused = match parsed.pool_discovery {
        PoolDiscovery::Registrations => WITH_MCS,
        PoolDiscovery::Karmada | PoolDiscovery::Mcs => WITH_REGISTRATIONS,
    };
    let given = command
        .get_arguments()
        .find(|arg| {
            arg.get_help_heading() == Some(refused)
                && matches.value_source(arg.get_id().as_str()) == Some(ValueSource::CommandLine)
        })
        .and_then(|arg| arg.get_long())
        .map(str::to_owned);
    match given {
        Some(flag) => Err(command.error(
            ErrorKind::ArgumentConflict,
            format!("--{flag} works only {}", refused.to_lowercase()),
        )),
        None => Ok(parsed),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = parse_args(std::env::args_os()).unwrap_or_else(|error| error.exit());
    let backend = match args.pool_discovery {
        PoolDiscovery::Registrations => registrations_backend(&args)?,
        PoolDiscovery::Karmada => mcs_backend(&args, McsImplementation::Karmada).await?,
        PoolDiscovery::Mcs => mcs_backend(&args, McsImplementation::Standard).await?,
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
    service::serve(listener, backend, tls, shutdown)
        .await
        .context("serve the Global Discovery Service")
}

/// Pools from registrations; router replicas from `--router`.
fn registrations_backend(args: &Args) -> Result<DiscoveryBackend> {
    let lease = Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")?;
    let registrations = match args.registration_storage {
        RegistrationStorage::Memory => Arc::new(MemoryPoolStore::new(lease)),
    };
    let mut ids = HashSet::new();
    if let Some(repeated) = args
        .routers
        .iter()
        .find(|router| !ids.insert(&router.replica))
    {
        bail!("--router {} is given twice", repeated.replica.0);
    }
    let pools: Arc<PoolDirectory> = registrations.clone();
    let registrar: Arc<dyn PoolRegistrar> = registrations;
    Ok(DiscoveryBackend {
        pools,
        routers: Arc::new(StaticReplicaDirectory::new(args.routers.clone())),
        registrar: Some(registrar),
    })
}

/// Pools from the PoolRelay Services that MCS imports, and router replicas
/// from the router Service's.
async fn mcs_backend(args: &Args, mcs: McsImplementation) -> Result<DiscoveryBackend> {
    let (Some(namespace), Some(name), Some(port_name)) = (
        &args.router_namespace,
        &args.router_service,
        &args.router_port_name,
    ) else {
        bail!(
            "--pool-discovery karmada and mcs need --router-namespace, --router-service and \
             --router-port-name"
        );
    };
    let client = kubernetes_client()
        .await
        .context("create the Kubernetes client")?;
    let router = ExportedService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    Ok(DiscoveryBackend {
        pools: Arc::new(McsPoolDirectory::new(
            client.clone(),
            mcs,
            args.pool_service_suffix.clone(),
        )),
        routers: Arc::new(McsReplicaDirectory::new(client, &router, mcs)),
        registrar: None,
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

    const ROUTER: &str = "--router=aws-ohio/global-router-0=http://10.42.0.5:50051";

    #[test]
    fn pools_come_from_registrations_kept_in_memory_by_default() {
        let args = parse_args(["gds"]).unwrap();
        assert_eq!(args.pool_discovery, PoolDiscovery::Registrations);
        assert_eq!(args.registration_storage, RegistrationStorage::Memory);
    }

    #[test]
    fn a_flag_for_another_pool_discovery_fails() {
        assert!(parse_args(["gds", ROUTER]).is_ok());
        let error = parse_args(["gds", "--pool-discovery", "karmada", ROUTER]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("--router works only with --pool-discovery registrations"),
            "{error}"
        );
        assert!(
            parse_args([
                "gds",
                "--pool-discovery",
                "mcs",
                "--router-service",
                "global-router"
            ])
            .is_ok()
        );
        assert!(parse_args(["gds", "--router-service", "global-router"]).is_err());
        assert!(
            parse_args([
                "gds",
                "--pool-discovery",
                "karmada",
                "--lease-duration-secs",
                "30"
            ])
            .is_err()
        );
        // A flag left at its default is not refused.
        assert!(parse_args(["gds", "--pool-discovery", "karmada"]).is_ok());
    }

    #[test]
    fn pool_service_suffix_can_start_with_a_hyphen_but_not_be_empty() {
        let args = parse_args([
            "gds",
            "--pool-discovery",
            "karmada",
            "--pool-service-suffix",
            "-pool-gateway",
        ])
        .unwrap();
        assert_eq!(args.pool_service_suffix, "-pool-gateway");
        // An empty suffix would make every imported Service a pool.
        assert!(
            parse_args([
                "gds",
                "--pool-discovery",
                "karmada",
                "--pool-service-suffix="
            ])
            .is_err()
        );
    }

    #[test]
    fn a_lease_is_at_most_a_day() {
        let day = MAX_LEASE_DURATION_SECS.to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &day]).is_ok());
        let longer = (MAX_LEASE_DURATION_SECS + 1).to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &longer]).is_err());
    }
}
