// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! Two flags choose the backend behind the API:
//! - `--pool-discovery`: where the pool and router lists come from. Either
//!   the registrations PoolRelays send (`registrations`), or the
//!   EndpointSlices that MCS imports, named the Karmada way (`karmada`) or
//!   the standard way (`mcs`), joined with registrations.
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
use clap::builder::NonEmptyStringValueParser;
use clap::error::ErrorKind;
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum};
use dynamo_global_common::Lease;
use dynamo_global_discovery_service::backends::kubernetes::{
    ExportedService, FrontendService, KARMADA_MAX_SERVICE_NAME, McsImplementation,
    McsPoolDirectory, McsReplicaDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::backends::registered::RegisteredPoolDirectory;
use dynamo_global_discovery_service::peers::PeerSync;
use dynamo_global_discovery_service::pool_discovery::{
    PoolDirectory, PoolRegistrar, RouterReplica, RouterReplicaId,
};
use dynamo_global_discovery_service::service::{self, PoolDiscoveryBackend, Tls};
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
    /// Where the pool and router lists come from.
    #[arg(long, value_enum, default_value_t = PoolDiscovery::Registrations)]
    pool_discovery: PoolDiscovery,
    /// Where registrations are kept.
    #[arg(long, value_enum, default_value_t = RegistrationStorage::Memory)]
    registration_storage: RegistrationStorage,
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
    /// A Global Router replica as `<id>=<url>`. Repeat it for each replica.
    #[arg(long = "router", value_parser = parse_router, help_heading = WITH_REGISTRATIONS)]
    routers: Vec<RouterReplica>,
    /// Each PoolRelay Service is named `<DGD name><suffix>`. Give a suffix
    /// that starts with `-` after an `=`: `--pool-service-suffix=-relay`.
    #[arg(
        long,
        default_value = "-pool-relay",
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS
    )]
    pool_service_suffix: String,
    /// Each pool's frontend Service is named `<DGD name><suffix>`.
    #[arg(
        long,
        default_value = "-frontend",
        allow_hyphen_values = true,
        value_parser = clap::builder::NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS
    )]
    frontend_service_suffix: String,
    /// The frontend Service's named port for HTTP.
    #[arg(long, default_value = "http", help_heading = WITH_MCS)]
    frontend_port_name: String,
    /// Namespace of the Global Router Service.
    #[arg(
        long,
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS,
        required_if_eq_any = MCS_VALUES
    )]
    router_namespace: Option<String>,
    /// The Global Router Service, exported through MCS.
    #[arg(
        long,
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS,
        required_if_eq_any = MCS_VALUES
    )]
    router_service: Option<String>,
    /// The router Service's named port for state streams.
    #[arg(
        long,
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS,
        required_if_eq_any = MCS_VALUES
    )]
    router_port_name: Option<String>,
    /// This service's own Service, exported through MCS. With it, the
    /// instances in every cluster share their registrations, over plain
    /// HTTP, so not with `--tls-cert-path` yet. The lease must be more than
    /// twice the heartbeat interval. Other regions then drop a pool whose
    /// PoolRelay is more than the lease minus two heartbeat intervals late
    /// (5 s with the defaults); its own region allows the lease minus one.
    #[arg(long, help_heading = WITH_MCS)]
    peer_service: Option<String>,
    /// Namespace of `--peer-service`. Defaults to `--router-namespace`.
    #[arg(long, help_heading = WITH_MCS, requires = "peer_service")]
    peer_namespace: Option<String>,
    /// The named HTTP port of `--peer-service`.
    #[arg(
        long,
        default_value = "http",
        help_heading = WITH_MCS,
        requires = "peer_service"
    )]
    peer_port_name: String,
}

/// The `--pool-discovery` values that read MCS, for clap's
/// `required_if_eq_any`.
const MCS_VALUES: [(&str, &str); 2] = [("pool_discovery", "karmada"), ("pool_discovery", "mcs")];

/// Where the pool and router lists come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PoolDiscovery {
    /// The pools PoolRelays register (`POST /pools`), while their leases
    /// last. Router replicas come from `--router`.
    Registrations,
    /// The pools whose PoolRelay Services Karmada imports, joined with their
    /// registrations, and the replicas of the router Service, from
    /// EndpointSlices named the Karmada way.
    Karmada,
    /// As `karmada`, from EndpointSlices with the standard
    /// `multicluster.kubernetes.io` labels.
    Mcs,
}

impl PoolDiscovery {
    /// How imported EndpointSlices are named, for the values that read MCS.
    fn mcs(self) -> Option<McsImplementation> {
        match self {
            Self::Registrations => None,
            Self::Karmada => Some(McsImplementation::Karmada),
            Self::Mcs => Some(McsImplementation::Standard),
        }
    }
}

/// Where registrations are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RegistrationStorage {
    /// In this process: each instance keeps only the registrations it
    /// receives. With MCS, `--peer-service` shares them between instances;
    /// otherwise run one replica.
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
    // Without `http://`, `Url` reads a host such as `localhost:50051` as the
    // scheme.
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "expected an http or https URL, such as http://10.42.0.5:50051, not {url}"
        ));
    }
    Ok(RouterReplica {
        replica: RouterReplicaId(id.to_owned()),
        url,
    })
}

/// Parses the flags. A flag set under the `--help` heading of another
/// `--pool-discovery` value fails, as clap's own errors do. A flag left at
/// its default is not refused. The checks that span flags are here too, so
/// that every flag mistake gets clap's usage and exit code.
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
                && matches
                    .value_source(arg.get_id().as_str())
                    .is_some_and(|source| source != ValueSource::DefaultValue)
        })
        .map(|arg| arg.get_long().unwrap_or(arg.get_id().as_str()).to_owned());
    if let Some(flag) = given {
        return Err(command.error(
            ErrorKind::ArgumentConflict,
            format!("--{flag} works only {}", refused.to_lowercase()),
        ));
    }
    if let Err(error) = Lease::new(parsed.heartbeat_interval_secs, parsed.lease_duration_secs) {
        return Err(command.error(
            ErrorKind::ValueValidation,
            format!("check --heartbeat-interval-secs and --lease-duration-secs: {error}"),
        ));
    }
    let mut ids = HashSet::new();
    if let Some(repeated) = parsed
        .routers
        .iter()
        .find(|router| !ids.insert(&router.replica))
    {
        return Err(command.error(
            ErrorKind::ValueValidation,
            format!("--router {} is given twice", repeated.replica.0),
        ));
    }
    // Karmada imports no Service with a longer name, and the API server
    // refuses to list one, so the service would never be ready.
    if let (Some(mcs), Some(service)) = (parsed.pool_discovery.mcs(), &parsed.router_service)
        && !mcs.can_import(service)
    {
        return Err(command.error(
            ErrorKind::ValueValidation,
            format!(
                "--router-service has {} characters, and Karmada imports a Service name of at \
                 most {KARMADA_MAX_SERVICE_NAME}",
                service.len()
            ),
        ));
    }
    Ok(parsed)
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = parse_args(std::env::args_os()).unwrap_or_else(|error| error.exit());
    let shutdown = CancellationToken::new();
    let backend = match args.pool_discovery.mcs() {
        None => registrations_backend(&args)?,
        Some(mcs) => mcs_backend(&args, mcs, &shutdown).await?,
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
        .map(|(cert, key)| Tls { cert, key });
    service::serve(listener, backend, tls, shutdown)
        .await
        .context("serve the Global Discovery Service")
}

fn lease(args: &Args) -> Result<Lease> {
    Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")
}

/// Where registrations go, as `--registration-storage` says.
fn registration_storage(args: &Args, lease: Lease) -> Arc<MemoryPoolStore> {
    match args.registration_storage {
        RegistrationStorage::Memory => Arc::new(MemoryPoolStore::new(lease)),
    }
}

/// Pools from registrations; router replicas from `--router`.
fn registrations_backend(args: &Args) -> Result<PoolDiscoveryBackend> {
    let registrations = registration_storage(args, lease(args)?);
    if args.routers.is_empty() {
        tracing::warn!(
            "no --router given, so GET /routers lists no Global Router replica for PoolRelays to \
             connect to"
        );
    }
    let pools: Arc<PoolDirectory> = registrations.clone();
    let registrar: Arc<dyn PoolRegistrar> = registrations;
    Ok(PoolDiscoveryBackend {
        pools,
        routers: Arc::new(StaticReplicaDirectory::new(args.routers.clone())),
        registrar: Some(registrar),
        sharing: None,
    })
}

/// Pools from what MCS imports, joined with registrations: a pool is listed
/// while MCS imports a Ready PoolRelay endpoint for it and its PoolRelay has
/// registered within the lease. MCS gives the addresses; the registration
/// gives the model, capabilities, capacity and location. Router replicas
/// come from the router Service's imports.
async fn mcs_backend(
    args: &Args,
    mcs: McsImplementation,
    shutdown: &CancellationToken,
) -> Result<PoolDiscoveryBackend> {
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
    let lease = lease(args)?;
    if args.peer_service.is_some() {
        if args.tls_cert_path.is_some() {
            bail!(
                "--peer-service does not work with --tls-cert-path yet: instances ask each other \
                 over plain HTTP"
            );
        }
        if lease.duration() <= lease.heartbeat_interval() * 2 {
            bail!(
                "with --peer-service, --lease-duration-secs must be more than twice \
                 --heartbeat-interval-secs, so that copies stay alive between refreshes"
            );
        }
    }
    let registrations = registration_storage(args, lease);
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
                lease,
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
    Ok(PoolDiscoveryBackend {
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
    // With a kubeconfig, kube also takes a proxy from `HTTPS_PROXY`, ignoring
    // `NO_PROXY`. This build has no proxy support, so the client would fail
    // here; connect to the API server directly instead. Only the host is
    // logged: a proxy URL can hold a password.
    if let Some(proxy) = config.proxy_url.take() {
        tracing::warn!(
            proxy_host = proxy.host(),
            "this build cannot use a proxy, so it connects to the Kubernetes API server directly"
        );
    }
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

    /// `--pool-discovery karmada` with the router flags it needs, then `extra`.
    fn karmada<'a>(extra: &[&'a str]) -> Vec<&'a str> {
        let mut args = vec![
            "gds",
            "--pool-discovery",
            "karmada",
            "--router-namespace",
            "global-discovery",
            "--router-service",
            "global-router",
            "--router-port-name",
            "http",
        ];
        args.extend_from_slice(extra);
        args
    }

    /// `args` fail because `--{flag}` works only with `--pool-discovery {values}`.
    fn assert_refused(args: &[&str], flag: &str, values: &str) {
        let error = parse_args(args.iter().copied()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict, "{error}");
        let reason = format!("--{flag} works only with --pool-discovery {values}");
        assert!(error.to_string().contains(&reason), "{error}");
    }

    #[test]
    fn pools_come_from_registrations_kept_in_memory_by_default() {
        let args = parse_args(["gds"]).unwrap();
        assert_eq!(args.pool_discovery, PoolDiscovery::Registrations);
        assert_eq!(args.registration_storage, RegistrationStorage::Memory);
    }

    #[test]
    fn every_flag_says_which_pool_discovery_it_works_with() {
        // The flags that work with every --pool-discovery value. Any other
        // flag needs one of the two headings, which `parse_args` enforces.
        const EVERY_VALUE: &[&str] = &[
            "listen",
            "tls_cert_path",
            "tls_key_path",
            "pool_discovery",
            "registration_storage",
            "heartbeat_interval_secs",
            "lease_duration_secs",
            "help",
        ];
        for arg in Args::command().get_arguments() {
            let id = arg.get_id().as_str();
            let headed = matches!(arg.get_help_heading(), Some(WITH_REGISTRATIONS | WITH_MCS));
            assert_ne!(
                EVERY_VALUE.contains(&id),
                headed,
                "--{id}: list it in EVERY_VALUE or give it a --pool-discovery heading"
            );
        }
    }

    #[test]
    fn mcs_needs_the_router_service() {
        let error = parse_args(["gds", "--pool-discovery", "mcs"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{error}");
        let all = karmada(&[]);
        assert!(parse_args(all.clone()).is_ok());
        for flag in [
            "--router-namespace",
            "--router-service",
            "--router-port-name",
        ] {
            let mut args = all.clone();
            let at = args.iter().position(|arg| *arg == flag).unwrap();
            args.drain(at..at + 2);
            let error = parse_args(args).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{error}");
            assert!(error.to_string().contains(flag), "{error}");
        }
    }

    #[test]
    fn peer_flags_need_peer_service() {
        for flag in ["--peer-namespace=other", "--peer-port-name=grpc"] {
            let error = parse_args(karmada(&[flag])).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{error}");
            assert!(error.to_string().contains("--peer-service"), "{error}");
            assert!(parse_args(karmada(&["--peer-service=gds", flag])).is_ok());
        }
    }

    #[test]
    fn a_flag_for_another_pool_discovery_fails() {
        assert!(parse_args(["gds", ROUTER]).is_ok());
        assert_refused(&karmada(&[ROUTER]), "router", "registrations");
        assert_refused(
            &["gds", "--router-service", "global-router"],
            "router-service",
            "karmada or mcs",
        );
        assert_refused(
            &["gds", "--frontend-service-suffix=-web"],
            "frontend-service-suffix",
            "karmada or mcs",
        );
        assert_refused(
            &["gds", "--peer-service", "global-discovery-service"],
            "peer-service",
            "karmada or mcs",
        );
        // Every --pool-discovery value takes registrations.
        assert!(
            parse_args(karmada(&[
                "--lease-duration-secs",
                "30",
                "--registration-storage",
                "memory"
            ]))
            .is_ok()
        );
        // Flags left at their defaults are not refused, whatever their heading.
        assert!(parse_args(karmada(&[])).is_ok());
    }

    #[test]
    fn service_suffixes_can_start_with_a_hyphen_but_not_be_empty() {
        let args = parse_args(karmada(&[
            "--pool-service-suffix=-pool-gateway",
            "--frontend-service-suffix",
            "-web",
        ]))
        .unwrap();
        assert_eq!(args.pool_service_suffix, "-pool-gateway");
        assert_eq!(args.frontend_service_suffix, "-web");
        // An empty suffix would make every imported Service a pool.
        for empty in ["--pool-service-suffix=", "--frontend-service-suffix="] {
            assert!(parse_args(karmada(&[empty])).is_err());
        }
        // A suffix left out fails, rather than taking the next flag.
        let error = parse_args(karmada(&[
            "--pool-service-suffix",
            "--listen=127.0.0.1:18496",
        ]))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidValue, "{error}");
    }

    #[test]
    fn router_flags_cannot_be_empty() {
        for flag in [
            "--router-namespace",
            "--router-service",
            "--router-port-name",
        ] {
            let mut args = karmada(&[]);
            let at = args.iter().position(|arg| *arg == flag).unwrap();
            args[at + 1] = "";
            let error = parse_args(args).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidValue, "{flag}: {error}");
        }
    }

    #[test]
    fn karmada_refuses_a_router_service_it_cannot_import() {
        let name = "r".repeat(KARMADA_MAX_SERVICE_NAME + 1);
        let mut args = karmada(&[]);
        let at = args
            .iter()
            .position(|arg| *arg == "--router-service")
            .unwrap();
        args[at + 1] = &name;
        let error = parse_args(args.clone()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation, "{error}");
        // Standard MCS imports it.
        let at = args.iter().position(|arg| *arg == "karmada").unwrap();
        args[at] = "mcs";
        assert!(parse_args(args).is_ok());
    }

    #[test]
    fn a_router_needs_an_http_url() {
        for url in ["http://10.42.0.5:50051", "https://global-router-0.svc:8443"] {
            let router = parse_router(&format!("r0={url}")).unwrap();
            assert_eq!(router.url.as_str(), format!("{url}/"));
        }
        // Without `http://`, a host name would be read as the scheme. An IP
        // address fails to parse at all.
        for url in [
            "localhost:50051",
            "global-router-0.svc:50051",
            "10.42.0.5:50051",
            "mailto:x@y",
        ] {
            assert!(parse_router(&format!("r0={url}")).is_err(), "{url}");
        }
    }

    #[test]
    fn a_bad_lease_or_a_repeated_router_is_a_usage_error() {
        for args in [
            // The default lease is 15 s.
            vec!["gds", "--heartbeat-interval-secs", "20"],
            vec!["gds", "--lease-duration-secs", "0"],
            vec!["gds", ROUTER, ROUTER],
        ] {
            let error = parse_args(args.clone()).unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::ValueValidation,
                "{args:?}: {error}"
            );
            assert_eq!(error.exit_code(), 2, "{args:?}");
        }
    }

    #[test]
    fn a_lease_is_at_most_a_day() {
        let day = MAX_LEASE_DURATION_SECS.to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &day]).is_ok());
        let longer = (MAX_LEASE_DURATION_SECS + 1).to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &longer]).is_err());
    }
}
