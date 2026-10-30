// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! Two flags choose the backend behind the API:
//! - `--pool-discovery`: where the pool and router lists come from. Either
//!   the registrations PoolRelays send (`registrations`), or the
//!   EndpointSlices that MCS imports, named the Karmada way (`karmada`) or
//!   the standard way (`mcs`), joined with registrations.
//! - `--registration-storage`: where registrations are kept: in this process
//!   (`memory`), or in a DynamoDB table that every instance reads
//!   (`dynamodb`, in builds with the `dynamodb` feature).
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
#[cfg(feature = "dynamodb")]
use dynamo_global_discovery_service::backends::dynamodb::{DynamoDbPoolStore, TableLocation};
use dynamo_global_discovery_service::backends::kubernetes::{
    ExportedService, FrontendService, KARMADA_MAX_SERVICE_NAME, McsImplementation,
    McsPoolDirectory, McsReplicaDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::backends::registered::RegisteredPoolDirectory;
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

/// The `--help` heading for the flags of `--registration-storage dynamodb`.
/// Such a flag fails at startup with another storage.
const WITH_DYNAMODB: &str = "With --registration-storage dynamodb";

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
    /// Each pool's frontend Service is named `<DGD name><suffix>`. Give a
    /// suffix that starts with `-` after an `=`: `--frontend-service-suffix=-web`.
    #[arg(
        long,
        default_value = "-frontend",
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS
    )]
    frontend_service_suffix: String,
    /// The frontend Service's named port for HTTP.
    #[arg(
        long,
        default_value = "http",
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_MCS
    )]
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
    /// The DynamoDB table that keeps registrations.
    #[arg(
        long,
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_DYNAMODB,
        required_if_eq("registration_storage", "dynamodb")
    )]
    dynamodb_table: Option<String>,
    /// The table's AWS region. Without it, the region comes from the AWS
    /// environment, such as `AWS_REGION`.
    #[arg(
        long,
        value_parser = NonEmptyStringValueParser::new(),
        help_heading = WITH_DYNAMODB
    )]
    dynamodb_region: Option<String>,
    /// Another DynamoDB endpoint than AWS's, such as DynamoDB Local's.
    #[arg(long, value_parser = parse_http_url, help_heading = WITH_DYNAMODB)]
    dynamodb_endpoint_url: Option<Url>,
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
    /// In this process. Run one replica: each keeps only the registrations it
    /// receives.
    Memory,
    /// In a DynamoDB table that every instance reads, such as a Global Table
    /// with a replica in each region, so instances can run in every cluster.
    /// Needs a build with the `dynamodb` feature, as the image has.
    Dynamodb,
}

fn parse_router(value: &str) -> Result<RouterReplica, String> {
    let (id, url) = value
        .split_once('=')
        .ok_or("expected <id>=<url>".to_owned())?;
    if id.is_empty() {
        return Err("the replica id is empty".to_owned());
    }
    Ok(RouterReplica {
        replica: RouterReplicaId(id.to_owned()),
        url: parse_http_url(url)?,
    })
}

/// An http or https URL. Without `http://`, `Url` reads a host such as
/// `localhost:50051` as the scheme.
fn parse_http_url(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|error| error.to_string())?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "expected an http or https URL, such as http://10.42.0.5:50051, not {url}"
        ));
    }
    Ok(url)
}

/// Parses the flags. A flag set under the `--help` heading of another
/// `--pool-discovery` value, or of another `--registration-storage`, fails,
/// as clap's own errors do. A flag left at
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
    let mut refused = vec![match parsed.pool_discovery {
        PoolDiscovery::Registrations => WITH_MCS,
        PoolDiscovery::Karmada | PoolDiscovery::Mcs => WITH_REGISTRATIONS,
    }];
    if parsed.registration_storage != RegistrationStorage::Dynamodb {
        refused.push(WITH_DYNAMODB);
    }
    let given = command.get_arguments().find_map(|arg| {
        let heading = refused
            .iter()
            .copied()
            .find(|heading| arg.get_help_heading() == Some(*heading))?;
        matches
            .value_source(arg.get_id().as_str())
            .is_some_and(|source| source != ValueSource::DefaultValue)
            .then(|| {
                let flag = arg.get_long().unwrap_or(arg.get_id().as_str()).to_owned();
                (flag, heading)
            })
    });
    if let Some((flag, heading)) = given {
        return Err(command.error(
            ErrorKind::ArgumentConflict,
            format!("--{flag} works only {}", heading.to_lowercase()),
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
    let backend = match args.pool_discovery.mcs() {
        None => registrations_backend(&args).await?,
        Some(mcs) => mcs_backend(&args, mcs).await?,
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
        .map(|(cert, key)| Tls { cert, key });
    service::serve(listener, backend, tls, shutdown)
        .await
        .context("serve the Global Discovery Service")
}

fn lease(args: &Args) -> Result<Lease> {
    Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")
}

/// Where registrations go: one store takes them and lists them.
struct RegistrationStore {
    pools: Arc<PoolDirectory>,
    registrar: Arc<dyn PoolRegistrar>,
}

/// The store `--registration-storage` names.
async fn registration_storage(args: &Args, lease: Lease) -> Result<RegistrationStore> {
    match args.registration_storage {
        RegistrationStorage::Memory => {
            let store = Arc::new(MemoryPoolStore::new(lease));
            Ok(RegistrationStore {
                pools: store.clone(),
                registrar: store,
            })
        }
        RegistrationStorage::Dynamodb => dynamodb_storage(args, lease).await,
    }
}

#[cfg(feature = "dynamodb")]
async fn dynamodb_storage(args: &Args, lease: Lease) -> Result<RegistrationStore> {
    let Some(name) = &args.dynamodb_table else {
        bail!("--registration-storage dynamodb needs --dynamodb-table");
    };
    let table = TableLocation {
        name: name.clone(),
        region: args.dynamodb_region.clone(),
        endpoint_url: args.dynamodb_endpoint_url.clone(),
    };
    let store = Arc::new(
        DynamoDbPoolStore::connect(&table, lease)
            .await
            .context("connect to DynamoDB")?,
    );
    Ok(RegistrationStore {
        pools: store.clone(),
        registrar: store,
    })
}

#[cfg(not(feature = "dynamodb"))]
async fn dynamodb_storage(_args: &Args, _lease: Lease) -> Result<RegistrationStore> {
    bail!(
        "this build has no DynamoDB support: build it with `--features dynamodb`, as the image is"
    )
}

/// Pools from registrations; router replicas from `--router`.
async fn registrations_backend(args: &Args) -> Result<PoolDiscoveryBackend> {
    let registrations = registration_storage(args, lease(args)?).await?;
    if args.routers.is_empty() {
        tracing::warn!(
            "no --router given, so GET /routers lists no Global Router replica for PoolRelays to \
             connect to"
        );
    }
    Ok(PoolDiscoveryBackend {
        pools: registrations.pools,
        routers: Arc::new(StaticReplicaDirectory::new(args.routers.clone())),
        registrar: Some(registrations.registrar),
    })
}

/// Pools from what MCS imports, joined with registrations: a pool is listed
/// while MCS imports a Ready PoolRelay endpoint for it and its PoolRelay has
/// registered within the lease. MCS gives the addresses; the registration
/// gives the model, capabilities, capacity and location. Router replicas
/// come from the router Service's imports.
async fn mcs_backend(args: &Args, mcs: McsImplementation) -> Result<PoolDiscoveryBackend> {
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
    let registrations = registration_storage(args, lease).await?;
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
    let pools = RegisteredPoolDirectory::new(Arc::new(imported), registrations.pools, lease);
    let router = ExportedService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    Ok(PoolDiscoveryBackend {
        pools: Arc::new(pools),
        routers: Arc::new(McsReplicaDirectory::new(client, &router, mcs)),
        registrar: Some(registrations.registrar),
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
            let headed = matches!(
                arg.get_help_heading(),
                Some(WITH_REGISTRATIONS | WITH_MCS | WITH_DYNAMODB)
            );
            assert_ne!(
                EVERY_VALUE.contains(&id),
                headed,
                "--{id}: list it in EVERY_VALUE or give it a heading"
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
            "--frontend-service-suffix=-web",
        ]))
        .unwrap();
        assert_eq!(args.pool_service_suffix, "-pool-gateway");
        assert_eq!(args.frontend_service_suffix, "-web");
        // An empty suffix would make every imported Service a pool.
        for empty in ["--pool-service-suffix=", "--frontend-service-suffix="] {
            assert!(parse_args(karmada(&[empty])).is_err());
        }
        // A suffix left out fails, rather than taking the next flag.
        for flag in ["--pool-service-suffix", "--frontend-service-suffix"] {
            let error = parse_args(karmada(&[flag, "--listen=127.0.0.1:18496"])).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidValue, "{flag}: {error}");
        }
    }

    #[test]
    fn frontend_port_name_cannot_be_empty() {
        let error = parse_args(karmada(&["--frontend-port-name="])).unwrap_err();
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
    fn dynamodb_flags_need_dynamodb_storage() {
        let error = parse_args(["gds", "--registration-storage", "dynamodb"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{error}");
        assert!(error.to_string().contains("--dynamodb-table"), "{error}");
        let dynamodb = [
            "--registration-storage=dynamodb",
            "--dynamodb-table=registrations",
            "--dynamodb-region=us-west-2",
            "--dynamodb-endpoint-url=http://localhost:8000",
        ];
        let args = parse_args(["gds"].into_iter().chain(dynamodb)).unwrap();
        assert_eq!(args.registration_storage, RegistrationStorage::Dynamodb);
        assert_eq!(args.dynamodb_table.as_deref(), Some("registrations"));
        // Every --pool-discovery value takes it.
        assert!(parse_args(karmada(&dynamodb)).is_ok());
        for flag in &dynamodb[1..] {
            let error = parse_args(["gds", flag]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ArgumentConflict, "{flag}: {error}");
            assert!(
                error
                    .to_string()
                    .contains("works only with --registration-storage dynamodb"),
                "{error}"
            );
        }
        for wrong in [
            "--dynamodb-table=",
            "--dynamodb-endpoint-url=localhost:8000",
        ] {
            let args = [
                "gds",
                "--registration-storage=dynamodb",
                "--dynamodb-table=t",
                wrong,
            ];
            assert!(parse_args(args).is_err(), "{wrong}");
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
