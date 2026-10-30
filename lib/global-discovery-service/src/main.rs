// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service. See `dynamo_global_discovery_service::service`.
//!
//! Two flags choose the backend behind the API:
//! - `--pool-discovery`: where the pool and router lists come from. Either
//!   the registrations PoolRelays send (`registrations`), or the
//!   EndpointSlices that MCS imports, named the Karmada way (`karmada`) or
//!   the standard way (`mcs`).
//! - `--registration-storage`: where registrations are kept. Either in this
//!   process (`memory`), so run one replica, or in a DynamoDB table that
//!   every replica shares (`dynamodb`), in one region or, through a Global
//!   Table, in several.
//!
//! ```bash
//! cargo run -p dynamo-global-discovery-service -- \
//!     --pool-discovery registrations --router aws-ohio/global-router-0=http://10.42.0.5:50051
//! ```
//!
//! With DynamoDB, here dynamodb-local, which takes any credentials:
//!
//! ```bash
//! AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local \
//! cargo run -p dynamo-global-discovery-service -- \
//!     --router aws-ohio/global-router-0=http://10.42.0.5:50051 \
//!     --registration-storage dynamodb --dynamodb-table global-discovery-pools \
//!     --dynamodb-region us-west-2 --dynamodb-endpoint http://localhost:8000 \
//!     --dynamodb-ensure-table
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
use dynamo_global_discovery_service::backends::dynamodb::{DynamoDbConfig, DynamoDbPoolStore};
use dynamo_global_discovery_service::backends::kubernetes::{
    ExportedService, McsImplementation, McsPoolDirectory, McsReplicaDirectory,
};
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
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
    /// Also require callers to present a certificate that this CA signed
    /// (mTLS). Probes need one too, so probe readiness with TCP or a command.
    #[arg(long, requires = "tls_cert_path")]
    tls_client_ca_cert_path: Option<PathBuf>,
    /// Where the pool and router lists come from.
    #[arg(long, value_enum, default_value_t = PoolDiscovery::Registrations)]
    pool_discovery: PoolDiscovery,
    /// Where registrations are kept. Only `dynamodb` lets several replicas
    /// serve the same pools.
    #[arg(
        long,
        value_enum,
        default_value_t = RegistrationStorage::Memory,
        help_heading = WITH_REGISTRATIONS
    )]
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
    /// The DynamoDB table that keeps registrations. Needed with
    /// `--registration-storage dynamodb`.
    #[arg(
        long,
        required_if_eq("registration_storage", "dynamodb"),
        help_heading = WITH_REGISTRATIONS
    )]
    dynamodb_table: Option<String>,
    /// The region of the DynamoDB table (replica) to use. By default the AWS
    /// SDK's: `AWS_REGION`, then the shared config.
    #[arg(long, help_heading = WITH_REGISTRATIONS)]
    dynamodb_region: Option<String>,
    /// Send DynamoDB requests here instead of to the region's endpoint, for
    /// example to dynamodb-local.
    #[arg(long, help_heading = WITH_REGISTRATIONS)]
    dynamodb_endpoint: Option<Url>,
    /// How often a watch reads the DynamoDB table again. Less than the
    /// heartbeat interval.
    #[arg(
        long,
        default_value_t = 2,
        value_parser = clap::value_parser!(u64).range(1..),
        help_heading = WITH_REGISTRATIONS
    )]
    dynamodb_poll_interval_secs: u64,
    /// Create the DynamoDB table if it does not exist. For development: it
    /// needs `dynamodb:CreateTable` and `dynamodb:UpdateTimeToLive`.
    #[arg(long, help_heading = WITH_REGISTRATIONS)]
    dynamodb_ensure_table: bool,
    /// How far the clocks of replicas sharing a DynamoDB table may disagree.
    /// A pool stays listed this much longer than its lease.
    #[arg(long, default_value_t = 2, help_heading = WITH_REGISTRATIONS)]
    lease_skew_secs: u64,
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
    #[arg(long, help_heading = WITH_MCS, required_if_eq_any = MCS_VALUES)]
    router_namespace: Option<String>,
    /// The Global Router Service, exported through MCS.
    #[arg(long, help_heading = WITH_MCS, required_if_eq_any = MCS_VALUES)]
    router_service: Option<String>,
    /// The router Service's named port for state streams.
    #[arg(long, help_heading = WITH_MCS, required_if_eq_any = MCS_VALUES)]
    router_port_name: Option<String>,
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
    /// In the DynamoDB table `--dynamodb-table`, which every replica shares.
    /// Lists are complete from the start.
    Dynamodb,
}

/// The flags only `--registration-storage dynamodb` reads.
const DYNAMODB_ONLY: [&str; 6] = [
    "dynamodb_table",
    "dynamodb_region",
    "dynamodb_endpoint",
    "dynamodb_poll_interval_secs",
    "dynamodb_ensure_table",
    "lease_skew_secs",
];

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

/// Parses the flags. A flag set under the `--help` heading of another
/// `--pool-discovery` value fails, as clap's own errors do, and so does a
/// DynamoDB flag without `--registration-storage dynamodb`. A flag left at
/// its default is not refused.
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
    let given = |refuse: &dyn Fn(&clap::Arg) -> bool| {
        command
            .get_arguments()
            .find(|arg| {
                refuse(arg)
                    && matches
                        .value_source(arg.get_id().as_str())
                        .is_some_and(|source| source != ValueSource::DefaultValue)
            })
            .map(|arg| arg.get_long().unwrap_or(arg.get_id().as_str()).to_owned())
    };
    let conflict = if let Some(flag) = given(&|arg| arg.get_help_heading() == Some(refused)) {
        Some(format!("--{flag} works only {}", refused.to_lowercase()))
    } else if parsed.registration_storage != RegistrationStorage::Dynamodb {
        given(&|arg| DYNAMODB_ONLY.contains(&arg.get_id().as_str()))
            .map(|flag| format!("--{flag} works only with --registration-storage dynamodb"))
    } else {
        None
    };
    match conflict {
        Some(message) => Err(command.error(ErrorKind::ArgumentConflict, message)),
        None => Ok(parsed),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = parse_args(std::env::args_os()).unwrap_or_else(|error| error.exit());
    let backend = match args.pool_discovery {
        PoolDiscovery::Registrations => registrations_backend(&args).await?,
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

/// Pools from registrations; router replicas from `--router`. With
/// DynamoDB, fails unless the table can be used.
async fn registrations_backend(args: &Args) -> Result<PoolDiscoveryBackend> {
    let lease = Lease::new(args.heartbeat_interval_secs, args.lease_duration_secs)
        .context("check --heartbeat-interval-secs and --lease-duration-secs")?;
    let mut ids = HashSet::new();
    if let Some(repeated) = args
        .routers
        .iter()
        .find(|router| !ids.insert(&router.replica))
    {
        bail!("--router {} is given twice", repeated.replica.0);
    }
    let (pools, registrar): (Arc<PoolDirectory>, Arc<dyn PoolRegistrar>) =
        match args.registration_storage {
            RegistrationStorage::Memory => {
                let store = Arc::new(MemoryPoolStore::new(lease));
                (store.clone(), store)
            }
            RegistrationStorage::Dynamodb => {
                let config = dynamodb_config(args)?;
                let table = config.table.clone();
                let store = DynamoDbPoolStore::connect(lease, config)
                    .await
                    .with_context(|| format!("use DynamoDB table {table} for registrations"))?;
                (store.clone(), store)
            }
        };
    Ok(PoolDiscoveryBackend {
        pools,
        routers: Arc::new(StaticReplicaDirectory::new(args.routers.clone())),
        registrar: Some(registrar),
    })
}

/// The DynamoDB flags as the store's configuration. A watch polls more often
/// than PoolRelays register, so it sees each change before the next
/// heartbeat.
fn dynamodb_config(args: &Args) -> Result<DynamoDbConfig> {
    let Some(table) = args.dynamodb_table.clone() else {
        bail!("--registration-storage dynamodb needs --dynamodb-table");
    };
    if args.dynamodb_poll_interval_secs >= args.heartbeat_interval_secs {
        bail!(
            "--dynamodb-poll-interval-secs {} must be less than --heartbeat-interval-secs {}",
            args.dynamodb_poll_interval_secs,
            args.heartbeat_interval_secs
        );
    }
    Ok(DynamoDbConfig {
        table,
        region: args.dynamodb_region.clone(),
        endpoint: args.dynamodb_endpoint.clone(),
        poll_interval: Duration::from_secs(args.dynamodb_poll_interval_secs),
        lease_skew: Duration::from_secs(args.lease_skew_secs),
        ensure_table: args.dynamodb_ensure_table,
    })
}

/// Pools from the PoolRelay Services that MCS imports, and router replicas
/// from the router Service's.
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
    let client = kubernetes_client()
        .await
        .context("create the Kubernetes client")?;
    let router = ExportedService {
        namespace: namespace.clone(),
        name: name.clone(),
        port_name: port_name.clone(),
    };
    Ok(PoolDiscoveryBackend {
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
            "tls_client_ca_cert_path",
            "pool_discovery",
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
    fn a_flag_for_another_pool_discovery_fails() {
        assert!(parse_args(["gds", ROUTER]).is_ok());
        assert_refused(&karmada(&[ROUTER]), "router", "registrations");
        assert_refused(
            &karmada(&["--lease-duration-secs", "30"]),
            "lease-duration-secs",
            "registrations",
        );
        assert_refused(
            &karmada(&["--registration-storage", "memory"]),
            "registration-storage",
            "registrations",
        );
        assert_refused(
            &["gds", "--router-service", "global-router"],
            "router-service",
            "karmada or mcs",
        );
        assert_refused(
            &["gds", "--pool-service-suffix=-relay"],
            "pool-service-suffix",
            "karmada or mcs",
        );
        assert_refused(
            &karmada(&["--dynamodb-table", "pools"]),
            "dynamodb-table",
            "registrations",
        );
        // Flags left at their defaults are not refused, whatever their heading.
        assert!(parse_args(karmada(&[])).is_ok());
    }

    /// `--registration-storage dynamodb --dynamodb-table pools`, then `extra`.
    fn dynamodb<'a>(extra: &[&'a str]) -> Vec<&'a str> {
        let mut args = vec![
            "gds",
            "--registration-storage",
            "dynamodb",
            "--dynamodb-table",
            "pools",
        ];
        args.extend_from_slice(extra);
        args
    }

    #[test]
    fn dynamodb_needs_a_table_and_has_defaults() {
        let error = parse_args(["gds", "--registration-storage", "dynamodb"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument, "{error}");
        assert!(error.to_string().contains("--dynamodb-table"), "{error}");

        let args = parse_args(dynamodb(&[])).unwrap();
        assert_eq!(args.registration_storage, RegistrationStorage::Dynamodb);
        assert_eq!(
            dynamodb_config(&args).unwrap(),
            DynamoDbConfig {
                table: "pools".to_owned(),
                region: None,
                endpoint: None,
                poll_interval: Duration::from_secs(2),
                lease_skew: Duration::from_secs(2),
                ensure_table: false,
            }
        );
        assert_eq!(
            dynamodb_config(&args).unwrap(),
            DynamoDbConfig::new("pools")
        );
    }

    #[test]
    fn dynamodb_flags_set_the_store() {
        let args = parse_args(dynamodb(&[
            "--dynamodb-region",
            "us-east-2",
            "--dynamodb-endpoint",
            "http://localhost:8000",
            "--dynamodb-poll-interval-secs",
            "1",
            "--dynamodb-ensure-table",
            "--lease-skew-secs",
            "0",
        ]))
        .unwrap();
        assert_eq!(
            dynamodb_config(&args).unwrap(),
            DynamoDbConfig {
                table: "pools".to_owned(),
                region: Some("us-east-2".to_owned()),
                endpoint: Some(Url::parse("http://localhost:8000").unwrap()),
                poll_interval: Duration::from_secs(1),
                lease_skew: Duration::ZERO,
                ensure_table: true,
            }
        );
        assert!(parse_args(dynamodb(&["--dynamodb-endpoint", "localhost"])).is_err());
        assert!(parse_args(dynamodb(&["--dynamodb-poll-interval-secs", "0"])).is_err());
    }

    #[test]
    fn a_dynamodb_flag_needs_dynamodb() {
        for flag in [
            &["--dynamodb-table", "pools"][..],
            &["--dynamodb-region", "us-east-2"],
            &["--dynamodb-endpoint", "http://localhost:8000"],
            &["--dynamodb-poll-interval-secs", "1"],
            &["--dynamodb-ensure-table"],
            &["--lease-skew-secs", "3"],
        ] {
            let mut args = vec!["gds"];
            args.extend_from_slice(flag);
            let error = parse_args(args).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ArgumentConflict, "{error}");
            let reason = format!(
                "{} works only with --registration-storage dynamodb",
                flag[0]
            );
            assert!(error.to_string().contains(&reason), "{error}");
        }
        // Every DynamoDB flag is listed, and only those.
        for arg in Args::command().get_arguments() {
            let id = arg.get_id().as_str();
            let listed = DYNAMODB_ONLY.contains(&id);
            assert_eq!(
                listed,
                id.starts_with("dynamodb_") || id == "lease_skew_secs",
                "{id}"
            );
        }
    }

    #[test]
    fn a_dynamodb_watch_polls_faster_than_relays_register() {
        let config = |poll: &str| {
            dynamodb_config(
                &parse_args(dynamodb(&["--dynamodb-poll-interval-secs", poll])).unwrap(),
            )
        };
        assert!(config("4").is_ok());
        let error = config("5").unwrap_err().to_string();
        assert!(
            error.contains("must be less than --heartbeat-interval-secs 5"),
            "{error}"
        );
    }

    #[test]
    fn pool_service_suffix_can_start_with_a_hyphen_but_not_be_empty() {
        let args = parse_args(karmada(&["--pool-service-suffix", "-pool-gateway"])).unwrap();
        assert_eq!(args.pool_service_suffix, "-pool-gateway");
        // An empty suffix would make every imported Service a pool.
        assert!(parse_args(karmada(&["--pool-service-suffix="])).is_err());
    }

    #[test]
    fn a_lease_is_at_most_a_day() {
        let day = MAX_LEASE_DURATION_SECS.to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &day]).is_ok());
        let longer = (MAX_LEASE_DURATION_SECS + 1).to_string();
        assert!(parse_args(["gds", "--lease-duration-secs", &longer]).is_err());
    }
}
