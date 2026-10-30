// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_frontend` and the crate's README.
//!
//! `--selection core` selects through the embedded router core instead of
//! the Global Router's own selectors, with each pool as one of the core's
//! workers and every `--router-policy` as one of its plugins. With
//! `--replica-sync grpc`, the core shares its requests in flight and
//! session pins with the other Global Frontends over the gRPC `ReplicaSync`
//! stream, served on the API port itself from startup, before any peer is
//! known. Peers are the other frontends' API addresses, listed from Global
//! Discovery (`--global-discovery`, with `--replica-url` naming this
//! frontend's own entry) or given statically (`--peer`); either implies
//! `--replica-sync grpc`:
//!
//! ```bash
//! cargo run -p dynamo-global-frontend -- --port 8000 --selection core \
//!     --router-policy affinity --region ohio \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000 \
//!     --pool-relay ohio/dynamo/mocker-1=http://10.42.0.6:50051 \
//!     --global-discovery http://discovery.dynamo:8080 \
//!     --replica-url http://10.42.0.7:8000
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_frontend::{
    Metrics, PoolSnapshot, app, follow_relays, measure_round_trips, poll_replica_peers, serve,
};
use dynamo_global_router::{
    AffinityConfig, AffinitySelector, CorePolicy, CoreSelector, GrpcReplicaSync, InFlightTracker,
    LeastInFlightSelector, Pool, PoolSignals, ReplicaTransport, RequestSelector,
    RoundRobinSelector,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

/// How often the round trip to each pool is measured.
const ROUND_TRIP_INTERVAL: Duration = Duration::from_secs(30);
/// How often expired session pins are dropped.
const PIN_SWEEP_INTERVAL: Duration = Duration::from_secs(60);
/// How often the replicas are listed from Global Discovery: the cadence of
/// the frontend's other periodic reads of the fleet (`ROUND_TRIP_INTERVAL`).
const DISCOVERY_POLL_INTERVAL: Duration = ROUND_TRIP_INTERVAL;
const DISCOVERY_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// How each request picks a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RouterPolicy {
    /// Every pool serving the model in turn.
    RoundRobin,
    /// The pool with the fewest of this process's requests in flight.
    LeastInFlight,
    /// Keep each session on one pool, nearest region first, and move it only
    /// for a pool with much more room (see `--headroom-factor`). Reads each
    /// pool's load from its Relay (`--pool-relay`).
    Affinity,
}

/// What implements the router policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SelectionPath {
    /// The Global Router's own selectors, as today.
    Current,
    /// The embedded router core, with each pool as one of its workers.
    Core,
}

/// How the core shares its requests in flight and session pins with the
/// other Global Frontends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ReplicaSyncMode {
    /// Not at all: this frontend sees only its own requests.
    Off,
    /// The gRPC `ReplicaSync` stream, served on the API port and dialed at
    /// each peer's API address.
    Grpc,
}

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// A pool as `<cluster>/<namespace>/<DGD name>=<model>@<frontend URL>`,
    /// where the cluster is the ID the pool's PoolRelay runs with. Separate
    /// several frontend URLs with commas, and repeat the flag for each pool.
    /// The cluster also names the pool's region.
    #[arg(long = "pool", value_parser = parse_pool, required = true)]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
    #[arg(long, value_enum, default_value_t = RouterPolicy::RoundRobin)]
    router_policy: RouterPolicy,
    #[arg(long, value_enum, default_value_t = SelectionPath::Current)]
    selection: SelectionPath,
    /// The region this process runs in, named as `--pool` names clusters.
    /// `affinity` tries pools in it first, and metrics tell requests kept in
    /// it from the rest.
    #[arg(long, required_if_eq("router_policy", "affinity"))]
    region: Option<String>,
    /// A pool's PoolRelay, which streams the pool's load, as
    /// `<cluster>/<namespace>/<DGD name>=<URL>`. Repeat the flag for each pool;
    /// each pool needs a Relay of its own, since a Relay reports the load of
    /// everything it watches as one.
    #[arg(long = "pool-relay", value_parser = parse_pool_relay)]
    pool_relays: Vec<(PoolKey, Url)>,
    /// `affinity`: a request leaves its pool only for one with more than this
    /// many times its spare capacity, or when its pool is full.
    #[arg(long, default_value_t = 2.0)]
    headroom_factor: f64,
    /// `affinity`: how long a session keeps its pool without a request.
    /// The core keeps at most 65,536 sessions per model; while the table is
    /// full, a new session is placed without a pin until idle ones expire.
    #[arg(long, default_value_t = 1800)]
    session_idle_secs: u64,
    /// `--selection core`: share this replica's requests in flight and
    /// session pins with the other Global Frontends. `grpc` serves the
    /// `ReplicaSync` stream on the API port from startup, with or without
    /// peers. `--peer` and `--global-discovery` imply `grpc`; `off` given
    /// with either is an error.
    #[arg(long, value_enum)]
    replica_sync: Option<ReplicaSyncMode>,
    /// Another Global Frontend to share requests in flight and session pins
    /// with, as its API address `http://host:port`. Repeat the flag for
    /// each peer. Static peers stay whatever Global Discovery lists.
    #[arg(long = "peer", value_parser = parse_peer)]
    peers: Vec<Url>,
    /// The Global Discovery service's base URL. Its `/routers` lists every
    /// Global Frontend's API URL; every entry but `--replica-url` becomes a
    /// peer, and the list is followed as it changes.
    #[arg(long, value_parser = parse_discovery_url, requires = "replica_url")]
    global_discovery: Option<Url>,
    /// This frontend's own API URL as Global Discovery lists it, so it does
    /// not dial itself.
    #[arg(long, value_parser = parse_peer, requires = "global_discovery")]
    replica_url: Option<Url>,
}

fn parse_key(key: &str) -> Result<PoolKey, String> {
    let [cluster, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(format!(
            "expected <cluster>/<namespace>/<DGD name>, got {key}"
        ));
    };
    PoolKey::new(cluster, namespace, dgd).map_err(|error| error.to_string())
}

fn parse_url(url: &str, what: &str) -> Result<Url, String> {
    let parsed = url
        .parse::<Url>()
        .map_err(|error| format!("{url}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{what} must be an HTTP(S) URL: {url}"));
    }
    Ok(parsed)
}

fn parse_pool_relay(value: &str) -> Result<(PoolKey, Url), String> {
    let (key, relay) = value
        .split_once('=')
        .ok_or_else(|| format!("expected <cluster>/<namespace>/<DGD name>=<URL>, got {value}"))?;
    let relay = parse_url(relay, "pool relay")?;
    // The Relay client is built without TLS.
    if relay.scheme() != "http" {
        return Err(format!("pool relay must be an http:// URL: {relay}"));
    }
    Ok((parse_key(key)?, relay))
}

fn parse_peer(value: &str) -> Result<Url, String> {
    dynamo_global_router::replica_sync::peer_url(value).map_err(|error| error.to_string())
}

fn parse_discovery_url(value: &str) -> Result<Url, String> {
    parse_url(value, "global discovery")
}

/// Gives each pool its Relay. Every Relay must name a configured pool, and
/// no two pools may share one.
fn attach_relays(pools: &mut [Pool], relays: &[(PoolKey, Url)]) -> Result<()> {
    for (key, relay) in relays {
        if pools.iter().any(|pool| pool.relay.as_ref() == Some(relay)) {
            bail!("--pool-relay {relay} is given for two pools; each pool needs its own Relay");
        }
        let id = V1PoolIdDeriver.derive(key);
        let Some(pool) = pools.iter_mut().find(|pool| pool.id == id) else {
            bail!("--pool-relay {relay} names a pool no --pool gives");
        };
        if pool.relay.replace(relay.clone()).is_some() {
            bail!("--pool-relay is given twice for pool {id}");
        }
    }
    Ok(())
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape = || {
        format!("expected <cluster>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}")
    };
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    if model.trim().is_empty() {
        return Err("pool model must not be empty".to_owned());
    }
    let key = parse_key(key)?;
    let frontends = frontends
        .split(',')
        .map(|url| parse_url(url, "pool frontend"))
        .collect::<Result<Vec<_>, _>>()?;
    // Until pools come from Global Discovery, which reports each pool's
    // region, the cluster names it.
    Ok(Pool::new(&key, model, key.cluster_id(), frontends))
}

fn replica_transport(args: &Args) -> Result<Option<ReplicaTransport>> {
    let implied_by = match (args.peers.is_empty(), args.global_discovery.is_some()) {
        (false, _) => Some("--peer"),
        (true, true) => Some("--global-discovery"),
        (true, false) => None,
    };
    match (args.replica_sync, implied_by) {
        (Some(ReplicaSyncMode::Off), Some(flag)) => {
            bail!("--replica-sync off conflicts with {flag}, which needs the gRPC transport")
        }
        (Some(ReplicaSyncMode::Off), None) | (None, None) => return Ok(None),
        (Some(ReplicaSyncMode::Grpc), _) | (None, Some(_)) => {}
    }
    if args.selection != SelectionPath::Core {
        let flag = implied_by.unwrap_or("--replica-sync grpc");
        bail!("{flag} needs --selection core: only the core replicates its bookings");
    }
    Ok(Some(ReplicaTransport::Grpc(GrpcReplicaSync {
        peers: args.peers.clone(),
    })))
}

/// Starts following each pool's Relay and measuring each region's round
/// trip, and returns the signals they fill, for the affinity policy on
/// either path.
fn load_signals(
    args: &Args,
    pools: &watch::Receiver<Arc<PoolSnapshot>>,
    in_flight: Arc<InFlightTracker>,
    shutdown: &CancellationToken,
) -> Arc<PoolSignals> {
    if args.pool_relays.is_empty() {
        tracing::warn!(
            "no --pool-relay is given, so affinity knows no pool's load \
             and keeps each session on its nearest pool"
        );
    }
    let signals = Arc::new(PoolSignals::new(in_flight));
    tokio::spawn(follow_relays(
        pools.clone(),
        signals.clone(),
        shutdown.child_token(),
    ));
    tokio::spawn(measure_round_trips(
        pools.clone(),
        signals.clone(),
        ROUND_TRIP_INTERVAL,
        shutdown.child_token(),
    ));
    signals
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let mut args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }
    attach_relays(&mut args.pools, &args.pool_relays)?;
    if !(args.headroom_factor >= 1.0 && args.headroom_factor.is_finite()) {
        bail!("--headroom-factor must be at least 1");
    }
    if args.router_policy != RouterPolicy::Affinity && !args.pool_relays.is_empty() {
        tracing::warn!("--pool-relay is used only by --router-policy affinity");
    }
    let replica_transport = replica_transport(&args)?;

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });

    // The pools never change; the sender must live as long as the process,
    // since the Relay followers stop once it drops.
    let (_catalog, pools) =
        watch::channel(Arc::new(PoolSnapshot::new(std::mem::take(&mut args.pools))));
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let in_flight = Arc::new(InFlightTracker::new());
    let metrics = Metrics::new(args.region.clone(), in_flight.clone());
    let idle = Duration::from_secs(args.session_idle_secs);
    let mut replica_sync_routes = None;
    let selector: Arc<dyn RequestSelector> = match (args.selection, args.router_policy) {
        (SelectionPath::Core, policy) => {
            let policy = match policy {
                RouterPolicy::RoundRobin => CorePolicy::RoundRobin,
                RouterPolicy::LeastInFlight => CorePolicy::LeastRequests,
                RouterPolicy::Affinity => CorePolicy::Affinity(AffinityConfig {
                    signals: load_signals(&args, &pools, in_flight.clone(), &shutdown),
                    region: args.region.clone().context("--region is required")?,
                    headroom_factor: args.headroom_factor,
                    idle,
                }),
            };
            let core = Arc::new(CoreSelector::with_transport(policy, replica_transport).await?);
            // Serve only once the configured pools are the core's workers;
            // a request before that would get a spurious 503.
            let (ready, follower) = core.follow(pools.clone(), shutdown.child_token());
            tokio::spawn(follower);
            ready
                .await
                .context("the router core stopped before taking the first pool snapshot")?;
            replica_sync_routes = core.replica_sync_routes();
            // The replica streams are long-lived; ending them as shutdown
            // begins lets the graceful drain finish with the requests.
            let replication = core.clone();
            let stopping = shutdown.clone();
            tokio::spawn(async move {
                stopping.cancelled().await;
                replication.stop_replication();
            });
            if let Some(discovery) = args.global_discovery.clone() {
                let client = GlobalDiscoveryClient::new(discovery, DISCOVERY_REQUEST_TIMEOUT)
                    .context("build the Global Discovery client")?;
                let (listed, peers) = watch::channel(Vec::new());
                tokio::spawn(poll_replica_peers(
                    client,
                    args.replica_url.clone(),
                    DISCOVERY_POLL_INTERVAL,
                    listed,
                    shutdown.child_token(),
                ));
                let follow = core
                    .follow_peers(peers, shutdown.child_token())
                    .context("--global-discovery needs the gRPC replica transport")?;
                tokio::spawn(follow);
            }
            core
        }
        (SelectionPath::Current, RouterPolicy::RoundRobin) => Arc::new(RoundRobinSelector::new()),
        (SelectionPath::Current, RouterPolicy::LeastInFlight) => {
            Arc::new(LeastInFlightSelector::new(in_flight.clone()))
        }
        (SelectionPath::Current, RouterPolicy::Affinity) => {
            let region = args.region.clone().context("--region is required")?;
            let signals = load_signals(&args, &pools, in_flight.clone(), &shutdown);
            let selector = Arc::new(AffinitySelector::new(
                signals,
                region,
                args.headroom_factor,
                idle,
            ));
            tokio::spawn(sweep_pins(selector.clone(), shutdown.child_token()));
            selector
        }
    };
    let mut app = app(pools, selector, in_flight, http, metrics);
    if let Some(routes) = replica_sync_routes {
        // The gRPC stream shares the listener: its paths are the service's,
        // beside `/v1/...`, and the listener speaks HTTP/1 and h2c alike.
        app = app.merge(routes);
    }
    let address = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .with_context(|| format!("listen on {address}"))?;
    tracing::info!(%address, "Global Frontend listening");
    serve(listener, app, shutdown)
        .await
        .context("serve the Global Frontend")
}

/// Drops expired session pins once a `PIN_SWEEP_INTERVAL`, off the request
/// path, until `cancel` fires.
async fn sweep_pins(selector: Arc<AffinitySelector>, cancel: CancellationToken) {
    let mut ticks = tokio::time::interval(PIN_SWEEP_INTERVAL);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => selector.sweep(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_paths_and_multiple_frontends() {
        let pool =
            parse_pool("ohio/dynamo/blue=Qwen/Qwen3@http://blue:8000,https://green:8000").unwrap();
        assert_eq!(pool.model, "Qwen/Qwen3");
        assert_eq!(pool.frontends.len(), 2);
    }

    #[test]
    fn rejects_missing_models_and_non_http_frontends() {
        for value in [
            "ohio/ns/pool=@http://pool:8000",
            "ohio/ns/pool=m@file:///tmp/pool",
            "ohio/ns/pool=m@mailto:user@example.com",
        ] {
            assert!(parse_pool(value).is_err(), "{value}");
        }
    }

    #[test]
    fn parses_pool_relays_and_requires_a_region_for_affinity() {
        let (key, relay) = parse_pool_relay("ohio/dynamo/blue=http://relay:50051").unwrap();
        assert_eq!(key.dgd_name(), "blue");
        assert_eq!(relay.as_str(), "http://relay:50051/");
        assert!(parse_pool_relay("ohio/dynamo=http://relay:50051").is_err());
        assert!(parse_pool_relay("ohio/dynamo/blue=https://relay:50051").is_err());

        // Each pool needs its own Relay.
        let mut pools = ["blue", "green"]
            .map(|dgd| parse_pool(&format!("ohio/dynamo/{dgd}=qwen@http://{dgd}:8000")).unwrap());
        let relays = ["blue", "green"]
            .map(|dgd| parse_pool_relay(&format!("ohio/dynamo/{dgd}=http://relay:50051")).unwrap());
        assert!(attach_relays(&mut pools, &relays).is_err());
        assert!(attach_relays(&mut pools[1..], &relays[1..]).is_ok());

        let pool = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";
        let affinity = ["global-frontend", pool, "--router-policy", "affinity"];
        assert!(Args::try_parse_from(affinity).is_err());
        let args = Args::try_parse_from([&affinity[..], &["--region", "ohio"]].concat()).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::Affinity);
        let args = Args::try_parse_from(["global-frontend", pool]).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::RoundRobin);
    }

    const POOL: &str = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";
    const CORE: [&str; 2] = ["--selection", "core"];
    const DISCOVERY: [&str; 4] = [
        "--global-discovery",
        "http://discovery:8080",
        "--replica-url",
        "http://gf-a:8000",
    ];

    fn transport(rest: &[&str]) -> Result<Option<ReplicaTransport>> {
        let args = Args::try_parse_from([&["global-frontend", POOL], rest].concat())?;
        replica_transport(&args)
    }

    fn grpc(peers: &[&str]) -> Option<ReplicaTransport> {
        Some(ReplicaTransport::Grpc(GrpcReplicaSync {
            peers: peers.iter().map(|peer| peer.parse().unwrap()).collect(),
        }))
    }

    #[test]
    fn replica_sync_is_off_unless_asked_for() {
        let args = Args::try_parse_from(["global-frontend", POOL]).unwrap();
        assert_eq!(args.selection, SelectionPath::Current);
        assert_eq!(args.replica_sync, None);
        assert_eq!(replica_transport(&args).unwrap(), None);
        assert_eq!(transport(&CORE).unwrap(), None);
        assert_eq!(
            transport(&[&CORE[..], &["--replica-sync", "off"]].concat()).unwrap(),
            None
        );
    }

    #[test]
    fn grpc_serves_with_or_without_peers() {
        // The stream is served from startup, so a replica can be dialed
        // before it knows anyone.
        let grpc_flag = ["--replica-sync", "grpc"];
        assert_eq!(
            transport(&[&CORE[..], &grpc_flag[..]].concat()).unwrap(),
            grpc(&[])
        );
        let peers = [
            "--peer",
            "http://10.0.0.2:8000",
            "--peer",
            "http://gf-b:8000",
        ];
        let expected = grpc(&["http://10.0.0.2:8000", "http://gf-b:8000"]);
        assert_eq!(
            transport(&[&CORE[..], &grpc_flag[..], &peers[..]].concat()).unwrap(),
            expected
        );
        assert_eq!(
            transport(&[&CORE[..], &peers[..]].concat()).unwrap(),
            expected
        );
    }

    #[test]
    fn discovery_implies_grpc_and_names_this_replica() {
        let args =
            Args::try_parse_from([&["global-frontend", POOL], &CORE[..], &DISCOVERY[..]].concat())
                .unwrap();
        assert_eq!(
            args.global_discovery.as_ref().map(Url::as_str),
            Some("http://discovery:8080/")
        );
        assert_eq!(
            args.replica_url.as_ref().map(Url::as_str),
            Some("http://gf-a:8000/")
        );
        assert_eq!(replica_transport(&args).unwrap(), grpc(&[]));
        // Static peers ride along; discovery never removes them.
        let with_peer = [&CORE[..], &DISCOVERY[..], &["--peer", "http://gf-c:8000"]].concat();
        assert_eq!(transport(&with_peer).unwrap(), grpc(&["http://gf-c:8000"]));
        assert!(transport(&[&CORE[..], &DISCOVERY[..2]].concat()).is_err());
        assert!(transport(&[&CORE[..], &DISCOVERY[2..]].concat()).is_err());
    }

    #[test]
    fn off_conflicts_with_peers_and_discovery() {
        let off = ["--replica-sync", "off"];
        assert!(
            transport(&[&CORE[..], &off[..], &["--peer", "http://gf-b:8000"]].concat()).is_err()
        );
        assert!(transport(&[&CORE[..], &off[..], &DISCOVERY[..]].concat()).is_err());
    }

    #[test]
    fn replica_sync_needs_the_core() {
        assert!(transport(&["--replica-sync", "grpc"]).is_err());
        assert!(transport(&["--peer", "http://10.0.0.2:8000"]).is_err());
        assert!(transport(&DISCOVERY).is_err());
    }

    #[test]
    fn peers_are_plain_http_api_addresses() {
        assert_eq!(
            parse_peer("http://peer:8000"),
            Ok("http://peer:8000".parse().unwrap())
        );
        // TLS peers wait for the transport to carry TLS.
        assert!(parse_peer("https://peer:8000").is_err());
        assert!(parse_peer("tcp://peer:5600").is_err());
        assert!(parse_peer("http://").is_err());
        assert!(parse_peer("peer:8000").is_err());
        // Discovery itself is reached over HTTP(S).
        assert!(parse_discovery_url("https://discovery:8080").is_ok());
        assert!(parse_discovery_url("tcp://discovery:8080").is_err());
    }

    #[test]
    fn requires_a_pool() {
        assert!(Args::try_parse_from(["global-frontend"]).is_err());
    }
}
