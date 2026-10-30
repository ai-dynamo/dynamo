// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_frontend` and the crate's README.
//!
//! `--selection core` selects through the embedded router core instead of
//! the Global Router's own selectors, with each pool as one of the core's
//! workers and every `--router-policy` as one of its plugins. With `--peer`,
//! the core shares its requests in flight and session pins with the other
//! Global Frontends, over one of two transports. The gRPC `ReplicaSync`
//! stream is served on the API port itself, and peers are the other
//! frontends' API addresses:
//!
//! ```bash
//! cargo run -p dynamo-global-frontend -- --port 8000 --selection core \
//!     --router-policy affinity --region ohio \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000 \
//!     --pool-relay ohio/dynamo/mocker-1=http://10.42.0.6:50051 \
//!     --peer http://10.42.1.5:8000
//! ```
//!
//! The core's own ZMQ mesh publishes on a port of its own instead, with
//! `--replica-sync-port 5600 --peer tcp://10.42.1.5:5600`. A frontend uses
//! one transport or the other: `tcp://` peers need `--replica-sync-port`,
//! and `http://` peers refuse it.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use dynamo_global_frontend::{
    Metrics, PoolSnapshot, app, follow_relays, measure_round_trips, serve,
};
use dynamo_global_router::{
    AffinityConfig, AffinitySelector, CorePolicy, CoreSelector, GrpcReplicaSync, InFlightTracker,
    LeastInFlightSelector, Pool, PoolSignals, ReplicaSync, ReplicaTransport, RequestSelector,
    RoundRobinSelector,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

/// How often the round trip to each pool is measured.
const ROUND_TRIP_INTERVAL: Duration = Duration::from_secs(30);
/// How often expired session pins are dropped.
const PIN_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

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
    /// `--selection core`: publish this replica's requests in flight and
    /// session pins over the core's own mesh on this port, for `tcp://`
    /// peers. Without it, `http://` peers subscribe on the API port.
    #[arg(long)]
    replica_sync_port: Option<u16>,
    /// `--selection core`: another Global Frontend to share requests in
    /// flight and session pins with. `http://host:port` is its API address,
    /// subscribed to over gRPC; `tcp://host:port` is its
    /// `--replica-sync-port` endpoint, subscribed to over the core's mesh.
    /// Repeat the flag for each peer; every peer must use one scheme.
    #[arg(long = "peer", value_parser = parse_peer)]
    peers: Vec<Peer>,
}

/// A replica peer, as one of the two transports dials it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Peer {
    /// A `--replica-sync-port` endpoint of the core's mesh.
    Mesh(String),
    /// An API address the gRPC stream is served on.
    Grpc(Url),
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

fn parse_peer(value: &str) -> Result<Peer, String> {
    if value.starts_with("http://") || value.starts_with("https://") {
        return dynamo_global_router::replica_sync::peer_url(value)
            .map(Peer::Grpc)
            .map_err(|error| error.to_string());
    }
    let Some(address) = value.strip_prefix("tcp://") else {
        return Err(format!(
            "expected http://host:port or tcp://host:port, got {value}"
        ));
    };
    match address.rsplit_once(':') {
        Some((host, port))
            if !host.is_empty() && port.parse::<u16>().is_ok_and(|port| port > 0) =>
        {
            Ok(Peer::Mesh(value.to_owned()))
        }
        _ => Err(format!("expected tcp://host:port, got {value}")),
    }
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

/// The replica transport `args` asks for, if any: the mesh with
/// `--replica-sync-port`, gRPC with `http://` peers alone.
fn replica_transport(args: &Args) -> Result<Option<ReplicaTransport>> {
    let mut mesh = Vec::new();
    let mut grpc = Vec::new();
    for peer in &args.peers {
        match peer {
            Peer::Mesh(endpoint) => mesh.push(endpoint.clone()),
            Peer::Grpc(url) => grpc.push(url.clone()),
        }
    }
    if !mesh.is_empty() && !grpc.is_empty() {
        bail!("--peer takes either tcp:// or http:// peers, not both: one transport per replica");
    }
    if !args.peers.is_empty() && args.selection != SelectionPath::Core {
        bail!("--peer needs --selection core");
    }
    match args.replica_sync_port {
        Some(_) if args.selection != SelectionPath::Core => {
            bail!("--replica-sync-port needs --selection core")
        }
        Some(_) if !grpc.is_empty() => bail!(
            "--replica-sync-port is the mesh's port; http:// peers use the gRPC transport, \
             which serves on the API port. Drop --replica-sync-port or give tcp:// peers"
        ),
        Some(port) => Ok(Some(ReplicaTransport::Mesh(ReplicaSync {
            port,
            peers: mesh,
        }))),
        None if !mesh.is_empty() => {
            bail!("tcp:// peers need --replica-sync-port, the port this replica publishes on")
        }
        None if !grpc.is_empty() => Ok(Some(ReplicaTransport::Grpc(GrpcReplicaSync {
            peers: grpc,
        }))),
        None => Ok(None),
    }
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

    /// The transport the command line `rest` asks for, after the pool.
    fn transport(rest: &[&str]) -> Result<Option<ReplicaTransport>> {
        let args = Args::try_parse_from([&["global-frontend", POOL], rest].concat())?;
        replica_transport(&args)
    }

    #[test]
    fn selection_defaults_to_legacy_and_the_core_takes_mesh_peers() {
        let args = Args::try_parse_from(["global-frontend", POOL]).unwrap();
        assert_eq!(args.selection, SelectionPath::Legacy);
        assert_eq!(replica_transport(&args).unwrap(), None);

        let args = Args::try_parse_from([
            "global-frontend",
            POOL,
            "--selection",
            "core",
            "--replica-sync-port",
            "5600",
            "--peer",
            "tcp://10.0.0.2:5600",
            "--peer",
            "tcp://10.0.0.3:5600",
        ])
        .unwrap();
        assert_eq!(args.selection, SelectionPath::Core);
        assert_eq!(
            replica_transport(&args).unwrap(),
            Some(ReplicaTransport::Mesh(ReplicaSync {
                port: 5600,
                peers: vec![
                    "tcp://10.0.0.2:5600".to_owned(),
                    "tcp://10.0.0.3:5600".to_owned()
                ],
            }))
        );
        // The mesh publishes on its own port whether or not it has peers yet.
        assert_eq!(
            transport(&["--selection", "core", "--replica-sync-port", "5600"]).unwrap(),
            Some(ReplicaTransport::Mesh(ReplicaSync {
                port: 5600,
                peers: Vec::new(),
            }))
        );
    }

    #[test]
    fn http_peers_select_the_grpc_transport_on_the_api_port() {
        let core = ["--selection", "core"];
        let peers = [
            "--peer",
            "http://10.0.0.2:8000",
            "--peer",
            "http://gf-b:8000",
        ];
        assert_eq!(
            transport(&[&core[..], &peers[..]].concat()).unwrap(),
            Some(ReplicaTransport::Grpc(GrpcReplicaSync {
                peers: vec![
                    "http://10.0.0.2:8000".parse().unwrap(),
                    "http://gf-b:8000".parse().unwrap(),
                ],
            }))
        );
        // The gRPC stream is served on the API port, not the mesh's.
        let with_port = [&core[..], &peers[..], &["--replica-sync-port", "5600"]].concat();
        assert!(transport(&with_port).is_err());
    }

    #[test]
    fn peers_need_the_core_and_one_transport() {
        // Any peer needs the core.
        assert!(transport(&["--peer", "http://10.0.0.2:8000"]).is_err());
        assert!(
            transport(&[
                "--peer",
                "tcp://10.0.0.2:5600",
                "--replica-sync-port",
                "5600"
            ])
            .is_err()
        );
        assert!(transport(&["--replica-sync-port", "5600"]).is_err());
        // A tcp:// peer needs the port this replica publishes on.
        assert!(transport(&["--selection", "core", "--peer", "tcp://10.0.0.2:5600"]).is_err());
        // One transport per replica.
        assert!(
            transport(&[
                "--selection",
                "core",
                "--replica-sync-port",
                "5600",
                "--peer",
                "tcp://10.0.0.2:5600",
                "--peer",
                "http://10.0.0.3:8000",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_peers_of_either_scheme() {
        assert_eq!(
            parse_peer("tcp://peer:5600"),
            Ok(Peer::Mesh("tcp://peer:5600".to_owned()))
        );
        assert_eq!(
            parse_peer("http://peer:8000"),
            Ok(Peer::Grpc("http://peer:8000".parse().unwrap()))
        );
        // TLS peers wait for the transport to carry TLS.
        assert!(parse_peer("https://peer:8000").is_err());
        assert!(parse_peer("tcp://10.0.0.2").is_err());
        assert!(parse_peer("tcp://:5600").is_err());
        assert!(parse_peer("http://").is_err());
        assert!(parse_peer("peer:8000").is_err());
    }

    #[test]
    fn requires_a_pool() {
        assert!(Args::try_parse_from(["global-frontend"]).is_err());
    }
}
